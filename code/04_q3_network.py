#!/usr/bin/env python3
"""Problem 3: Champion Network Structure & Team-Composition Communities.

Uses the pair tables produced by 03_q2_pairings.py to analyze the normalized
champion co-selection network with Louvain communities, weighted PageRank,
maximal cliques, clique percolation, Girvan-Newman, K-Means++, hierarchical
clustering, DBSCAN, heatmaps, and profile similarity.

Run from the project root:
    python code/04_q3_network.py --overwrite

Outputs:
    data/analysis/q3/tables/
    data/analysis/q3/figures/report/
    data/analysis/q3/figures/supplementary/
    data/analysis/q3/summary.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# Analysis thresholds control support; display limits only keep figures readable.
MIN_PAIR_GAMES = 500
STRONG_EDGE_QUANTILE = 0.80
MIN_CLUSTER_DEGREE = 3
K_RANGE = range(2, 7)
DBSCAN_MIN_SAMPLES = 4
NETWORK_NODES = 45
NETWORK_EDGES = 120
HEATMAP_CHAMPIONS = 30
TOP_N = 15
RANDOM_STATE = 67978


def project_root() -> Path:
    """Return the repository root inferred from this script location."""
    return Path(__file__).resolve().parents[1]


def prepare_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory, optionally replacing existing generated contents."""
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as CSV, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_plot(fig: plt.Figure, path: Path, dpi: int = 260) -> None:
    """Save and close a transparent report-quality Matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def largest_component(graph: nx.Graph) -> nx.Graph:
    """Return a copy of the graph’s largest connected component."""
    if graph.number_of_nodes() == 0:
        return graph.copy()
    nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(nodes).copy()


def load_q2_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Problem 2 champion, pair, and role tables required by Problem 3."""
    required = {
        "champion_stats.csv": input_dir / "champion_stats.csv",
        "pair_stats.csv": input_dir / "pair_stats.csv",
        "role_counts.csv": input_dir / "role_counts.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Problem 3 requires Problem 2 tables first. Run "
            "`python code/03_q2_pairings.py --overwrite`. Missing:\n  "
            + "\n  ".join(missing)
        )
    return (
        pd.read_csv(required["champion_stats.csv"]),
        pd.read_csv(required["pair_stats.csv"]),
        pd.read_csv(required["role_counts.csv"]),
    )


def build_graphs(champions: pd.DataFrame, pairs: pd.DataFrame) -> tuple[nx.Graph, nx.Graph]:
    """Build raw-frequency and supported positive-association graphs."""
    # Raw graph uses counts; normalized graph keeps supported positive-association edges.
    frequency = nx.Graph()
    association = nx.Graph()
    for row in champions.itertuples(index=False):
        frequency.add_node(int(row.champion_id), name=row.champion_name, pick_rate=float(row.pick_rate))
    for row in pairs.itertuples(index=False):
        a, b = int(row.champion_a_id), int(row.champion_b_id)
        frequency.add_edge(a, b, weight=float(row.games_together))
        if row.games_together >= MIN_PAIR_GAMES and row.association > 0:
            association.add_node(a, name=row.champion_a)
            association.add_node(b, name=row.champion_b)
            association.add_edge(
                a,
                b,
                weight=float(row.association),
                games=int(row.games_together),
                win_surplus_pp=float(row.win_surplus_pp),
            )
    return frequency, association


def centrality_table(
    champions: pd.DataFrame,
    frequency: nx.Graph,
    association: nx.Graph,
) -> pd.DataFrame:
    # Compare popularity-driven PageRank with normalized-association PageRank.
    """Compute weighted PageRank and weighted degree on raw and normalized graphs."""
    raw_page = nx.pagerank(frequency, weight="weight")
    assoc_core = largest_component(association)
    assoc_page = nx.pagerank(assoc_core, weight="weight") if assoc_core.number_of_nodes() else {}
    raw_degree = dict(frequency.degree(weight="weight"))
    assoc_degree = dict(association.degree(weight="weight"))

    out = champions[["champion_id", "champion_name", "pick_rate"]].copy()
    out["pagerank_frequency"] = out["champion_id"].map(raw_page).fillna(0.0)
    out["pagerank_association"] = out["champion_id"].map(assoc_page).fillna(0.0)
    out["degree_frequency"] = out["champion_id"].map(raw_degree).fillna(0.0)
    out["degree_association"] = out["champion_id"].map(assoc_degree).fillna(0.0)
    return out.sort_values("pagerank_association", ascending=False)


def louvain_communities(
    champions: pd.DataFrame,
    association: nx.Graph,
) -> tuple[pd.DataFrame, float, nx.Graph]:
    """Detect weighted Louvain communities on the largest association component."""
    core = largest_component(association)
    if core.number_of_nodes() == 0:
        raise RuntimeError("Association graph is empty; lower MIN_PAIR_GAMES or inspect Q2 pair statistics.")
    # Louvain finds dense weighted communities in the largest connected component.
    groups = nx.community.louvain_communities(core, weight="weight", seed=RANDOM_STATE)
    membership = {
        node: number
        for number, group in enumerate(sorted(groups, key=len, reverse=True), start=1)
        for node in group
    }
    out = champions[["champion_id", "champion_name"]].copy()
    out["community"] = out["champion_id"].map(membership).fillna(0).astype(int)
    modularity = nx.community.modularity(core, groups, weight="weight")
    return out, float(modularity), core


def girvan_newman(
    core: nx.Graph,
    centrality: pd.DataFrame,
) -> tuple[pd.DataFrame, float, nx.Graph]:
    """Run Girvan-Newman on the high-centrality core and keep the best modularity partition."""
    top_ids = set(centrality.nlargest(40, "pagerank_association")["champion_id"])
    graph = largest_component(core.subgraph(top_ids).copy())
    if graph.number_of_nodes() < 3:
        return pd.DataFrame(), np.nan, graph

    best_partition = [set(graph.nodes)]
    best_modularity = -np.inf
    # Girvan-Newman is limited to the high-centrality core because it is expensive.
    generator = nx.community.girvan_newman(graph)
    for _ in range(6):
        try:
            partition = [set(group) for group in next(generator)]
        except StopIteration:
            break
        score = nx.community.modularity(graph, partition, weight="weight")
        if score > best_modularity:
            best_modularity, best_partition = score, partition

    membership = {
        node: number
        for number, group in enumerate(sorted(best_partition, key=len, reverse=True), start=1)
        for node in group
    }
    out = pd.DataFrame({
        "champion_id": list(graph.nodes),
        "girvan_newman_community": [membership[node] for node in graph.nodes],
    })
    return out, float(best_modularity), graph


def clique_analysis(association: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame, nx.Graph]:
    """Find strong maximal cliques and overlapping 3-clique-percolation communities."""
    core = largest_component(association)
    weights = [data["weight"] for _, _, data in core.edges(data=True)]
    if not weights:
        return pd.DataFrame(), pd.DataFrame(), nx.Graph()

    # Keep the strongest edges, then search for fully connected champion groups.
    cutoff = float(np.quantile(weights, STRONG_EDGE_QUANTILE))
    strong = nx.Graph(
        (a, b, data)
        for a, b, data in core.edges(data=True)
        if data["weight"] >= cutoff
    )
    names = nx.get_node_attributes(core, "name")
    rows = []
    for group in nx.find_cliques(strong):
        if len(group) < 3:
            continue
        edges = [
            strong[a][b]
            for i, a in enumerate(group)
            for b in group[i + 1 :]
            if strong.has_edge(a, b)
        ]
        rows.append({
            "size": len(group),
            "champions": " | ".join(sorted(names[node] for node in group)),
            "mean_association": float(np.mean([edge["weight"] for edge in edges])),
            "minimum_games": int(min(edge["games"] for edge in edges)),
            "mean_win_surplus_pp": float(np.mean([edge["win_surplus_pp"] for edge in edges])),
        })
    cliques = pd.DataFrame(rows)
    if not cliques.empty:
        cliques = cliques.sort_values(["mean_association", "minimum_games"], ascending=False)

    overlap = []
    for number, group in enumerate(sorted(nx.community.k_clique_communities(strong, 3), key=len, reverse=True), start=1):
        overlap.append({
            "clique_community": number,
            "size": len(group),
            "champions": " | ".join(sorted(names[node] for node in group)),
        })
    return cliques, pd.DataFrame(overlap), strong


def profile_data(
    champions: pd.DataFrame,
    pairs: pd.DataFrame,
    association: nx.Graph,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build champion-by-champion co-pick profiles for clustering and similarity analysis."""
    active_ids = sorted(node for node, degree in association.degree() if degree >= MIN_CLUSTER_DEGREE)
    if len(active_ids) < 3:
        raise RuntimeError("Not enough well-supported champions for clustering.")

    # Each champion becomes a vector of normalized associations to other champions.
    index = {champion_id: i for i, champion_id in enumerate(active_ids)}
    matrix = np.zeros((len(active_ids), len(active_ids)), dtype=float)
    for row in pairs[pairs["games_together"] >= MIN_PAIR_GAMES].itertuples(index=False):
        a, b = index.get(int(row.champion_a_id)), index.get(int(row.champion_b_id))
        if a is None or b is None:
            continue
        matrix[a, b] = float(row.association)
        matrix[b, a] = float(row.association)

    info = (
        champions[champions["champion_id"].isin(active_ids)][["champion_id", "champion_name"]]
        .set_index("champion_id")
        .loc[active_ids]
        .reset_index()
    )
    return info, matrix, StandardScaler().fit_transform(matrix)


def clusterings(
    info: pd.DataFrame,
    scaled: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float]:
    # Pick K-Means k by silhouette, then compare hierarchical clustering and DBSCAN.
    """Run K-Means++, hierarchical clustering, DBSCAN, and a two-dimensional PCA projection."""
    scores, best_model, best_score = [], None, -np.inf
    max_k = min(max(K_RANGE), len(info) - 1)
    for k in [value for value in K_RANGE if value <= max_k]:
        model = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        scores.append({"k": k, "silhouette": float(score), "inertia": float(model.inertia_)})
        if score > best_score:
            best_score, best_model = score, model
    if best_model is None:
        raise RuntimeError("No valid K-Means model could be fitted.")

    best_k = int(best_model.n_clusters)
    hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(scaled)
    neighbours = min(DBSCAN_MIN_SAMPLES, len(info))
    distances = NearestNeighbors(n_neighbors=neighbours).fit(scaled).kneighbors(scaled)[0][:, -1]
    eps = float(np.quantile(distances, 0.85))
    dbscan = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(scaled)
    coords = PCA(n_components=2).fit_transform(scaled)

    assignments = info.copy()
    assignments["pca_1"] = coords[:, 0]
    assignments["pca_2"] = coords[:, 1]
    assignments["kmeans_cluster"] = best_model.labels_ + 1
    assignments["hierarchical_cluster"] = hierarchical + 1
    assignments["dbscan_cluster"] = dbscan
    return assignments, pd.DataFrame(scores), linkage(scaled, method="ward"), eps


def profile_similarity(info: pd.DataFrame, raw_profiles: np.ndarray) -> pd.DataFrame:
    # Cosine similarity compares whole teammate profiles, not direct pair strength.
    """Rank champion pairs by cosine similarity of their complete co-pick profiles."""
    similarity = cosine_similarity(raw_profiles)
    rows = [
        {
            "champion_a": info.iloc[i]["champion_name"],
            "champion_b": info.iloc[j]["champion_name"],
            "cosine_similarity": float(similarity[i, j]),
        }
        for i in range(len(info))
        for j in range(i + 1, len(info))
    ]
    out = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)
    out["pair"] = out["champion_a"] + " ↔ " + out["champion_b"]
    return out


def community_role_table(roles: pd.DataFrame, communities: pd.DataFrame) -> pd.DataFrame:
    """Summarize the role composition of each Louvain community as percentages."""
    valid_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    # Role shares make the unsupervised Louvain communities interpretable.
    merged = roles.merge(communities[["champion_id", "community"]], on="champion_id", how="left")
    merged = merged[(merged["community"] > 0) & merged["team_position"].isin(valid_roles)]
    table = merged.pivot_table(
        index="community",
        columns="team_position",
        values="appearances",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=valid_roles, fill_value=0)
    return table.div(table.sum(axis=1), axis=0) * 100.0


def association_heatmap_data(
    pairs: pd.DataFrame,
    centrality: pd.DataFrame,
    communities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    # Restrict the matrix to central champions and order them by community.
    """Build a normalized-association matrix for the most central community champions."""
    selected = (
        centrality.merge(communities[["champion_id", "community"]], on="champion_id", how="left")
        .query("community > 0")
        .nlargest(HEATMAP_CHAMPIONS, "pagerank_association")
        .sort_values(["community", "pagerank_association"], ascending=[True, False])
    )
    ids = selected["champion_id"].astype(int).tolist()
    names = selected.set_index("champion_id")["champion_name"]
    matrix = pd.DataFrame(0.0, index=ids, columns=ids)
    relevant = pairs[pairs["champion_a_id"].isin(ids) & pairs["champion_b_id"].isin(ids)]
    for row in relevant.itertuples(index=False):
        a, b = int(row.champion_a_id), int(row.champion_b_id)
        matrix.loc[a, b] = row.association
        matrix.loc[b, a] = row.association
    return matrix, names


def bar_plot(
    data: pd.DataFrame,
    label: str,
    value: str,
    title: str,
    xlabel: str,
    output: Path,
    top_n: int = TOP_N,
) -> None:
    """Create a reusable horizontal ranking plot for one network statistic."""
    top = data.nlargest(top_n, value).sort_values(value)
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(top))
    ax.barh(y, top[value], alpha=0.86)
    ax.set_yticks(y, top[label])
    for yi, number in zip(y, top[value]):
        text = f" {number:.4f}" if "pagerank" in value else f" {number:.2f}"
        ax.text(number, yi, text, va="center", fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    save_plot(fig, output)


def heatmap_plot(
    matrix: pd.DataFrame,
    title: str,
    output: Path,
    cmap: str,
    fmt: str | None = None,
    colorbar_label: str = "",
) -> None:
    """Render a labeled matrix heatmap with an optional diverging color scale."""
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * matrix.shape[1]), max(5, 0.55 * matrix.shape[0])))
    values = matrix.to_numpy(dtype=float)
    if cmap == "coolwarm":
        maximum = max(float(np.nanmax(np.abs(values))), 1e-9)
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-maximum, vmax=maximum)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    if fmt:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, format(values[row, col], fmt), ha="center", va="center", fontsize=8)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    save_plot(fig, output)


def community_network_plot(
    core: nx.Graph,
    centrality: pd.DataFrame,
    communities: pd.DataFrame,
    output: Path,
) -> None:
    # Display only central nodes/strong edges; the analysis itself uses the full graph.
    """Draw a readable high-centrality view of the Louvain association network."""
    top_ids = set(centrality.nlargest(NETWORK_NODES, "pagerank_association")["champion_id"])
    graph = core.subgraph(top_ids).copy()
    strongest = sorted(graph.edges(data=True), key=lambda edge: edge[2]["weight"], reverse=True)[:NETWORK_EDGES]
    display = nx.Graph()
    display.add_nodes_from(graph.nodes(data=True))
    display.add_edges_from(strongest)
    display.remove_nodes_from(list(nx.isolates(display)))

    page = dict(zip(centrality["champion_id"], centrality["pagerank_association"]))
    community = dict(zip(communities["champion_id"], communities["community"]))
    positions = nx.spring_layout(display, seed=RANDOM_STATE, weight="weight", iterations=300, k=1.0)
    weights = np.array([display.edges[edge]["weight"] for edge in display.edges], dtype=float)
    widths = 1.0 if len(weights) == 0 else 0.7 + 2.8 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(15, 11))
    nx.draw_networkx_edges(display, positions, width=widths, alpha=0.28, edge_color="#607089", ax=ax)
    nx.draw_networkx_nodes(
        display,
        positions,
        node_size=[450 + 85000 * page.get(node, 0.0) for node in display.nodes],
        node_color=[community.get(node, 0) for node in display.nodes],
        cmap="tab20",
        alpha=0.94,
        edgecolors="white",
        linewidths=0.8,
        ax=ax,
    )
    nx.draw_networkx_labels(
        display,
        positions,
        labels={node: display.nodes[node]["name"] for node in display.nodes},
        font_size=8.5,
        font_weight="bold",
        ax=ax,
    )
    # One title only: avoids the old overlapping title/subtitle problem.
    ax.set_title("Champion co-selection communities", loc="left", fontsize=18, fontweight="bold", pad=16)
    ax.axis("off")
    save_plot(fig, output)


def girvan_newman_plot(graph: nx.Graph, memberships: pd.DataFrame, output: Path) -> None:
    """Draw the Girvan-Newman partition on the selected high-centrality core."""
    if graph.number_of_nodes() == 0 or memberships.empty:
        return
    groups = dict(zip(memberships["champion_id"], memberships["girvan_newman_community"]))
    positions = nx.spring_layout(graph, seed=RANDOM_STATE, weight="weight", iterations=250)
    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(graph, positions, alpha=0.22, edge_color="#607089", ax=ax)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[groups[node] for node in graph.nodes],
        cmap="tab20",
        node_size=700,
        edgecolors="white",
        linewidths=0.7,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, labels=nx.get_node_attributes(graph, "name"), font_size=8, font_weight="bold", ax=ax)
    ax.set_title("Girvan-Newman communities on the high-centrality core", loc="left", fontsize=15, fontweight="bold")
    ax.axis("off")
    save_plot(fig, output)


def cluster_scatter(assignments: pd.DataFrame, column: str, title: str, output: Path) -> None:
    """Plot PCA coordinates colored by one clustering assignment."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(
        assignments["pca_1"],
        assignments["pca_2"],
        c=assignments[column],
        cmap="tab10",
        s=58,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.6,
    )
    distance = assignments["pca_1"] ** 2 + assignments["pca_2"] ** 2
    for row in assignments.loc[distance.nlargest(18).index].itertuples(index=False):
        ax.annotate(row.champion_name, (row.pca_1, row.pca_2), xytext=(4, 4), textcoords="offset points", fontsize=7.8)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    save_plot(fig, output)


def kmeans_selection_plots(scores: pd.DataFrame, output_dir: Path) -> None:
    """Plot K-Means inertia and silhouette scores across candidate k values."""
    for value, ylabel, title, filename in [
        ("inertia", "Within-cluster SSE / inertia", "K-Means elbow plot", "kmeans_elbow.png"),
        ("silhouette", "Silhouette score", "K-Means relative validation", "kmeans_silhouette.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(scores["k"], scores[value], marker="o")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        save_plot(fig, output_dir / filename)


def dendrogram_plot(hierarchy: np.ndarray, names: list[str], output: Path) -> None:
    """Plot the Ward hierarchical-clustering dendrogram for champion profiles."""
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(hierarchy, labels=names, leaf_rotation=90, leaf_font_size=6, ax=ax)
    ax.set_title("Hierarchical clustering of champion co-pick profiles", loc="left", fontsize=15, fontweight="bold")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    save_plot(fig, output)


def parse_args() -> argparse.Namespace:
    """Parse Problem 2 input-table and Problem 3 output paths."""
    root = project_root()
    p = argparse.ArgumentParser(description="Run Problem 3 champion-network analysis.")
    p.add_argument("--q2-tables", type=Path, default=root / "data/analysis/q2/tables")
    p.add_argument("--output", type=Path, default=root / "data/analysis/q3")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    """Run the complete Problem 3 analysis.

    Loads Q2 pair tables, builds the champion association network, runs graph and
    clustering comparisons, then saves report/supplementary outputs and a summary.
    """
    # 1) Prepare outputs, then load the compact tables produced by Problem 2.
    args = parse_args()
    prepare_dir(args.output, args.overwrite)
    tables = args.output / "tables"
    report = args.output / "figures/report"
    supplementary = args.output / "figures/supplementary"
    for path in (tables, report, supplementary):
        path.mkdir(parents=True, exist_ok=True)

    # Q3 starts from Q2 tables, so the large participant corpus is not scanned again.
    # 2) Build the weighted co-selection graph and graph-based structures.
    print("[Q3] Loading Problem 2 tables...", flush=True)
    champions, pairs, roles = load_q2_tables(args.q2_tables)
    print("[Q3] Building graph and communities...", flush=True)
    frequency, association = build_graphs(champions, pairs)
    centrality = centrality_table(champions, frequency, association)
    communities, modularity, core = louvain_communities(champions, association)
    girvan, girvan_modularity, girvan_graph = girvan_newman(core, centrality)
    cliques, clique_overlap, _ = clique_analysis(association)

    # 3) Compare graph communities with conventional profile-based clustering.
    print("[Q3] Running clustering comparisons...", flush=True)
    info, raw_profiles, scaled_profiles = profile_data(champions, pairs, association)
    assignments, k_scores, hierarchy, dbscan_eps = clusterings(info, scaled_profiles)
    similarity = profile_similarity(info, raw_profiles)
    role_table = community_role_table(roles, communities)
    assoc_matrix, assoc_names = association_heatmap_data(pairs, centrality, communities)

    # Save method outputs first; figures below are visual summaries of these tables.
    # 4) Save numerical method outputs before rendering their visual summaries.
    save_csv(centrality, tables / "centrality.csv")
    save_csv(communities, tables / "louvain_communities.csv")
    save_csv(girvan, tables / "girvan_newman_communities.csv")
    save_csv(cliques, tables / "maximal_cliques.csv")
    save_csv(clique_overlap, tables / "clique_percolation_communities.csv")
    save_csv(assignments, tables / "cluster_assignments.csv")
    save_csv(k_scores, tables / "kmeans_model_selection.csv")
    save_csv(similarity.head(100), tables / "profile_similarity.csv")
    save_csv(role_table.reset_index(), tables / "community_role_percent.csv")

    # Main report figures: communities -> roles -> PageRank -> higher-order cliques.
    community_network_plot(core, centrality, communities, report / "figure_8_louvain_community_network.png")
    role_display = role_table.rename(columns={"TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid", "BOTTOM": "ADC", "UTILITY": "Support"})
    role_display.index = [f"C{value}" for value in role_display.index]
    heatmap_plot(
        role_display,
        "Role composition of Louvain communities",
        report / "figure_9_community_role_heatmap.png",
        cmap="viridis",
        fmt=".0f",
        colorbar_label="Share of community appearances (%)",
    )
    bar_plot(
        centrality[centrality["pagerank_association"] > 0],
        "champion_name",
        "pagerank_association",
        "Most central champions in the normalized association network",
        "Association-weighted PageRank",
        report / "figure_10_pagerank_association.png",
    )
    if not cliques.empty:
        bar_plot(
            cliques,
            "champions",
            "mean_association",
            "Strongest tightly connected champion cliques",
            "Mean normalized association",
            report / "figure_11_strongest_cliques.png",
            top_n=12,
        )

    # Supplementary / method-comparison figures.
    assoc_display = assoc_matrix.copy()
    assoc_display.index = [assoc_names.loc[x] for x in assoc_display.index]
    assoc_display.columns = assoc_display.index
    heatmap_plot(
        assoc_display,
        "Normalized co-pick association among central champions",
        supplementary / "association_heatmap.png",
        cmap="coolwarm",
        colorbar_label="log2(lift)",
    )
    if not clique_overlap.empty:
        overlap = clique_overlap.assign(community_label="CP" + clique_overlap["clique_community"].astype(str))
        bar_plot(overlap, "community_label", "size", "Overlapping 3-clique communities", "Champions in clique-percolation community", supplementary / "clique_percolation.png", top_n=12)
    kmeans_selection_plots(k_scores, supplementary)
    cluster_scatter(assignments, "kmeans_cluster", "K-Means++ clusters of champion co-pick profiles", supplementary / "kmeans_clusters.png")
    cluster_scatter(assignments, "hierarchical_cluster", "Agglomerative hierarchical clusters of champion co-pick profiles", supplementary / "hierarchical_clusters.png")
    dendrogram_plot(hierarchy, assignments["champion_name"].tolist(), supplementary / "hierarchical_dendrogram.png")
    cluster_scatter(assignments, "dbscan_cluster", "DBSCAN clusters and noise in champion co-pick profiles", supplementary / "dbscan_clusters.png")
    girvan_newman_plot(girvan_graph, girvan, supplementary / "girvan_newman_network.png")
    bar_plot(similarity, "pair", "cosine_similarity", "Champions with the most similar co-pick profiles", "Cosine similarity", supplementary / "profile_similarity.png")
    bar_plot(centrality, "champion_name", "pagerank_frequency", "Most central champions in the raw co-pick network", "Weighted PageRank", supplementary / "pagerank_frequency.png")

    # 6) Persist one compact summary of graph structure and clustering results.
    dbscan_labels = assignments["dbscan_cluster"]
    summary = {
        "problem": "Champion Network Structure & Team-Composition Communities",
        "minimum_pair_games": MIN_PAIR_GAMES,
        "association_graph": {
            "nodes": association.number_of_nodes(),
            "edges": association.number_of_edges(),
            "components": nx.number_connected_components(association) if association.number_of_nodes() else 0,
            "largest_component": core.number_of_nodes(),
        },
        "louvain": {
            "communities": int(communities.query("community > 0")["community"].nunique()),
            "modularity": modularity,
        },
        "girvan_newman": {
            "communities": int(girvan["girvan_newman_community"].nunique()) if not girvan.empty else 0,
            "modularity": girvan_modularity,
        },
        "cliques": {
            "maximal_cliques_size_3plus": int(len(cliques)),
            "clique_percolation_communities": int(len(clique_overlap)),
        },
        "clustering": {
            "champions_used": int(len(assignments)),
            "kmeans_selected_k": int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"]),
            "dbscan_eps": dbscan_eps,
            "dbscan_clusters": len(set(dbscan_labels) - {-1}),
            "dbscan_noise_points": int((dbscan_labels == -1).sum()),
        },
        "report_figures": [
            "figure_8_louvain_community_network.png",
            "figure_9_community_role_heatmap.png",
            "figure_10_pagerank_association.png",
            "figure_11_strongest_cliques.png",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nQ3 COMPLETE")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

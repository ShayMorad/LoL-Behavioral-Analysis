#!/usr/bin/env python3
"""Problem 2: Champion Pairings & Combo Performance.

Build same-team champion pairs from the canonical participant Parquet files,
separate raw popularity from normalized co-selection (log2 lift), and compare
co-selection strength with descriptive pair performance.

Run from the project root:
    python code/03_q2_pairings.py --overwrite

Outputs:
    data/analysis/q2/tables/
    data/analysis/q2/figures/report/
    data/analysis/q2/figures/supplementary/
    data/analysis/q2/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Q2 uses the full canonical corpus, not the tracked-player subset from Q1.
REGION_FOLDERS = {"NA": "full_na", "KR": "full_kr", "EU": "full_eu"}
QUEUE_ID = 420
MIN_DURATION_S = 600
MIN_PAIR_GAMES = 500
MIN_PERFORMANCE_GAMES = 1000
TOP_N = 15


def project_root() -> Path:
    """Return the repository root inferred from this script location."""
    return Path(__file__).resolve().parents[1]


def sql_path(path: Path) -> str:
    """Convert a filesystem path to a DuckDB-safe absolute POSIX string."""
    return path.resolve().as_posix().replace("'", "''")


def prepare_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory, optionally replacing existing generated contents."""
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parquet_glob(path: Path) -> str:
    """Validate a Parquet directory and return its DuckDB glob."""
    if not path.exists() or not any(path.glob("*.parquet")):
        raise FileNotFoundError(f"No Parquet files found in: {path}")
    return sql_path(path / "*.parquet")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as CSV, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_plot(fig: plt.Figure, path: Path, dpi: int = 260) -> None:
    """Save and close a transparent report-quality Matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def load_valid_players(con: duckdb.DuckDBPyConnection, processed: Path) -> None:
    """Materialize valid five-champion Ranked Solo/Duo teams across all regions."""
    queries = []
    for source, folder in REGION_FOLDERS.items():
        participants = parquet_glob(processed / folder / "participants")
        queries.append(
            f"""
            SELECT
                '{source}' AS source,
                match_id,
                team_id,
                CAST(win AS BOOLEAN) AS win,
                CAST(champion_id AS BIGINT) AS champion_id,
                CAST(champion_name AS VARCHAR) AS champion_name,
                COALESCE(NULLIF(CAST(team_position AS VARCHAR), ''), 'UNKNOWN') AS team_position
            FROM read_parquet('{participants}', union_by_name=true)
            WHERE queue_id = {QUEUE_ID}
              AND game_duration_s >= {MIN_DURATION_S}
              AND champion_id IS NOT NULL
              AND team_id IS NOT NULL
              AND win IS NOT NULL
            """
        )

    # Union all regions, then keep only complete five-champion teams.
    con.execute("DROP TABLE IF EXISTS players_raw")
    con.execute(f"CREATE TEMP TABLE players_raw AS {' UNION ALL '.join(queries)}")
    con.execute(
        """
        CREATE TEMP TABLE valid_teams AS
        SELECT source, match_id, team_id
        FROM players_raw
        GROUP BY source, match_id, team_id
        HAVING COUNT(*) = 5
           AND COUNT(DISTINCT champion_id) = 5
           AND COUNT(DISTINCT win) = 1
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE players AS
        SELECT p.*
        FROM players_raw p
        INNER JOIN valid_teams v USING (source, match_id, team_id)
        """
    )


def champion_stats(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute champion appearance counts, pick rates, and observed win rates."""
    teams = int(con.execute("SELECT COUNT(*) FROM valid_teams").fetchone()[0])
    out = con.execute(
        """
        SELECT
            champion_id,
            ANY_VALUE(champion_name) AS champion_name,
            COUNT(*)::BIGINT AS appearances,
            AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) AS win_rate
        FROM players
        GROUP BY champion_id
        ORDER BY appearances DESC
        """
    ).fetchdf()
    out["pick_rate"] = out["appearances"] / float(teams)
    return out


def role_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Count each champion’s observed appearances by team position."""
    return con.execute(
        """
        SELECT
            champion_id,
            ANY_VALUE(champion_name) AS champion_name,
            team_position,
            COUNT(*)::BIGINT AS appearances
        FROM players
        GROUP BY champion_id, team_position
        ORDER BY champion_id, team_position
        """
    ).fetchdf()


def pair_stats(
    con: duckdb.DuckDBPyConnection,
    champions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pair support, log2 lift, pair win rate, and descriptive win surplus."""
    teams = int(con.execute("SELECT COUNT(*) FROM valid_teams").fetchone()[0])
    # Self-join each team once per unordered champion pair (5 champions -> 10 pairs).
    pairs = con.execute(
        """
        SELECT
            a.champion_id AS champion_a_id,
            ANY_VALUE(a.champion_name) AS champion_a,
            b.champion_id AS champion_b_id,
            ANY_VALUE(b.champion_name) AS champion_b,
            COUNT(*)::BIGINT AS games_together,
            AVG(CASE WHEN a.win THEN 1.0 ELSE 0.0 END) AS pair_win_rate
        FROM players a
        INNER JOIN players b
            ON a.source = b.source
           AND a.match_id = b.match_id
           AND a.team_id = b.team_id
           AND a.champion_id < b.champion_id
        GROUP BY a.champion_id, b.champion_id
        """
    ).fetchdf()

    base = champions[["champion_id", "appearances", "win_rate"]]
    pairs = pairs.merge(
        base.rename(columns={
            "champion_id": "champion_a_id",
            "appearances": "appearances_a",
            "win_rate": "win_rate_a",
        }),
        on="champion_a_id",
        validate="many_to_one",
    ).merge(
        base.rename(columns={
            "champion_id": "champion_b_id",
            "appearances": "appearances_b",
            "win_rate": "win_rate_b",
        }),
        on="champion_b_id",
        validate="many_to_one",
    )

    # Lift = observed pair count / expected count under independent champion picks.
    expected = pairs["appearances_a"] * pairs["appearances_b"] / float(teams)
    pairs["lift"] = pairs["games_together"] / expected
    pairs["association"] = np.log2(pairs["lift"].clip(lower=1e-12))
    # Win surplus is descriptive: pair win rate minus average individual win rate.
    baseline = (pairs["win_rate_a"] + pairs["win_rate_b"]) / 2.0
    pairs["win_surplus_pp"] = 100.0 * (pairs["pair_win_rate"] - baseline)
    pairs["pair"] = pairs["champion_a"] + " + " + pairs["champion_b"]
    return pairs.sort_values("games_together", ascending=False)


def pair_rankings_plot(pairs: pd.DataFrame, output: Path) -> None:
    """Report Figure 6: raw frequency beside normalized co-selection."""
    # Left: most common pairs. Right: most above expectation after normalization.
    raw = pairs.nlargest(TOP_N, "games_together").sort_values("games_together")
    norm = (
        pairs[pairs["games_together"] >= MIN_PERFORMANCE_GAMES]
        .nlargest(TOP_N, "association")
        .sort_values("association")
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, data, value, title, xlabel in [
        (axes[0], raw, "games_together", "Most common champion combinations", "Games together"),
        (axes[1], norm, "association", "Strongest normalized co-pick associations", "log2(lift)"),
    ]:
        y = np.arange(len(data))
        ax.barh(y, data[value], alpha=0.86)
        ax.set_yticks(y, data["pair"])
        for yi, number in zip(y, data[value]):
            text = f" {int(number):,}" if value == "games_together" else f" {number:.2f}"
            ax.text(number, yi, text, va="center", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=10)
        ax.grid(axis="x", alpha=0.15)
        ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        "Raw popularity and normalized association identify different champion pairs",
        x=0.02,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_plot(fig, output)


def combo_landscape_plot(pairs: pd.DataFrame, output: Path) -> None:
    """Report Figure 7: normalized association versus descriptive pair win surplus."""
    # Only well-supported pairs enter the association/performance landscape.
    data = pairs[pairs["games_together"] >= MIN_PAIR_GAMES].copy()
    fig, ax = plt.subplots(figsize=(11, 8))
    points = ax.scatter(
        data["association"],
        data["win_surplus_pp"],
        c=np.log10(data["games_together"]),
        cmap="viridis",
        s=34,
        alpha=0.68,
        edgecolor="none",
    )

    interesting = pd.concat([
        data.nlargest(4, "association"),
        data[data["games_together"] >= MIN_PERFORMANCE_GAMES].nlargest(4, "win_surplus_pp"),
        data[data["games_together"] >= MIN_PERFORMANCE_GAMES].nsmallest(4, "win_surplus_pp"),
    ]).drop_duplicates("pair")
    for row in interesting.itertuples(index=False):
        ax.annotate(row.pair, (row.association, row.win_surplus_pp), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.axhline(0, color="#666666", ls="--", lw=1)
    ax.axvline(0, color="#666666", ls="--", lw=1)
    ax.set_xlabel("Normalized co-pick association: log2(lift)")
    ax.set_ylabel("Pair win surplus (percentage points)")
    ax.set_title("Champion combo landscape", loc="left", fontsize=16, fontweight="bold", pad=12)
    fig.colorbar(points, ax=ax, label="log10(games together)")
    ax.grid(alpha=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    save_plot(fig, output)


def win_surplus_plot(pairs: pd.DataFrame, output: Path) -> None:
    """Plot high-support champion pairs with the largest descriptive win surplus."""
    eligible = pairs[pairs["games_together"] >= MIN_PERFORMANCE_GAMES]
    top = eligible.nlargest(TOP_N, "win_surplus_pp").sort_values("win_surplus_pp")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["pair"], top["win_surplus_pp"], alpha=0.86)
    ax.axvline(0, color="#666666", ls="--", lw=1)
    ax.set_xlabel("Win surplus (percentage points)")
    ax.set_title("High-support pairs with the largest descriptive win surplus", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    save_plot(fig, output)


def parse_args() -> argparse.Namespace:
    """Parse canonical processed-data paths, output location, and DuckDB resource settings."""
    root = project_root()
    p = argparse.ArgumentParser(description="Run Problem 2 champion-pair analysis.")
    p.add_argument("--processed", type=Path, default=root / "data/processed")
    p.add_argument("--output", type=Path, default=root / "data/analysis/q2")
    p.add_argument("--duckdb-memory-limit", default="4GB")
    p.add_argument("--duckdb-threads", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    """Run the complete Problem 2 analysis.

    Builds valid team/pair statistics from canonical participants, saves the
    reusable Q2 tables, and generates the pair-ranking and combo-performance figures.
    """
    # 1) Prepare clean output folders and resource settings.
    args = parse_args()
    prepare_dir(args.output, args.overwrite)
    tables = args.output / "tables"
    report = args.output / "figures/report"
    supplementary = args.output / "figures/supplementary"
    for path in (tables, report, supplementary):
        path.mkdir(parents=True, exist_ok=True)

    # DuckDB counts pairs directly from Parquet without loading the full corpus into pandas.
    # 2) Let DuckDB scan the canonical Parquet corpus without loading it all into RAM.
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{args.duckdb_memory_limit}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")
    try:
        print("[Q2] Loading valid Ranked Solo/Duo teams...", flush=True)
        load_valid_players(con, args.processed)
        teams = int(con.execute("SELECT COUNT(*) FROM valid_teams").fetchone()[0])
        print("[Q2] Computing champion and pair statistics...", flush=True)
        champions = champion_stats(con)
        roles = role_counts(con)
        pairs = pair_stats(con, champions)
    finally:
        con.close()

    # These tables are also the input contract for Problem 3.
    # 3) Save reusable pair/champion/role tables; Problem 3 consumes these directly.
    save_csv(champions, tables / "champion_stats.csv")
    save_csv(roles, tables / "role_counts.csv")
    save_csv(pairs, tables / "pair_stats.csv")

    # Main report figures answer Q2; the extra win-surplus ranking stays supplementary.
    # 4) Generate the two main report figures plus one supplementary ranking.
    pair_rankings_plot(pairs, report / "figure_6_pair_rankings.png")
    combo_landscape_plot(pairs, report / "figure_7_combo_landscape.png")
    win_surplus_plot(pairs, supplementary / "pair_win_surplus.png")

    # 5) Record the analysis scope and output counts for reproducibility.
    summary = {
        "problem": "Champion Pairings & Combo Performance",
        "queue_id": QUEUE_ID,
        "minimum_duration_seconds": MIN_DURATION_S,
        "valid_teams": teams,
        "champions": int(len(champions)),
        "champion_pairs": int(len(pairs)),
        "minimum_pair_games_for_landscape": MIN_PAIR_GAMES,
        "minimum_pair_games_for_ranked_performance": MIN_PERFORMANCE_GAMES,
        "report_figures": ["figure_6_pair_rankings.png", "figure_7_combo_landscape.png"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nQ2 COMPLETE")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

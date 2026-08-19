#!/usr/bin/env python3
r"""
07_statistical_analysis.py

Formal within-player statistical analysis for the main behavioral question:

    How are session depth, recent ranked-game volume, and post-loss requeue
    timing associated with performance in a player's subsequent Solo/Duo match?

The script uses a target-centric timeline produced by 05_build_player_timelines.py.

PRIMARY DESIGN
--------------
Outcome:
    target_win (binary, modeled with a Linear Probability Model for direct
    percentage-point interpretation).

Primary sample:
    - queue 420 targets (already guaranteed by the timeline table)
    - target duration >= 10 minutes
    - at least one prior observed ranked match

Main exposures:
    H1: ranked session depth using a 30-minute boundary
    H2: requeue gap after a valid previous ranked loss
    H3: observed ranked games in previous 6 hours

Adjustment:
    - player fixed effects via within-player demeaning
    - dynamic pre-target history controls
    - patch controls
    - complementary behavioral controls

Uncertainty:
    Two-way cluster-robust covariance by:
        1. player_id
        2. physical match_id

This explicitly acknowledges both repeated measurements of the same player and
the fact that multiple tracked players may appear in the same physical match.

SENSITIVITY
-----------
H1 session thresholds: 45 / 60 / 90 minutes
H3 recent-volume windows: 3 / 12 / 24 hours
H2 also receives a continuous log-gap specification.

IMPORTANT
---------
This is observational association, not causal inference.
target_* performance columns are outcomes and are never predictors.

Outputs:
    data/analysis/statistics/
    ├── tables/
    │   ├── model_summary.csv
    │   ├── behavior_effects.csv
    │   └── all_coefficients.csv
    ├── figures/
    │   ├── H1_session_depth_adjusted_effects.png
    │   ├── H2_post_loss_requeue_adjusted_effects.png
    │   └── H3_recent_volume_adjusted_effects.png
    └── audit/
        └── statistical_analysis_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_SESSION_THRESHOLD = 30
SESSION_SENSITIVITY = (45, 60, 90)

PRIMARY_VOLUME_WINDOW = 6
VOLUME_SENSITIVITY = (3, 12, 24)

SESSION_CAP = 8
VOLUME_CAP = 6

POST_LOSS_BINS = (
    (-np.inf, 5, "<=5m"),
    (5, 10, "5-10m"),
    (10, 20, "10-20m"),
    (20, 30, "20-30m"),
    (30, 60, "30-60m"),
    (60, 120, "1-2h"),
    (120, 360, "2-6h"),
    (360, 1440, "6-24h"),
    (1440, np.inf, ">24h"),
)

REGIONS = ("NA", "KR", "EU")


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {path}. Use --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--timelines",
        type=Path,
        required=True,
        help="Folder containing regional Solo420 target Parquet files.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duckdb-memory-limit", default="4GB")
    p.add_argument("--duckdb-threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def region_file(folder: Path, source: str) -> Path:
    p = folder / f"{source}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing timeline file: {p}")
    return p


def normal_pvalue(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def factor_codes(values: Sequence) -> tuple[np.ndarray, np.ndarray]:
    codes, uniques = pd.factorize(values, sort=False)
    if (codes < 0).any():
        raise ValueError("Cluster/fixed-effect groups contain missing values.")
    return codes.astype(np.int64), np.asarray(uniques)


def within_demean(
    y: np.ndarray,
    X: np.ndarray,
    group_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove player-specific means without creating thousands of dummy columns.
    Equivalent to including a fixed intercept for every player in OLS.
    """
    n_groups = int(group_codes.max()) + 1
    counts = np.bincount(group_codes, minlength=n_groups).astype(float)

    y_sums = np.bincount(group_codes, weights=y, minlength=n_groups)
    y_mean = y_sums / counts
    yw = y - y_mean[group_codes]

    Xw = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        sums = np.bincount(
            group_codes, weights=X[:, j], minlength=n_groups
        )
        means = sums / counts
        Xw[:, j] = X[:, j] - means[group_codes]

    return yw, Xw


def cluster_covariance(
    X: np.ndarray,
    resid: np.ndarray,
    bread: np.ndarray,
    codes: np.ndarray,
    p: int,
) -> np.ndarray:
    """
    One-way CR0 covariance with a standard finite-sample cluster correction.
    """
    n = X.shape[0]
    g = int(codes.max()) + 1

    scores = X * resid[:, None]
    aggregated = np.zeros((g, p), dtype=float)
    np.add.at(aggregated, codes, scores)

    meat = aggregated.T @ aggregated

    if g > 1 and n > p:
        correction = (g / (g - 1.0)) * ((n - 1.0) / (n - p))
    else:
        correction = 1.0

    return correction * (bread @ meat @ bread)


def fit_within_player_lpm(
    df: pd.DataFrame,
    X: pd.DataFrame,
    *,
    source: str,
    hypothesis: str,
    specification: str,
    parameter_setting: str,
    behavior_columns: set[str],
    reference: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Player-fixed-effect linear probability model with two-way clustered SEs.

    Two-way covariance:
        V(player) + V(match) - V(player x match)

    Since each (player, match) row is unique, the intersection cluster is
    effectively the observation-level robust covariance.
    """
    if len(df) != len(X):
        raise ValueError("Data and design matrix lengths differ.")

    work = pd.concat(
        [
            df[
                ["player_id", "match_id", "target_win"]
            ].reset_index(drop=True),
            X.reset_index(drop=True),
        ],
        axis=1,
    ).dropna()

    if work.empty:
        raise RuntimeError(f"No complete rows for {source}/{hypothesis}.")

    y = work["target_win"].astype(float).to_numpy()
    design_names = list(X.columns)
    Xm = work[design_names].astype(float).to_numpy()

    player_codes, player_levels = factor_codes(work["player_id"])
    match_codes, match_levels = factor_codes(work["match_id"])

    # Audit uniqueness at the exact regression sample level.
    if work.duplicated(["player_id", "match_id"]).any():
        raise RuntimeError(
            f"{source}/{hypothesis}: duplicate player-match rows in regression sample."
        )

    yw, Xw = within_demean(y, Xm, player_codes)

    # Drop columns with no within-player variation.
    within_ss = np.sum(Xw * Xw, axis=0)
    keep = within_ss > 1e-12

    dropped = [name for name, flag in zip(design_names, keep) if not flag]
    if dropped:
        design_names = [name for name, flag in zip(design_names, keep) if flag]
        Xw = Xw[:, keep]

    n, p = Xw.shape
    if p == 0:
        raise RuntimeError(f"{source}/{hypothesis}: no estimable predictors.")

    beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    if rank < p:
        raise RuntimeError(
            f"{source}/{hypothesis}: rank-deficient design ({rank} < {p})."
        )

    resid = yw - Xw @ beta
    xtx = Xw.T @ Xw
    bread = np.linalg.pinv(xtx)

    v_player = cluster_covariance(
        Xw, resid, bread, player_codes, p
    )
    v_match = cluster_covariance(
        Xw, resid, bread, match_codes, p
    )

    # Unique observation-level intersection of player and physical match.
    obs_codes = np.arange(n, dtype=np.int64)
    v_obs = cluster_covariance(
        Xw, resid, bread, obs_codes, p
    )

    cov = v_player + v_match - v_obs
    variances = np.diag(cov)
    variances = np.where(variances >= 0, variances, np.nan)
    se = np.sqrt(variances)

    sst = float(np.sum(yw * yw))
    sse = float(np.sum(resid * resid))
    within_r2 = 1.0 - sse / sst if sst > 0 else np.nan

    rows = []
    for name, b, s in zip(design_names, beta, se):
        z = b / s if s and np.isfinite(s) and s > 0 else np.nan
        rows.append(
            {
                "source": source,
                "hypothesis": hypothesis,
                "specification": specification,
                "parameter_setting": parameter_setting,
                "term": name,
                "estimate": float(b),
                "estimate_percentage_points": float(100.0 * b),
                "std_error": float(s) if np.isfinite(s) else np.nan,
                "ci_low": float(b - 1.959963984540054 * s)
                if np.isfinite(s)
                else np.nan,
                "ci_high": float(b + 1.959963984540054 * s)
                if np.isfinite(s)
                else np.nan,
                "ci_low_percentage_points": float(
                    100.0 * (b - 1.959963984540054 * s)
                )
                if np.isfinite(s)
                else np.nan,
                "ci_high_percentage_points": float(
                    100.0 * (b + 1.959963984540054 * s)
                )
                if np.isfinite(s)
                else np.nan,
                "z_value": float(z) if np.isfinite(z) else np.nan,
                "p_value_two_sided": normal_pvalue(z),
                "is_behavior_term": name in behavior_columns,
                "reference_category": reference if name in behavior_columns else "",
            }
        )

    coef_df = pd.DataFrame(rows)

    summary = pd.DataFrame(
        [
            {
                "source": source,
                "hypothesis": hypothesis,
                "specification": specification,
                "parameter_setting": parameter_setting,
                "n_rows": int(n),
                "n_players": int(len(player_levels)),
                "n_physical_matches": int(len(match_levels)),
                "n_predictors_after_within_transform": int(p),
                "within_r2": within_r2,
                "dropped_no_within_variation": ";".join(dropped),
            }
        ]
    )

    return coef_df, summary


def ordered_dummies(
    series: pd.Series,
    categories: Sequence[str],
    reference: str,
    prefix: str,
) -> tuple[pd.DataFrame, set[str]]:
    cat = pd.Categorical(series, categories=categories, ordered=True)
    d = pd.get_dummies(cat, prefix=prefix, dtype=float)

    # pd.get_dummies(Pandas Categorical) creates a fresh RangeIndex.
    # Preserve the source Series index so filtered regression samples
    # (especially the post-loss subset) stay perfectly aligned.
    d.index = series.index

    ref_col = f"{prefix}_{reference}"
    if ref_col not in d.columns:
        raise RuntimeError(f"Reference dummy not present: {ref_col}")

    d = d.drop(columns=[ref_col])
    return d, set(d.columns)


def patch_dummies(series: pd.Series) -> pd.DataFrame:
    values = series.astype(str)
    categories = sorted(values.dropna().unique().tolist())
    cat = pd.Categorical(values, categories=categories)
    d = pd.get_dummies(cat, prefix="patch", dtype=float)

    # Preserve the original row index for safe concatenation with controls
    # after filtering to a subset of observations.
    d.index = series.index

    if len(d.columns) <= 1:
        return pd.DataFrame(index=series.index)

    # Reference = lexicographically first observed patch.
    return d.iloc[:, 1:]


def common_controls(
    df: pd.DataFrame,
    *,
    include_prev_win: bool,
    include_gap: bool,
    session_control_col: str | None = None,
    volume_control_col: str | None = None,
) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)

    x["prior_win_rate"] = df["prior_ranked_win_rate"].astype(float)
    x["log1p_prior_matches"] = np.log1p(
        df["prior_ranked_matches"].astype(float)
    )
    x["log1p_prev_kda"] = np.log1p(
        df["prev_ranked_kda"].clip(lower=0).astype(float)
    )
    x["log1p_prev_loss_streak"] = np.log1p(
        df["prev_ranked_loss_streak"].fillna(0).clip(lower=0).astype(float)
    )
    x["prev_queue_was_flex440"] = (
        df["prev_ranked_queue_id"] == 440
    ).astype(float)

    if include_prev_win:
        x["prev_ranked_win"] = df["prev_ranked_win"].astype(float)

    if include_gap:
        x["log1p_gap_from_prev_ranked"] = np.log1p(
            df["gap_from_prev_ranked_min"].clip(lower=0).astype(float)
        )

    if session_control_col:
        x["session_depth_control"] = (
            df[session_control_col].clip(upper=SESSION_CAP).astype(float)
        )

    if volume_control_col:
        x["recent_volume_control"] = (
            df[volume_control_col].clip(upper=VOLUME_CAP).astype(float)
        )

    x = pd.concat([x, patch_dummies(df["target_patch"])], axis=1)
    return x


def session_categories(series: pd.Series) -> pd.Series:
    v = series.astype(int).clip(upper=SESSION_CAP)
    return v.map(
        lambda z: str(z) if z < SESSION_CAP else f"{SESSION_CAP}+"
    )


def volume_categories(series: pd.Series) -> pd.Series:
    v = series.astype(int).clip(upper=VOLUME_CAP)
    return v.map(
        lambda z: str(z) if z < VOLUME_CAP else f"{VOLUME_CAP}+"
    )


def requeue_category(value: float) -> str | None:
    if pd.isna(value):
        return None
    for lo, hi, label in POST_LOSS_BINS:
        if value > lo and value <= hi:
            return label
    return None


def primary_sample_query(path: Path) -> str:
    return f"""
        SELECT *
        FROM read_parquet('{sql_path(path)}')
        WHERE has_prior_ranked_match
          AND target_win IS NOT NULL
          AND target_duration_s >= 600
    """


def load_region(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    columns = [
        "source",
        "player_id",
        "match_id",
        "target_win",
        "target_patch",
        "target_duration_s",
        "target_start_ms",
        "prior_ranked_matches",
        "prior_ranked_win_rate",
        "prev_ranked_win",
        "prev_ranked_kda",
        "prev_ranked_loss_streak",
        "prev_ranked_queue_id",
        "prev_ranked_duration_s",
        "gap_from_prev_ranked_min",
        "post_loss_ranked_requeue_gap_min",
    ]

    for t in (30, 45, 60, 90):
        columns.append(f"ranked_session_game_no_{t}m")

    for h in (3, 6, 12, 24):
        columns.append(f"ranked_games_prev_{h}h")

    available = set(
        con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_path(path)}')"
        )
        .fetchdf()["column_name"]
        .astype(str)
    )

    missing = sorted(set(columns) - available)
    if missing:
        raise RuntimeError(f"{path.name}: missing columns: {missing}")

    col_sql = ", ".join(columns)
    return con.execute(
        f"""
        SELECT {col_sql}
        FROM ({primary_sample_query(path)})
        """
    ).fetchdf()


def fit_h1(
    df: pd.DataFrame,
    source: str,
    threshold: int,
    specification: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure_col = f"ranked_session_game_no_{threshold}m"
    cats = [str(i) for i in range(1, SESSION_CAP)] + [f"{SESSION_CAP}+"]
    exposure = session_categories(df[exposure_col])

    dummies, behavior_cols = ordered_dummies(
        exposure,
        cats,
        reference="1",
        prefix="session_depth",
    )

    if specification == "behavior_only":
        X = dummies
    else:
        controls = common_controls(
            df,
            include_prev_win=True,
            include_gap=True,
            volume_control_col=f"ranked_games_prev_{PRIMARY_VOLUME_WINDOW}h",
        )
        X = pd.concat([dummies, controls], axis=1)

    return fit_within_player_lpm(
        df,
        X,
        source=source,
        hypothesis="H1_session_depth",
        specification=specification,
        parameter_setting=f"{threshold}m_session_boundary",
        behavior_columns=behavior_cols,
        reference="session depth 1",
    )


def fit_h3(
    df: pd.DataFrame,
    source: str,
    window_hours: int,
    specification: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure_col = f"ranked_games_prev_{window_hours}h"
    cats = [str(i) for i in range(0, VOLUME_CAP)] + [f"{VOLUME_CAP}+"]
    exposure = volume_categories(df[exposure_col])

    dummies, behavior_cols = ordered_dummies(
        exposure,
        cats,
        reference="0",
        prefix="recent_games",
    )

    if specification == "behavior_only":
        X = dummies
    else:
        controls = common_controls(
            df,
            include_prev_win=True,
            include_gap=True,
            session_control_col=f"ranked_session_game_no_{PRIMARY_SESSION_THRESHOLD}m",
        )
        X = pd.concat([dummies, controls], axis=1)

    return fit_within_player_lpm(
        df,
        X,
        source=source,
        hypothesis="H3_recent_volume",
        specification=specification,
        parameter_setting=f"{window_hours}h_window",
        behavior_columns=behavior_cols,
        reference="0 recent ranked games",
    )


def fit_h2(
    df: pd.DataFrame,
    source: str,
    specification: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[
        (df["prev_ranked_win"] == False)
        & df["prev_ranked_duration_s"].notna()
        & (df["prev_ranked_duration_s"] >= 600)
        & df["post_loss_ranked_requeue_gap_min"].notna()
        & (df["post_loss_ranked_requeue_gap_min"] >= 0)
    ].copy()

    sub["requeue_category"] = sub[
        "post_loss_ranked_requeue_gap_min"
    ].map(requeue_category)

    cats = [x[2] for x in POST_LOSS_BINS]
    dummies, behavior_cols = ordered_dummies(
        sub["requeue_category"],
        cats,
        reference="<=5m",
        prefix="post_loss_gap",
    )

    if specification == "behavior_only":
        X = dummies
    else:
        controls = common_controls(
            sub,
            include_prev_win=False,
            include_gap=False,
            session_control_col=f"ranked_session_game_no_{PRIMARY_SESSION_THRESHOLD}m",
            volume_control_col=f"ranked_games_prev_{PRIMARY_VOLUME_WINDOW}h",
        )
        X = pd.concat([dummies, controls], axis=1)

    if not X.index.equals(sub.index):
        raise RuntimeError(
            f"{source}/H2: design-matrix index is not aligned with the "
            "filtered post-loss sample."
        )

    return fit_within_player_lpm(
        sub,
        X,
        source=source,
        hypothesis="H2_post_loss_requeue",
        specification=specification,
        parameter_setting="categorical_requeue_gap",
        behavior_columns=behavior_cols,
        reference="<=5m",
    )


def fit_h2_continuous(
    df: pd.DataFrame,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[
        (df["prev_ranked_win"] == False)
        & df["prev_ranked_duration_s"].notna()
        & (df["prev_ranked_duration_s"] >= 600)
        & df["post_loss_ranked_requeue_gap_min"].notna()
        & (df["post_loss_ranked_requeue_gap_min"] >= 0)
    ].copy()

    X = pd.DataFrame(index=sub.index)
    X["log2_1plus_post_loss_gap"] = np.log2(
        1.0 + sub["post_loss_ranked_requeue_gap_min"].astype(float)
    )
    behavior_cols = {"log2_1plus_post_loss_gap"}

    controls = common_controls(
        sub,
        include_prev_win=False,
        include_gap=False,
        session_control_col=f"ranked_session_game_no_{PRIMARY_SESSION_THRESHOLD}m",
        volume_control_col=f"ranked_games_prev_{PRIMARY_VOLUME_WINDOW}h",
    )
    X = pd.concat([X, controls], axis=1)

    return fit_within_player_lpm(
        sub,
        X,
        source=source,
        hypothesis="H2_post_loss_requeue",
        specification="adjusted_continuous_sensitivity",
        parameter_setting="log2_1plus_gap",
        behavior_columns=behavior_cols,
        reference="continuous: effect per doubling of (1+gap minutes)",
    )


def plot_primary_effects(
    effects: pd.DataFrame,
    *,
    hypothesis: str,
    parameter_setting: str,
    term_order: Sequence[str],
    labels: Sequence[str],
    title: str,
    output: Path,
) -> None:
    sub = effects[
        (effects["hypothesis"] == hypothesis)
        & (effects["specification"] == "adjusted")
        & (effects["parameter_setting"] == parameter_setting)
        & (effects["is_behavior_term"])
    ].copy()

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(term_order), dtype=float)
    offsets = {"NA": -0.18, "KR": 0.0, "EU": 0.18}

    for source in REGIONS:
        r = sub[sub["source"] == source].set_index("term").reindex(term_order)
        y = r["estimate_percentage_points"].to_numpy(dtype=float)
        lo = r["ci_low_percentage_points"].to_numpy(dtype=float)
        hi = r["ci_high_percentage_points"].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)

        if valid.any():
            ax.errorbar(
                x[valid] + offsets[source],
                y[valid],
                yerr=np.vstack(
                    [y[valid] - lo[valid], hi[valid] - y[valid]]
                ),
                marker="o",
                linestyle="none",
                capsize=3,
                label=source,
            )

    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Adjusted change in target win probability (percentage points)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_dir(args.output, args.overwrite)

    table_dir = args.output / "tables"
    figure_dir = args.output / "figures"
    audit_dir = args.output / "audit"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{args.duckdb_memory_limit}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")

    coef_frames = []
    summary_frames = []

    try:
        for source in REGIONS:
            print(f"[statistics] {source}: loading primary >=10m sample", flush=True)
            df = load_region(con, region_file(args.timelines, source))

            print(
                f"[statistics] {source}: {len(df):,} target rows, "
                f"{df['player_id'].nunique():,} players",
                flush=True,
            )

            # H1 primary: behavior-only FE and adjusted FE.
            for spec in ("behavior_only", "adjusted"):
                c, s = fit_h1(
                    df,
                    source,
                    PRIMARY_SESSION_THRESHOLD,
                    spec,
                )
                coef_frames.append(c)
                summary_frames.append(s)

            # H1 threshold sensitivity: adjusted only.
            for threshold in SESSION_SENSITIVITY:
                c, s = fit_h1(df, source, threshold, "adjusted_sensitivity")
                coef_frames.append(c)
                summary_frames.append(s)

            # H2 primary categorical: behavior-only FE and adjusted FE.
            for spec in ("behavior_only", "adjusted"):
                c, s = fit_h2(df, source, spec)
                coef_frames.append(c)
                summary_frames.append(s)

            # H2 continuous sensitivity.
            c, s = fit_h2_continuous(df, source)
            coef_frames.append(c)
            summary_frames.append(s)

            # H3 primary: behavior-only FE and adjusted FE.
            for spec in ("behavior_only", "adjusted"):
                c, s = fit_h3(
                    df,
                    source,
                    PRIMARY_VOLUME_WINDOW,
                    spec,
                )
                coef_frames.append(c)
                summary_frames.append(s)

            # H3 window sensitivity: adjusted only.
            for window in VOLUME_SENSITIVITY:
                c, s = fit_h3(df, source, window, "adjusted_sensitivity")
                coef_frames.append(c)
                summary_frames.append(s)

            del df

    finally:
        con.close()

    all_coef = pd.concat(coef_frames, ignore_index=True)
    model_summary = pd.concat(summary_frames, ignore_index=True)
    behavior_effects = all_coef[all_coef["is_behavior_term"]].copy()

    all_coef.to_csv(table_dir / "all_coefficients.csv", index=False)
    behavior_effects.to_csv(table_dir / "behavior_effects.csv", index=False)
    model_summary.to_csv(table_dir / "model_summary.csv", index=False)

    # Primary coefficient plots.
    session_terms = [
        f"session_depth_{x}"
        for x in [str(i) for i in range(2, SESSION_CAP)] + [f"{SESSION_CAP}+"]
    ]
    session_labels = [str(i) for i in range(2, SESSION_CAP)] + [f"{SESSION_CAP}+"]

    plot_primary_effects(
        behavior_effects,
        hypothesis="H1_session_depth",
        parameter_setting=f"{PRIMARY_SESSION_THRESHOLD}m_session_boundary",
        term_order=session_terms,
        labels=session_labels,
        title=(
            "Within-player adjusted association: observed session depth "
            "vs next Solo/Duo win"
        ),
        output=figure_dir / "H1_session_depth_adjusted_effects.png",
    )

    requeue_terms = [
        f"post_loss_gap_{x}"
        for x in [v[2] for v in POST_LOSS_BINS][1:]
    ]
    requeue_labels = [v[2] for v in POST_LOSS_BINS][1:]

    plot_primary_effects(
        behavior_effects,
        hypothesis="H2_post_loss_requeue",
        parameter_setting="categorical_requeue_gap",
        term_order=requeue_terms,
        labels=requeue_labels,
        title=(
            "Within-player adjusted association: post-loss requeue gap "
            "vs next Solo/Duo win"
        ),
        output=figure_dir / "H2_post_loss_requeue_adjusted_effects.png",
    )

    volume_terms = [
        f"recent_games_{x}"
        for x in [str(i) for i in range(1, VOLUME_CAP)] + [f"{VOLUME_CAP}+"]
    ]
    volume_labels = [str(i) for i in range(1, VOLUME_CAP)] + [f"{VOLUME_CAP}+"]

    plot_primary_effects(
        behavior_effects,
        hypothesis="H3_recent_volume",
        parameter_setting=f"{PRIMARY_VOLUME_WINDOW}h_window",
        term_order=volume_terms,
        labels=volume_labels,
        title=(
            "Within-player adjusted association: recent ranked volume "
            "vs next Solo/Duo win"
        ),
        output=figure_dir / "H3_recent_volume_adjusted_effects.png",
    )

    payload = {
        "primary_outcome": "target_win",
        "primary_sample": (
            "Solo/Duo target with prior ranked history and target duration >=10m."
        ),
        "primary_session_boundary_minutes": PRIMARY_SESSION_THRESHOLD,
        "session_sensitivity_minutes": list(SESSION_SENSITIVITY),
        "primary_recent_volume_window_hours": PRIMARY_VOLUME_WINDOW,
        "recent_volume_sensitivity_hours": list(VOLUME_SENSITIVITY),
        "post_loss_previous_match_rule": "previous ranked match duration >=10m",
        "estimator": (
            "Linear probability model with player fixed effects implemented "
            "by within-player demeaning."
        ),
        "uncertainty": (
            "Two-way cluster-robust covariance by player_id and physical match_id."
        ),
        "interpretation": (
            "Behavior coefficients are within-player associations expressed "
            "as percentage-point changes in next-match win probability."
        ),
        "causal_warning": (
            "Observational analysis; fixed effects reduce stable player-level "
            "confounding but do not establish causality."
        ),
        "model_count": int(len(model_summary)),
    }
    (audit_dir / "statistical_analysis_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\nFORMAL STATISTICAL ANALYSIS COMPLETE\n")
    print(
        model_summary[
            [
                "source",
                "hypothesis",
                "specification",
                "parameter_setting",
                "n_rows",
                "n_players",
                "n_physical_matches",
                "within_r2",
            ]
        ].to_string(index=False)
    )
    print(f"\nTables:  {table_dir}")
    print(f"Figures: {figure_dir}")



if __name__ == "__main__":
    main()

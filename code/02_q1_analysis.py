#!/usr/bin/env python3
"""Question 1 analysis: EDA, inference, prediction, robustness, and report figures.

Run from the project root:
    python code/02_q1_analysis.py --overwrite

Input:  data/analysis/timelines/solo420_targets/{NA,KR,EU}.parquet
Output: data/analysis/q1/
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Sequence

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

REGIONS = ('NA', 'KR', 'EU')
PRIMARY_SESSION_THRESHOLD = 30
SESSION_SENSITIVITY = (45, 60, 90)
PRIMARY_VOLUME_WINDOW = 6
VOLUME_SENSITIVITY = (3, 12, 24)
SESSION_CAP = 8
VOLUME_CAP = 6
RANDOM_STATE = 67978
MAX_DEPTH_GRID = (2, 3, 4, 5, 6, 8)
MIN_SAMPLES_LEAF_GRID = (250, 1000, 3000)
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

# Separate feature groups let us measure the incremental predictive value of behavior.
HISTORY_BASE_FEATURES = [
    "prior_ranked_win_rate",
    "log1p_prior_ranked_matches",
    "prev_ranked_win",
    "log1p_prev_ranked_kda",
    "prev_ranked_cs_per_min",
    "prev_ranked_gold_per_min",
    "prev_ranked_damage_per_min",
    "prev_ranked_vision_per_min",
    "log1p_prev_ranked_loss_streak",
    "log1p_prev_ranked_win_streak",
    "prev_queue_was_flex440",
    "log1p_prev_ranked_duration_min",
    "prior_ranked_mean_kda",
    "prior_ranked_mean_cs_per_min",
    "prior_ranked_mean_gold_per_min",
    "prior_ranked_mean_damage_per_min",
]

BEHAVIOR_BASE_FEATURES = [
    "log1p_gap_from_prev_ranked_min",
    "session_depth_30m",
    "ranked_games_prev_6h_capped",
    "ranked_minutes_prev_6h",
    "previous_ranked_was_loss",
    "log1p_post_loss_gap_min",
    "champion_changed_from_prev_ranked",
    "role_changed_from_prev_ranked",
]

REGION_COLORS = {'NA': '#3B82F6', 'KR': '#EF4444', 'EU': '#D4A72C'}


def project_root() -> Path:
    """Return the project root inferred from this script location."""
    return Path(__file__).resolve().parents[1]


def sql_path(path: Path) -> str:
    """Convert a path to a DuckDB-safe absolute POSIX string."""
    return path.resolve().as_posix().replace("'", "''")


def prepare_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory, optionally replacing existing contents."""
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f'Output directory is not empty: {path}. Use --overwrite.')
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as CSV, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def transparent_save(fig: plt.Figure, path: Path, dpi: int = 220) -> None:
    """Save and close a Matplotlib figure with a transparent background."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)


def clean_feature_name(name: str) -> str:
    """Convert internal feature names to readable report labels."""
    replacements = {
        "prior_ranked_mean_damage_per_min": "Prior mean damage/min",
        "prior_ranked_mean_kda": "Prior mean KDA",
        "prior_ranked_mean_gold_per_min": "Prior mean gold/min",
        "prior_ranked_mean_cs_per_min": "Prior mean CS/min",
        "prior_ranked_win_rate": "Prior win rate",
        "prev_ranked_damage_per_min": "Previous damage/min",
        "prev_ranked_gold_per_min": "Previous gold/min",
        "prev_ranked_cs_per_min": "Previous CS/min",
        "log1p_prev_ranked_kda": "Previous KDA (log)",
        "champion_changed_from_prev_ranked": "Champion changed",
        "role_changed_from_prev_ranked": "Role changed",
        "session_depth_30m": "Session depth (30m)",
        "ranked_games_prev_6h_capped": "Games in previous 6h",
        "log1p_gap_from_prev_ranked_min": "Inter-match gap (log)",
        "ranked_minutes_prev_6h": "Minutes played in previous 6h",
        "previous_ranked_was_loss": "Previous ranked match was loss",
        "log1p_post_loss_gap_min": "Post-loss requeue gap (log)",
    }
    if name in replacements:
        return replacements[name]
    if name.startswith("patch_"):
        return "Patch " + name.removeprefix("patch_")
    if name.startswith("region_"):
        return "Region " + name.removeprefix("region_")
    return name.replace("_", " ").title()


def wilson_interval(wins: float, n: float, z: float = 1.959963984540054):
    """Compute a Wilson confidence interval for a binomial proportion."""
    if n is None or n <= 0:
        return (np.nan, np.nan)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (center - half, center + half)


def add_winrate_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Add win-rate and Wilson interval columns to an aggregated table."""
    out = df.copy()
    out['win_rate'] = out['wins'] / out['n']
    cis = [wilson_interval(w, n) for w, n in zip(out['wins'], out['n'])]
    out['win_rate_ci_low'] = [x[0] for x in cis]
    out['win_rate_ci_high'] = [x[1] for x in cis]
    return out


def bin_post_loss_gap(value):
    """Map a post-loss requeue gap to its descriptive time bin."""
    if pd.isna(value):
        return None
    for lo, hi, label in POST_LOSS_BINS:
        if value > lo and value <= hi:
            return label
    return None


def post_loss_order() -> list[str]:
    """Return post-loss gap labels in plotting order."""
    return [x[2] for x in POST_LOSS_BINS]


def depth_label(value: int) -> str:
    """Format capped session depth for tables and plots."""
    return str(value) if value < SESSION_CAP else f'{SESSION_CAP}+'


def volume_label(value: int) -> str:
    """Format capped recent-volume counts for tables and plots."""
    return str(value) if value < VOLUME_CAP else f'{VOLUME_CAP}+'


def region_file(folder: Path, source: str) -> Path:
    """Return the regional timeline file and fail clearly if it is missing."""
    p = folder / f'{source}.parquet'
    if not p.exists():
        raise FileNotFoundError(f'Missing timeline file: {p}')
    return p


def normal_pvalue(z: float) -> float:
    """Return a two-sided normal-approximation p-value from a z statistic."""
    if not np.isfinite(z):
        return np.nan
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def factor_codes(values: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Convert grouping labels to dense integer codes."""
    codes, uniques = pd.factorize(values, sort=False)
    if (codes < 0).any():
        raise ValueError('Cluster/fixed-effect groups contain missing values.')
    return (codes.astype(np.int64), np.asarray(uniques))


def within_demean(y: np.ndarray, X: np.ndarray, group_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove player means; equivalent to player fixed intercepts in OLS."""
    n_groups = int(group_codes.max()) + 1
    counts = np.bincount(group_codes, minlength=n_groups).astype(float)
    y_sums = np.bincount(group_codes, weights=y, minlength=n_groups)
    y_mean = y_sums / counts
    yw = y - y_mean[group_codes]
    Xw = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        sums = np.bincount(group_codes, weights=X[:, j], minlength=n_groups)
        means = sums / counts
        Xw[:, j] = X[:, j] - means[group_codes]
    return (yw, Xw)


def cluster_covariance(X: np.ndarray, resid: np.ndarray, bread: np.ndarray, codes: np.ndarray, p: int) -> np.ndarray:
    """Compute one-way cluster-robust covariance with finite-sample correction."""
    n = X.shape[0]
    g = int(codes.max()) + 1
    scores = X * resid[:, None]
    aggregated = np.zeros((g, p), dtype=float)
    np.add.at(aggregated, codes, scores)
    meat = aggregated.T @ aggregated
    if g > 1 and n > p:
        correction = g / (g - 1.0) * ((n - 1.0) / (n - p))
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

    # Player demeaning removes all time-invariant player-level differences.
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

    # Two-way clustering handles repeated players and shared physical matches.
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
    """Create ordered dummy variables while preserving the original row index."""
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
    """Create patch controls using the first observed patch as the reference."""
    values = series.astype(str)
    categories = sorted(values.dropna().unique().tolist())
    cat = pd.Categorical(values, categories=categories)
    d = pd.get_dummies(cat, prefix='patch', dtype=float)
    d.index = series.index
    if len(d.columns) <= 1:
        return pd.DataFrame(index=series.index)
    return d.iloc[:, 1:]


def common_controls(
        df: pd.DataFrame,
        *,
        include_prev_win: bool,
        include_gap: bool,
        session_control_col: str | None = None,
        volume_control_col: str | None = None,
) -> pd.DataFrame:
    """Build the shared pre-target control matrix used by adjusted models."""
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
    """Convert session depth to capped ordered categories."""
    v = series.astype(int).clip(upper=SESSION_CAP)
    return v.map(lambda z: str(z) if z < SESSION_CAP else f'{SESSION_CAP}+')


def volume_categories(series: pd.Series) -> pd.Series:
    """Convert recent game count to capped ordered categories."""
    v = series.astype(int).clip(upper=VOLUME_CAP)
    return v.map(lambda z: str(z) if z < VOLUME_CAP else f'{VOLUME_CAP}+')


def requeue_category(value: float) -> str | None:
    """Return the categorical post-loss requeue label for one gap."""
    if pd.isna(value):
        return None
    for lo, hi, label in POST_LOSS_BINS:
        if value > lo and value <= hi:
            return label
    return None


def primary_sample_query(path: Path) -> str:
    """Return the SQL query defining the primary >=10-minute target sample."""
    return f"""
        SELECT *
        FROM read_parquet('{sql_path(path)}')
        WHERE has_prior_ranked_match
          AND target_win IS NOT NULL
          AND target_duration_s >= 600
    """


def load_region(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    """Load the regional columns required by the statistical models."""
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
    columns += [f"ranked_session_game_no_{t}m" for t in (30, 45, 60, 90)]
    columns += [f"ranked_games_prev_{h}h" for h in (3, 6, 12, 24)]

    return con.execute(
        f"""
        SELECT {", ".join(columns)}
        FROM ({primary_sample_query(path)})
        """
    ).fetchdf()


def fit_h1(
        df: pd.DataFrame,
        source: str,
        threshold: int,
        specification: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the session-depth model for one region and session threshold."""
    # H1 exposure: game number within the observed session.
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
    """Fit the recent-volume model for one region and time window."""
    # H3 exposure: recent ranked-game count in the chosen activity window.
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
    """Fit the categorical post-loss requeue model for one region."""
    # H2 applies only after a valid previous loss lasting at least 10 minutes.
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
    """Fit the continuous post-loss gap sensitivity model."""
    # Sensitivity version estimates one effect per doubling of (1 + gap minutes).
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


def timeline_glob(folder: Path) -> str:
    """Return a Parquet glob for the prepared regional timeline files."""
    files = sorted(folder.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'No Parquet files found in: {folder}')
    return sql_path(folder / '*.parquet')


def load_primary_sample(
        con: duckdb.DuckDBPyConnection,
        folder: Path,
) -> pd.DataFrame:
    """Load the primary >=10-minute sample used for prediction."""
    columns = sorted(
        {
            "source",
            "player_id",
            "match_id",
            "target_win",
            "target_patch",
            "target_start_ms",
            "target_duration_s",
            "has_prior_ranked_match",
            "prior_ranked_matches",
            "prior_ranked_win_rate",
            "prev_ranked_win",
            "prev_ranked_kda",
            "prev_ranked_cs_per_min",
            "prev_ranked_gold_per_min",
            "prev_ranked_damage_per_min",
            "prev_ranked_vision_per_min",
            "prev_ranked_loss_streak",
            "prev_ranked_win_streak",
            "prev_ranked_queue_id",
            "prev_ranked_duration_s",
            "prior_ranked_mean_kda",
            "prior_ranked_mean_cs_per_min",
            "prior_ranked_mean_gold_per_min",
            "prior_ranked_mean_damage_per_min",
            "gap_from_prev_ranked_min",
            "post_loss_ranked_requeue_gap_min",
            "ranked_session_game_no_30m",
            "ranked_games_prev_6h",
            "ranked_minutes_played_prev_6h",
            "champion_changed_from_prev_ranked",
            "role_changed_from_prev_ranked",
        }
    )
    glob = timeline_glob(folder)

    return con.execute(
        f"""
        SELECT {", ".join(columns)}
        FROM read_parquet('{glob}')
        WHERE has_prior_ranked_match
          AND target_win IS NOT NULL
          AND target_duration_s >= 600
        """
    ).fetchdf()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create strictly pre-target predictors used by the decision trees."""
    out = df.copy()

    # Historical-performance features use only information available before the target.
    out["prior_ranked_win_rate"] = pd.to_numeric(
        out["prior_ranked_win_rate"], errors="coerce"
    )

    out["log1p_prior_ranked_matches"] = np.log1p(
        pd.to_numeric(out["prior_ranked_matches"], errors="coerce").clip(lower=0)
    )
    out["prev_ranked_win"] = out["prev_ranked_win"].astype(float)
    out["log1p_prev_ranked_kda"] = np.log1p(
        pd.to_numeric(out["prev_ranked_kda"], errors="coerce").clip(lower=0)
    )
    out["log1p_prev_ranked_loss_streak"] = np.log1p(
        pd.to_numeric(out["prev_ranked_loss_streak"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    out["log1p_prev_ranked_win_streak"] = np.log1p(
        pd.to_numeric(out["prev_ranked_win_streak"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    out["prev_queue_was_flex440"] = (
            out["prev_ranked_queue_id"] == 440
    ).astype(float)
    out["log1p_prev_ranked_duration_min"] = np.log1p(
        pd.to_numeric(out["prev_ranked_duration_s"], errors="coerce")
        .clip(lower=0)
        / 60.0
    )

    # Behavioral timing/activity features for behavior-only and combined trees.
    out["log1p_gap_from_prev_ranked_min"] = np.log1p(
        pd.to_numeric(out["gap_from_prev_ranked_min"], errors="coerce")
        .clip(lower=0)
    )
    out["session_depth_30m"] = (
        pd.to_numeric(out["ranked_session_game_no_30m"], errors="coerce")
        .clip(lower=1, upper=SESSION_CAP)
    )
    out["ranked_games_prev_6h_capped"] = (
        pd.to_numeric(out["ranked_games_prev_6h"], errors="coerce")
        .clip(lower=0, upper=VOLUME_CAP)
    )
    out["ranked_minutes_prev_6h"] = pd.to_numeric(
        out["ranked_minutes_played_prev_6h"], errors="coerce"
    ).clip(lower=0)

    out["previous_ranked_was_loss"] = (
            out["prev_ranked_win"] == 0
    ).astype(float)

    raw_post_loss_gap = pd.to_numeric(
        out["post_loss_ranked_requeue_gap_min"], errors="coerce"
    )
    out["log1p_post_loss_gap_min"] = np.where(
        out["previous_ranked_was_loss"] == 1,
        np.log1p(raw_post_loss_gap.clip(lower=0).fillna(0)),
        0.0,
    )

    for col in (
            "champion_changed_from_prev_ranked",
            "role_changed_from_prev_ranked",
    ):
        out[col] = out[col].astype(float)

    out["target_win"] = out["target_win"].astype(int)
    out["source"] = out["source"].astype(str)
    out["target_patch"] = out["target_patch"].astype(str)

    return out


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return data with split labels plus a compact per-region threshold table.
    """
    out = df.copy()
    threshold_rows = []

    split = pd.Series(index=out.index, dtype="object")

    # Split chronologically inside each region so future matches never enter training.
    for source in REGIONS:
        mask = out["source"] == source
        times = out.loc[mask, "target_start_ms"].astype(np.int64)

        q70 = float(times.quantile(0.70))
        q85 = float(times.quantile(0.85))

        split.loc[mask & (out["target_start_ms"] < q70)] = "train"
        split.loc[
            mask
            & (out["target_start_ms"] >= q70)
            & (out["target_start_ms"] < q85)
            ] = "validation"
        split.loc[
            mask & (out["target_start_ms"] >= q85)
            ] = "test"

        threshold_rows.append(
            {
                "source": source,
                "train_validation_boundary_ms": int(q70),
                "validation_test_boundary_ms": int(q85),
            }
        )

    out["split"] = split

    if out["split"].isna().any():
        raise RuntimeError("Some rows were not assigned to a chronological split.")

    # Same physical match must never cross splits.
    cross_split = (
        out.groupby(["source", "match_id"])["split"]
        .nunique()
        .gt(1)
        .sum()
    )
    if cross_split:
        raise RuntimeError(
            f"{cross_split} physical matches cross chronological split boundaries."
        )

    return out, pd.DataFrame(threshold_rows)


def fit_context_schema(train: pd.DataFrame) -> dict:
    """Learn region and patch categories from training data only."""
    return {
        "regions": sorted(train["source"].dropna().unique().tolist()),
        "patches": sorted(train["target_patch"].dropna().unique().tolist()),
    }


def make_context_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Create one-hot region and patch controls from a train-defined schema."""
    out = pd.DataFrame(index=df.index)

    # Use train-defined categories only. Unknown future patches/regions become
    # all-zero for that category family rather than leaking test schema.
    for region in schema["regions"][1:]:
        out[f"region_{region}"] = (df["source"] == region).astype(float)

    for patch in schema["patches"][1:]:
        out[f"patch_{patch}"] = (df["target_patch"] == patch).astype(float)

    return out


def fill_train_medians(
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Impute numeric features using medians learned from the training set."""
    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    for col in train.columns:
        median = float(pd.to_numeric(train[col], errors="coerce").median())
        if not np.isfinite(median):
            median = 0.0

        train[col] = pd.to_numeric(train[col], errors="coerce").fillna(median)
        validation[col] = pd.to_numeric(
            validation[col], errors="coerce"
        ).fillna(median)
        test[col] = pd.to_numeric(test[col], errors="coerce").fillna(median)

    return train, validation, test


def metrics_row(
        y_true: np.ndarray,
        prob: np.ndarray,
        *,
        model_name: str,
        subgroup: str,
) -> dict:
    """Calculate classification metrics for one model and subgroup."""
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = np.nan

    return {
        "model": model_name,
        "subgroup": subgroup,
        "n": int(len(y_true)),
        "observed_win_rate": float(np.mean(y_true)),
        "mean_predicted_win_probability": float(np.mean(prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(auc) if np.isfinite(auc) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_probabilities(
        frame: pd.DataFrame,
        prob: np.ndarray,
        model_name: str,
) -> list[dict]:
    """Evaluate predicted probabilities overall and separately by region."""
    rows = []
    y = frame["target_win"].to_numpy(dtype=int)

    rows.append(
        metrics_row(y, prob, model_name=model_name, subgroup="ALL")
    )

    for source in REGIONS:
        mask = frame["source"].to_numpy() == source
        if mask.any():
            rows.append(
                metrics_row(
                    y[mask],
                    prob[mask],
                    model_name=model_name,
                    subgroup=source,
                )
            )

    return rows


def decision_tree_grid(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_set: str,
) -> tuple[DecisionTreeClassifier, pd.DataFrame]:
    """Select pre-pruned entropy-tree settings using validation ROC-AUC."""
    rows = []

    # Small pre-pruning grid; validation AUC chooses the final entropy tree.
    for depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_SAMPLES_LEAF_GRID:
            model = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=depth,
                min_samples_leaf=min_leaf,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            rows.append(
                {
                    "feature_set": feature_set,
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                    "train_auc": train_auc,
                    "validation_auc": val_auc,
                    "validation_minus_train_auc": val_auc - train_auc,
                    "nodes": model.tree_.node_count,
                    "leaves": model.tree_.n_leaves,
                }
            )

    grid = pd.DataFrame(rows)
    best = grid.sort_values(
        ["validation_auc", "max_depth", "min_samples_leaf"],
        ascending=[False, True, False],
    ).iloc[0]

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=int(best["max_depth"]),
        min_samples_leaf=int(best["min_samples_leaf"]),
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model, grid



def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Apply Holm family-wise-error correction to a sequence of p-values."""
    p = pd.to_numeric(p_values, errors='coerce').to_numpy(dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return pd.Series(out, index=p_values.index)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adjusted_sorted = []
    running_max = 0.0
    for rank, original_idx in enumerate(order):
        multiplier = m - rank
        adj = min(1.0, multiplier * p[original_idx])
        running_max = max(running_max, adj)
        adjusted_sorted.append(running_max)
    for original_idx, adj in zip(order, adjusted_sorted):
        out[original_idx] = adj
    return pd.Series(out, index=p_values.index)


def sample_variants(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the primary, alias-confirmed, and >=5-minute robustness samples."""
    target_duration = pd.to_numeric(df['target_duration_s'], errors='coerce')
    ge10 = df[target_duration >= 600].copy()
    alias_ge10 = ge10[ge10['is_alias_confirmed'].fillna(False).astype(bool)].copy()
    ge5 = df[target_duration >= 300].copy()
    return {'authoritative_ge10m': ge10, 'alias_confirmed_ge10m': alias_ge10, 'authoritative_ge5m': ge5}


def compare_to_primary(
        effects: pd.DataFrame,
) -> pd.DataFrame:
    """Compare robustness-sample effects with the primary >=10-minute estimates."""
    primary = effects[
        effects["sample_variant"] == "authoritative_ge10m"
        ][
        [
            "source",
            "hypothesis",
            "parameter_setting",
            "term",
            "estimate_percentage_points",
            "ci_low_percentage_points",
            "ci_high_percentage_points",
            "p_value_holm",
        ]
    ].rename(
        columns={
            "estimate_percentage_points": "primary_effect_pp",
            "ci_low_percentage_points": "primary_ci_low_pp",
            "ci_high_percentage_points": "primary_ci_high_pp",
            "p_value_holm": "primary_p_holm",
        }
    )

    others = effects[
        effects["sample_variant"] != "authoritative_ge10m"
        ].copy()

    merged = others.merge(
        primary,
        on=["source", "hypothesis", "parameter_setting", "term"],
        how="left",
        validate="many_to_one",
    )

    merged["effect_difference_pp"] = (
            merged["estimate_percentage_points"] - merged["primary_effect_pp"]
    )

    merged["same_sign_as_primary"] = (
            np.sign(merged["estimate_percentage_points"])
            == np.sign(merged["primary_effect_pp"])
    )

    return merged


def run_eda(con: duckdb.DuckDBPyConnection, timelines: Path, out: Path) -> dict:
    """Create descriptive Q1 tables and the session-gap/report EDA figures."""
    tables = out / "tables"
    report_figs = out / "figures" / "report"
    supp_figs = out / "figures" / "supplementary"
    for d in (tables, report_figs, supp_figs):
        d.mkdir(parents=True, exist_ok=True)

    glob = timeline_glob(timelines)
    # Descriptive sample: observable prior ranked history and known target outcome.
    con.execute("DROP TABLE IF EXISTS q1_sample")
    con.execute(
        f"""
        CREATE TEMP TABLE q1_sample AS
        SELECT * FROM read_parquet('{glob}')
        WHERE has_prior_ranked_match AND target_win IS NOT NULL
        """
    )

    # Sample size and target-outcome summary by region.
    overview = con.execute(
        """
        SELECT source,
               COUNT(*)::BIGINT AS target_rows, COUNT(DISTINCT player_id)::BIGINT AS players, COUNT(DISTINCT match_id)::BIGINT AS physical_matches, SUM(prev_ranked_win = FALSE)::BIGINT AS targets_after_ranked_loss, AVG(CASE WHEN target_win THEN 1.0 ELSE 0.0 END) AS win_rate,
               AVG(target_kda)    AS mean_kda,
               MEDIAN(target_kda) AS median_kda,
               SUM(target_duration_s < 300)::BIGINT AS targets_under_5m, SUM(target_duration_s < 600) ::BIGINT AS targets_under_10m
        FROM q1_sample
        GROUP BY source
        ORDER BY source
        """
    ).fetchdf()
    save_csv(overview, tables / "sample_overview.csv")

    # Gap distribution supports the session-threshold sensitivity analysis.
    gap_quantiles = con.execute(
        """
        SELECT source,
               COUNT(gap_from_prev_ranked_min)::BIGINT AS n, quantile_cont(gap_from_prev_ranked_min, 0.25) AS p25_min,
               MEDIAN(gap_from_prev_ranked_min)              AS median_min,
               quantile_cont(gap_from_prev_ranked_min, 0.75) AS p75_min,
               quantile_cont(gap_from_prev_ranked_min, 0.90) AS p90_min,
               quantile_cont(gap_from_prev_ranked_min, 0.95) AS p95_min,
               quantile_cont(gap_from_prev_ranked_min, 0.99) AS p99_min
        FROM q1_sample
        WHERE gap_from_prev_ranked_min IS NOT NULL
          AND gap_from_prev_ranked_min >= 0
        GROUP BY source
        ORDER BY source
        """
    ).fetchdf()
    save_csv(gap_quantiles, tables / "gap_quantiles.csv")

    coverage_rows = []
    for threshold in (10, 15, 20, 30, 45, 60, 90, 120, 180, 360, 720, 1440):
        q = con.execute(
            f"""
            SELECT source,
                   COUNT(*)::BIGINT AS n_gaps,
                   SUM(gap_from_prev_ranked_min<={threshold})::BIGINT AS n_at_or_below
            FROM q1_sample
            WHERE gap_from_prev_ranked_min IS NOT NULL AND gap_from_prev_ranked_min>=0
            GROUP BY source
            """
        ).fetchdf()
        q["threshold_minutes"] = threshold
        q["pct_at_or_below"] = 100.0 * q["n_at_or_below"] / q["n_gaps"]
        coverage_rows.append(q)
    gap_coverage = pd.concat(coverage_rows, ignore_index=True)
    save_csv(gap_coverage, tables / "gap_threshold_coverage.csv")

    # Session depth for all sensitivity thresholds.
    session_rows = []
    for threshold in (30, 45, 60, 90):
        col = f"ranked_session_game_no_{threshold}m"
        df = con.execute(
            f"""
            SELECT source,
                   LEAST({col},{SESSION_CAP})::BIGINT AS depth_bucket_num,
                   COUNT(*)::BIGINT AS n,
                   SUM(CASE WHEN target_win THEN 1 ELSE 0 END)::BIGINT AS wins,
                   AVG(target_kda) AS mean_kda,
                   AVG(target_damage_to_champions_per_min) AS mean_damage_per_min
            FROM q1_sample
            GROUP BY source,depth_bucket_num
            """
        ).fetchdf()
        df["threshold_minutes"] = threshold
        df["session_depth"] = df["depth_bucket_num"].map(depth_label)
        session_rows.append(add_winrate_ci(df))
    session_depth = pd.concat(session_rows, ignore_index=True)
    save_csv(session_depth, tables / "session_depth_outcomes.csv")

    # Descriptive post-loss outcomes by requeue-delay bin.
    post_loss = con.execute(
        """
        SELECT source,
               post_loss_ranked_requeue_gap_min,
               target_win,
               target_kda,
               target_damage_to_champions_per_min,
               target_cs_per_min,
               target_gold_per_min
        FROM q1_sample
        WHERE prev_ranked_win = FALSE
          AND post_loss_ranked_requeue_gap_min IS NOT NULL
          AND post_loss_ranked_requeue_gap_min >= 0
        """
    ).fetchdf()
    post_loss["requeue_bin"] = pd.Categorical(
        post_loss["post_loss_ranked_requeue_gap_min"].map(bin_post_loss_gap),
        categories=post_loss_order(), ordered=True,
    )
    post_loss_table = (
        post_loss.groupby(["source", "requeue_bin"], observed=True)
        .agg(
            n=("target_win", "size"), wins=("target_win", "sum"),
            mean_kda=("target_kda", "mean"),
            mean_damage_per_min=("target_damage_to_champions_per_min", "mean"),
            median_gap_min=("post_loss_ranked_requeue_gap_min", "median"),
        ).reset_index()
    )
    post_loss_table = add_winrate_ci(post_loss_table)
    save_csv(post_loss_table, tables / "post_loss_requeue_outcomes.csv")

    # Descriptive recent-volume outcomes across all candidate windows.
    recent_rows = []
    for h in (3, 6, 12, 24):
        col = f"ranked_games_prev_{h}h"
        df = con.execute(
            f"""
            SELECT source,LEAST({col},{VOLUME_CAP})::BIGINT AS volume_bucket_num,
                   COUNT(*)::BIGINT AS n,
                   SUM(CASE WHEN target_win THEN 1 ELSE 0 END)::BIGINT AS wins,
                   AVG(target_kda) AS mean_kda,
                   AVG(target_damage_to_champions_per_min) AS mean_damage_per_min
            FROM q1_sample GROUP BY source,volume_bucket_num
            """
        ).fetchdf()
        df["window_hours"] = h
        df["recent_games"] = df["volume_bucket_num"].map(volume_label)
        recent_rows.append(add_winrate_ci(df))
    recent_volume = pd.concat(recent_rows, ignore_index=True)
    save_csv(recent_volume, tables / "recent_volume_outcomes.csv")

    # Figure 1: ECDF. This is a methodological figure, not decorative EDA.
    gaps = con.execute(
        """
        SELECT source, gap_from_prev_ranked_min
        FROM q1_sample
        WHERE gap_from_prev_ranked_min BETWEEN 0 AND 1440
        """
    ).fetchdf()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for source in REGIONS:
        x = np.sort(gaps.loc[gaps["source"] == source, "gap_from_prev_ranked_min"].astype(float).to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, lw=2.2, label=source, color=REGION_COLORS[source])
    for t in (30, 60, 90):
        ax.axvline(t, lw=0.9, ls="--", alpha=0.45)
    ax.set_xscale("log")
    ax.set_xlabel("Minutes from previous ranked match end to next start (log scale)")
    ax.set_ylabel("Cumulative share of observed gaps")
    ax.set_title(
        "Most requeues happen quickly, but there is no single natural session cutoff",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    transparent_save(fig, report_figs / "figure_1_inter_match_gap_ecdf.png")

    return None


def add_holm_by_family(effects: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Apply Holm correction separately within each specified hypothesis family."""
    out = effects.copy()
    out["p_value_holm"] = np.nan
    for _, idx in out.groupby(list(group_cols)).groups.items():
        idx = list(idx)
        out.loc[idx, "p_value_holm"] = holm_adjust(out.loc[idx, "p_value_two_sided"]).to_numpy()
    out["significant_raw_0_05"] = pd.to_numeric(out["p_value_two_sided"], errors="coerce") < 0.05
    out["significant_holm_0_05"] = pd.to_numeric(out["p_value_holm"], errors="coerce") < 0.05
    return out


def run_statistics(con: duckdb.DuckDBPyConnection, timelines: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run primary and sensitivity H1-H3 models for all three regions."""
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    coef_frames, summary_frames = [], []

    # Fit every primary model plus threshold/window sensitivities by region.
    for source in REGIONS:
        print(f"[q1 statistics] {source}", flush=True)
        df = load_region(con, region_file(timelines, source))

        for spec in ("behavior_only", "adjusted"):
            c, s = fit_h1(df, source, PRIMARY_SESSION_THRESHOLD, spec)
            coef_frames.append(c)
            summary_frames.append(s)
        for threshold in SESSION_SENSITIVITY:
            c, s = fit_h1(df, source, threshold, "adjusted_sensitivity")
            coef_frames.append(c)
            summary_frames.append(s)

        for spec in ("behavior_only", "adjusted"):
            c, s = fit_h2(df, source, spec)
            coef_frames.append(c)
            summary_frames.append(s)
        c, s = fit_h2_continuous(df, source)
        coef_frames.append(c)
        summary_frames.append(s)

        for spec in ("behavior_only", "adjusted"):
            c, s = fit_h3(df, source, PRIMARY_VOLUME_WINDOW, spec)
            coef_frames.append(c)
            summary_frames.append(s)
        for window in VOLUME_SENSITIVITY:
            c, s = fit_h3(df, source, window, "adjusted_sensitivity")
            coef_frames.append(c)
            summary_frames.append(s)
        del df
        gc.collect()

    all_coef = pd.concat(coef_frames, ignore_index=True)
    model_summary = pd.concat(summary_frames, ignore_index=True)
    effects = all_coef[all_coef["is_behavior_term"]].copy()
    effects = add_holm_by_family(
        effects,
        ["source", "hypothesis", "specification", "parameter_setting"],
    )
    save_csv(all_coef, tables / "all_coefficients.csv")
    save_csv(effects, tables / "behavior_effects.csv")
    save_csv(model_summary, tables / "statistical_model_summary.csv")
    return effects, model_summary


def _primary_effect_data(effects: pd.DataFrame, hypothesis: str, setting: str) -> pd.DataFrame:
    """Select one primary adjusted effect family for plotting."""
    return effects[
        (effects["hypothesis"] == hypothesis)
        & (effects["specification"] == "adjusted")
        & (effects["parameter_setting"] == setting)
        ].copy()


def plot_adjusted_effects(effects: pd.DataFrame, output: Path) -> None:
    """Create the three-panel adjusted-effect figure for H1-H3."""
    panels = [
        ("H1_session_depth", f"{PRIMARY_SESSION_THRESHOLD}m_session_boundary",
         [f"session_depth_{i}" for i in range(2, SESSION_CAP)] + [f"session_depth_{SESSION_CAP}+"],
         [str(i) for i in range(2, SESSION_CAP)] + [f"{SESSION_CAP}+"],
         "A  Session depth", "Game number in observed session\n(reference: game 1)"),
        ("H2_post_loss_requeue", "categorical_requeue_gap",
         [f"post_loss_gap_{x[2]}" for x in POST_LOSS_BINS[1:]],
         [x[2] for x in POST_LOSS_BINS[1:]],
         "B  Post-loss requeue", "Delay after prior loss\n(reference: ≤5m)"),
        ("H3_recent_volume", f"{PRIMARY_VOLUME_WINDOW}h_window",
         [f"recent_games_{i}" for i in range(1, VOLUME_CAP)] + [f"recent_games_{VOLUME_CAP}+"],
         [str(i) for i in range(1, VOLUME_CAP)] + [f"{VOLUME_CAP}+"],
         "C  Recent ranked volume", "Games in previous 6h\n(reference: 0)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharex=False)
    for ax, (hyp, setting, terms, labels, title, xlabel) in zip(axes, panels):
        sub = _primary_effect_data(effects, hyp, setting)
        y = np.arange(len(terms))
        offsets = {"NA": -0.18, "KR": 0.0, "EU": 0.18}
        for source in REGIONS:
            s = sub[sub["source"] == source].set_index("term").reindex(terms)
            est = s["estimate_percentage_points"].astype(float).to_numpy()
            lo = s["ci_low_percentage_points"].astype(float).to_numpy()
            hi = s["ci_high_percentage_points"].astype(float).to_numpy()
            yy = y + offsets[source]
            ax.errorbar(est, yy, xerr=np.vstack([est - lo, hi - est]), fmt="o", capsize=2.5,
                        ms=4.8, lw=1.2, color=REGION_COLORS[source], label=source)
        ax.axvline(0, color="#333333", lw=1, ls="--", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Change in next-match win probability (percentage points)")
        ax.grid(axis="x", alpha=0.15)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    fig.suptitle("Adjusted within-player effects are small and inconsistent across regions",
                 x=0.02, ha="left", fontweight="bold", fontsize=14)
    fig.text(
        0.02,
        0.01,
        "Points are adjusted linear-probability estimates; bars are 95% CIs. "
        "Reference categories are omitted.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    transparent_save(fig, output)


def run_prediction(con: duckdb.DuckDBPyConnection, timelines: Path, out: Path) -> dict:
    """Train baselines and entropy trees on a chronological split and evaluate on held-out test data."""
    tables = out / 'tables'
    preds_dir = out / 'predictions'
    for d in (tables, preds_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Build the leakage-safe prediction sample, then split it chronologically.
    raw = load_primary_sample(con, timelines)
    data = add_engineered_features(raw)
    del raw
    gc.collect()
    data, split_thresholds = chronological_split(data)
    split_summary = (
        data.groupby(["source", "split"])
        .agg(
            rows=("target_win", "size"),
            players=("player_id", "nunique"),
            physical_matches=("match_id", "nunique"),
            win_rate=("target_win", "mean"),
            min_time_ms=("target_start_ms", "min"),
            max_time_ms=("target_start_ms", "max"),
        )
        .reset_index()
        .merge(split_thresholds, on="source", how="left")
    )
    save_csv(split_summary, tables / 'prediction_split_summary.csv')
    train = data[data['split'] == 'train'].copy()
    validation = data[data['split'] == 'validation'].copy()
    test = data[data['split'] == 'test'].copy()
    # Region/patch encoding and numeric imputation are learned from training only.
    context_schema = fit_context_schema(train)
    train_context = make_context_features(train, context_schema)
    val_context = make_context_features(validation, context_schema)
    test_context = make_context_features(test, context_schema)
    history_features = HISTORY_BASE_FEATURES + list(train_context.columns)
    behavior_features = BEHAVIOR_BASE_FEATURES + list(train_context.columns)
    combined_features = HISTORY_BASE_FEATURES + BEHAVIOR_BASE_FEATURES + list(train_context.columns)
    train_full = pd.concat([train[HISTORY_BASE_FEATURES + BEHAVIOR_BASE_FEATURES], train_context], axis=1)
    val_full = pd.concat([validation[HISTORY_BASE_FEATURES + BEHAVIOR_BASE_FEATURES], val_context], axis=1)
    test_full = pd.concat([test[HISTORY_BASE_FEATURES + BEHAVIOR_BASE_FEATURES], test_context], axis=1)
    train_full, val_full, test_full = fill_train_medians(
        train_full, val_full, test_full
    )
    y_train = train['target_win'].astype(int)
    y_val = validation['target_win'].astype(int)
    # Compare history-only, behavior-only, and combined information directly.
    feature_sets = {
        "history_tree": history_features,
        "behavior_tree": behavior_features,
        "combined_tree": combined_features,
    }
    models = {}
    grids = []
    for model_name, features in feature_sets.items():
        print(f'[q1 prediction] tuning {model_name}', flush=True)
        model, grid = decision_tree_grid(
            train_full[features],
            y_train,
            val_full[features],
            y_val,
            feature_set=model_name,
        )
        models[model_name] = model
        grids.append(grid)
    validation_grid = pd.concat(grids, ignore_index=True)
    save_csv(validation_grid, tables / 'validation_grid.csv')
    train_base_rate = float(y_train.mean())
    # Baselines make the held-out tree metrics meaningful.
    predictions: Dict[str, np.ndarray] = {
        "majority_baseline": np.full(len(test), train_base_rate, dtype=float),
        "historical_win_rate_baseline": (
            pd.to_numeric(test["prior_ranked_win_rate"], errors="coerce")
            .fillna(train_base_rate)
            .clip(0.001, 0.999)
            .to_numpy(dtype=float)
        ),
    }
    for name, model in models.items():
        predictions[name] = model.predict_proba(test_full[feature_sets[name]])[:, 1]
    metric_rows = []
    for name, prob in predictions.items():
        metric_rows.extend(evaluate_probabilities(test, prob, name))
    metrics = pd.DataFrame(metric_rows)
    save_csv(metrics, tables / 'test_metrics.csv')
    save_csv(metrics[['model', 'subgroup', 'n', 'tn', 'fp', 'fn', 'tp']], tables / 'confusion_matrices.csv')
    # Quantify the held-out gain from adding behavioral features to history.
    incremental = []
    for subgroup in ['ALL', *REGIONS]:
        h = metrics[(metrics['model'] == 'history_tree') & (metrics['subgroup'] == subgroup)].iloc[0]
        c = metrics[(metrics['model'] == 'combined_tree') & (metrics['subgroup'] == subgroup)].iloc[0]
        incremental.append(
            {
                "subgroup": subgroup,
                "history_auc": h["roc_auc"],
                "combined_auc": c["roc_auc"],
                "delta_auc_combined_minus_history": c["roc_auc"] - h["roc_auc"],
                "history_accuracy": h["accuracy"],
                "combined_accuracy": c["accuracy"],
                "delta_accuracy_combined_minus_history": c["accuracy"] - h["accuracy"],
                "history_f1": h["f1"],
                "combined_f1": c["f1"],
                "delta_f1_combined_minus_history": c["f1"] - h["f1"],
            }
        )
    incremental_df = pd.DataFrame(incremental)
    save_csv(incremental_df, tables / 'incremental_value.csv')
    # Importance describes tree usage; it is not a causal importance measure.
    importance = pd.DataFrame(
        {
            "feature": combined_features,
            "importance": models["combined_tree"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    save_csv(importance, tables / 'feature_importance.csv')
    pred_out = test[
        [
            "source",
            "player_id",
            "match_id",
            "target_start_ms",
            "target_win",
            "session_depth_30m",
            "ranked_games_prev_6h_capped",
            "previous_ranked_was_loss",
        ]
    ].copy()
    for name, prob in predictions.items():
        pred_out[f'prob_{name}'] = prob
    pred_out.to_parquet(preds_dir / 'test_predictions.parquet', index=False, compression='zstd')
    result = {
        "metrics": metrics,
        "incremental": incremental_df,
        "importance": importance,
        "test": test,
        "predictions": predictions,
        "models": models,
        "combined_features": combined_features,
    }
    del data, train, validation, train_full, val_full, test_full, train_context, val_context, test_context
    gc.collect()
    return result


def plot_prediction_evaluation(pred: dict, output: Path) -> None:
    """Create the ROC and normalized confusion-matrix report figure."""
    test = pred["test"]
    predictions = pred["predictions"]
    y = test["target_win"].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    styles = {
        "historical_win_rate_baseline": "Historical win rate",
        "history_tree": "History tree",
        "behavior_tree": "Behavior tree",
        "combined_tree": "Combined tree",
    }
    for name, label in styles.items():
        fpr, tpr, _ = roc_curve(y, predictions[name])
        auc = roc_auc_score(y, predictions[name])
        lw = 2.6 if name == "combined_tree" else 1.7
        ax.plot(fpr, tpr, lw=lw, label=f"{label}  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#666666")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("A  Chronological held-out ROC", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.12)
    ax.spines[["top", "right"]].set_visible(False)

    prob = predictions["combined_tree"]
    cls = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, cls, labels=[0, 1]).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )
    im = axes[1].imshow(norm, vmin=0, vmax=1, cmap="Blues")
    axes[1].set_xticks([0, 1], labels=["Predicted loss", "Predicted win"])
    axes[1].set_yticks([0, 1], labels=["Actual loss", "Actual win"])
    axes[1].set_title("B  Combined-tree confusion matrix", loc="left", fontweight="bold")
    for i in range(2):
        for j in range(2):
            axes[1].text(
                j,
                i,
                f"{norm[i, j] * 100:.1f}%\n(n={int(cm[i, j]):,})",
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if norm[i, j] > 0.55 else "#222222",
            )
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Row-normalized share")
    fig.suptitle(
        "Behavior adds only marginal ranking information, and losses remain hard to identify",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    transparent_save(fig, output)


def load_region_all_durations(
        con: duckdb.DuckDBPyConnection,
        path: Path,
) -> pd.DataFrame:
    """Load all target durations required by the robustness analyses."""
    columns = [
        "source",
        "player_id",
        "match_id",
        "target_win",
        "target_patch",
        "target_duration_s",
        "target_start_ms",
        "is_alias_confirmed",
        "prior_ranked_matches",
        "prior_ranked_win_rate",
        "prev_ranked_win",
        "prev_ranked_kda",
        "prev_ranked_loss_streak",
        "prev_ranked_queue_id",
        "prev_ranked_duration_s",
        "gap_from_prev_ranked_min",
        "post_loss_ranked_requeue_gap_min",
        "ranked_session_game_no_30m",
        "ranked_session_game_no_45m",
        "ranked_session_game_no_60m",
        "ranked_session_game_no_90m",
        "ranked_games_prev_3h",
        "ranked_games_prev_6h",
        "ranked_games_prev_12h",
        "ranked_games_prev_24h",
    ]
    parquet = sql_path(path)

    return con.execute(
        f"""
        SELECT {", ".join(columns)}
        FROM read_parquet('{parquet}')
        WHERE has_prior_ranked_match
          AND target_win IS NOT NULL
        """
    ).fetchdf()


def run_primary_models_local(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the three primary adjusted H1-H3 models on one sample."""
    coefs = []
    models = []
    for fn, args in (
            (fit_h1, (df, source, PRIMARY_SESSION_THRESHOLD, "adjusted")),
            (fit_h2, (df, source, "adjusted")),
            (fit_h3, (df, source, PRIMARY_VOLUME_WINDOW, "adjusted")),
    ):
        c, s = fn(*args)
        coefs.append(c)
        models.append(s)
    return pd.concat(coefs, ignore_index=True), pd.concat(models, ignore_index=True)


def plot_sensitivity_summary(primary_effects: pd.DataFrame, output: Path) -> None:
    # Useful supplementary course-style heatmap: maximum absolute adjusted
    # behavioral coefficient within each parameterization. It visualizes scale
    # and sensitivity without pretending that a single coefficient summarizes
    # direction.
    """Plot a heatmap of effect magnitudes across sensitivity settings."""
    x = primary_effects[primary_effects["specification"].isin(["adjusted", "adjusted_sensitivity"])].copy()
    rows = []
    for (source, hyp, setting), sub in x.groupby(["source", "hypothesis", "parameter_setting"]):
        rows.append(
            {
                "source": source,
                "hypothesis": hyp,
                "setting": setting,
                "max_abs_pp": sub["estimate_percentage_points"].abs().max(),
            }
        )
    d = pd.DataFrame(rows)
    # Create labels that remain readable in Word.
    d["label"] = (
            d["hypothesis"]
            .str.replace("H1_session_depth", "H1 session", regex=False)
            .str.replace("H2_post_loss_requeue", "H2 requeue", regex=False)
            .str.replace("H3_recent_volume", "H3 volume", regex=False)
            + " | "
            + d["setting"].astype(str)
    )
    pivot = d.pivot_table(index="label", columns="source", values="max_abs_pp", aggfunc="first")
    pivot = pivot.reindex(columns=list(REGIONS))
    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.38 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index, fontsize=8)
    ax.set_title("Sensitivity: largest adjusted effect within each specification", loc="left", fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j]
            if pd.notna(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Maximum |effect| (percentage points)")
    fig.tight_layout()
    transparent_save(fig, output)


def short_tree_feature_name(name: str) -> str:
    """Short labels used only inside tree nodes."""
    names = {
        "prior_ranked_mean_damage_per_min": "Prior dmg/min",
        "prior_ranked_mean_kda": "Prior KDA",
        "prior_ranked_mean_gold_per_min": "Prior gold/min",
        "prior_ranked_mean_cs_per_min": "Prior CS/min",
        "prior_ranked_win_rate": "Prior win rate",
        "prev_ranked_damage_per_min": "Prev dmg/min",
        "prev_ranked_gold_per_min": "Prev gold/min",
        "prev_ranked_cs_per_min": "Prev CS/min",
        "log1p_prev_ranked_kda": "Prev KDA (log)",
        "champion_changed_from_prev_ranked": "Champion changed",
        "role_changed_from_prev_ranked": "Role changed",
        "session_depth_30m": "Session depth",
        "ranked_games_prev_6h_capped": "Games prev 6h",
        "log1p_gap_from_prev_ranked_min": "Match gap (log)",
        "ranked_minutes_prev_6h": "Minutes prev 6h",
        "previous_ranked_was_loss": "Previous loss",
        "log1p_post_loss_gap_min": "Post-loss gap (log)",
    }
    if name in names:
        return names[name]
    if name.startswith("patch_"):
        return "Patch " + name.removeprefix("patch_")
    if name.startswith("region_"):
        return "Region " + name.removeprefix("region_")
    return name.replace("_", " ").title()


def plot_feature_importance(pred: dict, output: Path) -> None:
    """Plot the most important features used by the combined decision tree."""
    importance = pred["importance"].head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.barh([clean_feature_name(x) for x in importance["feature"]], importance["importance"])
    ax.set_xlabel("Decision-tree feature importance")
    ax.set_title("The combined tree relies mostly on prior performance", loc="left", fontweight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    transparent_save(fig, output)


def plot_tree_top_levels(pred: dict, output: Path) -> None:
    """Draw a simplified, presentation-focused view of the tree's top levels."""
    model = pred["models"]["combined_tree"]
    features = [short_tree_feature_name(x) for x in pred["combined_features"]]
    tree = model.tree_

    # Only the first three levels are shown. Unlike sklearn.plot_tree, this
    # custom view intentionally keeps just the split rule inside each node so
    # the figure remains readable when inserted into the report.
    max_display_depth = 2
    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    positions: dict[int, tuple[float, float]] = {}
    y_by_depth = {0: 0.84, 1: 0.53, 2: 0.22}

    def place(node_id: int, depth: int, left: float, right: float) -> None:
        """Assign balanced display coordinates to the visible tree nodes."""
        x = (left + right) / 2.0
        positions[node_id] = (x, y_by_depth[depth])

        if depth >= max_display_depth:
            return

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if left_child == right_child:
            return

        place(left_child, depth + 1, left, x)
        place(right_child, depth + 1, x, right)

    # Leave the right side free for the readability note.
    place(0, 0, 0.035, 0.745)

    # Draw connecting branches behind the node boxes.
    for node_id, (x, y) in positions.items():
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        for child in (left_child, right_child):
            if child not in positions:
                continue
            child_x, child_y = positions[child]
            ax.annotate(
                "",
                xy=(child_x, child_y + 0.055),
                xytext=(x, y - 0.055),
                arrowprops={
                    "arrowstyle": "-",
                    "linewidth": 1.6,
                    "color": "#777777",
                },
            )

    # Keep a single large piece of information in each node: the split rule.
    for node_id, (x, y) in positions.items():
        feature_index = tree.feature[node_id]

        if feature_index >= 0:
            feature_name = features[feature_index]
            threshold = float(tree.threshold[node_id])

            # Integer/binary-style thresholds are easier to read without
            # unnecessary decimal precision; continuous thresholds keep 2 dp.
            if abs(threshold - round(threshold)) < 0.005:
                threshold_text = f"{threshold:.0f}"
            else:
                threshold_text = f"{threshold:.2f}"

            label = f"{feature_name}\n≤ {threshold_text}"
        else:
            label = "Terminal node"

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.65,rounding_size=0.15",
                "facecolor": "#F5E6C8",
                "edgecolor": "#8A6B2B",
                "linewidth": 1.5,
            },
        )

    fig.suptitle(
        "How the combined tree makes its first decisions",
        x=0.04,
        y=0.98,
        ha="left",
        fontweight="bold",
        fontsize=17,
        color="#222222",
    )
    fig.text(
        0.04,
        0.925,
        "Top three levels of the validated pre-pruned entropy tree",
        ha="left",
        fontsize=11,
        color="#555555",
    )

    note = (
        "* For readability, each node shows only its split rule.\n"
        "  Sample counts, class proportions, and impurity are omitted.\n"
        "  The trained tree continues below the levels shown.\n"
        "  Left branch = rule true; right branch = rule false."
    )
    fig.text(
        0.79,
        0.38,
        note,
        ha="left",
        va="center",
        fontsize=10.5,
        color="#444444",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#F7F7F7",
            "edgecolor": "#BBBBBB",
            "linewidth": 1.0,
        },
    )

    fig.tight_layout(rect=[0, 0.02, 0.98, 0.90])
    transparent_save(fig, output)


def run_robustness(con: duckdb.DuckDBPyConnection, timelines: Path, out: Path) -> dict:
    """Repeat primary H1-H3 models for the alias-confirmed and >=5-minute samples."""
    tables = out / "tables"
    effect_frames, sample_rows = [], []
    # Refit the same primary models on provenance and short-game robustness samples.
    for source in REGIONS:
        df = load_region_all_durations(con, region_file(timelines, source))
        for variant, sub in sample_variants(df).items():
            sample_rows.append({
                "source": source,
                "sample_variant": variant,
                "rows": len(sub),
                "players": sub["player_id"].nunique(),
                "physical_matches": sub["match_id"].nunique(),
                "win_rate": sub["target_win"].astype(float).mean(),
            })
            coef, _ = run_primary_models_local(sub, source)
            coef["sample_variant"] = variant
            effect_frames.append(coef[coef["is_behavior_term"]].copy())
        del df
        gc.collect()

    effects = pd.concat(effect_frames, ignore_index=True)
    effects = add_holm_by_family(effects, ["sample_variant", "source", "hypothesis", "parameter_setting"])
    samples = pd.DataFrame(sample_rows)
    comparison = compare_to_primary(effects)
    save_csv(samples, tables / "robustness_sample_sizes.csv")
    save_csv(effects, tables / "robustness_behavior_effects.csv")
    save_csv(comparison, tables / "robustness_comparison_to_primary.csv")

    alias = comparison[comparison["sample_variant"] == "alias_confirmed_ge10m"]
    ge5 = comparison[comparison["sample_variant"] == "authoritative_ge5m"]
    return {
        "alias_max_abs_change_pp": float(alias["effect_difference_pp"].abs().max()) if not alias.empty else None,
        "ge5_max_abs_change_pp": float(ge5["effect_difference_pp"].abs().max()) if not ge5.empty else None,
        "holm_significant_total": int(effects["significant_holm_0_05"].sum()),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and default project paths."""
    root = project_root()
    p = argparse.ArgumentParser(description="Run Question 1 analysis and create report figures.")
    p.add_argument("--timelines", type=Path, default=root / "data/analysis/timelines/solo420_targets")
    p.add_argument("--output", type=Path, default=root / "data/analysis/q1")
    p.add_argument("--duckdb-memory-limit", default="4GB")
    p.add_argument("--duckdb-threads", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    """Run the complete Problem 1 analysis.

    Produces descriptive tables, adjusted within-player inference, chronological
    decision-tree evaluation, robustness checks, and the final report figures.
    """
    # 1) Prepare output folders and the in-memory DuckDB connection.
    args = parse_args()
    prepare_dir(args.output, args.overwrite)
    output_dirs = (
        args.output / "tables",
        args.output / "figures/report",
        args.output / "figures/supplementary",
        args.output / "predictions",
        args.output / "audit",
    )
    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    # DuckDB scans Parquet; pandas/sklearn handle the compact modeling samples.
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{args.duckdb_memory_limit}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")
    # 2) Run the four scientific stages in report order.
    try:
        print("\n=== Q1: descriptive analysis ===", flush=True)
        run_eda(con, args.timelines, args.output)

        print("\n=== Q1: within-player statistical analysis ===", flush=True)
        effects, model_summary = run_statistics(con, args.timelines, args.output)
        plot_adjusted_effects(effects, args.output / "figures/report/figure_2_adjusted_behavior_effects.png")
        plot_sensitivity_summary(
            effects,
            args.output
            / "figures/supplementary/supplementary_parameter_sensitivity_heatmap.png",
        )

        print("\n=== Q1: chronological predictive modeling ===", flush=True)
        pred = run_prediction(con, args.timelines, args.output)
        plot_prediction_evaluation(pred, args.output / "figures/report/figure_3_prediction_evaluation.png")
        plot_feature_importance(pred, args.output / "figures/report/figure_4_feature_importance.png")
        plot_tree_top_levels(pred, args.output / "figures/report/figure_5_tree_top_levels.png")

        print("\n=== Q1: robustness ===", flush=True)
        robustness = run_robustness(con, args.timelines, args.output)
    finally:
        con.close()

    # 3) Collect the small set of headline values used in the report/reproduction check.
    metrics = pred["metrics"]
    overall = metrics[metrics["subgroup"] == "ALL"].set_index("model")
    primary = model_summary[
        (model_summary["hypothesis"] == "H1_session_depth")
        & (model_summary["specification"] == "adjusted")
        & (model_summary["parameter_setting"] == "30m_session_boundary")
        ]
    key_results = pd.DataFrame([
        {"metric": "Primary target observations", "value": int(primary["n_rows"].sum())},
        {"metric": "History tree test ROC-AUC", "value": float(overall.loc["history_tree", "roc_auc"])},
        {"metric": "Behavior tree test ROC-AUC", "value": float(overall.loc["behavior_tree", "roc_auc"])},
        {"metric": "Combined tree test ROC-AUC", "value": float(overall.loc["combined_tree", "roc_auc"])},
        {
            "metric": "Combined - history AUC",
            "value": float(
                pred["incremental"]
                .loc[
                    pred["incremental"]["subgroup"] == "ALL",
                    "delta_auc_combined_minus_history",
                ]
                .iloc[0]
            ),
        },
        {"metric": "Holm-significant robustness terms", "value": robustness["holm_significant_total"]},
        {"metric": "Max >=5m coefficient change (pp)", "value": robustness["ge5_max_abs_change_pp"]},
    ])
    save_csv(key_results, args.output / "tables/key_results_for_report.csv")

    # 4) Save a compact audit summary listing the frozen Q1 settings and figures.
    summary = {
        "question": (
            "How are recent competitive volume, session depth, and post-loss requeue "
            "timing associated with subsequent Ranked Solo/Duo performance?"
        ),
        "primary_session_boundary_minutes": PRIMARY_SESSION_THRESHOLD,
        "session_sensitivity_minutes": list(SESSION_SENSITIVITY),
        "primary_volume_window_hours": PRIMARY_VOLUME_WINDOW,
        "volume_sensitivity_hours": list(VOLUME_SENSITIVITY),
        "primary_target_duration_min": 10,
        "report_figures": [
            "figure_1_inter_match_gap_ecdf.png",
            "figure_2_adjusted_behavior_effects.png",
            "figure_3_prediction_evaluation.png",
            "figure_4_feature_importance.png",
            "figure_5_tree_top_levels.png",
        ],
        "robustness": robustness,
    }
    (args.output / "audit/q1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nQ1 ANALYSIS COMPLETE")
    print(key_results.to_string(index=False))
    print(f"\nReport figures: {args.output / 'figures/report'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

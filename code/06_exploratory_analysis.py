#!/usr/bin/env python3
r"""
06_exploratory_analysis.py

Research-driven EDA for the main League of Legends behavioral/temporal question:

    How are recent ranked-game volume, session depth, and post-loss requeue
    timing associated with performance in a player's subsequent ranked match?

INPUT
-----
Analysis-ready target-centric timelines created by:
    05_build_player_timelines.py

Expected input folder:
    data/analysis/timelines/solo420_targets/

OUTPUT
------
data/analysis/eda/
├── audit/
│   └── eda_validation_summary.csv
├── tables/
│   ├── sample_overview.csv
│   ├── gap_quantiles.csv
│   ├── gap_threshold_coverage.csv
│   ├── duration_sensitivity.csv
│   ├── session_threshold_summary.csv
│   ├── session_depth_outcomes.csv
│   ├── post_loss_requeue_outcomes.csv
│   ├── recent_volume_outcomes.csv
│   └── baseline_outcomes_by_region.csv
└── figures/
    ├── inter_match_gap_ecdf.png
    ├── post_loss_requeue_winrate.png
    ├── recent_volume_6h_winrate.png
    ├── session_depth_30m_winrate.png
    ├── session_depth_45m_winrate.png
    ├── session_depth_60m_winrate.png
    └── session_depth_90m_winrate.png

PURPOSE
-------
This script is intentionally descriptive/exploratory.

It does NOT:
- perform causal inference;
- choose the final session threshold;
- choose the final remake/short-game policy;
- fit the final predictive model;
- report p-values.

It DOES:
- validate the timeline table again;
- quantify candidate session definitions;
- show the inter-match-gap structure;
- describe post-loss requeue behavior;
- describe session-depth and recent-volume relationships with next-match outcomes;
- report sample sizes and uncertainty;
- provide evidence for later methodological choices.

Important:
- Target is queue 420.
- Predictor/history columns are pre-target features produced by script 05.
- target_* performance columns are outcomes, never predictors.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SESSION_THRESHOLDS = (30, 45, 60, 90)
RECENT_WINDOWS_HOURS = (3, 6, 12, 24)

# Descriptive bins only. These are not the final inferential parameterization.
POST_LOSS_BINS = [
    (-np.inf, 5, "<=5m"),
    (5, 10, "5-10m"),
    (10, 20, "10-20m"),
    (20, 30, "20-30m"),
    (30, 60, "30-60m"),
    (60, 120, "1-2h"),
    (120, 360, "2-6h"),
    (360, 1440, "6-24h"),
    (1440, np.inf, ">24h"),
]

SESSION_DEPTH_CAP = 8
RECENT_VOLUME_CAP = 6


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
        help="Folder containing NA/KR/EU solo420 target Parquet files.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duckdb-memory-limit", default="4GB")
    p.add_argument("--duckdb-threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parquet_glob(folder: Path) -> str:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in: {folder}")
    return sql_path(folder / "*.parquet")


def wilson_interval(wins: float, n: float, z: float = 1.959963984540054):
    if n is None or n <= 0:
        return np.nan, np.nan
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
        / denom
    )
    return center - half, center + half


def add_winrate_ci(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["win_rate"] = out["wins"] / out["n"]
    cis = [wilson_interval(w, n) for w, n in zip(out["wins"], out["n"])]
    out["win_rate_ci_low"] = [x[0] for x in cis]
    out["win_rate_ci_high"] = [x[1] for x in cis]
    return out


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def bin_post_loss_gap(value):
    if pd.isna(value):
        return None
    for lo, hi, label in POST_LOSS_BINS:
        if value > lo and value <= hi:
            return label
    return None


def post_loss_order() -> list[str]:
    return [x[2] for x in POST_LOSS_BINS]


def depth_label(value: int) -> str:
    return str(value) if value < SESSION_DEPTH_CAP else f"{SESSION_DEPTH_CAP}+"


def volume_label(value: int) -> str:
    return str(value) if value < RECENT_VOLUME_CAP else f"{RECENT_VOLUME_CAP}+"


def plot_gap_ecdf(sample: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for source in ("NA", "KR", "EU"):
        x = (
            sample.loc[
                (sample["source"] == source)
                & sample["gap_from_prev_ranked_min"].notna()
                & (sample["gap_from_prev_ranked_min"] >= 0)
                & (sample["gap_from_prev_ranked_min"] <= 1440),
                "gap_from_prev_ranked_min",
            ]
            .astype(float)
            .sort_values()
            .to_numpy()
        )
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=source)

    ax.set_xscale("log")
    ax.set_xlabel("Gap from previous ranked match end to target start (minutes, log scale)")
    ax.set_ylabel("Cumulative share of observed gaps")
    ax.set_title("Observed ranked inter-match gaps (up to 24 hours)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_winrate_table(
    df: pd.DataFrame,
    x_col: str,
    order: Iterable[str],
    title: str,
    xlabel: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    order = list(order)
    x = np.arange(len(order))

    for source in ("NA", "KR", "EU"):
        sub = (
            df[df["source"] == source]
            .set_index(x_col)
            .reindex(order)
            .reset_index()
        )
        y = sub["win_rate"].astype(float).to_numpy()
        low = sub["win_rate_ci_low"].astype(float).to_numpy()
        high = sub["win_rate_ci_high"].astype(float).to_numpy()
        valid = ~np.isnan(y)

        if valid.any():
            ax.errorbar(
                x[valid],
                y[valid],
                yerr=np.vstack([y[valid] - low[valid], high[valid] - y[valid]]),
                marker="o",
                capsize=3,
                label=source,
            )

    ax.axhline(0.5, linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Target-match win rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_dir(args.output, args.overwrite)

    audit_dir = args.output / "audit"
    table_dir = args.output / "tables"
    figure_dir = args.output / "figures"
    audit_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    glob = parquet_glob(args.timelines)

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{args.duckdb_memory_limit}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")

    # ------------------------------------------------------------------
    # 1. Validate input and materialize the descriptive sample.
    # ------------------------------------------------------------------
    required = {
        "source",
        "player_id",
        "match_id",
        "target_queue_id",
        "target_start_ms",
        "target_duration_s",
        "target_win",
        "target_kda",
        "target_damage_to_champions_per_min",
        "target_cs_per_min",
        "target_gold_per_min",
        "target_game_complete",
        "target_under_5_min",
        "target_under_10_min",
        "has_prior_ranked_match",
        "gap_from_prev_ranked_min",
        "prev_ranked_win",
        "post_loss_ranked_requeue_gap_min",
        "prev_ranked_loss_streak",
        "prior_ranked_win_rate",
        "ranked_games_prev_3h",
        "ranked_games_prev_6h",
        "ranked_games_prev_12h",
        "ranked_games_prev_24h",
        "ranked_session_game_no_30m",
        "ranked_session_game_no_45m",
        "ranked_session_game_no_60m",
        "ranked_session_game_no_90m",
        "solo_session_game_no_30m",
        "solo_session_game_no_45m",
        "solo_session_game_no_60m",
        "solo_session_game_no_90m",
    }

    cols = set(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob}')")
        .fetchdf()["column_name"]
        .astype(str)
    )
    missing = sorted(required - cols)
    if missing:
        raise RuntimeError(f"Timeline input is missing required columns: {missing}")

    validation_queries = {
        "rows_total": f"SELECT COUNT(*) FROM read_parquet('{glob}')",
        "duplicate_source_player_match": f"""
            SELECT COUNT(*) FROM (
                SELECT source, player_id, match_id
                FROM read_parquet('{glob}')
                GROUP BY source, player_id, match_id
                HAVING COUNT(*) > 1
            )
        """,
        "non_420_target_rows": f"""
            SELECT COUNT(*) FROM read_parquet('{glob}')
            WHERE target_queue_id <> 420 OR target_queue_id IS NULL
        """,
        "negative_ranked_gap_rows": f"""
            SELECT COUNT(*) FROM read_parquet('{glob}')
            WHERE gap_from_prev_ranked_min < 0
        """,
        "null_target_win_rows": f"""
            SELECT COUNT(*) FROM read_parquet('{glob}')
            WHERE target_win IS NULL
        """,
        "rows_with_prior_ranked": f"""
            SELECT COUNT(*) FROM read_parquet('{glob}')
            WHERE has_prior_ranked_match
        """,
    }

    validation_rows = []
    for check, query in validation_queries.items():
        validation_rows.append(
            {"check": check, "value": int(con.execute(query).fetchone()[0])}
        )

    validation_df = pd.DataFrame(validation_rows)
    save_csv(validation_df, audit_dir / "eda_validation_summary.csv")

    problems = {
        x["check"]: x["value"]
        for x in validation_rows
        if x["check"]
        in {
            "duplicate_source_player_match",
            "non_420_target_rows",
            "negative_ranked_gap_rows",
        }
        and x["value"] != 0
    }
    if problems:
        raise RuntimeError(f"EDA input validation failed: {problems}")

    con.execute("DROP TABLE IF EXISTS sample")
    con.execute(
        f"""
        CREATE TEMP TABLE sample AS
        SELECT *
        FROM read_parquet('{glob}')
        WHERE has_prior_ranked_match
          AND target_win IS NOT NULL
        """
    )

    # ------------------------------------------------------------------
    # 2. High-level sample overview.
    # ------------------------------------------------------------------
    overview = con.execute(
        """
        SELECT
            source,
            COUNT(*)::BIGINT AS target_rows,
            COUNT(DISTINCT player_id)::BIGINT AS players,
            COUNT(DISTINCT match_id)::BIGINT AS physical_matches,
            SUM(prev_ranked_win = FALSE)::BIGINT AS targets_after_ranked_loss,
            100.0 * SUM(prev_ranked_win = FALSE) / COUNT(*) AS pct_after_ranked_loss,
            SUM(target_under_5_min)::BIGINT AS targets_under_5_min,
            SUM(target_under_10_min)::BIGINT AS targets_under_10_min,
            100.0 * SUM(target_under_5_min) / COUNT(*) AS pct_under_5_min,
            100.0 * SUM(target_under_10_min) / COUNT(*) AS pct_under_10_min,
            AVG(CASE WHEN target_win THEN 1.0 ELSE 0.0 END) AS win_rate,
            AVG(target_kda) AS mean_target_kda,
            MEDIAN(target_kda) AS median_target_kda
        FROM sample
        GROUP BY source
        ORDER BY source
        """
    ).fetchdf()
    save_csv(overview, table_dir / "sample_overview.csv")

    baseline = con.execute(
        """
        SELECT
            source,
            COUNT(*)::BIGINT AS n,
            SUM(CASE WHEN target_win THEN 1 ELSE 0 END)::BIGINT AS wins,
            AVG(target_kda) AS mean_kda,
            MEDIAN(target_kda) AS median_kda,
            AVG(target_damage_to_champions_per_min) AS mean_damage_per_min,
            AVG(target_cs_per_min) AS mean_cs_per_min,
            AVG(target_gold_per_min) AS mean_gold_per_min
        FROM sample
        GROUP BY source
        ORDER BY source
        """
    ).fetchdf()
    baseline = add_winrate_ci(baseline)
    save_csv(baseline, table_dir / "baseline_outcomes_by_region.csv")

    # ------------------------------------------------------------------
    # 3. Gap distribution and threshold coverage.
    # ------------------------------------------------------------------
    gap_rows = []
    for scope, col in (
        ("ranked_420_plus_440", "gap_from_prev_ranked_min"),
        ("solo420_only", "gap_from_prev_solo_min"),
    ):
        if col not in cols:
            continue
        df = con.execute(
            f"""
            SELECT
                source,
                '{scope}' AS history_scope,
                COUNT({col})::BIGINT AS n,
                MIN({col}) AS min_min,
                quantile_cont({col}, 0.01) AS p01_min,
                quantile_cont({col}, 0.05) AS p05_min,
                quantile_cont({col}, 0.10) AS p10_min,
                quantile_cont({col}, 0.25) AS p25_min,
                MEDIAN({col}) AS median_min,
                quantile_cont({col}, 0.75) AS p75_min,
                quantile_cont({col}, 0.90) AS p90_min,
                quantile_cont({col}, 0.95) AS p95_min,
                quantile_cont({col}, 0.99) AS p99_min,
                MAX({col}) AS max_min
            FROM sample
            WHERE {col} IS NOT NULL AND {col} >= 0
            GROUP BY source
            """
        ).fetchdf()
        gap_rows.append(df)
    gap_quantiles = pd.concat(gap_rows, ignore_index=True)
    save_csv(gap_quantiles, table_dir / "gap_quantiles.csv")

    threshold_rows = []
    for scope, col in (
        ("ranked_420_plus_440", "gap_from_prev_ranked_min"),
        ("solo420_only", "gap_from_prev_solo_min"),
    ):
        if col not in cols:
            continue
        for threshold in (10, 15, 20, 30, 45, 60, 90, 120, 180, 360, 720, 1440):
            q = con.execute(
                f"""
                SELECT
                    source,
                    COUNT(*)::BIGINT AS n_gaps,
                    SUM({col} <= {threshold})::BIGINT AS n_at_or_below
                FROM sample
                WHERE {col} IS NOT NULL AND {col} >= 0
                GROUP BY source
                """
            ).fetchdf()
            q["history_scope"] = scope
            q["threshold_minutes"] = threshold
            q["pct_at_or_below"] = 100.0 * q["n_at_or_below"] / q["n_gaps"]
            threshold_rows.append(q)
    gap_thresholds = pd.concat(threshold_rows, ignore_index=True)
    save_csv(gap_thresholds, table_dir / "gap_threshold_coverage.csv")

    # For the ECDF plot we only need raw gaps and source.
    gap_sample = con.execute(
        """
        SELECT source, gap_from_prev_ranked_min
        FROM sample
        WHERE gap_from_prev_ranked_min IS NOT NULL
          AND gap_from_prev_ranked_min >= 0
          AND gap_from_prev_ranked_min <= 1440
        """
    ).fetchdf()
    plot_gap_ecdf(gap_sample, figure_dir / "inter_match_gap_ecdf.png")

    # ------------------------------------------------------------------
    # 4. Short-game/remake sensitivity.
    # This does not choose the final policy.
    # ------------------------------------------------------------------
    duration_sensitivity = con.execute(
        """
        WITH variants AS (
            SELECT source, 'all_targets' AS sample_variant, * EXCLUDE(source) FROM sample
            UNION ALL
            SELECT source, 'target_at_least_5m', * EXCLUDE(source)
            FROM sample WHERE NOT target_under_5_min
            UNION ALL
            SELECT source, 'target_at_least_10m', * EXCLUDE(source)
            FROM sample WHERE NOT target_under_10_min
            UNION ALL
            SELECT source, 'game_complete_only', * EXCLUDE(source)
            FROM sample WHERE target_game_complete
        )
        SELECT
            source,
            sample_variant,
            COUNT(*)::BIGINT AS n,
            SUM(CASE WHEN target_win THEN 1 ELSE 0 END)::BIGINT AS wins,
            AVG(target_kda) AS mean_kda,
            MEDIAN(target_kda) AS median_kda,
            AVG(target_damage_to_champions_per_min) AS mean_damage_per_min
        FROM variants
        GROUP BY source, sample_variant
        ORDER BY source, sample_variant
        """
    ).fetchdf()
    duration_sensitivity = add_winrate_ci(duration_sensitivity)
    save_csv(duration_sensitivity, table_dir / "duration_sensitivity.csv")

    # ------------------------------------------------------------------
    # 5. Candidate session definitions.
    # Report both all-ranked history and Solo-only history.
    # ------------------------------------------------------------------
    session_summary_rows = []
    session_depth_rows = []

    for scope, prefix in (
        ("ranked_420_plus_440", "ranked"),
        ("solo420_only", "solo"),
    ):
        for threshold in SESSION_THRESHOLDS:
            col = f"{prefix}_session_game_no_{threshold}m"
            if col not in cols:
                continue

            s = con.execute(
                f"""
                SELECT
                    source,
                    COUNT(*)::BIGINT AS target_rows,
                    SUM({col} = 1)::BIGINT AS first_game_targets,
                    SUM({col} >= 2)::BIGINT AS deeper_session_targets,
                    SUM({col} >= 3)::BIGINT AS depth_3plus_targets,
                    SUM({col} >= 5)::BIGINT AS depth_5plus_targets,
                    AVG({col}) AS mean_session_game_no,
                    MEDIAN({col}) AS median_session_game_no,
                    quantile_cont({col}, 0.90) AS p90_session_game_no,
                    MAX({col})::BIGINT AS max_session_game_no
                FROM sample
                GROUP BY source
                """
            ).fetchdf()
            s["history_scope"] = scope
            s["threshold_minutes"] = threshold
            s["pct_deeper_session"] = 100.0 * s["deeper_session_targets"] / s["target_rows"]
            s["pct_depth_3plus"] = 100.0 * s["depth_3plus_targets"] / s["target_rows"]
            s["pct_depth_5plus"] = 100.0 * s["depth_5plus_targets"] / s["target_rows"]
            session_summary_rows.append(s)

            d = con.execute(
                f"""
                SELECT
                    source,
                    LEAST({col}, {SESSION_DEPTH_CAP})::BIGINT AS depth_bucket_num,
                    COUNT(*)::BIGINT AS n,
                    SUM(CASE WHEN target_win THEN 1 ELSE 0 END)::BIGINT AS wins,
                    AVG(target_kda) AS mean_kda,
                    MEDIAN(target_kda) AS median_kda,
                    AVG(target_damage_to_champions_per_min) AS mean_damage_per_min,
                    AVG(target_cs_per_min) AS mean_cs_per_min,
                    AVG(target_gold_per_min) AS mean_gold_per_min
                FROM sample
                GROUP BY source, depth_bucket_num
                """
            ).fetchdf()
            d["history_scope"] = scope
            d["threshold_minutes"] = threshold
            d["session_depth"] = d["depth_bucket_num"].map(depth_label)
            d = add_winrate_ci(d)
            session_depth_rows.append(d)

    session_summary = pd.concat(session_summary_rows, ignore_index=True)
    session_depth = pd.concat(session_depth_rows, ignore_index=True)

    save_csv(session_summary, table_dir / "session_threshold_summary.csv")
    save_csv(session_depth, table_dir / "session_depth_outcomes.csv")

    # Plot only ranked-history variants at this exploratory stage.
    for threshold in SESSION_THRESHOLDS:
        p = session_depth[
            (session_depth["history_scope"] == "ranked_420_plus_440")
            & (session_depth["threshold_minutes"] == threshold)
        ].copy()

        plot_winrate_table(
            p,
            x_col="session_depth",
            order=[str(x) for x in range(1, SESSION_DEPTH_CAP)]
            + [f"{SESSION_DEPTH_CAP}+"],
            title=f"Target win rate by observed ranked-session depth ({threshold}m boundary)",
            xlabel="Game number within observed ranked session",
            output=figure_dir / f"session_depth_{threshold}m_winrate.png",
        )

    # ------------------------------------------------------------------
    # 6. Post-loss requeue behavior.
    # ------------------------------------------------------------------
    post_loss = con.execute(
        """
        SELECT
            source,
            post_loss_ranked_requeue_gap_min,
            target_win,
            target_kda,
            target_damage_to_champions_per_min,
            target_cs_per_min,
            target_gold_per_min
        FROM sample
        WHERE prev_ranked_win = FALSE
          AND post_loss_ranked_requeue_gap_min IS NOT NULL
          AND post_loss_ranked_requeue_gap_min >= 0
        """
    ).fetchdf()

    post_loss["requeue_bin"] = post_loss["post_loss_ranked_requeue_gap_min"].map(
        bin_post_loss_gap
    )
    post_loss["requeue_bin"] = pd.Categorical(
        post_loss["requeue_bin"],
        categories=post_loss_order(),
        ordered=True,
    )

    post_loss_table = (
        post_loss.groupby(["source", "requeue_bin"], observed=True)
        .agg(
            n=("target_win", "size"),
            wins=("target_win", "sum"),
            mean_kda=("target_kda", "mean"),
            median_kda=("target_kda", "median"),
            mean_damage_per_min=("target_damage_to_champions_per_min", "mean"),
            mean_cs_per_min=("target_cs_per_min", "mean"),
            mean_gold_per_min=("target_gold_per_min", "mean"),
            median_gap_min=("post_loss_ranked_requeue_gap_min", "median"),
        )
        .reset_index()
    )
    post_loss_table = add_winrate_ci(post_loss_table)
    save_csv(post_loss_table, table_dir / "post_loss_requeue_outcomes.csv")

    plot_winrate_table(
        post_loss_table,
        x_col="requeue_bin",
        order=post_loss_order(),
        title="Next Solo/Duo win rate after a ranked loss, by requeue gap",
        xlabel="Gap after previous ranked loss",
        output=figure_dir / "post_loss_requeue_winrate.png",
    )

    # ------------------------------------------------------------------
    # 7. Recent ranked volume.
    # ------------------------------------------------------------------
    recent_rows = []
    for h in RECENT_WINDOWS_HOURS:
        col = f"ranked_games_prev_{h}h"
        df = con.execute(
            f"""
            SELECT
                source,
                LEAST({col}, {RECENT_VOLUME_CAP})::BIGINT AS volume_bucket_num,
                COUNT(*)::BIGINT AS n,
                SUM(CASE WHEN target_win THEN 1 ELSE 0 END)::BIGINT AS wins,
                AVG(target_kda) AS mean_kda,
                MEDIAN(target_kda) AS median_kda,
                AVG(target_damage_to_champions_per_min) AS mean_damage_per_min,
                AVG(target_cs_per_min) AS mean_cs_per_min,
                AVG(target_gold_per_min) AS mean_gold_per_min
            FROM sample
            GROUP BY source, volume_bucket_num
            """
        ).fetchdf()
        df["window_hours"] = h
        df["recent_games"] = df["volume_bucket_num"].map(volume_label)
        df = add_winrate_ci(df)
        recent_rows.append(df)

    recent_volume = pd.concat(recent_rows, ignore_index=True)
    save_csv(recent_volume, table_dir / "recent_volume_outcomes.csv")

    p6 = recent_volume[recent_volume["window_hours"] == 6].copy()
    plot_winrate_table(
        p6,
        x_col="recent_games",
        order=[str(x) for x in range(0, RECENT_VOLUME_CAP)]
        + [f"{RECENT_VOLUME_CAP}+"],
        title="Target Solo/Duo win rate by ranked games played in previous 6 hours",
        xlabel="Observed ranked games in previous 6 hours",
        output=figure_dir / "recent_volume_6h_winrate.png",
    )

    # ------------------------------------------------------------------
    # 8. Save a machine-readable EDA summary.
    # ------------------------------------------------------------------
    payload = {
        "purpose": "Research-driven descriptive EDA before inferential modeling.",
        "main_target_queue": 420,
        "analysis_sample_rule": (
            "Target rows with a prior ranked match and non-null target win."
        ),
        "candidate_session_thresholds_minutes": list(SESSION_THRESHOLDS),
        "recent_volume_windows_hours": list(RECENT_WINDOWS_HOURS),
        "post_loss_bins": post_loss_order(),
        "important_notes": [
            "All figures are descriptive and unadjusted.",
            "Do not interpret win-rate differences as causal effects.",
            "Final session threshold must be justified from gap/session evidence and sensitivity.",
            "Final short-game/remake policy must be justified from duration sensitivity.",
            "Later inference should account for repeated players and shared physical matches.",
            "Later prediction should use chronological train/test separation and explicit baselines.",
        ],
    }
    (audit_dir / "eda_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    con.close()

    print("\nRESEARCH EDA COMPLETE\n")
    print(overview.to_string(index=False))
    print("\nCreated tables:")
    for p in sorted(table_dir.glob("*.csv")):
        print(f"  {p}")
    print("\nCreated figures:")
    for p in sorted(figure_dir.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()

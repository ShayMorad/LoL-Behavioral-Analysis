#!/usr/bin/env python3
r"""
09_q1_robustness.py

Final targeted robustness / closure stage for Question 1.

PURPOSE
-------
Question 1 has already been analyzed through:
    06_exploratory_analysis.py
    07_statistical_analysis.py
    08_predictive_modeling.py

This script does NOT open a new analysis branch. It checks whether the primary
within-player conclusions survive two important methodological changes:

1. Tracking-cohort robustness:
       authoritative >=10m targets
       vs
       alias-confirmed >=10m targets

2. Short-game robustness:
       authoritative >=10m targets
       vs
       authoritative >=5m targets

It also summarizes the session-boundary and recent-volume-window sensitivity
that was already computed in 07_statistical_analysis.py.

PRIMARY Q1 SPECIFICATIONS
-------------------------
H1 session depth:
    30-minute ranked-session boundary

H2 post-loss requeue:
    categorical post-loss requeue gap;
    previous ranked loss must itself be >=10 minutes

H3 recent ranked volume:
    ranked games in previous 6 hours

All models use the same adjusted within-player specification implemented in
07_statistical_analysis.py.

OUTPUT
------
data/analysis/q1_robustness/
├── tables/
│   ├── robustness_sample_sizes.csv
│   ├── robustness_behavior_effects.csv
│   ├── robustness_comparison_to_primary.csv
│   ├── robustness_family_summary.csv
│   └── parameter_sensitivity_summary.csv
└── audit/
    └── q1_robustness_summary.json

The goal is to decide whether Question 1 can be frozen for the final report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


REGIONS = ("NA", "KR", "EU")

TIMELINE_COLUMNS = [
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
    p.add_argument(
        "--statistics",
        type=Path,
        required=True,
        help="Existing data/analysis/statistics folder produced by script 07.",
    )
    p.add_argument(
        "--statistics-script",
        type=Path,
        default=None,
        help=(
            "Path to 07_statistical_analysis.py. If omitted, the script "
            "expects it next to this file."
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duckdb-memory-limit", default="4GB")
    p.add_argument("--duckdb-threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_statistics_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing statistical-analysis script: {path}")

    spec = importlib.util.spec_from_file_location("q1_stats07", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """
    Holm family-wise-error correction.
    """
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
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


def region_file(folder: Path, source: str) -> Path:
    p = folder / f"{source}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing timeline file: {p}")
    return p


def load_region(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> pd.DataFrame:
    available = set(
        con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_path(path)}')"
        )
        .fetchdf()["column_name"]
        .astype(str)
    )

    missing = sorted(set(TIMELINE_COLUMNS) - available)
    if missing:
        raise RuntimeError(f"{path.name}: missing columns: {missing}")

    cols = ", ".join(TIMELINE_COLUMNS)

    return con.execute(
        f"""
        SELECT {cols}
        FROM read_parquet('{sql_path(path)}')
        WHERE has_prior_ranked_match
          AND target_win IS NOT NULL
        """
    ).fetchdf()


def sample_variants(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    target_duration = pd.to_numeric(
        df["target_duration_s"], errors="coerce"
    )

    ge10 = df[target_duration >= 600].copy()
    alias_ge10 = ge10[
        ge10["is_alias_confirmed"].fillna(False).astype(bool)
    ].copy()
    ge5 = df[target_duration >= 300].copy()

    return {
        "authoritative_ge10m": ge10,
        "alias_confirmed_ge10m": alias_ge10,
        "authoritative_ge5m": ge5,
    }


def run_primary_models(stats07, df: pd.DataFrame, source: str):
    coef_frames = []
    summary_frames = []

    c, s = stats07.fit_h1(
        df,
        source,
        stats07.PRIMARY_SESSION_THRESHOLD,
        "adjusted",
    )
    coef_frames.append(c)
    summary_frames.append(s)

    c, s = stats07.fit_h2(
        df,
        source,
        "adjusted",
    )
    coef_frames.append(c)
    summary_frames.append(s)

    c, s = stats07.fit_h3(
        df,
        source,
        stats07.PRIMARY_VOLUME_WINDOW,
        "adjusted",
    )
    coef_frames.append(c)
    summary_frames.append(s)

    return (
        pd.concat(coef_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )


def add_family_holm(effect_df: pd.DataFrame) -> pd.DataFrame:
    out = effect_df.copy()
    out["p_value_holm"] = np.nan

    group_cols = [
        "sample_variant",
        "source",
        "hypothesis",
        "parameter_setting",
    ]

    for _, idx in out.groupby(group_cols).groups.items():
        idx = list(idx)
        out.loc[idx, "p_value_holm"] = holm_adjust(
            out.loc[idx, "p_value_two_sided"]
        ).to_numpy()

    out["significant_raw_0_05"] = out["p_value_two_sided"] < 0.05
    out["significant_holm_0_05"] = out["p_value_holm"] < 0.05
    return out


def compare_to_primary(
    effects: pd.DataFrame,
) -> pd.DataFrame:
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


def family_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for keys, sub in effects.groupby(
        ["sample_variant", "source", "hypothesis", "parameter_setting"]
    ):
        variant, source, hypothesis, setting = keys

        rows.append(
            {
                "sample_variant": variant,
                "source": source,
                "hypothesis": hypothesis,
                "parameter_setting": setting,
                "behavior_terms": len(sub),
                "max_abs_effect_pp": float(
                    sub["estimate_percentage_points"].abs().max()
                ),
                "median_abs_effect_pp": float(
                    sub["estimate_percentage_points"].abs().median()
                ),
                "raw_p_lt_0_05_terms": int(
                    sub["significant_raw_0_05"].sum()
                ),
                "holm_p_lt_0_05_terms": int(
                    sub["significant_holm_0_05"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_parameter_sensitivity(
    primary_effects_path: Path,
) -> pd.DataFrame:
    """
    Summarize already-computed sensitivity specifications from script 07.
    No models are rerun here.
    """
    df = pd.read_csv(primary_effects_path, keep_default_na=False)

    df = df[df["is_behavior_term"].astype(str).str.lower() == "true"].copy()

    # Holm within source/hypothesis/specification/setting family.
    df["p_value_holm"] = np.nan
    groups = [
        "source",
        "hypothesis",
        "specification",
        "parameter_setting",
    ]

    for _, idx in df.groupby(groups).groups.items():
        idx = list(idx)
        df.loc[idx, "p_value_holm"] = holm_adjust(
            pd.to_numeric(
                df.loc[idx, "p_value_two_sided"], errors="coerce"
            )
        ).to_numpy()

    rows = []
    for keys, sub in df.groupby(groups):
        source, hypothesis, specification, setting = keys

        rows.append(
            {
                "source": source,
                "hypothesis": hypothesis,
                "specification": specification,
                "parameter_setting": setting,
                "behavior_terms": len(sub),
                "max_abs_effect_pp": float(
                    pd.to_numeric(
                        sub["estimate_percentage_points"],
                        errors="coerce",
                    ).abs().max()
                ),
                "median_abs_effect_pp": float(
                    pd.to_numeric(
                        sub["estimate_percentage_points"],
                        errors="coerce",
                    ).abs().median()
                ),
                "raw_p_lt_0_05_terms": int(
                    (
                        pd.to_numeric(
                            sub["p_value_two_sided"], errors="coerce"
                        )
                        < 0.05
                    ).sum()
                ),
                "holm_p_lt_0_05_terms": int(
                    (sub["p_value_holm"] < 0.05).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    prepare_dir(args.output, args.overwrite)

    table_dir = args.output / "tables"
    audit_dir = args.output / "audit"
    table_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    stats_script = (
        args.statistics_script
        if args.statistics_script is not None
        else Path(__file__).resolve().parent / "07_statistical_analysis.py"
    )

    stats07 = load_statistics_module(stats_script)

    primary_effects_path = (
        args.statistics / "tables" / "behavior_effects.csv"
    )
    if not primary_effects_path.exists():
        raise FileNotFoundError(
            f"Missing script-07 behavior effects: {primary_effects_path}"
        )

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{args.duckdb_memory_limit}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")

    coef_frames = []
    model_frames = []
    sample_rows = []

    try:
        for source in REGIONS:
            print(f"[Q1 robustness] {source}: loading timeline", flush=True)
            df = load_region(con, region_file(args.timelines, source))

            for variant_name, sub in sample_variants(df).items():
                print(
                    f"  {variant_name}: {len(sub):,} rows, "
                    f"{sub['player_id'].nunique():,} players",
                    flush=True,
                )

                sample_rows.append(
                    {
                        "source": source,
                        "sample_variant": variant_name,
                        "rows": len(sub),
                        "players": sub["player_id"].nunique(),
                        "physical_matches": sub["match_id"].nunique(),
                        "win_rate": sub["target_win"].astype(float).mean(),
                    }
                )

                coef, models = run_primary_models(stats07, sub, source)
                coef["sample_variant"] = variant_name
                models["sample_variant"] = variant_name

                coef_frames.append(coef)
                model_frames.append(models)

    finally:
        con.close()

    all_coef = pd.concat(coef_frames, ignore_index=True)
    all_models = pd.concat(model_frames, ignore_index=True)

    effects = all_coef[all_coef["is_behavior_term"]].copy()
    effects = add_family_holm(effects)

    samples = pd.DataFrame(sample_rows)
    comparison = compare_to_primary(effects)
    family = family_summary(effects)

    parameter_sensitivity = summarize_parameter_sensitivity(
        primary_effects_path
    )

    samples.to_csv(
        table_dir / "robustness_sample_sizes.csv", index=False
    )
    effects.to_csv(
        table_dir / "robustness_behavior_effects.csv", index=False
    )
    comparison.to_csv(
        table_dir / "robustness_comparison_to_primary.csv", index=False
    )
    family.to_csv(
        table_dir / "robustness_family_summary.csv", index=False
    )
    parameter_sensitivity.to_csv(
        table_dir / "parameter_sensitivity_summary.csv", index=False
    )
    all_models.to_csv(
        table_dir / "robustness_model_summary.csv", index=False
    )

    # Compact closure diagnostics.
    alias_comparison = comparison[
        comparison["sample_variant"] == "alias_confirmed_ge10m"
    ]
    ge5_comparison = comparison[
        comparison["sample_variant"] == "authoritative_ge5m"
    ]

    payload = {
        "purpose": "Final targeted robustness check for Question 1.",
        "primary_sample": "authoritative tracked cohort, target duration >=10m",
        "robustness_samples": [
            "alias-confirmed tracked cohort, target duration >=10m",
            "authoritative tracked cohort, target duration >=5m",
        ],
        "primary_behavioral_settings": {
            "H1_session_depth": "30-minute session boundary",
            "H2_post_loss_requeue": (
                "categorical requeue gap after previous ranked loss >=10m"
            ),
            "H3_recent_volume": "ranked games in previous 6 hours",
        },
        "already_existing_parameter_sensitivity": {
            "session_boundaries_minutes": [45, 60, 90],
            "recent_volume_windows_hours": [3, 12, 24],
            "post_loss": "continuous log2(1+gap) sensitivity",
        },
        "alias_confirmed_max_abs_change_from_primary_pp": (
            float(alias_comparison["effect_difference_pp"].abs().max())
            if not alias_comparison.empty
            else None
        ),
        "ge5_max_abs_change_from_primary_pp": (
            float(ge5_comparison["effect_difference_pp"].abs().max())
            if not ge5_comparison.empty
            else None
        ),
        "holm_significant_terms_by_family": family[
            [
                "sample_variant",
                "source",
                "hypothesis",
                "parameter_setting",
                "holm_p_lt_0_05_terms",
            ]
        ].to_dict("records"),
        "interpretation_rule": (
            "Question 1 can be frozen if the strict tracking subset and >=5m "
            "target sensitivity do not create a stable, cross-region monotonic "
            "behavioral pattern contradicting the primary analysis."
        ),
    }

    (audit_dir / "q1_robustness_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\nQ1 ROBUSTNESS COMPLETE\n")
    print(samples.to_string(index=False))

    print("\nFAMILY SUMMARY\n")
    print(
        family[
            [
                "sample_variant",
                "source",
                "hypothesis",
                "max_abs_effect_pp",
                "raw_p_lt_0_05_terms",
                "holm_p_lt_0_05_terms",
            ]
        ].to_string(index=False)
    )

    print(f"\nTables: {table_dir}")


if __name__ == "__main__":
    main()

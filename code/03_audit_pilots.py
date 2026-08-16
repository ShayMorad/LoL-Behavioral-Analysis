#!/usr/bin/env python3
"""
03_audit_pilots.py

Formal reality-check audit for the Match-V5 pilot outputs produced by
02_extract_match_v5.py.

This script is intentionally NOT the cleaning step. It measures the data first,
creates reproducible audit tables/figures, and surfaces candidate cleaning rules
for later justification.

PowerShell example (run from the project root):
    python .\\code\03_audit_pilots.py `
      --input "NA=.\data\processed\pilot_na" `
              "KR=.\data\processed\pilot_kr" `
              "EU=.\data\processed\pilot_euw" `
      --output ".\data\processed\pilot_audit" `
      --overwrite

Outputs
-------
pilot_audit/
    audit_summary.json
    pilot_comparison.csv
    platform_distribution.csv
    queue_distribution.csv
    patch_distribution.csv
    duration_summary.csv
    player_coverage.csv
    player_match_count_quantiles.csv
    inter_match_gap_summary.csv
    role_distribution.csv
    selected_missingness.csv
    data_quality_checks.csv
    figures/
        ... PNG reality-check figures ...

Notes
-----
* The pilots produced with --max-matches 1000 are FIRST-N traversal samples,
  not random samples. Use them to validate structure and feasibility, not to
  estimate population proportions.
* Plot bin widths use NumPy's Freedman-Diaconis ("fd") rule when possible,
  avoiding an arbitrary fixed bin count.
* Figures use matplotlib only, honest axes, labeled units, and no decorative
  chart junk.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IMPORTANT_PARTICIPANT_COLUMNS = [
    "player_id",
    "match_id",
    "game_start_ms",
    "game_end_ms",
    "game_duration_s",
    "queue_id",
    "platform_id",
    "patch",
    "team_id",
    "participant_id",
    "win",
    "champion_id",
    "team_position",
    "individual_position",
    "kills",
    "deaths",
    "assists",
    "gold_earned",
    "total_damage_dealt_to_champions",
    "total_minions_killed",
    "neutral_minions_killed",
    "vision_score",
    "derived_kda",
    "derived_cs_per_min",
    "derived_gold_per_min",
    "derived_damage_to_champions_per_min",
    "challenge_kill_participation",
    "challenge_team_damage_percentage",
    "challenge_early_laning_phase_gold_exp_advantage",
    "challenge_laning_phase_gold_exp_advantage",
    "challenge_max_cs_advantage_on_lane_opponent",
    "challenge_max_level_lead_lane_opponent",
    "challenge_had_afk_teammate",
]

EXPECTED_POSITIONS = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}


def parse_named_path(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f'Input must have the form NAME=PATH, got: "{text}"'
        )
    name, raw_path = text.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError(
            f'Input must have the form NAME=PATH, got: "{text}"'
        )
    return name, Path(raw_path)


def parquet_files(root: Path, table: str) -> List[Path]:
    table_dir = root / table
    if not table_dir.exists():
        raise FileNotFoundError(f"Missing table directory: {table_dir}")
    files = sorted(table_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in: {table_dir}")
    return files


def read_table(root: Path, table: str) -> pd.DataFrame:
    files = parquet_files(root, table)
    return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)


def safe_pct(numer: int | float, denom: int | float) -> float:
    return (100.0 * numer / denom) if denom else float("nan")


def utc_string(ms: float | int | None) -> str | None:
    if ms is None or pd.isna(ms):
        return None
    return pd.to_datetime(int(ms), unit="ms", utc=True).isoformat()


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fd_bins(values: np.ndarray) -> str | int:
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.all(values == values[0]):
        return 10
    return "fd"


def save_duration_histogram(label: str, matches: pd.DataFrame, figures: Path) -> None:
    values = pd.to_numeric(matches["game_duration_min"], errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return

    median = float(np.median(values))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=fd_bins(values))
    ax.set_xlabel("Match duration (minutes)")
    ax.set_ylabel("Number of matches")
    ax.set_title(f"{label}: median match duration = {median:.1f} minutes")
    fig.tight_layout()
    fig.savefig(figures / f"{label}_match_duration_histogram.png", dpi=160)
    plt.close(fig)


def save_player_count_histogram(
    label: str, participant_counts: pd.Series, figures: Path
) -> None:
    values = participant_counts.astype(float).to_numpy()
    if len(values) == 0:
        return

    repeat_pct = safe_pct(int((participant_counts >= 2).sum()), len(participant_counts))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=fd_bins(values))
    if np.nanmax(values) > 10:
        ax.set_xscale("log")
        x_label = "Observed matches per player (log scale)"
    else:
        x_label = "Observed matches per player"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of players")
    ax.set_title(f"{label}: {repeat_pct:.1f}% of players appear in at least 2 pilot matches")
    fig.tight_layout()
    fig.savefig(figures / f"{label}_matches_per_player.png", dpi=160)
    plt.close(fig)


def save_gap_histogram(label: str, gaps_minutes: pd.Series, figures: Path) -> None:
    values = pd.to_numeric(gaps_minutes, errors="coerce").dropna()
    values = values[values > 0].to_numpy()
    if len(values) == 0:
        return

    median = float(np.median(values))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=fd_bins(values))
    ax.set_xscale("log")
    ax.set_xlabel("Gap since previous observed match (minutes, log scale)")
    ax.set_ylabel("Number of consecutive observed player-match pairs")
    ax.set_title(f"{label}: median positive observed gap = {median:.1f} minutes")
    fig.tight_layout()
    fig.savefig(figures / f"{label}_inter_match_gap_histogram.png", dpi=160)
    plt.close(fig)


def save_queue_bar(label: str, matches: pd.DataFrame, figures: Path) -> None:
    counts = matches["queue_id"].astype("Int64").astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Riot queue ID")
    ax.set_ylabel("Number of matches")
    ax.set_title(f"{label}: observed ranked queue composition in the pilot")
    fig.tight_layout()
    fig.savefig(figures / f"{label}_queue_distribution.png", dpi=160)
    plt.close(fig)


def save_patch_bar(label: str, matches: pd.DataFrame, figures: Path) -> None:
    counts = matches["patch"].astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Patch")
    ax.set_ylabel("Number of matches")
    ax.set_title(f"{label}: patch coverage in the first-N pilot")
    fig.tight_layout()
    fig.savefig(figures / f"{label}_patch_distribution.png", dpi=160)
    plt.close(fig)


def compute_inter_match_gaps(participants: pd.DataFrame) -> pd.DataFrame:
    cols = ["player_id", "match_id", "game_start_ms", "game_end_ms", "queue_id", "platform_id"]
    d = participants[cols].dropna(subset=["player_id", "game_start_ms"]).copy()
    d = d.sort_values(["player_id", "game_start_ms", "match_id"])
    d["previous_game_end_ms"] = d.groupby("player_id")["game_end_ms"].shift(1)
    d["previous_match_id"] = d.groupby("player_id")["match_id"].shift(1)
    d["gap_minutes"] = (
        pd.to_numeric(d["game_start_ms"], errors="coerce")
        - pd.to_numeric(d["previous_game_end_ms"], errors="coerce")
    ) / 60_000.0
    return d


def audit_one(label: str, root: Path, output: Path) -> Dict[str, object]:
    matches = read_table(root, "matches")
    participants = read_table(root, "participants")
    teams = read_table(root, "teams")

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # ---------------- Basic distributions ----------------
    platform_counts = matches["platform_id"].astype(str).value_counts(dropna=False)
    queue_counts = matches["queue_id"].astype("Int64").astype(str).value_counts(dropna=False)
    patch_counts = matches["patch"].astype(str).value_counts(dropna=False)

    # ---------------- Player coverage ----------------
    player_counts = (
        participants.dropna(subset=["player_id"])
        .groupby("player_id", observed=True)
        .size()
        .sort_values(ascending=False)
    )

    thresholds = [1, 2, 3, 5, 10, 20, 50, 100]
    player_coverage_rows = []
    for threshold in thresholds:
        count = int((player_counts >= threshold).sum())
        player_coverage_rows.append({
            "source": label,
            "min_observed_matches": threshold,
            "players": count,
            "percent_of_observed_players": safe_pct(count, len(player_counts)),
        })

    quantiles = player_counts.quantile(
        [0, .25, .5, .75, .9, .95, .99, 1]
    ) if len(player_counts) else pd.Series(dtype=float)

    # ---------------- Time / durations ----------------
    duration = pd.to_numeric(matches["game_duration_min"], errors="coerce")
    duration_summary = {
        "source": label,
        "n": int(duration.notna().sum()),
        "mean_min": float(duration.mean()),
        "median_min": float(duration.median()),
        "std_min": float(duration.std()),
        "min_min": float(duration.min()),
        "p01_min": float(duration.quantile(.01)),
        "p05_min": float(duration.quantile(.05)),
        "p95_min": float(duration.quantile(.95)),
        "p99_min": float(duration.quantile(.99)),
        "max_min": float(duration.max()),
    }

    start_ms = pd.to_numeric(matches["game_start_ms"], errors="coerce")
    time_min = start_ms.min()
    time_max = start_ms.max()

    # ---------------- Consecutive-match gaps ----------------
    gap_df = compute_inter_match_gaps(participants)
    gaps = pd.to_numeric(gap_df["gap_minutes"], errors="coerce").dropna()
    positive_gaps = gaps[gaps >= 0]

    gap_summary = {
        "source": label,
        "consecutive_pairs": int(gaps.notna().sum()),
        "negative_gap_pairs": int((gaps < 0).sum()),
        "zero_or_positive_pairs": int((gaps >= 0).sum()),
        "median_positive_gap_min": float(positive_gaps.median()) if len(positive_gaps) else None,
        "p25_positive_gap_min": float(positive_gaps.quantile(.25)) if len(positive_gaps) else None,
        "p75_positive_gap_min": float(positive_gaps.quantile(.75)) if len(positive_gaps) else None,
        "p90_positive_gap_min": float(positive_gaps.quantile(.90)) if len(positive_gaps) else None,
        "p95_positive_gap_min": float(positive_gaps.quantile(.95)) if len(positive_gaps) else None,
        "p99_positive_gap_min": float(positive_gaps.quantile(.99)) if len(positive_gaps) else None,
    }

    # ---------------- Missingness ----------------
    missing_rows = []
    for col in IMPORTANT_PARTICIPANT_COLUMNS:
        if col not in participants.columns:
            missing_rows.append({
                "source": label,
                "column": col,
                "present_in_table": False,
                "missing_rows": len(participants),
                "missing_percent": 100.0,
            })
        else:
            missing = int(participants[col].isna().sum())
            missing_rows.append({
                "source": label,
                "column": col,
                "present_in_table": True,
                "missing_rows": missing,
                "missing_percent": safe_pct(missing, len(participants)),
            })

    # ---------------- Roles ----------------
    role_rows = []
    for col in ["team_position", "individual_position", "lane", "role"]:
        if col not in participants.columns:
            continue
        counts = participants[col].fillna("<MISSING>").astype(str).value_counts(dropna=False)
        for value, count in counts.items():
            role_rows.append({
                "source": label,
                "field": col,
                "value": value,
                "rows": int(count),
                "percent": safe_pct(count, len(participants)),
            })

    # ---------------- Data-quality checks ----------------
    complete_mask = matches["end_of_game_result"].eq("GameComplete")
    complete_ids = set(matches.loc[complete_mask, "match_id"])

    win_counts = (
        participants[participants["match_id"].isin(complete_ids)]
        .assign(win_int=lambda x: x["win"].astype("boolean").astype("Int64"))
        .groupby("match_id")["win_int"]
        .sum(min_count=1)
    )

    match_participant_counts = participants.groupby("match_id").size()
    match_unique_players = participants.groupby("match_id")["player_id"].nunique(dropna=True)
    team_counts_from_teams = teams.groupby("match_id").size()

    position_counts = (
        participants.groupby(["match_id", "team_position"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    complete_position_set_matches = 0
    if all(pos in position_counts.columns for pos in EXPECTED_POSITIONS):
        complete_position_set_matches = int(
            (position_counts[list(EXPECTED_POSITIONS)] == 2).all(axis=1).sum()
        )

    checks = [
        ("unique_match_ids", int(matches["match_id"].nunique()), len(matches),
         "Expected one unique match_id per matches row"),
        ("duplicate_match_rows", int(matches["match_id"].duplicated().sum()), 0,
         "Expected no duplicate match rows after extraction"),
        ("matches_with_10_participant_rows", int((match_participant_counts == 10).sum()), len(matches),
         "Expected standard Summoner's Rift matches to have 10 participant rows"),
        ("matches_with_10_unique_player_ids", int((match_unique_players == 10).sum()), len(matches),
         "Checks duplicate/missing player identity within a match"),
        ("matches_with_2_team_rows", int((team_counts_from_teams == 2).sum()), len(matches),
         "Expected two team rows per match"),
        ("complete_matches_with_5_winners", int((win_counts == 5).sum()), int(complete_mask.sum()),
         "For completed 5v5 games, exactly five participant rows should be winners"),
        ("missing_player_id_rows", int(participants["player_id"].isna().sum()), 0,
         "Longitudinal analysis requires player_id"),
        ("missing_game_start_rows", int(participants["game_start_ms"].isna().sum()), 0,
         "Temporal analysis requires game_start_ms"),
        ("nonpositive_match_durations", int((pd.to_numeric(matches["game_duration_s"], errors="coerce") <= 0).sum()), 0,
         "Match duration must be positive"),
        ("negative_inter_match_gap_pairs", int((gaps < 0).sum()), 0,
         "A player should not have overlapping ranked matches"),
        ("matches_with_two_of_each_standard_position", complete_position_set_matches, len(matches),
         "Diagnostic only: checks TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY assignment"),
    ]

    quality_rows = [{
        "source": label,
        "check": name,
        "observed": observed,
        "reference": reference,
        "note": note,
    } for name, observed, reference, note in checks]

    # ---------------- Save per-source tabular outputs ----------------
    platform_df = pd.DataFrame([
        {"source": label, "platform_id": str(k), "matches": int(v),
         "percent": safe_pct(v, len(matches))}
        for k, v in platform_counts.items()
    ])
    queue_df = pd.DataFrame([
        {"source": label, "queue_id": str(k), "matches": int(v),
         "percent": safe_pct(v, len(matches))}
        for k, v in queue_counts.items()
    ])
    patch_df = pd.DataFrame([
        {"source": label, "patch": str(k), "matches": int(v),
         "percent": safe_pct(v, len(matches))}
        for k, v in patch_counts.items()
    ])

    # ---------------- Figures ----------------
    save_duration_histogram(label, matches, figures)
    save_player_count_histogram(label, player_counts, figures)
    save_gap_histogram(label, gap_df["gap_minutes"], figures)
    save_queue_bar(label, matches, figures)
    save_patch_bar(label, matches, figures)

    return {
        "summary": {
            "source": label,
            "pilot_root": str(root.resolve()),
            "matches": int(len(matches)),
            "participants": int(len(participants)),
            "teams": int(len(teams)),
            "unique_players": int(participants["player_id"].nunique(dropna=True)),
            "unique_matches": int(matches["match_id"].nunique()),
            "game_complete_matches": int(complete_mask.sum()),
            "non_complete_matches": int((~complete_mask).sum()),
            "time_start_utc": utc_string(time_min),
            "time_end_utc": utc_string(time_max),
            "time_span_days": float((time_max - time_min) / 86_400_000.0)
                if pd.notna(time_min) and pd.notna(time_max) else None,
            "queue_420_matches": int((pd.to_numeric(matches["queue_id"], errors="coerce") == 420).sum()),
            "queue_440_matches": int((pd.to_numeric(matches["queue_id"], errors="coerce") == 440).sum()),
            "platforms": {str(k): int(v) for k, v in platform_counts.items()},
            "patches": {str(k): int(v) for k, v in patch_counts.items()},
        },
        "platform_rows": platform_df.to_dict("records"),
        "queue_rows": queue_df.to_dict("records"),
        "patch_rows": patch_df.to_dict("records"),
        "duration_row": duration_summary,
        "player_coverage_rows": player_coverage_rows,
        "player_quantile_rows": [
            {"source": label, "quantile": float(q), "observed_matches_per_player": float(v)}
            for q, v in quantiles.items()
        ],
        "gap_row": gap_summary,
        "role_rows": role_rows,
        "missing_rows": missing_rows,
        "quality_rows": quality_rows,
    }


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {output}\nUse --overwrite for a fresh audit."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        type=parse_named_path,
        required=True,
        help='Pilot roots in NAME=PATH form, e.g. "NA=.\\data\\processed\\pilot_na"',
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Audit output directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output(args.output, args.overwrite)

    results = []
    for label, root in args.input:
        print(f"[audit] {label}: {root}", flush=True)
        results.append(audit_one(label, root, args.output))

    # Combine all sources into compact audit tables.
    write_df(pd.DataFrame([r["summary"] for r in results]), args.output / "pilot_comparison.csv")
    write_df(pd.DataFrame([x for r in results for x in r["platform_rows"]]), args.output / "platform_distribution.csv")
    write_df(pd.DataFrame([x for r in results for x in r["queue_rows"]]), args.output / "queue_distribution.csv")
    write_df(pd.DataFrame([x for r in results for x in r["patch_rows"]]), args.output / "patch_distribution.csv")
    write_df(pd.DataFrame([r["duration_row"] for r in results]), args.output / "duration_summary.csv")
    write_df(pd.DataFrame([x for r in results for x in r["player_coverage_rows"]]), args.output / "player_coverage.csv")
    write_df(pd.DataFrame([x for r in results for x in r["player_quantile_rows"]]), args.output / "player_match_count_quantiles.csv")
    write_df(pd.DataFrame([r["gap_row"] for r in results]), args.output / "inter_match_gap_summary.csv")
    write_df(pd.DataFrame([x for r in results for x in r["role_rows"]]), args.output / "role_distribution.csv")
    write_df(pd.DataFrame([x for r in results for x in r["missing_rows"]]), args.output / "selected_missingness.csv")
    write_df(pd.DataFrame([x for r in results for x in r["quality_rows"]]), args.output / "data_quality_checks.csv")

    json_summary = {
        "important_sampling_note": (
            "These pilots were created with --max-matches and therefore represent the first "
            "N matches encountered by filesystem traversal, not a random sample. Distribution "
            "percentages are diagnostic only until the full corpus is processed."
        ),
        "sources": [r["summary"] for r in results],
    }
    with (args.output / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)

    print("\nPilot audit complete.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

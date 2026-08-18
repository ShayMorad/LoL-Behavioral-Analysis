#!/usr/bin/env python3
r"""
04_audit_dataset.py

Final analytical-readiness audit for the League of Legends project.

This script audits the AUTHORITATIVE linked player-match cohort created by
03_audit_tracking_coverage.py, together with the canonical full_* match/team
tables.

It does not choose a final session threshold and does not build model features.
Instead it establishes whether the data are ready for chronological timeline
construction and next-match analysis.

Recommended PowerShell command
------------------------------
python .\code\04_audit_dataset.py `
  --processed "NA=.\data\processed\full_na" "KR=.\data\processed\full_kr" "EU=.\data\processed\full_eu" `
  --linked ".\data\processed\tracking\linked\authoritative" `
  --output ".\data\processed\analysis_audit" `
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd


HISTORY_THRESHOLDS = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
GAP_THRESHOLDS_MIN = [10, 15, 30, 45, 60, 90, 120, 180, 360, 720, 1440]

SELECTED_FIELDS = [
    "player_id",
    "match_id",
    "game_start_ms",
    "game_end_ms",
    "game_duration_s",
    "platform_id",
    "queue_id",
    "patch",
    "win",
    "champion_id",
    "team_position",
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


def parse_named_path(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f'Expected NAME=PATH, got "{text}"')
    name, raw = text.split("=", 1)
    name, raw = name.strip(), raw.strip()
    if not name or not raw:
        raise argparse.ArgumentTypeError(f'Expected NAME=PATH, got "{text}"')
    return name, Path(raw)


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def sql_text(text: str) -> str:
    return text.replace("'", "''")


def parquet_glob(root: Path, table: str) -> str:
    d = root / table
    if not d.exists() or not any(d.glob("*.parquet")):
        raise FileNotFoundError(f"Missing Parquet table: {d}")
    return sql_path(d / "*.parquet")


def linked_file(linked_root: Path, source: str) -> str:
    d = linked_root / source
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        raise FileNotFoundError(
            f"Missing authoritative linked Parquet dataset: {d}\n"
            "Run 03_audit_tracking_coverage.py first."
        )
    return sql_path(d / "*.parquet")


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def table_columns(con: duckdb.DuckDBPyConnection, parquet: str) -> set[str]:
    return set(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet}')")
        .fetchdf()["column_name"]
        .astype(str)
    )


def utc_iso(ms):
    if ms is None or pd.isna(ms):
        return None
    return pd.to_datetime(int(ms), unit="ms", utc=True).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", nargs="+", type=parse_named_path, required=True)
    p.add_argument("--linked", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duckdb-memory-limit", default="4GB")
    p.add_argument("--duckdb-threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dir(args.output, args.overwrite)

    processed: Dict[str, Path] = dict(args.processed)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{sql_text(args.duckdb_memory_limit)}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")

    overview_rows = []
    quality_rows = []
    missing_rows = []
    history_coverage_rows = []
    history_quantile_frames = []
    history_span_frames = []
    queue_frames = []
    patch_frames = []
    role_frames = []
    duration_frames = []
    end_result_frames = []
    gap_summary_rows = []
    gap_sensitivity_rows = []
    queue_transition_frames = []
    target_feasibility_rows = []
    shared_match_rows = []

    try:
        for source, root in processed.items():
            print(f"[analysis audit] {source}", flush=True)

            m = parquet_glob(root, "matches")
            t = parquet_glob(root, "teams")
            l = linked_file(args.linked, source)
            lcols = table_columns(con, l)

            # ------------------------------------------------------------
            # Core counts and observation window.
            # ------------------------------------------------------------
            overview = con.execute(
                f"""
                WITH mm AS (
                    SELECT
                        COUNT(*)::BIGINT AS matches,
                        COUNT(DISTINCT match_id)::BIGINT AS unique_matches,
                        MIN(game_start_ms)::BIGINT AS min_start_ms,
                        MAX(game_start_ms)::BIGINT AS max_start_ms,
                        SUM(end_of_game_result = 'GameComplete')::BIGINT AS complete_matches,
                        SUM(end_of_game_result <> 'GameComplete'
                            OR end_of_game_result IS NULL)::BIGINT AS non_complete_matches
                    FROM read_parquet('{m}', union_by_name=true)
                ),
                ll AS (
                    SELECT
                        COUNT(*)::BIGINT AS authoritative_player_match_rows,
                        COUNT(DISTINCT player_id)::BIGINT AS authoritative_players_observed,
                        COUNT(DISTINCT match_id)::BIGINT AS matches_with_authoritative_players
                    FROM read_parquet('{l}')
                )
                SELECT * FROM mm, ll
                """
            ).fetchdf().iloc[0].to_dict()

            min_ms = overview.pop("min_start_ms")
            max_ms = overview.pop("max_start_ms")
            overview.update(
                {
                    "source": source,
                    "time_start_utc": utc_iso(min_ms),
                    "time_end_utc": utc_iso(max_ms),
                    "time_span_days": (
                        (int(max_ms) - int(min_ms)) / 86400000.0
                        if min_ms is not None and max_ms is not None
                        else None
                    ),
                }
            )
            overview_rows.append(overview)

            # ------------------------------------------------------------
            # Structural / chronology / linkage quality.
            # ------------------------------------------------------------
            checks = {
                "duplicate_canonical_match_ids": f"""
                    SELECT COUNT(*) FROM (
                        SELECT match_id FROM read_parquet('{m}')
                        GROUP BY match_id HAVING COUNT(*) > 1
                    )
                """,
                "canonical_matches_not_2_teams": f"""
                    SELECT COUNT(*) FROM (
                        SELECT match_id FROM read_parquet('{t}')
                        GROUP BY match_id HAVING COUNT(*) <> 2
                    )
                """,
                "duplicate_authoritative_player_match_rows": f"""
                    SELECT COUNT(*) FROM (
                        SELECT player_id, match_id
                        FROM read_parquet('{l}')
                        GROUP BY player_id, match_id
                        HAVING COUNT(*) > 1
                    )
                """,
                "null_player_ids_in_linked": f"""
                    SELECT COUNT(*) FROM read_parquet('{l}') WHERE player_id IS NULL
                """,
                "null_match_ids_in_linked": f"""
                    SELECT COUNT(*) FROM read_parquet('{l}') WHERE match_id IS NULL
                """,
                "null_start_times_in_linked": f"""
                    SELECT COUNT(*) FROM read_parquet('{l}') WHERE game_start_ms IS NULL
                """,
                "nonpositive_duration_in_linked": f"""
                    SELECT COUNT(*) FROM read_parquet('{l}')
                    WHERE game_duration_s IS NULL OR game_duration_s <= 0
                """,
                "linked_rows_missing_canonical_match": f"""
                    SELECT COUNT(*)
                    FROM read_parquet('{l}') x
                    LEFT JOIN read_parquet('{m}') y USING (match_id)
                    WHERE y.match_id IS NULL
                """,
            }
            for name, query in checks.items():
                quality_rows.append(
                    {
                        "source": source,
                        "check": name,
                        "problem_rows_or_groups": int(con.execute(query).fetchone()[0]),
                    }
                )

            # ------------------------------------------------------------
            # Missingness in fields we expect to use later.
            # ------------------------------------------------------------
            linked_count = int(overview["authoritative_player_match_rows"])
            for col in SELECTED_FIELDS:
                if col not in lcols:
                    missing_rows.append(
                        {
                            "source": source,
                            "column": col,
                            "present": False,
                            "missing_rows": linked_count,
                            "missing_percent": 100.0 if linked_count else None,
                        }
                    )
                else:
                    nmiss = int(
                        con.execute(
                            f"SELECT COUNT(*) FROM read_parquet('{l}') WHERE {col} IS NULL"
                        ).fetchone()[0]
                    )
                    missing_rows.append(
                        {
                            "source": source,
                            "column": col,
                            "present": True,
                            "missing_rows": nmiss,
                            "missing_percent": (
                                100.0 * nmiss / linked_count if linked_count else None
                            ),
                        }
                    )

            # ------------------------------------------------------------
            # Queue / patch / role distributions for authoritative player-rows.
            # ------------------------------------------------------------
            qdf = con.execute(
                f"""
                SELECT
                    queue_id,
                    COUNT(*)::BIGINT AS player_match_rows,
                    COUNT(DISTINCT player_id)::BIGINT AS players,
                    100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS percent
                FROM read_parquet('{l}')
                GROUP BY queue_id
                ORDER BY player_match_rows DESC, queue_id
                """
            ).fetchdf()
            qdf.insert(0, "source", source)
            queue_frames.append(qdf)

            if "patch" in lcols:
                pdf = con.execute(
                    f"""
                    SELECT
                        CAST(patch AS VARCHAR) AS patch,
                        COUNT(*)::BIGINT AS player_match_rows,
                        100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS percent
                    FROM read_parquet('{l}')
                    GROUP BY patch
                    ORDER BY patch
                    """
                ).fetchdf()
                pdf.insert(0, "source", source)
                patch_frames.append(pdf)

            if "team_position" in lcols:
                rdf = con.execute(
                    f"""
                    SELECT
                        CAST(team_position AS VARCHAR) AS team_position,
                        COUNT(*)::BIGINT AS player_match_rows,
                        100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS percent
                    FROM read_parquet('{l}')
                    GROUP BY team_position
                    ORDER BY player_match_rows DESC
                    """
                ).fetchdf()
                rdf.insert(0, "source", source)
                role_frames.append(rdf)

            # ------------------------------------------------------------
            # Match duration / completion. We do NOT choose remake policy yet.
            # ------------------------------------------------------------
            ddf = con.execute(
                f"""
                SELECT
                    COUNT(*)::BIGINT AS matches,
                    AVG(game_duration_s / 60.0) AS mean_min,
                    MEDIAN(game_duration_s / 60.0) AS median_min,
                    MIN(game_duration_s / 60.0) AS min_min,
                    quantile_cont(game_duration_s / 60.0, 0.01) AS p01_min,
                    quantile_cont(game_duration_s / 60.0, 0.05) AS p05_min,
                    quantile_cont(game_duration_s / 60.0, 0.25) AS p25_min,
                    quantile_cont(game_duration_s / 60.0, 0.75) AS p75_min,
                    quantile_cont(game_duration_s / 60.0, 0.95) AS p95_min,
                    quantile_cont(game_duration_s / 60.0, 0.99) AS p99_min,
                    MAX(game_duration_s / 60.0) AS max_min,
                    SUM(game_duration_s < 300)::BIGINT AS under_5_min,
                    SUM(game_duration_s >= 300 AND game_duration_s < 600)::BIGINT AS min_5_to_10,
                    SUM(game_duration_s >= 600 AND game_duration_s < 900)::BIGINT AS min_10_to_15,
                    SUM(game_duration_s >= 900 AND game_duration_s < 1200)::BIGINT AS min_15_to_20,
                    SUM(game_duration_s >= 1200)::BIGINT AS at_least_20_min
                FROM read_parquet('{m}')
                """
            ).fetchdf()
            ddf.insert(0, "source", source)
            duration_frames.append(ddf)

            edf = con.execute(
                f"""
                SELECT
                    CAST(end_of_game_result AS VARCHAR) AS end_of_game_result,
                    COUNT(*)::BIGINT AS matches,
                    100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS percent
                FROM read_parquet('{m}')
                GROUP BY end_of_game_result
                ORDER BY matches DESC
                """
            ).fetchdf()
            edf.insert(0, "source", source)
            end_result_frames.append(edf)

            # ------------------------------------------------------------
            # Per-player histories.
            # all_ranked = queues 420 + 440
            # solo420 = queue 420 only
            # ------------------------------------------------------------
            con.execute("DROP TABLE IF EXISTS player_history")
            con.execute(
                f"""
                CREATE TEMP TABLE player_history AS
                SELECT
                    player_id,
                    COUNT(*) FILTER (WHERE queue_id IN (420, 440))::BIGINT AS all_ranked_matches,
                    COUNT(*) FILTER (WHERE queue_id = 420)::BIGINT AS solo420_matches,
                    COUNT(*) FILTER (WHERE queue_id = 440)::BIGINT AS flex440_matches,
                    MIN(game_start_ms) FILTER (WHERE queue_id IN (420, 440))::BIGINT AS first_ranked_start_ms,
                    MAX(game_start_ms) FILTER (WHERE queue_id IN (420, 440))::BIGINT AS last_ranked_start_ms
                FROM read_parquet('{l}')
                GROUP BY player_id
                """
            )

            nplayers = int(overview["authoritative_players_observed"])
            for threshold in HISTORY_THRESHOLDS:
                all_n = int(
                    con.execute(
                        f"SELECT COUNT(*) FROM player_history "
                        f"WHERE all_ranked_matches >= {threshold}"
                    ).fetchone()[0]
                )
                solo_n = int(
                    con.execute(
                        f"SELECT COUNT(*) FROM player_history "
                        f"WHERE solo420_matches >= {threshold}"
                    ).fetchone()[0]
                )
                history_coverage_rows.extend(
                    [
                        {
                            "source": source,
                            "history_scope": "ranked_420_plus_440",
                            "min_observed_matches": threshold,
                            "players": all_n,
                            "percent_of_authoritative_players": (
                                100.0 * all_n / nplayers if nplayers else None
                            ),
                        },
                        {
                            "source": source,
                            "history_scope": "solo_duo_420_only",
                            "min_observed_matches": threshold,
                            "players": solo_n,
                            "percent_of_authoritative_players": (
                                100.0 * solo_n / nplayers if nplayers else None
                            ),
                        },
                    ]
                )

            hq = con.execute(
                """
                WITH scopes AS (
                    SELECT 'ranked_420_plus_440' AS history_scope, all_ranked_matches AS n
                    FROM player_history
                    UNION ALL
                    SELECT 'solo_duo_420_only', solo420_matches FROM player_history
                )
                SELECT
                    history_scope,
                    MIN(n) AS min_matches,
                    quantile_cont(n, 0.25) AS p25_matches,
                    MEDIAN(n) AS median_matches,
                    quantile_cont(n, 0.75) AS p75_matches,
                    quantile_cont(n, 0.90) AS p90_matches,
                    quantile_cont(n, 0.95) AS p95_matches,
                    quantile_cont(n, 0.99) AS p99_matches,
                    MAX(n) AS max_matches
                FROM scopes
                GROUP BY history_scope
                """
            ).fetchdf()
            hq.insert(0, "source", source)
            history_quantile_frames.append(hq)

            hs = con.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE first_ranked_start_ms IS NOT NULL)::BIGINT AS players,
                    AVG((last_ranked_start_ms - first_ranked_start_ms) / 86400000.0) AS mean_span_days,
                    MEDIAN((last_ranked_start_ms - first_ranked_start_ms) / 86400000.0) AS median_span_days,
                    quantile_cont((last_ranked_start_ms - first_ranked_start_ms) / 86400000.0, 0.25) AS p25_span_days,
                    quantile_cont((last_ranked_start_ms - first_ranked_start_ms) / 86400000.0, 0.75) AS p75_span_days,
                    quantile_cont((last_ranked_start_ms - first_ranked_start_ms) / 86400000.0, 0.95) AS p95_span_days,
                    MAX((last_ranked_start_ms - first_ranked_start_ms) / 86400000.0) AS max_span_days
                FROM player_history
                """
            ).fetchdf()
            hs.insert(0, "source", source)
            history_span_frames.append(hs)

            # ------------------------------------------------------------
            # Consecutive observed ranked gaps.
            # Use previous END -> current START.
            # ------------------------------------------------------------
            for scope, where_clause in (
                ("ranked_420_plus_440", "queue_id IN (420, 440)"),
                ("solo_duo_420_only", "queue_id = 420"),
            ):
                con.execute("DROP TABLE IF EXISTS gaps")
                con.execute(
                    f"""
                    CREATE TEMP TABLE gaps AS
                    WITH ordered AS (
                        SELECT
                            player_id,
                            match_id,
                            queue_id,
                            game_start_ms,
                            game_end_ms,
                            LAG(match_id) OVER (
                                PARTITION BY player_id
                                ORDER BY game_start_ms, match_id
                            ) AS previous_match_id,
                            LAG(queue_id) OVER (
                                PARTITION BY player_id
                                ORDER BY game_start_ms, match_id
                            ) AS previous_queue_id,
                            LAG(game_end_ms) OVER (
                                PARTITION BY player_id
                                ORDER BY game_start_ms, match_id
                            ) AS previous_game_end_ms
                        FROM read_parquet('{l}')
                        WHERE {where_clause}
                    )
                    SELECT
                        *,
                        (game_start_ms - previous_game_end_ms) / 60000.0 AS gap_minutes
                    FROM ordered
                    WHERE previous_match_id IS NOT NULL
                    """
                )

                g = con.execute(
                    """
                    SELECT
                        COUNT(*)::BIGINT AS consecutive_pairs,
                        COUNT(DISTINCT player_id)::BIGINT AS players_with_pairs,
                        SUM(gap_minutes < 0)::BIGINT AS negative_gap_pairs,
                        SUM(gap_minutes = 0)::BIGINT AS zero_gap_pairs,
                        MIN(gap_minutes) AS min_gap_min,
                        quantile_cont(gap_minutes, 0.01) AS p01_gap_min,
                        quantile_cont(gap_minutes, 0.05) AS p05_gap_min,
                        quantile_cont(gap_minutes, 0.25) AS p25_gap_min,
                        MEDIAN(gap_minutes) AS median_gap_min,
                        quantile_cont(gap_minutes, 0.75) AS p75_gap_min,
                        quantile_cont(gap_minutes, 0.90) AS p90_gap_min,
                        quantile_cont(gap_minutes, 0.95) AS p95_gap_min,
                        quantile_cont(gap_minutes, 0.99) AS p99_gap_min,
                        MAX(gap_minutes) AS max_gap_min
                    FROM gaps
                    """
                ).fetchdf().iloc[0].to_dict()
                g.update({"source": source, "history_scope": scope})
                gap_summary_rows.append(g)

                total_pairs = int(g["consecutive_pairs"])
                for threshold in GAP_THRESHOLDS_MIN:
                    n = int(
                        con.execute(
                            f"SELECT COUNT(*) FROM gaps "
                            f"WHERE gap_minutes >= 0 AND gap_minutes <= {threshold}"
                        ).fetchone()[0]
                    )
                    gap_sensitivity_rows.append(
                        {
                            "source": source,
                            "history_scope": scope,
                            "threshold_minutes": threshold,
                            "pairs_at_or_below_threshold": n,
                            "percent_of_pairs": (
                                100.0 * n / total_pairs if total_pairs else None
                            ),
                        }
                    )

                # Feasibility for next-match outcome analysis is exactly the
                # number of consecutive rows under the chosen history scope.
                target_feasibility_rows.append(
                    {
                        "source": source,
                        "history_scope": scope,
                        "players_with_at_least_2_observed_matches": int(
                            g["players_with_pairs"]
                        ),
                        "next_match_prediction_pairs": total_pairs,
                        "negative_gap_pairs": int(g["negative_gap_pairs"]),
                    }
                )

                if scope == "ranked_420_plus_440":
                    qt = con.execute(
                        """
                        SELECT
                            previous_queue_id,
                            queue_id AS current_queue_id,
                            COUNT(*)::BIGINT AS transitions,
                            100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS percent
                        FROM gaps
                        GROUP BY previous_queue_id, queue_id
                        ORDER BY transitions DESC
                        """
                    ).fetchdf()
                    qt.insert(0, "source", source)
                    queue_transition_frames.append(qt)

            # ------------------------------------------------------------
            # Shared-match dependence: multiple tracked players can contribute
            # separate player observations from one physical match.
            # ------------------------------------------------------------
            sm = con.execute(
                f"""
                WITH x AS (
                    SELECT match_id, COUNT(*)::BIGINT AS tracked_players
                    FROM read_parquet('{l}')
                    GROUP BY match_id
                )
                SELECT
                    COUNT(*)::BIGINT AS covered_matches,
                    SUM(tracked_players = 1)::BIGINT AS matches_with_1_tracked,
                    SUM(tracked_players = 2)::BIGINT AS matches_with_2_tracked,
                    SUM(tracked_players >= 3)::BIGINT AS matches_with_3plus_tracked,
                    AVG(tracked_players) AS mean_tracked_players_per_covered_match,
                    MAX(tracked_players)::INTEGER AS max_tracked_players_in_match
                FROM x
                """
            ).fetchdf().iloc[0].to_dict()
            sm["source"] = source
            shared_match_rows.append(sm)

    finally:
        con.close()

    def save(df: pd.DataFrame, name: str) -> None:
        df.to_csv(args.output / name, index=False)

    overview_df = pd.DataFrame(overview_rows)
    quality_df = pd.DataFrame(quality_rows)
    gap_df = pd.DataFrame(gap_summary_rows)
    feasibility_df = pd.DataFrame(target_feasibility_rows)

    save(overview_df, "dataset_overview.csv")
    save(quality_df, "data_quality_checks.csv")
    save(pd.DataFrame(missing_rows), "selected_missingness.csv")
    save(pd.DataFrame(history_coverage_rows), "tracked_player_coverage.csv")
    save(pd.concat(history_quantile_frames, ignore_index=True), "tracked_player_match_count_quantiles.csv")
    save(pd.concat(history_span_frames, ignore_index=True), "tracked_player_history_span.csv")
    save(pd.concat(queue_frames, ignore_index=True), "tracked_queue_distribution.csv")
    if patch_frames:
        save(pd.concat(patch_frames, ignore_index=True), "tracked_patch_distribution.csv")
    if role_frames:
        save(pd.concat(role_frames, ignore_index=True), "tracked_role_distribution.csv")
    save(pd.concat(duration_frames, ignore_index=True), "duration_summary.csv")
    save(pd.concat(end_result_frames, ignore_index=True), "end_result_distribution.csv")
    save(gap_df, "tracked_inter_match_gap_summary.csv")
    save(pd.DataFrame(gap_sensitivity_rows), "tracked_gap_threshold_sensitivity.csv")
    save(pd.concat(queue_transition_frames, ignore_index=True), "tracked_queue_transitions.csv")
    save(feasibility_df, "next_match_analysis_feasibility.csv")
    save(pd.DataFrame(shared_match_rows), "shared_match_dependence.csv")

    serious_checks = {
        "duplicate_canonical_match_ids",
        "canonical_matches_not_2_teams",
        "duplicate_authoritative_player_match_rows",
        "null_player_ids_in_linked",
        "null_match_ids_in_linked",
        "null_start_times_in_linked",
        "linked_rows_missing_canonical_match",
    }
    serious_problems = quality_df[
        quality_df["check"].isin(serious_checks)
        & (quality_df["problem_rows_or_groups"] > 0)
    ]

    negative_gaps = int(gap_df["negative_gap_pairs"].fillna(0).sum())
    ready = serious_problems.empty and negative_gaps == 0

    payload = {
        "main_analysis_cohort": "authoritative",
        "ready_for_timeline_building": bool(ready),
        "serious_quality_problems": serious_problems.to_dict("records"),
        "negative_consecutive_gap_pairs_total": negative_gaps,
        "important_method_notes": [
            "The unit for longitudinal analysis is (source, player_id, match_id).",
            "Multiple authoritative tracked players may contribute separate observations from the same physical match.",
            "Session thresholds are not chosen in this audit; gap sensitivity is reported for later justification.",
            "Queue 420 Solo/Duo is intended as the primary analytical queue; queue 440 can be retained for sensitivity/history variants.",
            "Short games/remakes are audited but not removed here; final semantic handling belongs in timeline/analysis construction.",
            "All next-match features must later be computed strictly from matches preceding the target match.",
        ],
        "overview": overview_rows,
        "next_match_feasibility": target_feasibility_rows,
    }
    (args.output / "audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nFINAL ANALYTICAL AUDIT COMPLETE\n")
    print(
        overview_df[
            [
                "source",
                "matches",
                "authoritative_players_observed",
                "authoritative_player_match_rows",
                "matches_with_authoritative_players",
            ]
        ].to_string(index=False)
    )
    print("\nNEXT-MATCH FEASIBILITY\n")
    print(feasibility_df.to_string(index=False))

    if ready:
        print("\nAUDIT PASSED: ready to build chronological player timelines.")
    else:
        print("\nAUDIT NEEDS REVIEW before timeline construction.")
        if not serious_problems.empty:
            print(serious_problems.to_string(index=False))
        if negative_gaps:
            print(f"Negative consecutive gap pairs: {negative_gaps}")

    print(f"\nSaved audit to: {args.output}")


if __name__ == "__main__":
    main()

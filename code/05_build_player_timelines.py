#!/usr/bin/env python3
r"""
05_build_player_timelines.py

Build the analysis-ready chronological player timelines.

INPUT
-----
Permanent sharded authoritative player-match links created by:
    03_audit_tracking_coverage.py

Canonical match tables created by:
    01_extract_match_v5.py

OUTPUT
------
data/analysis/timelines/
├── ranked_history/
│   ├── NA.parquet
│   ├── KR.parquet
│   └── EU.parquet
├── solo420_targets/
│   ├── NA.parquet
│   ├── KR.parquet
│   └── EU.parquet
└── audit/
    ├── timeline_build_summary.csv
    ├── timeline_build_summary.json
    └── feature_manifest.csv

DESIGN
------
The row is TARGET-CENTRIC.

For a target match t:
- columns prefixed target_* describe match t and are outcomes/context;
- history features describe only matches before t;
- prev_* columns come from the immediately previous observed match;
- prior_* aggregates exclude t;
- recent-volume windows exclude t;
- session depth is computed from previous END -> current START gaps.

Main target:
    queue 420 (Ranked Solo/Duo)

History variants retained:
    1. all observed ranked queue 420 + 440 history;
    2. queue 420-only history.

Session thresholds are NOT fixed yet. We materialize 30/45/60/90 minute
variants so the later EDA/statistical stage can justify the final choice.

Important:
- Short/remake-like matches are flagged, not silently deleted.
- Non-complete matches are flagged, not silently deleted.
- The first observed session for each player is flagged as potentially
  left-censored.
- No Riot IGN or raw PUUID is needed at this stage.
- Unique longitudinal key: (source, player_id, match_id).

PowerShell
----------
python .\code\05_build_player_timelines.py `
  --processed "NA=.\data\processed\full_na" "KR=.\data\processed\full_kr" "EU=.\data\processed\full_eu" `
  --linked ".\data\processed\tracking\linked\authoritative" `
  --output ".\data\analysis\timelines" `
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


SESSION_THRESHOLDS_MIN = (30, 45, 60, 90)
RECENT_WINDOWS_HOURS = (3, 6, 12, 24)

REQUIRED_LINKED_COLUMNS = {
    "player_id",
    "match_id",
    "platform_id",
    "queue_id",
    "patch",
    "game_start_ms",
    "game_end_ms",
    "game_duration_s",
    "win",
    "champion_id",
    "team_position",
    "kills",
    "deaths",
    "assists",
    "derived_total_cs",
    "derived_kda",
    "derived_cs_per_min",
    "derived_gold_per_min",
    "derived_damage_to_champions_per_min",
    "derived_vision_score_per_min",
    "tracking_evidence",
    "is_alias_confirmed",
}

OPTIONAL_TARGET_COLUMNS = (
    "gold_earned",
    "total_damage_dealt_to_champions",
    "vision_score",
    "total_minions_killed",
    "neutral_minions_killed",
    "game_ended_in_early_surrender",
    "game_ended_in_surrender",
    "challenge_kill_participation",
    "challenge_team_damage_percentage",
    "challenge_early_laning_phase_gold_exp_advantage",
    "challenge_laning_phase_gold_exp_advantage",
    "challenge_max_cs_advantage_on_lane_opponent",
    "challenge_max_level_lead_lane_opponent",
)


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
            raise RuntimeError(
                f"Output directory is not empty: {path}. Use --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def table_columns(con: duckdb.DuckDBPyConnection, parquet: str) -> set[str]:
    return set(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet}')")
        .fetchdf()["column_name"]
        .astype(str)
    )


def copy_query_to_parquet(
    con: duckdb.DuckDBPyConnection, query: str, output_file: Path
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()
    con.execute(
        f"COPY ({query}) TO '{sql_path(output_file)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )


def scope_table_name(scope: str, stage: str) -> str:
    return f"{scope}_{stage}"


def build_scope_features(
    con: duckdb.DuckDBPyConnection,
    scope: str,
    where_clause: str,
    prefix: str,
) -> str:
    """
    Create a temporary feature table for one chronological history scope.

    prefix examples:
        ranked
        solo
    """
    s1 = scope_table_name(scope, "s1")
    s2 = scope_table_name(scope, "s2")
    s3 = scope_table_name(scope, "s3")
    s4 = scope_table_name(scope, "s4")
    final = scope_table_name(scope, "features")

    for table in (s1, s2, s3, s4, final):
        con.execute(f"DROP TABLE IF EXISTS {table}")

    recent_count_exprs = []
    recent_minutes_exprs = []
    for h in RECENT_WINDOWS_HOURS:
        ms = h * 60 * 60 * 1000
        recent_count_exprs.append(
            f"""
            COUNT(*) OVER (
                PARTITION BY player_id
                ORDER BY game_start_ms
                RANGE BETWEEN {ms} PRECEDING AND 1 PRECEDING
            )::{ "BIGINT" } AS {prefix}_games_prev_{h}h
            """
        )
        recent_minutes_exprs.append(
            f"""
            COALESCE(
                SUM(game_duration_s / 60.0) OVER (
                    PARTITION BY player_id
                    ORDER BY game_start_ms
                    RANGE BETWEEN {ms} PRECEDING AND 1 PRECEDING
                ),
                0.0
            ) AS {prefix}_minutes_played_prev_{h}h
            """
        )

    # Stage 1: sequence numbers, immediate lags, recent-load windows,
    # strictly-prior expanding baselines.
    con.execute(
        f"""
        CREATE TEMP TABLE {s1} AS
        SELECT
            player_id,
            match_id,
            queue_id,
            game_start_ms,
            game_end_ms,
            game_duration_s,
            end_of_game_result,
            win,
            champion_id,
            team_position,
            derived_kda,
            derived_cs_per_min,
            derived_gold_per_min,
            derived_damage_to_champions_per_min,
            derived_vision_score_per_min,

            ROW_NUMBER() OVER (
                PARTITION BY player_id
                ORDER BY game_start_ms, match_id
            )::BIGINT AS {prefix}_sequence_no,

            LAG(match_id) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_match_id,
            LAG(queue_id) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_queue_id,
            LAG(game_start_ms) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_start_ms,
            LAG(game_end_ms) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_end_ms,
            LAG(game_duration_s) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_duration_s,
            LAG(end_of_game_result) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_end_result,
            LAG(win) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_win,
            LAG(champion_id) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_champion_id,
            LAG(team_position) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_team_position,
            LAG(derived_kda) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_kda,
            LAG(derived_cs_per_min) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_cs_per_min,
            LAG(derived_gold_per_min) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_gold_per_min,
            LAG(derived_damage_to_champions_per_min) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_damage_per_min,
            LAG(derived_vision_score_per_min) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            ) AS prev_{prefix}_vision_per_min,

            {",".join(recent_count_exprs)},
            {",".join(recent_minutes_exprs)},

            AVG(
                CASE
                    WHEN win = TRUE THEN 1.0
                    WHEN win = FALSE THEN 0.0
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY player_id
                ORDER BY game_start_ms, match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_{prefix}_win_rate,

            AVG(derived_kda) OVER (
                PARTITION BY player_id
                ORDER BY game_start_ms, match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_{prefix}_mean_kda,

            AVG(derived_cs_per_min) OVER (
                PARTITION BY player_id
                ORDER BY game_start_ms, match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_{prefix}_mean_cs_per_min,

            AVG(derived_gold_per_min) OVER (
                PARTITION BY player_id
                ORDER BY game_start_ms, match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_{prefix}_mean_gold_per_min,

            AVG(derived_damage_to_champions_per_min) OVER (
                PARTITION BY player_id
                ORDER BY game_start_ms, match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_{prefix}_mean_damage_per_min,

            COUNT(*) OVER (
                PARTITION BY player_id, champion_id
                ORDER BY game_start_ms, match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )::BIGINT AS prior_{prefix}_games_on_target_champion,

            COUNT(*) OVER (
                PARTITION BY player_id, team_position
                ORDER BY game_start_ms, match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )::BIGINT AS prior_{prefix}_games_in_target_role

        FROM base
        WHERE {where_clause}
        """
    )

    # Stage 2: gaps + streak ending at current row.
    con.execute(
        f"""
        CREATE TEMP TABLE {s2} AS
        SELECT
            *,
            ({prefix}_sequence_no - 1)::BIGINT AS prior_{prefix}_matches,

            CASE
                WHEN prev_{prefix}_end_ms IS NULL THEN NULL
                ELSE (game_start_ms - prev_{prefix}_end_ms) / 60000.0
            END AS gap_from_prev_{prefix}_min,

            CASE
                WHEN prev_{prefix}_win = FALSE
                 AND prev_{prefix}_end_ms IS NOT NULL
                THEN (game_start_ms - prev_{prefix}_end_ms) / 60000.0
                ELSE NULL
            END AS post_loss_{prefix}_requeue_gap_min,

            CASE
                WHEN prev_{prefix}_champion_id IS NULL OR champion_id IS NULL THEN NULL
                ELSE champion_id <> prev_{prefix}_champion_id
            END AS champion_changed_from_prev_{prefix},

            CASE
                WHEN prev_{prefix}_team_position IS NULL OR team_position IS NULL THEN NULL
                ELSE team_position <> prev_{prefix}_team_position
            END AS role_changed_from_prev_{prefix},

            CASE
                WHEN win = FALSE THEN
                    {prefix}_sequence_no
                    - COALESCE(
                        MAX(
                            CASE
                                WHEN win = TRUE OR win IS NULL
                                THEN {prefix}_sequence_no
                                ELSE NULL
                            END
                        ) OVER (
                            PARTITION BY player_id
                            ORDER BY game_start_ms, match_id
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ),
                        0
                    )
                ELSE 0
            END::BIGINT AS ending_{prefix}_loss_streak,

            CASE
                WHEN win = TRUE THEN
                    {prefix}_sequence_no
                    - COALESCE(
                        MAX(
                            CASE
                                WHEN win = FALSE OR win IS NULL
                                THEN {prefix}_sequence_no
                                ELSE NULL
                            END
                        ) OVER (
                            PARTITION BY player_id
                            ORDER BY game_start_ms, match_id
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ),
                        0
                    )
                ELSE 0
            END::BIGINT AS ending_{prefix}_win_streak

        FROM {s1}
        """
    )

    # Stage 3: streaks entering current target + session boundaries.
    boundary_exprs = []
    for threshold in SESSION_THRESHOLDS_MIN:
        boundary_exprs.append(
            f"""
            CASE
                WHEN prev_{prefix}_match_id IS NULL THEN 1
                WHEN gap_from_prev_{prefix}_min > {threshold} THEN 1
                ELSE 0
            END::INTEGER AS {prefix}_new_session_{threshold}m
            """
        )

    con.execute(
        f"""
        CREATE TEMP TABLE {s3} AS
        SELECT
            *,
            LAG(ending_{prefix}_loss_streak) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            )::BIGINT AS prev_{prefix}_loss_streak,
            LAG(ending_{prefix}_win_streak) OVER (
                PARTITION BY player_id ORDER BY game_start_ms, match_id
            )::BIGINT AS prev_{prefix}_win_streak,
            {",".join(boundary_exprs)}
        FROM {s2}
        """
    )

    # Stage 4: cumulative session IDs.
    session_id_exprs = []
    for threshold in SESSION_THRESHOLDS_MIN:
        session_id_exprs.append(
            f"""
            SUM({prefix}_new_session_{threshold}m) OVER (
                PARTITION BY player_id
                ORDER BY game_start_ms, match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )::BIGINT AS {prefix}_session_id_{threshold}m
            """
        )

    con.execute(
        f"""
        CREATE TEMP TABLE {s4} AS
        SELECT
            *,
            {",".join(session_id_exprs)}
        FROM {s3}
        """
    )

    # Final: game number inside each candidate session definition.
    session_game_exprs = []
    session_left_censor_exprs = []
    for threshold in SESSION_THRESHOLDS_MIN:
        session_game_exprs.append(
            f"""
            ROW_NUMBER() OVER (
                PARTITION BY player_id, {prefix}_session_id_{threshold}m
                ORDER BY game_start_ms, match_id
            )::BIGINT AS {prefix}_session_game_no_{threshold}m
            """
        )
        session_left_censor_exprs.append(
            f"""
            ({prefix}_session_id_{threshold}m = 1)
                AS {prefix}_session_potentially_left_censored_{threshold}m
            """
        )

    con.execute(
        f"""
        CREATE TEMP TABLE {final} AS
        SELECT
            *,
            {",".join(session_game_exprs)},
            {",".join(session_left_censor_exprs)}
        FROM {s4}
        """
    )

    return final


def feature_columns(prefix: str) -> List[str]:
    cols = [
        f"{prefix}_sequence_no",
        f"prior_{prefix}_matches",
        f"prev_{prefix}_match_id",
        f"prev_{prefix}_queue_id",
        f"prev_{prefix}_start_ms",
        f"prev_{prefix}_end_ms",
        f"prev_{prefix}_duration_s",
        f"prev_{prefix}_end_result",
        f"prev_{prefix}_win",
        f"prev_{prefix}_champion_id",
        f"prev_{prefix}_team_position",
        f"prev_{prefix}_kda",
        f"prev_{prefix}_cs_per_min",
        f"prev_{prefix}_gold_per_min",
        f"prev_{prefix}_damage_per_min",
        f"prev_{prefix}_vision_per_min",
        f"gap_from_prev_{prefix}_min",
        f"post_loss_{prefix}_requeue_gap_min",
        f"champion_changed_from_prev_{prefix}",
        f"role_changed_from_prev_{prefix}",
        f"prev_{prefix}_loss_streak",
        f"prev_{prefix}_win_streak",
        f"prior_{prefix}_win_rate",
        f"prior_{prefix}_mean_kda",
        f"prior_{prefix}_mean_cs_per_min",
        f"prior_{prefix}_mean_gold_per_min",
        f"prior_{prefix}_mean_damage_per_min",
        f"prior_{prefix}_games_on_target_champion",
        f"prior_{prefix}_games_in_target_role",
    ]
    for h in RECENT_WINDOWS_HOURS:
        cols.extend(
            [
                f"{prefix}_games_prev_{h}h",
                f"{prefix}_minutes_played_prev_{h}h",
            ]
        )
    for threshold in SESSION_THRESHOLDS_MIN:
        cols.extend(
            [
                f"{prefix}_session_id_{threshold}m",
                f"{prefix}_session_game_no_{threshold}m",
                f"{prefix}_session_potentially_left_censored_{threshold}m",
            ]
        )
    return cols


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

    ranked_out = args.output / "ranked_history"
    solo_out = args.output / "solo420_targets"
    audit_out = args.output / "audit"
    ranked_out.mkdir(parents=True, exist_ok=True)
    solo_out.mkdir(parents=True, exist_ok=True)
    audit_out.mkdir(parents=True, exist_ok=True)

    processed: Dict[str, Path] = dict(args.processed)
    summary_rows = []

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{sql_text(args.duckdb_memory_limit)}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")

    try:
        for source, root in processed.items():
            print(f"[timeline] {source}: loading authoritative player-matches", flush=True)

            linked = linked_file(args.linked, source)
            matches = parquet_glob(root, "matches")

            cols = table_columns(con, linked)
            missing = sorted(REQUIRED_LINKED_COLUMNS - cols)
            if missing:
                raise RuntimeError(
                    f"{source}: linked Parquet is missing required columns: {missing}"
                )

            con.execute("DROP TABLE IF EXISTS base")
            con.execute(
                f"""
                CREATE TEMP TABLE base AS
                SELECT
                    l.*,
                    m.end_of_game_result
                FROM read_parquet('{linked}') l
                INNER JOIN read_parquet('{matches}', union_by_name=true) m
                    USING (match_id)
                WHERE l.queue_id IN (420, 440)
                """
            )

            # Safety: the linked table must still be one row per player-match.
            duplicate_base = int(
                con.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT player_id, match_id
                        FROM base
                        GROUP BY player_id, match_id
                        HAVING COUNT(*) > 1
                    )
                    """
                ).fetchone()[0]
            )
            if duplicate_base:
                raise RuntimeError(
                    f"{source}: duplicate (player_id, match_id) rows in ranked base: "
                    f"{duplicate_base}"
                )

            ranked_features = build_scope_features(
                con=con,
                scope="ranked",
                where_clause="queue_id IN (420, 440)",
                prefix="ranked",
            )
            solo_features = build_scope_features(
                con=con,
                scope="solo",
                where_clause="queue_id = 420",
                prefix="solo",
            )

            # Optional target fields are carried if the extractor contains them.
            optional_selects = []
            for col in OPTIONAL_TARGET_COLUMNS:
                if col in cols:
                    optional_selects.append(f"b.{col} AS target_{col}")

            target_selects = [
                f"'{sql_text(source)}' AS source",
                "b.player_id",
                "b.match_id",
                "b.platform_id AS target_platform_id",
                "b.queue_id AS target_queue_id",
                "b.patch AS target_patch",
                "b.game_start_ms AS target_start_ms",
                "b.game_end_ms AS target_end_ms",
                "b.game_duration_s AS target_duration_s",
                "b.end_of_game_result AS target_end_result",
                "(b.end_of_game_result = 'GameComplete') AS target_game_complete",
                "(b.game_duration_s < 300) AS target_under_5_min",
                "(b.game_duration_s < 600) AS target_under_10_min",
                "(b.game_duration_s < 900) AS target_under_15_min",
                "b.is_alias_confirmed",
                "b.tracking_evidence",
                "b.champion_id AS target_champion_id",
                "b.team_position AS target_team_position",
                "b.win AS target_win",
                "b.kills AS target_kills",
                "b.deaths AS target_deaths",
                "b.assists AS target_assists",
                "b.derived_total_cs AS target_total_cs",
                "b.derived_kda AS target_kda",
                "b.derived_cs_per_min AS target_cs_per_min",
                "b.derived_gold_per_min AS target_gold_per_min",
                (
                    "b.derived_damage_to_champions_per_min "
                    "AS target_damage_to_champions_per_min"
                ),
                "b.derived_vision_score_per_min AS target_vision_score_per_min",
            ] + optional_selects

            ranked_selects = [f"r.{c}" for c in feature_columns("ranked")]

            ranked_query = f"""
                SELECT
                    {", ".join(target_selects)},
                    {", ".join(ranked_selects)},
                    (r.prev_ranked_match_id IS NOT NULL) AS has_prior_ranked_match
                FROM base b
                INNER JOIN {ranked_features} r
                    USING (player_id, match_id)
                ORDER BY player_id, target_start_ms, match_id
            """

            ranked_file = ranked_out / f"{source}.parquet"
            copy_query_to_parquet(con, ranked_query, ranked_file)

            solo_selects = [f"s.{c}" for c in feature_columns("solo")]
            solo_query = f"""
                SELECT
                    {", ".join(target_selects)},
                    {", ".join(ranked_selects)},
                    {", ".join(solo_selects)},
                    (r.prev_ranked_match_id IS NOT NULL) AS has_prior_ranked_match,
                    (s.prev_solo_match_id IS NOT NULL) AS has_prior_solo_match
                FROM base b
                INNER JOIN {ranked_features} r
                    USING (player_id, match_id)
                INNER JOIN {solo_features} s
                    USING (player_id, match_id)
                WHERE b.queue_id = 420
                ORDER BY player_id, target_start_ms, match_id
            """

            solo_file = solo_out / f"{source}.parquet"
            copy_query_to_parquet(con, solo_query, solo_file)

            ranked_p = sql_path(ranked_file)
            solo_p = sql_path(solo_file)

            ranked_stats = con.execute(
                f"""
                SELECT
                    COUNT(*)::BIGINT AS ranked_rows,
                    COUNT(DISTINCT player_id)::BIGINT AS ranked_players,
                    COUNT(*) FILTER (
                        WHERE has_prior_ranked_match
                    )::BIGINT AS ranked_rows_with_prior,
                    COUNT(*) FILTER (
                        WHERE gap_from_prev_ranked_min < 0
                    )::BIGINT AS negative_ranked_gaps,
                    COUNT(*) FILTER (
                        WHERE prev_ranked_win = FALSE
                    )::BIGINT AS targets_after_ranked_loss
                FROM read_parquet('{ranked_p}')
                """
            ).fetchdf().iloc[0].to_dict()

            solo_stats = con.execute(
                f"""
                SELECT
                    COUNT(*)::BIGINT AS solo420_target_rows,
                    COUNT(DISTINCT player_id)::BIGINT AS solo420_players,
                    COUNT(*) FILTER (
                        WHERE has_prior_ranked_match
                    )::BIGINT AS solo_targets_with_prior_ranked,
                    COUNT(*) FILTER (
                        WHERE has_prior_solo_match
                    )::BIGINT AS solo_targets_with_prior_solo,
                    COUNT(*) FILTER (
                        WHERE prior_solo_matches >= 5
                    )::BIGINT AS solo_targets_with_5plus_prior_solo,
                    COUNT(*) FILTER (
                        WHERE prior_solo_matches >= 10
                    )::BIGINT AS solo_targets_with_10plus_prior_solo,
                    COUNT(*) FILTER (
                        WHERE prev_ranked_win = FALSE
                    )::BIGINT AS solo_targets_after_ranked_loss,
                    COUNT(*) FILTER (
                        WHERE post_loss_ranked_requeue_gap_min IS NOT NULL
                    )::BIGINT AS solo_targets_with_post_loss_gap,
                    COUNT(*) FILTER (
                        WHERE target_game_complete
                    )::BIGINT AS solo_complete_targets,
                    COUNT(*) FILTER (
                        WHERE target_under_5_min
                    )::BIGINT AS solo_targets_under_5_min,
                    COUNT(*) FILTER (
                        WHERE target_under_10_min
                    )::BIGINT AS solo_targets_under_10_min,
                    COUNT(*) FILTER (
                        WHERE gap_from_prev_ranked_min < 0
                    )::BIGINT AS negative_solo_target_ranked_gaps,
                    COUNT(*) - (
                        SELECT COUNT(*) FROM (
                            SELECT DISTINCT player_id, match_id
                            FROM read_parquet('{solo_p}')
                        )
                    ) AS duplicate_solo_player_match_rows
                FROM read_parquet('{solo_p}')
                """
            ).fetchdf().iloc[0].to_dict()

            if int(ranked_stats["negative_ranked_gaps"]) != 0:
                raise RuntimeError(f"{source}: negative ranked gaps created.")
            if int(solo_stats["negative_solo_target_ranked_gaps"]) != 0:
                raise RuntimeError(f"{source}: negative gaps in solo target table.")
            if int(solo_stats["duplicate_solo_player_match_rows"]) != 0:
                raise RuntimeError(f"{source}: duplicate solo player-match rows created.")

            summary_rows.append(
                {
                    "source": source,
                    **ranked_stats,
                    **solo_stats,
                    "ranked_history_file": str(ranked_file.resolve()),
                    "solo420_targets_file": str(solo_file.resolve()),
                }
            )

            print(
                f"[timeline] {source}: "
                f"{int(solo_stats['solo420_target_rows']):,} Solo/Duo targets, "
                f"{int(solo_stats['solo_targets_with_prior_ranked']):,} "
                "with prior ranked history",
                flush=True,
            )

    finally:
        con.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(audit_out / "timeline_build_summary.csv", index=False)

    manifest_rows = [
        {
            "feature_group": "identifier",
            "examples": "source, player_id, match_id",
            "availability": "identity/key",
            "model_default": "not predictor",
            "notes": "Unique longitudinal key is (source, player_id, match_id).",
        },
        {
            "feature_group": "target outcome",
            "examples": "target_win, target_kda, target_damage_to_champions_per_min",
            "availability": "known after target match",
            "model_default": "TARGET ONLY",
            "notes": "Never use target performance columns as predictors.",
        },
        {
            "feature_group": "target context",
            "examples": "target_patch, target_champion_id, target_team_position",
            "availability": "target-match context",
            "model_default": "use only when justified",
            "notes": "Champion may be pre-game-known; role field should be treated cautiously.",
        },
        {
            "feature_group": "immediate prior match",
            "examples": "prev_ranked_win, prev_ranked_kda, gap_from_prev_ranked_min",
            "availability": "strictly before target",
            "model_default": "predictor",
            "notes": "Primary post-loss requeue variables live here.",
        },
        {
            "feature_group": "recent competitive volume",
            "examples": "ranked_games_prev_3h/6h/12h/24h",
            "availability": "strictly before target",
            "model_default": "predictor",
            "notes": "Current target is excluded from all recent windows.",
        },
        {
            "feature_group": "recent competitive load",
            "examples": "ranked_minutes_played_prev_3h/6h/12h/24h",
            "availability": "strictly before target",
            "model_default": "predictor",
            "notes": "Sum of previous observed ranked match durations in the window.",
        },
        {
            "feature_group": "session depth",
            "examples": "ranked_session_game_no_30m/45m/60m/90m",
            "availability": "strictly based on pre-target chronology",
            "model_default": "predictor",
            "notes": "Threshold not chosen yet; retain sensitivity variants.",
        },
        {
            "feature_group": "streak/history",
            "examples": "prev_ranked_loss_streak, prior_ranked_win_rate",
            "availability": "strictly before target",
            "model_default": "predictor/control",
            "notes": "Expanding baselines exclude the target match.",
        },
        {
            "feature_group": "switch behavior",
            "examples": "champion_changed_from_prev_ranked, role_changed_from_prev_ranked",
            "availability": "comparison with immediately previous observed ranked match",
            "model_default": "predictor",
            "notes": "Target champion/role context is retained separately.",
        },
        {
            "feature_group": "data-quality flags",
            "examples": "target_game_complete, target_under_5_min, target_under_10_min",
            "availability": "known after target match",
            "model_default": "filter/sensitivity only",
            "notes": "No remake-duration policy is silently imposed by this builder.",
        },
    ]
    pd.DataFrame(manifest_rows).to_csv(
        audit_out / "feature_manifest.csv", index=False
    )

    payload = {
        "main_target_queue": 420,
        "history_scopes": ["ranked_420_plus_440", "solo_duo_420_only"],
        "session_thresholds_minutes": list(SESSION_THRESHOLDS_MIN),
        "recent_volume_windows_hours": list(RECENT_WINDOWS_HOURS),
        "target_centric_design": True,
        "leakage_rule": (
            "All predictor/history features are computed strictly from rows "
            "preceding the target match. target_* performance fields are outcomes."
        ),
        "short_game_policy": (
            "Flagged only; no final remake/short-game exclusion is imposed here."
        ),
        "sources": summary_rows,
    }
    (audit_out / "timeline_build_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPLAYER TIMELINES BUILT\n")
    print(
        summary_df[
            [
                "source",
                "ranked_rows",
                "ranked_players",
                "solo420_target_rows",
                "solo420_players",
                "solo_targets_with_prior_ranked",
                "solo_targets_after_ranked_loss",
                "negative_solo_target_ranked_gaps",
                "duplicate_solo_player_match_rows",
            ]
        ].to_string(index=False)
    )

    print(f"\nRanked history:  {ranked_out}")
    print(f"Solo420 targets: {solo_out}")
    print(f"Audit:           {audit_out}")


if __name__ == "__main__":
    main()

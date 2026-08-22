#!/usr/bin/env python3
"""
01_prepare_data.py

Cross-platform processed-data preparation for the League of Legends project.

This is the normal starting point for the submitted project. It assumes the
canonical regional Parquet tables and the FINAL tracked-player lookup tables
already exist under data/processed. It then:

1. validates canonical matches/participants/teams and tracked-player lookups;
2. rebuilds permanent tracked-player <-> match linkage for both cohorts;
3. writes compact coverage/readiness audits;
4. rebuilds the chronological, target-centric player timelines used by Q1.

Why the tracked-player lookup is an input
-----------------------------------------
The authoritative/alias-confirmed identities were reconstructed earlier from
raw Riot PUUIDs, original seed lists, and league_data.db. That identity step
cannot be reproduced from anonymized canonical Parquet alone. Because this
project is intentionally re-run from the processed-data stage, the small
tracked-player lookup tables are treated as part of the processed input while
all large linkage/timeline products are regenerated here.

No Windows-only shell syntax is used. Run from the project root on Linux,
Windows, or macOS:

    python code/01_prepare_data.py --overwrite

Outputs
-------
data/processed/tracking/linked/
data/processed/tracking/coverage_audit/
data/processed/analysis_audit/
data/analysis/timelines/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd


# Shared settings used when rebuilding longitudinal Q1 features.
REGIONS = ("NA", "KR", "EU")
SESSION_THRESHOLDS_MIN = (30, 45, 60, 90)
RECENT_WINDOWS_HOURS = (3, 6, 12, 24)

REQUIRED_MATCH_COLUMNS = {
    "match_id",
    "platform_id",
    "queue_id",
    "game_start_ms",
    "game_end_ms",
    "game_duration_s",
    "end_of_game_result",
}

REQUIRED_PARTICIPANT_COLUMNS = {
    "match_id",
    "player_id",
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
}

REQUIRED_LINKED_COLUMNS = REQUIRED_PARTICIPANT_COLUMNS | {
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


def project_root() -> Path:
    """Return the repository root inferred from this script location."""
    return Path(__file__).resolve().parents[1]


def parse_named_path(text: str) -> Tuple[str, Path]:
    """Parse a NAME=PATH command-line value into a normalized source name and Path."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f'Expected NAME=PATH, got "{text}"')
    name, raw = text.split("=", 1)
    name, raw = name.strip(), raw.strip()
    if not name or not raw:
        raise argparse.ArgumentTypeError(f'Expected NAME=PATH, got "{text}"')
    return name.upper(), Path(raw)


def sql_path(path: Path) -> str:
    """Convert a filesystem path to a DuckDB-safe absolute POSIX string."""
    return path.resolve().as_posix().replace("'", "''")


def sql_text(text: str) -> str:
    """Escape a plain string before embedding it in a DuckDB SQL literal."""
    return text.replace("'", "''")


def parquet_glob(root: Path, table: str) -> str:
    """Validate a canonical Parquet table directory and return its DuckDB glob."""
    d = root / table
    if not d.exists() or not any(d.glob("*.parquet")):
        raise FileNotFoundError(f"Missing Parquet table: {d}")
    return sql_path(d / "*.parquet")


def tracked_file(tracking_root: Path, cohort: str, source: str) -> Path:
    """Return one processed tracked-player lookup and fail clearly if it is missing."""
    p = tracking_root / cohort / source / "tracked_players.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing processed tracked-player lookup: {p}\n"
            "Keep data/processed/tracking/{authoritative,alias_confirmed}; "
            "they are inputs to the processed-stage reproduction pipeline."
        )
    return p


def linked_glob(linked_root: Path, source: str) -> str:
    """Return the regenerated linked-player Parquet glob for one region."""
    d = linked_root / source
    if not d.exists() or not any(d.glob("*.parquet")):
        raise FileNotFoundError(f"Missing linked Parquet dataset: {d}")
    return sql_path(d / "*.parquet")


def table_columns(con: duckdb.DuckDBPyConnection, parquet: str) -> set[str]:
    """Read the column names exposed by a Parquet dataset through DuckDB."""
    return set(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet}', union_by_name=true)")
        .fetchdf()["column_name"]
        .astype(str)
    )


def prepare_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory, optionally replacing existing generated contents."""
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_sharded_parquet(
    con: duckdb.DuckDBPyConnection,
    query: str,
    output_dir: Path,
    rows_per_file: int,
) -> tuple[str, int]:
    """Deterministic sharded export; avoids GitHub-size single files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.parquet"):
        old.unlink()

    # Stable row numbers make the large linkage export reproducible across shards.
    con.execute("DROP TABLE IF EXISTS linked_export_tmp")
    con.execute(
        f"""
        CREATE TEMP TABLE linked_export_tmp AS
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY player_id, game_start_ms, match_id
            )::BIGINT AS __export_rownum,
            *
        FROM ({query})
        """
    )

    total = int(con.execute("SELECT COUNT(*) FROM linked_export_tmp").fetchone()[0])
    if total == 0:
        raise RuntimeError(f"No linked rows produced for {output_dir}")

    parts = 0
    for offset in range(0, total, rows_per_file):
        first_row = offset + 1
        last_row = min(offset + rows_per_file, total)
        out_file = output_dir / f"part-{parts:05d}.parquet"
        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE (__export_rownum)
                FROM linked_export_tmp
                WHERE __export_rownum BETWEEN {first_row} AND {last_row}
                ORDER BY __export_rownum
            )
            TO '{sql_path(out_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        parts += 1

    con.execute("DROP TABLE linked_export_tmp")
    glob = sql_path(output_dir / "*.parquet")
    written = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0])
    if written != total:
        raise RuntimeError(
            f"Sharded export mismatch for {output_dir}: expected {total:,}, wrote {written:,}."
        )
    return glob, parts


def validate_and_link(
    con: duckdb.DuckDBPyConnection,
    processed: Dict[str, Path],
    tracking_root: Path,
    linked_root: Path,
    coverage_out: Path,
    rows_per_linked_file: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate processed inputs and rebuild tracked player-match linkage."""
    coverage_rows: list[dict] = []
    linkage_rows: list[dict] = []
    quality_rows: list[dict] = []
    uncovered_rows: list[dict] = []

    for source in REGIONS:
        root = processed[source]
        print(f"[prepare] {source}: validating canonical tables", flush=True)

        matches = parquet_glob(root, "matches")
        participants = parquet_glob(root, "participants")
        teams = parquet_glob(root, "teams")
        parquet_glob(root, "team_bans")  # existence check

        mcols = table_columns(con, matches)
        pcols = table_columns(con, participants)
        missing_m = sorted(REQUIRED_MATCH_COLUMNS - mcols)
        missing_p = sorted(REQUIRED_PARTICIPANT_COLUMNS - pcols)
        if missing_m or missing_p:
            raise RuntimeError(
                f"{source}: missing required columns. matches={missing_m}; participants={missing_p}"
            )

        # Validate match uniqueness, player IDs, and the expected 10-player/2-team structure.
        canonical = con.execute(
            f"""
            WITH m AS (
                SELECT COUNT(*)::BIGINT AS matches,
                       COUNT(DISTINCT match_id)::BIGINT AS unique_matches
                FROM read_parquet('{matches}', union_by_name=true)
            ), p AS (
                SELECT COUNT(*)::BIGINT AS participant_rows,
                       COUNT(DISTINCT player_id)::BIGINT AS unique_participants,
                       SUM(player_id IS NULL)::BIGINT AS null_player_ids,
                       SUM(CASE WHEN player_id IS NOT NULL
                                 AND NOT regexp_full_match(player_id, '[0-9a-f]{{32}}')
                                THEN 1 ELSE 0 END)::BIGINT AS malformed_player_ids
                FROM read_parquet('{participants}', union_by_name=true)
            ), bad_p AS (
                SELECT COUNT(*)::BIGINT AS bad_participant_structure FROM (
                    SELECT match_id
                    FROM read_parquet('{participants}', union_by_name=true)
                    GROUP BY match_id
                    HAVING COUNT(*) <> 10 OR COUNT(DISTINCT player_id) <> 10
                )
            ), bad_t AS (
                SELECT COUNT(*)::BIGINT AS bad_team_structure FROM (
                    SELECT match_id
                    FROM read_parquet('{teams}', union_by_name=true)
                    GROUP BY match_id
                    HAVING COUNT(*) <> 2
                )
            )
            SELECT * FROM m, p, bad_p, bad_t
            """
        ).fetchdf().iloc[0].to_dict()

        canonical_checks = {
            "duplicate_match_ids": int(canonical["matches"] - canonical["unique_matches"]),
            "null_player_ids": int(canonical["null_player_ids"]),
            "malformed_player_ids": int(canonical["malformed_player_ids"]),
            "matches_not_10_unique_participants": int(canonical["bad_participant_structure"]),
            "matches_not_2_teams": int(canonical["bad_team_structure"]),
        }
        for check, problems in canonical_checks.items():
            quality_rows.append({"source": source, "stage": "canonical", "check": check, "problems": problems})
        if any(canonical_checks.values()):
            raise RuntimeError(f"{source}: canonical validation failed: {canonical_checks}")

        alias_path = tracked_file(tracking_root, "alias_confirmed", source)
        auth_path = tracked_file(tracking_root, "authoritative", source)
        alias = sql_path(alias_path)
        auth = sql_path(auth_path)

        duplicate_alias = int(
            con.execute(
                f"SELECT COUNT(*) FROM (SELECT player_id FROM read_parquet('{alias}') GROUP BY player_id HAVING COUNT(*)>1)"
            ).fetchone()[0]
        )
        duplicate_auth = int(
            con.execute(
                f"SELECT COUNT(*) FROM (SELECT player_id FROM read_parquet('{auth}') GROUP BY player_id HAVING COUNT(*)>1)"
            ).fetchone()[0]
        )
        alias_not_auth = int(
            con.execute(
                f"""
                SELECT COUNT(*) FROM read_parquet('{alias}') a
                LEFT JOIN read_parquet('{auth}') u USING(player_id)
                WHERE u.player_id IS NULL
                """
            ).fetchone()[0]
        )
        lookup_checks = {
            "duplicate_alias_confirmed_ids": duplicate_alias,
            "duplicate_authoritative_ids": duplicate_auth,
            "alias_confirmed_not_in_authoritative": alias_not_auth,
        }
        for check, problems in lookup_checks.items():
            quality_rows.append({"source": source, "stage": "tracking_lookup", "check": check, "problems": problems})
        if any(lookup_checks.values()):
            raise RuntimeError(f"{source}: tracking lookup validation failed: {lookup_checks}")

        # Count tracked players from each cohort in every physical match.
        con.execute("DROP TABLE IF EXISTS match_cov")
        con.execute(
            f"""
            CREATE TEMP TABLE match_cov AS
            WITH alias_ids AS (SELECT DISTINCT player_id FROM read_parquet('{alias}')),
                 auth_ids AS (SELECT DISTINCT player_id FROM read_parquet('{auth}')),
                 counts AS (
                    SELECT p.match_id,
                           COUNT(*) FILTER (WHERE a.player_id IS NOT NULL) AS n_alias,
                           COUNT(*) FILTER (WHERE u.player_id IS NOT NULL) AS n_auth
                    FROM read_parquet('{participants}', union_by_name=true) p
                    LEFT JOIN alias_ids a USING(player_id)
                    LEFT JOIN auth_ids u USING(player_id)
                    GROUP BY p.match_id
                 )
            SELECT m.match_id, m.platform_id, m.queue_id, m.game_start_ms,
                   COALESCE(c.n_alias,0)::INTEGER AS n_alias,
                   COALESCE(c.n_auth,0)::INTEGER AS n_auth
            FROM read_parquet('{matches}', union_by_name=true) m
            LEFT JOIN counts c USING(match_id)
            """
        )
        cov = con.execute(
            """
            SELECT COUNT(*)::BIGINT AS total_matches,
                   SUM(n_alias>=1)::BIGINT AS alias_covered_matches,
                   SUM(n_auth>=1)::BIGINT AS authoritative_covered_matches,
                   AVG(n_alias) AS mean_alias_players_per_match,
                   AVG(n_auth) AS mean_authoritative_players_per_match
            FROM match_cov
            """
        ).fetchdf().iloc[0].to_dict()
        total_matches = int(cov["total_matches"])
        coverage_rows.append(
            {
                "source": source,
                **cov,
                "alias_confirmed_match_coverage_percent": 100.0 * int(cov["alias_covered_matches"]) / total_matches,
                "authoritative_match_coverage_percent": 100.0 * int(cov["authoritative_covered_matches"]) / total_matches,
            }
        )
        uncovered_rows.append(
            {
                "source": source,
                "authoritative_uncovered_matches": int(
                    con.execute("SELECT COUNT(*) FROM match_cov WHERE n_auth=0").fetchone()[0]
                ),
                "authoritative_uncovered_queue420": int(
                    con.execute("SELECT COUNT(*) FROM match_cov WHERE n_auth=0 AND queue_id=420").fetchone()[0]
                ),
                "authoritative_uncovered_queue440": int(
                    con.execute("SELECT COUNT(*) FROM match_cov WHERE n_auth=0 AND queue_id=440").fetchone()[0]
                ),
            }
        )

        # Materialize both cohorts; Q1 uses authoritative and keeps alias membership for robustness.
        auth_cols = table_columns(con, auth)
        auth_evidence_expr = (
            "CAST(tr.tracking_evidence AS VARCHAR)" if "tracking_evidence" in auth_cols else "'authoritative'"
        )

        for cohort, lookup_path in (("alias_confirmed", alias_path), ("authoritative", auth_path)):
            lookup = sql_path(lookup_path)
            out_dir = linked_root / cohort / source
            if cohort == "authoritative":
                alias_join = f"LEFT JOIN read_parquet('{alias}') ac USING(player_id)"
                alias_expr = "(ac.player_id IS NOT NULL)"
                evidence_expr = auth_evidence_expr
            else:
                alias_join = ""
                alias_expr = "TRUE"
                evidence_expr = "'fresh_seed_alias_unique'"

            query = f"""
                SELECT '{sql_text(source)}' AS source,
                       '{cohort}' AS tracked_cohort,
                       {alias_expr} AS is_alias_confirmed,
                       {evidence_expr} AS tracking_evidence,
                       p.*
                FROM read_parquet('{participants}', union_by_name=true) p
                INNER JOIN read_parquet('{lookup}') tr USING(player_id)
                {alias_join}
            """
            linked, parts = write_sharded_parquet(con, query, out_dir, rows_per_linked_file)
            stats = con.execute(
                f"""
                SELECT COUNT(*)::BIGINT AS player_match_rows,
                       COUNT(DISTINCT player_id)::BIGINT AS players,
                       COUNT(DISTINCT match_id)::BIGINT AS matches,
                       COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT player_id,match_id FROM read_parquet('{linked}'))) AS duplicate_player_match_rows
                FROM read_parquet('{linked}')
                """
            ).fetchdf().iloc[0].to_dict()
            if int(stats["duplicate_player_match_rows"]) != 0:
                raise RuntimeError(f"{source}/{cohort}: duplicate player-match linkage created")
            linkage_rows.append({"source": source, "cohort": cohort, **stats, "parquet_parts": parts})

    coverage_df = pd.DataFrame(coverage_rows)
    linkage_df = pd.DataFrame(linkage_rows)
    quality_df = pd.DataFrame(quality_rows)
    coverage_df.to_csv(coverage_out / "match_tracking_coverage_summary.csv", index=False)
    linkage_df.to_csv(coverage_out / "linked_player_match_summary.csv", index=False)
    quality_df.to_csv(coverage_out / "processed_input_quality_checks.csv", index=False)
    pd.DataFrame(uncovered_rows).to_csv(coverage_out / "uncovered_matches_summary.csv", index=False)
    return coverage_df, linkage_df, quality_df


def scope_table_name(scope: str, stage: str) -> str:
    """Build a consistent temporary-table name for one chronological feature scope."""
    return f"{scope}_{stage}"


def build_scope_features(
    con: duckdb.DuckDBPyConnection,
    scope: str,
    where_clause: str,
    prefix: str,
) -> str:
    """Build strictly pre-target chronological features for one history scope."""
    s1 = scope_table_name(scope, "s1")
    s2 = scope_table_name(scope, "s2")
    s3 = scope_table_name(scope, "s3")
    s4 = scope_table_name(scope, "s4")
    final = scope_table_name(scope, "features")
    for table in (s1, s2, s3, s4, final):
        con.execute(f"DROP TABLE IF EXISTS {table}")

    # Rolling windows include only matches strictly before the current match.
    recent_count_exprs = []
    recent_minutes_exprs = []
    for h in RECENT_WINDOWS_HOURS:
        ms = h * 60 * 60 * 1000
        recent_count_exprs.append(
            f"""
            COUNT(*) OVER (
                PARTITION BY player_id ORDER BY game_start_ms
                RANGE BETWEEN {ms} PRECEDING AND 1 PRECEDING
            )::BIGINT AS {prefix}_games_prev_{h}h
            """
        )
        recent_minutes_exprs.append(
            f"""
            COALESCE(SUM(game_duration_s / 60.0) OVER (
                PARTITION BY player_id ORDER BY game_start_ms
                RANGE BETWEEN {ms} PRECEDING AND 1 PRECEDING
            ), 0.0) AS {prefix}_minutes_played_prev_{h}h
            """
        )

    # Stage 1: lags, recent activity, and prior-history averages.
    con.execute(
        f"""
        CREATE TEMP TABLE {s1} AS
        SELECT
            player_id, match_id, queue_id, game_start_ms, game_end_ms,
            game_duration_s, end_of_game_result, win, champion_id, team_position,
            derived_kda, derived_cs_per_min, derived_gold_per_min,
            derived_damage_to_champions_per_min, derived_vision_score_per_min,

            ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id)::BIGINT AS {prefix}_sequence_no,
            LAG(match_id) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_match_id,
            LAG(queue_id) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_queue_id,
            LAG(game_start_ms) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_start_ms,
            LAG(game_end_ms) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_end_ms,
            LAG(game_duration_s) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_duration_s,
            LAG(end_of_game_result) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_end_result,
            LAG(win) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_win,
            LAG(champion_id) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_champion_id,
            LAG(team_position) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_team_position,
            LAG(derived_kda) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_kda,
            LAG(derived_cs_per_min) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_cs_per_min,
            LAG(derived_gold_per_min) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_gold_per_min,
            LAG(derived_damage_to_champions_per_min) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_damage_per_min,
            LAG(derived_vision_score_per_min) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_{prefix}_vision_per_min,

            {','.join(recent_count_exprs)},
            {','.join(recent_minutes_exprs)},

            AVG(CASE WHEN win=TRUE THEN 1.0 WHEN win=FALSE THEN 0.0 ELSE NULL END)
                OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS prior_{prefix}_win_rate,
            AVG(derived_kda) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS prior_{prefix}_mean_kda,
            AVG(derived_cs_per_min) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS prior_{prefix}_mean_cs_per_min,
            AVG(derived_gold_per_min) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS prior_{prefix}_mean_gold_per_min,
            AVG(derived_damage_to_champions_per_min) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS prior_{prefix}_mean_damage_per_min,
            COUNT(*) OVER (PARTITION BY player_id,champion_id ORDER BY game_start_ms,match_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)::BIGINT
                AS prior_{prefix}_games_on_target_champion,
            COUNT(*) OVER (PARTITION BY player_id,team_position ORDER BY game_start_ms,match_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)::BIGINT
                AS prior_{prefix}_games_in_target_role
        FROM base
        WHERE {where_clause}
        """
    )

    # Stage 2: turn lags into gaps, switches, and current streak lengths.
    con.execute(
        f"""
        CREATE TEMP TABLE {s2} AS
        SELECT *,
            ({prefix}_sequence_no - 1)::BIGINT AS prior_{prefix}_matches,
            CASE WHEN prev_{prefix}_end_ms IS NULL THEN NULL
                 ELSE (game_start_ms-prev_{prefix}_end_ms)/60000.0 END AS gap_from_prev_{prefix}_min,
            CASE WHEN prev_{prefix}_win=FALSE AND prev_{prefix}_end_ms IS NOT NULL
                 THEN (game_start_ms-prev_{prefix}_end_ms)/60000.0 ELSE NULL END AS post_loss_{prefix}_requeue_gap_min,
            CASE WHEN prev_{prefix}_champion_id IS NULL OR champion_id IS NULL THEN NULL
                 ELSE champion_id <> prev_{prefix}_champion_id END AS champion_changed_from_prev_{prefix},
            CASE WHEN prev_{prefix}_team_position IS NULL OR team_position IS NULL THEN NULL
                 ELSE team_position <> prev_{prefix}_team_position END AS role_changed_from_prev_{prefix},
            CASE WHEN win=FALSE THEN {prefix}_sequence_no - COALESCE(
                MAX(CASE WHEN win=TRUE OR win IS NULL THEN {prefix}_sequence_no ELSE NULL END)
                OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0)
                ELSE 0 END::BIGINT AS ending_{prefix}_loss_streak,
            CASE WHEN win=TRUE THEN {prefix}_sequence_no - COALESCE(
                MAX(CASE WHEN win=FALSE OR win IS NULL THEN {prefix}_sequence_no ELSE NULL END)
                OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0)
                ELSE 0 END::BIGINT AS ending_{prefix}_win_streak
        FROM {s1}
        """
    )

    # Stage 3: mark new sessions under each candidate inactivity threshold.
    boundaries = []
    for threshold in SESSION_THRESHOLDS_MIN:
        boundaries.append(
            f"""
            CASE WHEN prev_{prefix}_match_id IS NULL THEN 1
                 WHEN gap_from_prev_{prefix}_min > {threshold} THEN 1 ELSE 0 END::INTEGER
                 AS {prefix}_new_session_{threshold}m
            """
        )
    con.execute(
        f"""
        CREATE TEMP TABLE {s3} AS
        SELECT *,
            LAG(ending_{prefix}_loss_streak) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id)::BIGINT AS prev_{prefix}_loss_streak,
            LAG(ending_{prefix}_win_streak) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id)::BIGINT AS prev_{prefix}_win_streak,
            {','.join(boundaries)}
        FROM {s2}
        """
    )

    # Stage 4: cumulative boundary markers become session IDs.
    session_ids = []
    for threshold in SESSION_THRESHOLDS_MIN:
        session_ids.append(
            f"""
            SUM({prefix}_new_session_{threshold}m) OVER (
                PARTITION BY player_id ORDER BY game_start_ms,match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )::BIGINT AS {prefix}_session_id_{threshold}m
            """
        )
    con.execute(f"CREATE TEMP TABLE {s4} AS SELECT *, {','.join(session_ids)} FROM {s3}")

    # Final stage: number games within each session and flag left-censored first sessions.
    game_nos, censor_flags = [], []
    for threshold in SESSION_THRESHOLDS_MIN:
        game_nos.append(
            f"""
            ROW_NUMBER() OVER (
                PARTITION BY player_id,{prefix}_session_id_{threshold}m
                ORDER BY game_start_ms,match_id
            )::BIGINT AS {prefix}_session_game_no_{threshold}m
            """
        )
        censor_flags.append(
            f"({prefix}_session_id_{threshold}m=1) AS {prefix}_session_potentially_left_censored_{threshold}m"
        )
    con.execute(
        f"CREATE TEMP TABLE {final} AS SELECT *, {','.join(game_nos)}, {','.join(censor_flags)} FROM {s4}"
    )
    return final


def feature_columns(prefix: str) -> List[str]:
    """Return the chronological feature columns exported for one history scope."""
    cols = [
        f"{prefix}_sequence_no", f"prior_{prefix}_matches",
        f"prev_{prefix}_match_id", f"prev_{prefix}_queue_id",
        f"prev_{prefix}_start_ms", f"prev_{prefix}_end_ms",
        f"prev_{prefix}_duration_s", f"prev_{prefix}_end_result",
        f"prev_{prefix}_win", f"prev_{prefix}_champion_id",
        f"prev_{prefix}_team_position", f"prev_{prefix}_kda",
        f"prev_{prefix}_cs_per_min", f"prev_{prefix}_gold_per_min",
        f"prev_{prefix}_damage_per_min", f"prev_{prefix}_vision_per_min",
        f"gap_from_prev_{prefix}_min", f"post_loss_{prefix}_requeue_gap_min",
        f"champion_changed_from_prev_{prefix}", f"role_changed_from_prev_{prefix}",
        f"prev_{prefix}_loss_streak", f"prev_{prefix}_win_streak",
        f"prior_{prefix}_win_rate", f"prior_{prefix}_mean_kda",
        f"prior_{prefix}_mean_cs_per_min", f"prior_{prefix}_mean_gold_per_min",
        f"prior_{prefix}_mean_damage_per_min", f"prior_{prefix}_games_on_target_champion",
        f"prior_{prefix}_games_in_target_role",
    ]
    for h in RECENT_WINDOWS_HOURS:
        cols += [f"{prefix}_games_prev_{h}h", f"{prefix}_minutes_played_prev_{h}h"]
    for t in SESSION_THRESHOLDS_MIN:
        cols += [
            f"{prefix}_session_id_{t}m",
            f"{prefix}_session_game_no_{t}m",
            f"{prefix}_session_potentially_left_censored_{t}m",
        ]
    return cols


def copy_query_to_parquet(con: duckdb.DuckDBPyConnection, query: str, output_file: Path) -> None:
    """Execute a DuckDB query and write its result as compressed Parquet."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()
    con.execute(
        f"COPY ({query}) TO '{sql_path(output_file)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )


def build_timelines(
    con: duckdb.DuckDBPyConnection,
    processed: Dict[str, Path],
    linked_authoritative_root: Path,
    timeline_root: Path,
) -> pd.DataFrame:
    """Build authoritative ranked histories and target-centric Solo/Duo timelines for Q1."""
    ranked_out = timeline_root / "ranked_history"
    solo_out = timeline_root / "solo420_targets"
    audit_out = timeline_root / "audit"
    ranked_out.mkdir(parents=True, exist_ok=True)
    solo_out.mkdir(parents=True, exist_ok=True)
    audit_out.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for source in REGIONS:
        print(f"[prepare] {source}: building chronological timelines", flush=True)
        linked = linked_glob(linked_authoritative_root, source)
        matches = parquet_glob(processed[source], "matches")
        cols = table_columns(con, linked)
        missing = sorted(REQUIRED_LINKED_COLUMNS - cols)
        if missing:
            raise RuntimeError(f"{source}: linked data missing required columns: {missing}")

        # Q1 history uses ranked Solo/Duo and Flex only.
        con.execute("DROP TABLE IF EXISTS base")
        con.execute(
            f"""
            CREATE TEMP TABLE base AS
            SELECT l.*, m.end_of_game_result
            FROM read_parquet('{linked}') l
            INNER JOIN read_parquet('{matches}', union_by_name=true) m USING(match_id)
            WHERE l.queue_id IN (420,440)
            """
        )
        duplicate_base = int(
            con.execute(
                "SELECT COUNT(*) FROM (SELECT player_id,match_id FROM base GROUP BY player_id,match_id HAVING COUNT(*)>1)"
            ).fetchone()[0]
        )
        if duplicate_base:
            raise RuntimeError(f"{source}: duplicate player-match rows in ranked base: {duplicate_base}")

        # Build pre-target features for all ranked history and for Solo/Duo-only history.
        ranked_features = build_scope_features(con, "ranked", "queue_id IN (420,440)", "ranked")
        solo_features = build_scope_features(con, "solo", "queue_id=420", "solo")

        optional_selects = [f"b.{c} AS target_{c}" for c in OPTIONAL_TARGET_COLUMNS if c in cols]
        # Keep target fields separate from history fields so leakage is easy to audit.
        target_selects = [
            f"'{sql_text(source)}' AS source",
            "b.player_id", "b.match_id",
            "b.platform_id AS target_platform_id", "b.queue_id AS target_queue_id",
            "b.patch AS target_patch", "b.game_start_ms AS target_start_ms",
            "b.game_end_ms AS target_end_ms", "b.game_duration_s AS target_duration_s",
            "b.end_of_game_result AS target_end_result",
            "(b.end_of_game_result='GameComplete') AS target_game_complete",
            "(b.game_duration_s<300) AS target_under_5_min",
            "(b.game_duration_s<600) AS target_under_10_min",
            "(b.game_duration_s<900) AS target_under_15_min",
            "b.is_alias_confirmed", "b.tracking_evidence",
            "b.champion_id AS target_champion_id", "b.team_position AS target_team_position",
            "b.win AS target_win", "b.kills AS target_kills", "b.deaths AS target_deaths",
            "b.assists AS target_assists", "b.derived_total_cs AS target_total_cs",
            "b.derived_kda AS target_kda", "b.derived_cs_per_min AS target_cs_per_min",
            "b.derived_gold_per_min AS target_gold_per_min",
            "b.derived_damage_to_champions_per_min AS target_damage_to_champions_per_min",
            "b.derived_vision_score_per_min AS target_vision_score_per_min",
        ] + optional_selects
        ranked_selects = [f"r.{c}" for c in feature_columns("ranked")]

        # Full ranked timeline for each tracked player.
        ranked_query = f"""
            SELECT {', '.join(target_selects)}, {', '.join(ranked_selects)},
                   (r.prev_ranked_match_id IS NOT NULL) AS has_prior_ranked_match
            FROM base b INNER JOIN {ranked_features} r USING(player_id,match_id)
            ORDER BY player_id,target_start_ms,match_id
        """
        ranked_file = ranked_out / f"{source}.parquet"
        copy_query_to_parquet(con, ranked_query, ranked_file)

        # Solo/Duo target timeline used by Q1 modeling.
        solo_selects = [f"s.{c}" for c in feature_columns("solo")]
        solo_query = f"""
            SELECT {', '.join(target_selects)}, {', '.join(ranked_selects)}, {', '.join(solo_selects)},
                   (r.prev_ranked_match_id IS NOT NULL) AS has_prior_ranked_match,
                   (s.prev_solo_match_id IS NOT NULL) AS has_prior_solo_match
            FROM base b
            INNER JOIN {ranked_features} r USING(player_id,match_id)
            INNER JOIN {solo_features} s USING(player_id,match_id)
            WHERE b.queue_id=420
            ORDER BY player_id,target_start_ms,match_id
        """
        solo_file = solo_out / f"{source}.parquet"
        copy_query_to_parquet(con, solo_query, solo_file)

        ranked_p, solo_p = sql_path(ranked_file), sql_path(solo_file)
        stats = con.execute(
            f"""
            WITH r AS (
                SELECT COUNT(*)::BIGINT AS ranked_rows,
                       COUNT(DISTINCT player_id)::BIGINT AS ranked_players
                FROM read_parquet('{ranked_p}')
            ), s AS (
                SELECT COUNT(*)::BIGINT AS solo420_target_rows,
                       COUNT(DISTINCT player_id)::BIGINT AS solo420_players,
                       SUM(has_prior_ranked_match)::BIGINT AS solo_targets_with_prior_ranked,
                       SUM(prev_ranked_win=FALSE)::BIGINT AS solo_targets_after_ranked_loss,
                       SUM(gap_from_prev_ranked_min<0)::BIGINT AS negative_ranked_gaps,
                       COUNT(*)-(SELECT COUNT(*) FROM (SELECT DISTINCT player_id,match_id FROM read_parquet('{solo_p}'))) AS duplicate_player_match_rows
                FROM read_parquet('{solo_p}')
            ) SELECT * FROM r,s
            """
        ).fetchdf().iloc[0].to_dict()
        if int(stats["negative_ranked_gaps"]) or int(stats["duplicate_player_match_rows"]):
            raise RuntimeError(f"{source}: invalid timeline generated: {stats}")
        summary_rows.append({"source": source, **stats})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(audit_out / "timeline_build_summary.csv", index=False)
    (audit_out / "timeline_build_summary.json").write_text(
        json.dumps(
            {
                "target_queue": 420,
                "history_queues": [420, 440],
                "session_thresholds_minutes": list(SESSION_THRESHOLDS_MIN),
                "recent_windows_hours": list(RECENT_WINDOWS_HOURS),
                "leakage_rule": "All history/predictor features are computed strictly before the target match.",
                "sources": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def analytical_readiness(
    con: duckdb.DuckDBPyConnection,
    processed: Dict[str, Path],
    linked_authoritative_root: Path,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit linked data quality and verify that chronological next-match analysis is feasible."""
    overview_rows, checks, feasibility_rows = [], [], []
    for source in REGIONS:
        matches = parquet_glob(processed[source], "matches")
        linked = linked_glob(linked_authoritative_root, source)
        # Compact readiness audit: size, coverage, and chronological feasibility.
        overview = con.execute(
            f"""
            WITH m AS (
                SELECT COUNT(*)::BIGINT AS matches,
                       MIN(game_start_ms)::BIGINT AS min_start_ms,
                       MAX(game_start_ms)::BIGINT AS max_start_ms
                FROM read_parquet('{matches}', union_by_name=true)
            ), l AS (
                SELECT COUNT(*)::BIGINT AS player_match_rows,
                       COUNT(DISTINCT player_id)::BIGINT AS tracked_players,
                       COUNT(DISTINCT match_id)::BIGINT AS covered_matches
                FROM read_parquet('{linked}')
            ) SELECT * FROM m,l
            """
        ).fetchdf().iloc[0].to_dict()
        overview_rows.append({"source": source, **overview})

        check_queries = {
            "duplicate_player_match_rows": f"SELECT COUNT(*) FROM (SELECT player_id,match_id FROM read_parquet('{linked}') GROUP BY player_id,match_id HAVING COUNT(*)>1)",
            "null_player_ids": f"SELECT COUNT(*) FROM read_parquet('{linked}') WHERE player_id IS NULL",
            "null_match_ids": f"SELECT COUNT(*) FROM read_parquet('{linked}') WHERE match_id IS NULL",
            "null_start_times": f"SELECT COUNT(*) FROM read_parquet('{linked}') WHERE game_start_ms IS NULL",
            "nonpositive_duration": f"SELECT COUNT(*) FROM read_parquet('{linked}') WHERE game_duration_s IS NULL OR game_duration_s<=0",
        }
        for name, q in check_queries.items():
            checks.append({"source": source, "check": name, "problems": int(con.execute(q).fetchone()[0])})

        # Verify that consecutive-match gaps are chronological in both history scopes.
        for scope, where in (("ranked_420_plus_440", "queue_id IN (420,440)"), ("solo420_only", "queue_id=420")):
            f = con.execute(
                f"""
                WITH o AS (
                    SELECT player_id,match_id,game_start_ms,game_end_ms,
                           LAG(game_end_ms) OVER (PARTITION BY player_id ORDER BY game_start_ms,match_id) AS prev_end
                    FROM read_parquet('{linked}') WHERE {where}
                )
                SELECT COUNT(*) FILTER (WHERE prev_end IS NOT NULL)::BIGINT AS consecutive_pairs,
                       COUNT(DISTINCT player_id) FILTER (WHERE prev_end IS NOT NULL)::BIGINT AS players_with_pairs,
                       COUNT(*) FILTER (WHERE prev_end IS NOT NULL AND game_start_ms<prev_end)::BIGINT AS negative_gap_pairs
                FROM o
                """
            ).fetchdf().iloc[0].to_dict()
            feasibility_rows.append({"source": source, "scope": scope, **f})

    overview_df = pd.DataFrame(overview_rows)
    checks_df = pd.DataFrame(checks)
    feasibility_df = pd.DataFrame(feasibility_rows)
    overview_df.to_csv(output / "dataset_overview.csv", index=False)
    checks_df.to_csv(output / "data_quality_checks.csv", index=False)
    feasibility_df.to_csv(output / "next_match_analysis_feasibility.csv", index=False)
    if (checks_df["problems"] != 0).any() or (feasibility_df["negative_gap_pairs"] != 0).any():
        raise RuntimeError("Processed analytical-readiness audit failed; inspect data/processed/analysis_audit.")
    return overview_df, checks_df, feasibility_df


def parse_args() -> argparse.Namespace:
    """Parse processed-input paths, output locations, and DuckDB resource settings."""
    root = project_root()
    p = argparse.ArgumentParser(description="Validate processed data, rebuild tracking linkage, and build Q1 timelines.")
    p.add_argument(
        "--processed",
        nargs="+",
        type=parse_named_path,
        default=[
            ("NA", root / "data/processed/full_na"),
            ("KR", root / "data/processed/full_kr"),
            ("EU", root / "data/processed/full_eu"),
        ],
        help="Regional canonical roots as NAME=PATH. Defaults to data/processed/full_*.",
    )
    p.add_argument("--tracking", type=Path, default=root / "data/processed/tracking")
    p.add_argument("--timelines", type=Path, default=root / "data/analysis/timelines")
    p.add_argument("--analysis-audit", type=Path, default=root / "data/processed/analysis_audit")
    p.add_argument("--rows-per-linked-file", type=int, default=200_000)
    p.add_argument("--duckdb-memory-limit", default="4GB")
    p.add_argument("--duckdb-threads", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    """Run the processed-data preparation pipeline.

    Validates preserved inputs, rebuilds tracked-player linkage and audits,
    constructs leakage-safe Q1 timelines, and writes a compact pipeline summary.
    """
    # 1) Resolve preserved inputs and generated-output locations.
    args = parse_args()
    processed = dict(args.processed)
    if set(processed) != set(REGIONS):
        raise SystemExit(f"Expected processed sources {REGIONS}; got {sorted(processed)}")

    # These folders are generated from the preserved processed inputs.
    linked_root = args.tracking / "linked"
    coverage_out = args.tracking / "coverage_audit"
    for out in (linked_root, coverage_out, args.analysis_audit, args.timelines):
        prepare_dir(out, args.overwrite)

    # DuckDB performs the large Parquet joins and window calculations out of core.
    # 2) Configure DuckDB for the large Parquet joins and window calculations.
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{sql_text(args.duckdb_memory_limit)}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")

    # 3) Rebuild linkage, audit analytical readiness, then create Q1 timelines.
    try:
        coverage, linkage, input_quality = validate_and_link(
            con,
            processed,
            args.tracking,
            linked_root,
            coverage_out,
            args.rows_per_linked_file,
        )
        overview, readiness_checks, feasibility = analytical_readiness(
            con,
            processed,
            linked_root / "authoritative",
            args.analysis_audit,
        )
        timelines = build_timelines(
            con,
            processed,
            linked_root / "authoritative",
            args.timelines,
        )
    finally:
        con.close()

    # 4) Persist a compact machine-readable record of the completed preparation.
    payload = {
        "processed_starting_point": True,
        "regions": list(REGIONS),
        "coverage": coverage.to_dict("records"),
        "linkage": linkage.to_dict("records"),
        "timeline_summary": timelines.to_dict("records"),
        "all_input_quality_checks_passed": bool((input_quality["problems"] == 0).all()),
        "all_readiness_checks_passed": bool((readiness_checks["problems"] == 0).all()),
        "all_negative_gap_checks_passed": bool((feasibility["negative_gap_pairs"] == 0).all()),
    }
    (args.analysis_audit / "pipeline_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\nPROCESSED-DATA PREPARATION COMPLETE\n")
    print("Tracking coverage:")
    print(coverage[["source", "authoritative_match_coverage_percent"]].to_string(index=False))
    print("\nTimeline summary:")
    print(timelines.to_string(index=False))
    print(f"\nTimelines: {args.timelines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

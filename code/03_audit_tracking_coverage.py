#!/usr/bin/env python3
r"""
03_audit_tracking_coverage.py

Final tracking-provenance audit + persistent player-match linkage.

Cohort names used by the final project
--------------------------------------
alias_confirmed:
    Fresh seed-list Riot ID matched uniquely to one raw PUUID.

authoritative:
    alias_confirmed UNION DB seed PUUIDs that can be assigned unambiguously
    to exactly one supplied regional corpus.

This script:
1. Audits match coverage for both cohorts.
2. Verifies alias_confirmed is a subset of authoritative.
3. Links every tracked player to every match in which that player appears.
4. Materializes those links as reusable Parquet tables so later scripts do
   NOT need to redo identity matching or tracking joins.

The canonical full_* Parquet data are never modified.

Recommended PowerShell command
------------------------------
python .\code\03_audit_tracking_coverage.py `
  --processed "NA=.\data\processed\full_na" "KR=.\data\processed\full_kr" "EU=.\data\processed\full_eu" `
  --tracking ".\data\processed\tracking" `
  --output ".\data\processed\tracking\coverage_audit" `
  --linked-output ".\data\processed\tracking\linked" `
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


COHORTS = ("alias_confirmed", "authoritative")


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


def tracked_file(tracking_root: Path, cohort: str, source: str) -> Path:
    p = tracking_root / cohort / source / "tracked_players.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing tracked-player file: {p}\n"
            f"Expected final cohort names: {COHORTS}"
        )
    return p


def prepare_dir(path: Path, overwrite: bool) -> None:
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
    """
    Materialize a query as a multi-file Parquet dataset.

    Keeping linked outputs sharded makes them GitHub-friendly while remaining
    transparent to DuckDB/PyArrow/pandas readers that accept Parquet globs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for old in output_dir.glob("*.parquet"):
        old.unlink()

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

    total_rows = int(
        con.execute("SELECT COUNT(*) FROM linked_export_tmp").fetchone()[0]
    )

    if total_rows == 0:
        raise RuntimeError(f"No linked rows produced for {output_dir}")

    part = 0
    for offset in range(0, total_rows, rows_per_file):
        first_row = offset + 1
        last_row = min(offset + rows_per_file, total_rows)
        out_file = output_dir / f"part-{part:05d}.parquet"

        # Important: shard by an explicit deterministic row number.
        # Repeated LIMIT/OFFSET queries without ORDER BY are not guaranteed
        # to scan a DuckDB table in the same order and can overlap.
        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE (__export_rownum)
                FROM linked_export_tmp
                WHERE __export_rownum BETWEEN {int(first_row)} AND {int(last_row)}
                ORDER BY __export_rownum
            )
            TO '{sql_path(out_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        part += 1

    con.execute("DROP TABLE linked_export_tmp")

    glob_path = sql_path(output_dir / "*.parquet")
    written_rows = int(
        con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob_path}')").fetchone()[0]
    )
    if written_rows != total_rows:
        raise RuntimeError(
            f"Sharded export row-count mismatch for {output_dir}: "
            f"expected {total_rows:,}, wrote {written_rows:,}."
        )

    return glob_path, part


def get_columns(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> List[str]:
    p = sql_path(parquet_path)
    return (
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')")
        .fetchdf()["column_name"]
        .astype(str)
        .tolist()
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", nargs="+", type=parse_named_path, required=True)
    p.add_argument("--tracking", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--linked-output", type=Path, required=True)
    p.add_argument("--sample-uncovered", type=int, default=100)
    p.add_argument(
        "--rows-per-linked-file",
        type=int,
        default=200000,
        help=(
            "Rows per linked Parquet shard. Default 200,000 keeps the current "
            "EU linked datasets comfortably below GitHub's per-file limit."
        ),
    )
    p.add_argument("--duckdb-memory-limit", default="4GB")
    p.add_argument("--duckdb-threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dir(args.output, args.overwrite)
    prepare_dir(args.linked_output, args.overwrite)

    processed: Dict[str, Path] = dict(args.processed)

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{sql_text(args.duckdb_memory_limit)}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")

    coverage_rows = []
    dist_frames = []
    queue_frames = []
    platform_frames = []
    uncovered_rows = []
    uncovered_samples = []
    linkage_rows = []
    validation_rows = []

    try:
        for source, root in processed.items():
            print(f"[tracking audit] {source}", flush=True)

            matches = parquet_glob(root, "matches")
            participants = parquet_glob(root, "participants")

            alias_path = tracked_file(args.tracking, "alias_confirmed", source)
            auth_path = tracked_file(args.tracking, "authoritative", source)
            alias = sql_path(alias_path)
            auth = sql_path(auth_path)

            # ------------------------------------------------------------
            # Validate tracked lookup tables before using them.
            # ------------------------------------------------------------
            duplicate_alias = con.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT player_id
                    FROM read_parquet('{alias}')
                    GROUP BY player_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]

            duplicate_auth = con.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT player_id
                    FROM read_parquet('{auth}')
                    GROUP BY player_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]

            alias_not_auth = con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{alias}') a
                LEFT JOIN read_parquet('{auth}') b USING (player_id)
                WHERE b.player_id IS NULL
                """
            ).fetchone()[0]

            if duplicate_alias or duplicate_auth or alias_not_auth:
                raise RuntimeError(
                    f"{source}: invalid tracking lookup. "
                    f"duplicate_alias={duplicate_alias}, "
                    f"duplicate_authoritative={duplicate_auth}, "
                    f"alias_not_in_authoritative={alias_not_auth}"
                )

            validation_rows.extend(
                [
                    {
                        "source": source,
                        "check": "duplicate_alias_confirmed_player_ids",
                        "problems": int(duplicate_alias),
                    },
                    {
                        "source": source,
                        "check": "duplicate_authoritative_player_ids",
                        "problems": int(duplicate_auth),
                    },
                    {
                        "source": source,
                        "check": "alias_confirmed_not_in_authoritative",
                        "problems": int(alias_not_auth),
                    },
                ]
            )

            # ------------------------------------------------------------
            # One row per canonical match with tracked-player counts.
            # ------------------------------------------------------------
            con.execute("DROP TABLE IF EXISTS match_cov")
            con.execute(
                f"""
                CREATE TEMP TABLE match_cov AS
                WITH
                alias_ids AS (
                    SELECT DISTINCT player_id FROM read_parquet('{alias}')
                ),
                auth_ids AS (
                    SELECT DISTINCT player_id FROM read_parquet('{auth}')
                ),
                counts AS (
                    SELECT
                        p.match_id,
                        COUNT(*) FILTER (WHERE a.player_id IS NOT NULL) AS n_alias_confirmed,
                        COUNT(*) FILTER (WHERE u.player_id IS NOT NULL) AS n_authoritative
                    FROM read_parquet('{participants}', union_by_name=true) p
                    LEFT JOIN alias_ids a USING (player_id)
                    LEFT JOIN auth_ids u USING (player_id)
                    GROUP BY p.match_id
                )
                SELECT
                    m.match_id,
                    m.platform_id,
                    m.queue_id,
                    m.game_start_ms,
                    COALESCE(c.n_alias_confirmed, 0)::INTEGER AS n_alias_confirmed,
                    COALESCE(c.n_authoritative, 0)::INTEGER AS n_authoritative
                FROM read_parquet('{matches}', union_by_name=true) m
                LEFT JOIN counts c USING (match_id)
                """
            )

            row = con.execute(
                """
                SELECT
                    COUNT(*)::BIGINT AS total_matches,
                    SUM(n_alias_confirmed >= 1)::BIGINT AS alias_confirmed_covered_matches,
                    SUM(n_authoritative >= 1)::BIGINT AS authoritative_covered_matches,
                    SUM(n_alias_confirmed = 0)::BIGINT AS alias_confirmed_uncovered_matches,
                    SUM(n_authoritative = 0)::BIGINT AS authoritative_uncovered_matches,
                    AVG(n_alias_confirmed) AS mean_alias_confirmed_players_per_match,
                    AVG(n_authoritative) AS mean_authoritative_players_per_match,
                    MAX(n_alias_confirmed)::INTEGER AS max_alias_confirmed_players_in_match,
                    MAX(n_authoritative)::INTEGER AS max_authoritative_players_in_match
                FROM match_cov
                """
            ).fetchdf().iloc[0].to_dict()

            total = int(row["total_matches"])
            row["source"] = source
            row["alias_confirmed_match_coverage_percent"] = (
                100.0 * int(row["alias_confirmed_covered_matches"]) / total
            )
            row["authoritative_match_coverage_percent"] = (
                100.0 * int(row["authoritative_covered_matches"]) / total
            )
            coverage_rows.append(row)

            # ------------------------------------------------------------
            # Persistent linked player-match tables.
            #
            # These are the reusable bridge between identity/provenance and
            # later timeline/statistical/modeling code.
            # ------------------------------------------------------------
            auth_cols = set(get_columns(con, auth_path))
            auth_evidence_expr = (
                "CAST(tr.tracking_evidence AS VARCHAR)"
                if "tracking_evidence" in auth_cols
                else "'authoritative'"
            )

            for cohort, lookup_path in (
                ("alias_confirmed", alias_path),
                ("authoritative", auth_path),
            ):
                lookup = sql_path(lookup_path)
                linked_dir = args.linked_output / cohort / source

                if cohort == "authoritative":
                    alias_join = (
                        f"LEFT JOIN read_parquet('{alias}') ac USING (player_id)"
                    )
                    is_alias_expr = "(ac.player_id IS NOT NULL)"
                    evidence_expr = auth_evidence_expr
                else:
                    alias_join = ""
                    is_alias_expr = "TRUE"
                    evidence_expr = "'fresh_seed_alias_unique'"

                query = f"""
                    SELECT
                        '{sql_text(source)}' AS source,
                        '{cohort}' AS tracked_cohort,
                        {is_alias_expr} AS is_alias_confirmed,
                        {evidence_expr} AS tracking_evidence,
                        p.*
                    FROM read_parquet('{participants}', union_by_name=true) p
                    INNER JOIN read_parquet('{lookup}') tr USING (player_id)
                    {alias_join}
                """
                linked, shard_count = write_sharded_parquet(
                    con,
                    query,
                    linked_dir,
                    rows_per_file=args.rows_per_linked_file,
                )

                stats = con.execute(
                    f"""
                    SELECT
                        COUNT(*)::BIGINT AS player_match_rows,
                        COUNT(DISTINCT player_id)::BIGINT AS players_with_observations,
                        COUNT(DISTINCT match_id)::BIGINT AS matches_with_tracked_players,
                        COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT player_id, match_id FROM read_parquet('{linked}'))) AS duplicate_player_match_rows
                    FROM read_parquet('{linked}')
                    """
                ).fetchdf().iloc[0].to_dict()

                if int(stats["duplicate_player_match_rows"]) != 0:
                    raise RuntimeError(
                        f"{source}/{cohort}: duplicate (player_id, match_id) rows "
                        f"were materialized."
                    )

                stats.update(
                    {
                        "source": source,
                        "cohort": cohort,
                        "linked_dataset": str(linked_dir.resolve()),
                        "linked_parquet_parts": int(shard_count),
                    }
                )
                linkage_rows.append(stats)

            # ------------------------------------------------------------
            # Coverage distribution.
            # ------------------------------------------------------------
            dist = con.execute(
                """
                WITH x AS (
                    SELECT 'alias_confirmed' AS cohort, n_alias_confirmed AS n FROM match_cov
                    UNION ALL
                    SELECT 'authoritative', n_authoritative FROM match_cov
                )
                SELECT
                    cohort,
                    CASE
                        WHEN n = 0 THEN '0'
                        WHEN n = 1 THEN '1'
                        WHEN n = 2 THEN '2'
                        ELSE '3+'
                    END AS tracked_players_in_match,
                    COUNT(*)::BIGINT AS matches
                FROM x
                GROUP BY cohort, tracked_players_in_match
                ORDER BY cohort,
                    CASE tracked_players_in_match
                        WHEN '0' THEN 0
                        WHEN '1' THEN 1
                        WHEN '2' THEN 2
                        ELSE 3
                    END
                """
            ).fetchdf()
            dist.insert(0, "source", source)
            dist["percent"] = 100.0 * dist["matches"] / total
            dist_frames.append(dist)

            qdf = con.execute(
                """
                SELECT
                    queue_id,
                    COUNT(*)::BIGINT AS matches,
                    SUM(n_alias_confirmed >= 1)::BIGINT AS alias_confirmed_covered,
                    SUM(n_authoritative >= 1)::BIGINT AS authoritative_covered,
                    100.0 * SUM(n_alias_confirmed >= 1) / COUNT(*) AS alias_confirmed_coverage_percent,
                    100.0 * SUM(n_authoritative >= 1) / COUNT(*) AS authoritative_coverage_percent
                FROM match_cov
                GROUP BY queue_id
                ORDER BY matches DESC, queue_id
                """
            ).fetchdf()
            qdf.insert(0, "source", source)
            queue_frames.append(qdf)

            pdf = con.execute(
                """
                SELECT
                    platform_id,
                    COUNT(*)::BIGINT AS matches,
                    SUM(n_alias_confirmed >= 1)::BIGINT AS alias_confirmed_covered,
                    SUM(n_authoritative >= 1)::BIGINT AS authoritative_covered,
                    100.0 * SUM(n_alias_confirmed >= 1) / COUNT(*) AS alias_confirmed_coverage_percent,
                    100.0 * SUM(n_authoritative >= 1) / COUNT(*) AS authoritative_coverage_percent
                FROM match_cov
                GROUP BY platform_id
                ORDER BY matches DESC, platform_id
                """
            ).fetchdf()
            pdf.insert(0, "source", source)
            platform_frames.append(pdf)

            udf = con.execute(
                """
                SELECT
                    'alias_confirmed' AS cohort,
                    COUNT(*) FILTER (WHERE n_alias_confirmed = 0)::BIGINT AS uncovered_all,
                    COUNT(*) FILTER (WHERE n_alias_confirmed = 0 AND queue_id = 420)::BIGINT AS uncovered_queue420,
                    COUNT(*) FILTER (WHERE n_alias_confirmed = 0 AND queue_id = 440)::BIGINT AS uncovered_queue440
                FROM match_cov
                UNION ALL
                SELECT
                    'authoritative',
                    COUNT(*) FILTER (WHERE n_authoritative = 0)::BIGINT,
                    COUNT(*) FILTER (WHERE n_authoritative = 0 AND queue_id = 420)::BIGINT,
                    COUNT(*) FILTER (WHERE n_authoritative = 0 AND queue_id = 440)::BIGINT
                FROM match_cov
                """
            ).fetchdf()
            udf.insert(0, "source", source)
            uncovered_rows.extend(udf.to_dict("records"))

            if args.sample_uncovered > 0:
                sample = con.execute(
                    f"""
                    SELECT
                        '{sql_text(source)}' AS source,
                        match_id,
                        platform_id,
                        queue_id,
                        game_start_ms,
                        n_alias_confirmed,
                        n_authoritative
                    FROM match_cov
                    WHERE n_authoritative = 0
                    ORDER BY game_start_ms, match_id
                    LIMIT {int(args.sample_uncovered)}
                    """
                ).fetchdf()
                if not sample.empty:
                    uncovered_samples.append(sample)

    finally:
        con.close()

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(args.output / "match_tracking_coverage_summary.csv", index=False)

    pd.DataFrame(validation_rows).to_csv(
        args.output / "tracking_lookup_validation.csv", index=False
    )
    pd.DataFrame(linkage_rows).to_csv(
        args.output / "linked_player_match_summary.csv", index=False
    )
    pd.concat(dist_frames, ignore_index=True).to_csv(
        args.output / "tracked_players_per_match_distribution.csv", index=False
    )
    pd.concat(queue_frames, ignore_index=True).to_csv(
        args.output / "coverage_by_queue.csv", index=False
    )
    pd.concat(platform_frames, ignore_index=True).to_csv(
        args.output / "coverage_by_platform.csv", index=False
    )
    pd.DataFrame(uncovered_rows).to_csv(
        args.output / "uncovered_matches_summary.csv", index=False
    )

    if uncovered_samples:
        pd.concat(uncovered_samples, ignore_index=True).to_csv(
            args.output / "uncovered_match_ids_sample.csv", index=False
        )

    payload = {
        "cohort_definitions": {
            "alias_confirmed": (
                "Fresh seed-list alias uniquely resolved to a raw Match-V5 PUUID."
            ),
            "authoritative": (
                "alias_confirmed plus DB seed PUUIDs assigned unambiguously to one "
                "regional processed corpus."
            ),
        },
        "main_analysis_cohort": "authoritative",
        "linked_data_root": str(args.linked_output.resolve()),
        "coverage": coverage_rows,
        "linkage": linkage_rows,
    }
    (args.output / "match_tracking_coverage_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTRACKING COVERAGE + LINKAGE COMPLETE\n")
    cols = [
        "source",
        "total_matches",
        "alias_confirmed_match_coverage_percent",
        "authoritative_match_coverage_percent",
    ]
    print(coverage_df[cols].to_string(index=False))

    link_df = pd.DataFrame(linkage_rows)
    print("\nLINKED PLAYER-MATCH TABLES\n")
    print(
        link_df[
            [
                "source",
                "cohort",
                "player_match_rows",
                "players_with_observations",
                "matches_with_tracked_players",
                "duplicate_player_match_rows",
            ]
        ].to_string(index=False)
    )

    print(f"\nCoverage audit: {args.output}")
    print(f"Reusable links: {args.linked_output}")


if __name__ == "__main__":
    main()

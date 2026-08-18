#!/usr/bin/env python3
"""
00_verify_processed_data.py

One-time preflight check for the canonical Parquet extraction.

Purpose
-------
Verify that the existing full_* Parquet datasets are safe to keep after the
Riot-ID/seed-list issue was discovered.

This script checks that:
1. Analytical Parquet tables contain no Riot display-name / raw account-ID
   columns (unless you intentionally extracted raw PUUIDs).
2. participant.player_id is a 32-character lowercase SHA-256 prefix.
3. Core relational integrity still holds.
4. Optional raw-JSON spot checks prove:
       raw participant PUUID -> SHA256[:32] -> existing Parquet player_id
   without using riotIdGameName / riotIdTagline at all.
5. Extraction run_summary.json is inspected when available.

It NEVER modifies input data.

Example (processed-only):
python .\code\00_verify_processed_data.py `
  --processed "NA=.\data\processed\full_na" `
              "KR=.\data\processed\full_kr" `
              "EU=.\data\processed\full_eu" `
  --output ".\data\processed\preflight_verification" `
  --overwrite

Stronger version with raw spot checks (adjust raw paths if needed):
python .\code\00_verify_processed_data.py `
  --processed "NA=.\data\processed\full_na" `
              "KR=.\data\processed\full_kr" `
              "EU=.\data\processed\full_eu" `
  --raw "NA=.\data\raw\matches_raw_na" `
        "KR=.\data\raw\matches_raw_kr" `
        "EU=.\data\raw\matches_raw_euw" `
  --raw-sample-matches 50 `
  --output ".\data\processed\preflight_verification" `
  --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import duckdb
import pandas as pd

try:
    import orjson  # type: ignore
except ImportError:
    orjson = None


FORBIDDEN_IDENTITY_COLUMNS = {
    "puuid",
    "summoner_id",
    "summoner_name",
    "riot_id_game_name",
    "riot_id_tagline",
}

REQUIRED_PARTICIPANT_COLUMNS = {
    "match_id",
    "player_id",
    "platform_id",
    "queue_id",
    "game_start_ms",
    "game_end_ms",
    "win",
    "champion_id",
    "team_position",
    "kills",
    "deaths",
    "assists",
}

REQUIRED_MATCH_COLUMNS = {
    "match_id",
    "platform_id",
    "queue_id",
    "game_start_ms",
    "game_end_ms",
    "game_duration_s",
    "end_of_game_result",
}


def parse_named_path(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f'Expected NAME=PATH, got "{text}"')
    name, raw = text.split("=", 1)
    name = name.strip()
    raw = raw.strip()
    if not name or not raw:
        raise argparse.ArgumentTypeError(f'Expected NAME=PATH, got "{text}"')
    return name, Path(raw)


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def parquet_glob(root: Path, table: str) -> str:
    p = root / table
    if not p.exists():
        raise FileNotFoundError(f"Missing table directory: {p}")
    files = list(p.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in: {p}")
    return sql_path(p / "*.parquet")


def stable_player_id(puuid: str) -> str:
    return hashlib.sha256(puuid.encode("utf-8")).hexdigest()[:32]


def loads_json(raw: bytes):
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def iter_json_files(root: Path) -> Iterator[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif (
                            entry.is_file(follow_symlinks=False)
                            and entry.name.lower().endswith(".json")
                        ):
                            yield Path(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue


def read_schema_columns(
    con: duckdb.DuckDBPyConnection, pattern: str
) -> List[str]:
    df = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{pattern}', union_by_name=true)"
    ).fetchdf()
    return df["column_name"].astype(str).tolist()


def raw_spot_check(
    con: duckdb.DuckDBPyConnection,
    source: str,
    raw_root: Path,
    participants_pattern: str,
    sample_matches: int,
) -> Dict:
    """
    Deterministically inspect the first N parseable raw JSON files and verify
    every participant's PUUID-derived hash against the processed Parquet row
    for the same match.
    """
    expected_rows = []
    parse_errors = 0
    matches_sampled = 0
    raw_participants = 0
    raw_names_with_replacement_char = 0

    for path in sorted(iter_json_files(raw_root)):
        if matches_sampled >= sample_matches:
            break
        try:
            obj = loads_json(path.read_bytes())
        except Exception:
            parse_errors += 1
            continue

        if not isinstance(obj, dict):
            continue

        metadata = obj.get("metadata") or {}
        info = obj.get("info") or {}
        match_id = metadata.get("matchId")
        if match_id is None:
            match_id = info.get("gameId")
        if match_id is None:
            continue
        match_id = str(match_id)

        participants = info.get("participants") or []
        for p in participants:
            if not isinstance(p, dict):
                continue
            puuid = p.get("puuid")
            if not puuid:
                continue

            raw_participants += 1

            # This is diagnostic only. The hash verification below never uses names.
            game_name = str(p.get("riotIdGameName") or "")
            tag = str(p.get("riotIdTagline") or "")
            if "\ufffd" in game_name or "\ufffd" in tag:
                raw_names_with_replacement_char += 1

            expected_rows.append(
                {
                    "match_id": match_id,
                    "player_id": stable_player_id(str(puuid)),
                }
            )

        matches_sampled += 1

    if not expected_rows:
        return {
            "raw_matches_sampled": matches_sampled,
            "raw_participants_hashed": 0,
            "raw_hash_pairs_found_in_parquet": 0,
            "raw_hash_pairs_missing_from_parquet": 0,
            "raw_hash_match_percent": None,
            "raw_parse_errors_before_sample_limit": parse_errors,
            "raw_names_with_unicode_replacement_char": raw_names_with_replacement_char,
        }

    expected = pd.DataFrame(expected_rows).drop_duplicates()
    con.register("expected_raw_ids", expected)

    matched = con.execute(
        f"""
        SELECT COUNT(*) AS n
        FROM expected_raw_ids e
        INNER JOIN read_parquet('{participants_pattern}', union_by_name=true) p
          ON p.match_id = e.match_id
         AND p.player_id = e.player_id
        """
    ).fetchone()[0]

    total = len(expected)
    con.unregister("expected_raw_ids")

    return {
        "raw_matches_sampled": matches_sampled,
        "raw_participants_hashed": total,
        "raw_hash_pairs_found_in_parquet": int(matched),
        "raw_hash_pairs_missing_from_parquet": int(total - matched),
        "raw_hash_match_percent": 100.0 * matched / total if total else None,
        "raw_parse_errors_before_sample_limit": parse_errors,
        "raw_names_with_unicode_replacement_char": raw_names_with_replacement_char,
    }


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {path}. Use --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", nargs="+", type=parse_named_path, required=True)
    p.add_argument("--raw", nargs="*", type=parse_named_path, default=[])
    p.add_argument("--raw-sample-matches", type=int, default=50)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output(args.output, args.overwrite)

    processed: Dict[str, Path] = dict(args.processed)
    raw: Dict[str, Path] = dict(args.raw)

    unknown_raw_labels = set(raw) - set(processed)
    if unknown_raw_labels:
        raise ValueError(
            f"Raw labels missing from --processed: {sorted(unknown_raw_labels)}"
        )

    con = duckdb.connect(database=":memory:")
    rows = []
    failures = []

    try:
        for source, root in processed.items():
            print(f"[verify] {source}: {root}", flush=True)

            matches_pattern = parquet_glob(root, "matches")
            participants_pattern = parquet_glob(root, "participants")
            teams_pattern = parquet_glob(root, "teams")
            bans_pattern = parquet_glob(root, "team_bans")

            participant_cols = read_schema_columns(con, participants_pattern)
            match_cols = read_schema_columns(con, matches_pattern)
            team_cols = read_schema_columns(con, teams_pattern)
            ban_cols = read_schema_columns(con, bans_pattern)

            forbidden_present = sorted(
                FORBIDDEN_IDENTITY_COLUMNS.intersection(participant_cols)
            )
            suspicious_name_columns = sorted(
                c for c in participant_cols
                if "name" in c.lower() and c != "champion_name"
            )

            missing_participant_required = sorted(
                REQUIRED_PARTICIPANT_COLUMNS - set(participant_cols)
            )
            missing_match_required = sorted(
                REQUIRED_MATCH_COLUMNS - set(match_cols)
            )

            counts = con.execute(
                f"""
                WITH
                m AS (
                    SELECT
                        COUNT(*) AS matches,
                        COUNT(DISTINCT match_id) AS unique_matches
                    FROM read_parquet('{matches_pattern}', union_by_name=true)
                ),
                p AS (
                    SELECT
                        COUNT(*) AS participant_rows,
                        SUM(CASE WHEN player_id IS NULL THEN 1 ELSE 0 END) AS null_player_ids,
                        SUM(
                            CASE
                            WHEN player_id IS NOT NULL
                             AND NOT regexp_full_match(player_id, '[0-9a-f]{{32}}')
                            THEN 1 ELSE 0 END
                        ) AS malformed_player_ids
                    FROM read_parquet('{participants_pattern}', union_by_name=true)
                ),
                bad_match_participants AS (
                    SELECT COUNT(*) AS n
                    FROM (
                        SELECT match_id
                        FROM read_parquet('{participants_pattern}', union_by_name=true)
                        GROUP BY match_id
                        HAVING COUNT(*) <> 10
                           OR COUNT(DISTINCT player_id) <> 10
                    )
                ),
                bad_match_teams AS (
                    SELECT COUNT(*) AS n
                    FROM (
                        SELECT match_id
                        FROM read_parquet('{teams_pattern}', union_by_name=true)
                        GROUP BY match_id
                        HAVING COUNT(*) <> 2
                    )
                )
                SELECT
                    m.matches,
                    m.unique_matches,
                    p.participant_rows,
                    p.null_player_ids,
                    p.malformed_player_ids,
                    bad_match_participants.n AS matches_with_bad_participant_structure,
                    bad_match_teams.n AS matches_with_bad_team_structure
                FROM m, p, bad_match_participants, bad_match_teams
                """
            ).fetchdf().iloc[0].to_dict()

            run_summary_path = root / "audit" / "run_summary.json"
            run_keep_puuid = None
            run_parse_errors = None
            if run_summary_path.exists():
                try:
                    run_summary = json.loads(
                        run_summary_path.read_text(encoding="utf-8")
                    )
                    run_keep_puuid = run_summary.get("keep_puuid")
                    run_parse_errors = run_summary.get("parse_or_schema_errors")
                except Exception:
                    pass

            raw_stats = {
                "raw_matches_sampled": None,
                "raw_participants_hashed": None,
                "raw_hash_pairs_found_in_parquet": None,
                "raw_hash_pairs_missing_from_parquet": None,
                "raw_hash_match_percent": None,
                "raw_parse_errors_before_sample_limit": None,
                "raw_names_with_unicode_replacement_char": None,
            }
            if source in raw:
                raw_stats = raw_spot_check(
                    con=con,
                    source=source,
                    raw_root=raw[source],
                    participants_pattern=participants_pattern,
                    sample_matches=args.raw_sample_matches,
                )

            source_failures = []

            if forbidden_present:
                source_failures.append(
                    f"forbidden identity columns present: {forbidden_present}"
                )
            if suspicious_name_columns:
                source_failures.append(
                    f"unexpected name-like participant columns: {suspicious_name_columns}"
                )
            if missing_participant_required:
                source_failures.append(
                    f"missing participant columns: {missing_participant_required}"
                )
            if missing_match_required:
                source_failures.append(
                    f"missing match columns: {missing_match_required}"
                )
            if int(counts["matches"]) != int(counts["unique_matches"]):
                source_failures.append("duplicate match_id rows")
            if int(counts["null_player_ids"]) != 0:
                source_failures.append("null player_id values")
            if int(counts["malformed_player_ids"]) != 0:
                source_failures.append("malformed player_id values")
            if int(counts["matches_with_bad_participant_structure"]) != 0:
                source_failures.append("matches not containing 10 unique participant IDs")
            if int(counts["matches_with_bad_team_structure"]) != 0:
                source_failures.append("matches not containing exactly 2 team rows")
            if (
                raw_stats["raw_hash_pairs_missing_from_parquet"] is not None
                and int(raw_stats["raw_hash_pairs_missing_from_parquet"]) != 0
            ):
                source_failures.append(
                    "raw PUUID hashes did not map perfectly to processed player_id"
                )

            safe = len(source_failures) == 0
            if not safe:
                failures.append(f"{source}: " + "; ".join(source_failures))

            row = {
                "source": source,
                "safe_to_keep_processed_parquet": safe,
                "forbidden_identity_columns_present": ";".join(forbidden_present),
                "unexpected_name_like_columns": ";".join(suspicious_name_columns),
                "run_summary_keep_puuid": run_keep_puuid,
                "run_summary_parse_or_schema_errors": run_parse_errors,
                **counts,
                **raw_stats,
            }
            rows.append(row)

            # Save schemas for transparent inspection.
            pd.DataFrame({"column": participant_cols}).to_csv(
                args.output / f"{source}_participants_columns.csv", index=False
            )
            pd.DataFrame({"column": match_cols}).to_csv(
                args.output / f"{source}_matches_columns.csv", index=False
            )
            pd.DataFrame({"column": team_cols}).to_csv(
                args.output / f"{source}_teams_columns.csv", index=False
            )
            pd.DataFrame({"column": ban_cols}).to_csv(
                args.output / f"{source}_team_bans_columns.csv", index=False
            )

    finally:
        con.close()

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output / "preflight_summary.csv", index=False)

    payload = {
        "all_sources_passed": not failures,
        "failures": failures,
        "sources": rows,
        "interpretation": (
            "A passing raw spot check proves that player_id in the existing Parquet "
            "is derived from raw PUUID and does not depend on Riot display-name text."
        ),
    }
    (args.output / "preflight_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + summary_df.to_string(index=False))
    print(f"\nSaved: {args.output / 'preflight_summary.csv'}")

    if failures:
        print("\nPRE-FLIGHT FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(2)

    print("\nPRE-FLIGHT PASSED: existing canonical Parquet files are safe to keep.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
02_build_tracked_players.py

Build the FINAL tracked-player cohorts used by the project.

ALIAS_CONFIRMED cohort
----------------------
Only players whose Riot ID from the fresh original summoners_*_used.txt file
matches exactly one PUUID in that region's raw Match-V5 archive.

This is the strict alias-confirmed cohort retained for provenance sensitivity checks.

AUTHORITATIVE cohort
--------------------
ALIAS_CONFIRMED plus PUUIDs stored in league_data.db/player_mastery that can be assigned
to exactly one supplied regional processed corpus.

This is the main tracked-player cohort used for the project analysis.

Important
---------
- No match-count/history-length inference is used.
- Raw Riot names and raw PUUIDs are NOT written to tracked_players.parquet.
- Ambiguous aliases are excluded rather than guessed.
- DB PUUIDs appearing in more than one regional corpus are excluded from the
  expanded source assignment rather than guessed.
- Existing full_* Parquet tables are not modified.

Example (PowerShell)
--------------------
python .\code\02_build_tracked_players.py `
  --db ".\data\raw\Games of League of Legends\league_data.db" `
  --processed "NA=.\data\processed\full_na" "KR=.\data\processed\full_kr" "EU=.\data\processed\full_eu" `
  --raw "NA=.\data\raw\Games of League of Legends\matches_raw_na" "KR=.\data\raw\Games of League of Legends\matches_raw_kr" "EU=.\data\raw\Games of League of Legends\matches_raw_euw" `
  --seed-list "NA=.\data\raw\Games of League of Legends\summoners_list_na_used.txt" "KR=.\data\raw\Games of League of Legends\summoners_list_kr_used.txt" "EU=.\data\raw\Games of League of Legends\summoners_list_euw_used.txt" `
  --output ".\data\processed\tracking" `
  --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import duckdb
import pandas as pd

try:
    import orjson  # type: ignore
except ImportError:
    orjson = None


def parse_named_path(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f'Expected NAME=PATH, got "{text}"')
    name, raw = text.split("=", 1)
    name = name.strip()
    raw = raw.strip()
    if not name or not raw:
        raise argparse.ArgumentTypeError(f'Expected NAME=PATH, got "{text}"')
    return name, Path(raw)


def stable_player_id(puuid: str) -> str:
    return hashlib.sha256(puuid.encode("utf-8")).hexdigest()[:32]


def normalize_riot_id(game_name: object, tag_line: object) -> Optional[str]:
    if game_name is None or tag_line is None:
        return None
    name = unicodedata.normalize("NFKC", str(game_name).strip()).casefold()
    tag = unicodedata.normalize("NFKC", str(tag_line).strip()).casefold()
    if not name or not tag:
        return None
    return f"{name}#{tag}"


def loads_json(raw: bytes):
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def iter_json_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Raw path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(root)

    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError:
            continue

        # deterministic order
        for entry in sorted(entries, key=lambda e: e.name, reverse=True):
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
            except OSError:
                continue

        for entry in sorted(entries, key=lambda e: e.name):
            try:
                if (
                    entry.is_file(follow_symlinks=False)
                    and entry.name.lower().endswith(".json")
                ):
                    yield Path(entry.path)
            except OSError:
                continue


def read_seed_list(path: Path) -> Tuple[List[str], Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    aliases: List[str] = []
    display: Dict[str, str] = {}
    duplicate_aliases: Set[str] = set()

    # utf-8-sig works for normal UTF-8 and UTF-8 BOM files.
    # errors=strict prevents silent mojibake/replacement.
    with path.open("r", encoding="utf-8-sig", errors="strict") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if "#" not in line:
                raise ValueError(f"Malformed seed line in {path}: {line!r}")

            game_name, tag = line.split("#", 1)
            key = normalize_riot_id(game_name, tag)
            if key is None:
                continue

            if key in display:
                duplicate_aliases.add(key)
                continue

            aliases.append(key)
            display[key] = line

    if duplicate_aliases:
        print(
            f"[warning] {path.name}: {len(duplicate_aliases)} duplicate "
            "normalized aliases were ignored."
        )

    return aliases, display


def resolve_seed_aliases(
    source: str,
    seed_path: Path,
    raw_root: Path,
) -> Tuple[pd.DataFrame, Dict]:
    aliases, display = read_seed_list(seed_path)
    targets = set(aliases)

    alias_to_puuids: Dict[str, Set[str]] = defaultdict(set)
    json_files_scanned = 0
    parse_errors = 0
    participant_rows_seen = 0

    print(f"[{source}] scanning raw JSON for {len(aliases):,} fresh seed aliases...")

    for path in iter_json_files(raw_root):
        try:
            obj = loads_json(path.read_bytes())
        except Exception:
            parse_errors += 1
            continue

        if not isinstance(obj, dict):
            continue

        info = obj.get("info") or {}
        participants = info.get("participants") or []
        json_files_scanned += 1

        for p in participants:
            if not isinstance(p, dict):
                continue
            participant_rows_seen += 1

            alias = normalize_riot_id(
                p.get("riotIdGameName"),
                p.get("riotIdTagline"),
            )
            if alias is None or alias not in targets:
                continue

            puuid = p.get("puuid")
            if puuid:
                alias_to_puuids[alias].add(str(puuid))

    rows = []
    for alias in aliases:
        puuids = alias_to_puuids.get(alias, set())

        if len(puuids) == 0:
            status = "unresolved"
            player_id = None
        elif len(puuids) == 1:
            status = "resolved_unique"
            player_id = stable_player_id(next(iter(puuids)))
        else:
            status = "ambiguous_multiple_puuids"
            player_id = None

        rows.append(
            {
                "seed_riot_id": display[alias],
                "status": status,
                "distinct_puuids_seen": len(puuids),
                "player_id": player_id,
            }
        )

    detail = pd.DataFrame(rows)

    resolved_n = int((detail["status"] == "resolved_unique").sum())
    ambiguous_n = int((detail["status"] == "ambiguous_multiple_puuids").sum())
    unresolved_n = int((detail["status"] == "unresolved").sum())

    summary = {
        "source": source,
        "seed_aliases": len(aliases),
        "resolved_unique": resolved_n,
        "ambiguous": ambiguous_n,
        "unresolved": unresolved_n,
        "resolved_unique_percent": (
            100.0 * resolved_n / len(aliases) if aliases else None
        ),
        "raw_json_files_scanned": json_files_scanned,
        "participant_rows_seen": participant_rows_seen,
        "raw_parse_errors": parse_errors,
    }

    return detail, summary


def load_db_seed_ids(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "player_mastery" not in tables:
            raise RuntimeError(
                f"player_mastery not found in {db_path}; tables={sorted(tables)}"
            )

        rows = conn.execute(
            """
            SELECT DISTINCT puuid
            FROM player_mastery
            WHERE puuid IS NOT NULL
              AND TRIM(puuid) <> ''
            """
        ).fetchall()
    finally:
        conn.close()

    ids = sorted({stable_player_id(str(row[0])) for row in rows})
    return pd.DataFrame({"player_id": ids})


def parquet_pattern(root: Path, table: str) -> str:
    table_dir = root / table
    if not table_dir.exists() or not any(table_dir.glob("*.parquet")):
        raise FileNotFoundError(f"Missing Parquet table: {table_dir}")
    return (table_dir.resolve() / "*.parquet").as_posix().replace("'", "''")


def verify_ids_exist_in_source(
    con: duckdb.DuckDBPyConnection,
    processed_root: Path,
    ids: Set[str],
) -> Set[str]:
    if not ids:
        return set()

    df = pd.DataFrame({"player_id": sorted(ids)})
    con.register("candidate_ids", df)
    pattern = parquet_pattern(processed_root, "participants")

    matched = con.execute(
        f"""
        SELECT DISTINCT p.player_id
        FROM read_parquet('{pattern}', union_by_name=true) p
        INNER JOIN candidate_ids c USING (player_id)
        """
    ).fetchdf()

    con.unregister("candidate_ids")
    return set(matched["player_id"].astype(str))


def assign_db_ids_to_unique_sources(
    con: duckdb.DuckDBPyConnection,
    processed: Dict[str, Path],
    db_ids: pd.DataFrame,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    con.register("db_ids", db_ids)

    sources_by_id: Dict[str, Set[str]] = defaultdict(set)

    for source, root in processed.items():
        pattern = parquet_pattern(root, "participants")
        df = con.execute(
            f"""
            SELECT DISTINCT p.player_id
            FROM read_parquet('{pattern}', union_by_name=true) p
            INNER JOIN db_ids d USING (player_id)
            """
        ).fetchdf()

        for pid in df["player_id"].astype(str):
            sources_by_id[pid].add(source)

    con.unregister("db_ids")

    unique_by_source: Dict[str, Set[str]] = defaultdict(set)
    multi_source: Dict[str, Set[str]] = {}

    for pid, source_set in sources_by_id.items():
        if len(source_set) == 1:
            unique_by_source[next(iter(source_set))].add(pid)
        elif len(source_set) > 1:
            multi_source[pid] = source_set

    return unique_by_source, multi_source


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
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--processed", nargs="+", type=parse_named_path, required=True)
    p.add_argument("--raw", nargs="+", type=parse_named_path, required=True)
    p.add_argument("--seed-list", nargs="+", type=parse_named_path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output(args.output, args.overwrite)

    processed = dict(args.processed)
    raw = dict(args.raw)
    seed_lists = dict(args.seed_list)

    labels = set(processed)
    if set(raw) != labels or set(seed_lists) != labels:
        raise ValueError(
            "Source labels must match exactly across --processed, --raw and "
            f"--seed-list. processed={sorted(processed)}, raw={sorted(raw)}, "
            f"seed-list={sorted(seed_lists)}"
        )

    audit_dir = args.output / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")

    try:
        # 1) Resolve fresh seed aliases.
        alias_details: Dict[str, pd.DataFrame] = {}
        alias_summaries: Dict[str, Dict] = {}
        alias_confirmed_ids: Dict[str, Set[str]] = {}

        for source in processed:
            detail, summary = resolve_seed_aliases(
                source=source,
                seed_path=seed_lists[source],
                raw_root=raw[source],
            )

            alias_details[source] = detail
            alias_summaries[source] = summary

            detail.to_csv(
                audit_dir / f"{source}_seed_alias_resolution_detail.csv",
                index=False,
                encoding="utf-8-sig",
            )

            ids = set(
                detail.loc[
                    detail["status"] == "resolved_unique",
                    "player_id",
                ]
                .dropna()
                .astype(str)
            )

            # Strong safety check: every resolved raw PUUID hash must already
            # exist in that region's canonical participant Parquet corpus.
            found = verify_ids_exist_in_source(con, processed[source], ids)
            missing = ids - found
            if missing:
                raise RuntimeError(
                    f"{source}: {len(missing)} primary player_ids derived from raw "
                    "PUUIDs were not found in processed participants Parquet."
                )

            alias_confirmed_ids[source] = found

        # 2) Load DB seed PUUIDs and assign only unique-source IDs.
        db_ids = load_db_seed_ids(args.db)
        db_unique_by_source, db_multi_source = assign_db_ids_to_unique_sources(
            con, processed, db_ids
        )

        # 3) Write ALIAS_CONFIRMED and AUTHORITATIVE cohorts.
        summary_rows = []

        for source in processed:
            alias_confirmed = alias_confirmed_ids[source]
            db_unique = db_unique_by_source.get(source, set())
            authoritative = alias_confirmed | db_unique

            alias_dir = args.output / "alias_confirmed" / source
            authoritative_dir = args.output / "authoritative" / source
            alias_dir.mkdir(parents=True, exist_ok=True)
            authoritative_dir.mkdir(parents=True, exist_ok=True)

            alias_df = pd.DataFrame(
                {
                    "player_id": sorted(alias_confirmed),
                    "source": source,
                    "cohort": "alias_confirmed",
                    "tracking_evidence": "fresh_seed_alias_unique",
                }
            )
            alias_df.to_parquet(
                alias_dir / "tracked_players.parquet",
                index=False,
                compression="zstd",
            )

            authoritative_rows = []
            for pid in sorted(authoritative):
                by_alias = pid in alias_confirmed
                by_db = pid in db_unique

                if by_alias and by_db:
                    evidence = "fresh_seed_alias_unique+mastery_db_unique_source"
                elif by_alias:
                    evidence = "fresh_seed_alias_unique_only"
                else:
                    evidence = "mastery_db_unique_source_only"

                authoritative_rows.append(
                    {
                        "player_id": pid,
                        "source": source,
                        "cohort": "authoritative",
                        "fresh_seed_alias_unique": by_alias,
                        "mastery_db_unique_source": by_db,
                        "tracking_evidence": evidence,
                    }
                )

            authoritative_df = pd.DataFrame(authoritative_rows)
            authoritative_df.to_parquet(
                authoritative_dir / "tracked_players.parquet",
                index=False,
                compression="zstd",
            )

            both = len(alias_confirmed & db_unique)
            alias_only = len(alias_confirmed - db_unique)
            db_only = len(db_unique - alias_confirmed)

            s = alias_summaries[source]
            summary_rows.append(
                {
                    "source": source,
                    "seed_aliases": s["seed_aliases"],
                    "seed_aliases_resolved_unique": s["resolved_unique"],
                    "seed_aliases_ambiguous": s["ambiguous"],
                    "seed_aliases_unresolved": s["unresolved"],
                    "seed_alias_resolution_percent": s["resolved_unique_percent"],
                    "alias_confirmed_tracked_players": len(alias_confirmed),
                    "db_seed_ids_unique_to_source": len(db_unique),
                    "supported_by_both": both,
                    "alias_confirmed_only": alias_only,
                    "authoritative_db_only": db_only,
                    "authoritative_tracked_players": len(authoritative),
                    "raw_json_files_scanned": s["raw_json_files_scanned"],
                    "raw_parse_errors": s["raw_parse_errors"],
                }
            )

        # 4) Audit ambiguous DB source assignments.
        multi_rows = [
            {
                "player_id": pid,
                "sources_present": ";".join(sorted(source_set)),
                "source_count": len(source_set),
            }
            for pid, source_set in sorted(db_multi_source.items())
        ]
        pd.DataFrame(multi_rows).to_csv(
            audit_dir / "db_seed_ids_present_in_multiple_sources.csv",
            index=False,
        )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            audit_dir / "tracking_cohort_summary.csv",
            index=False,
        )

        summary_json = {
            "alias_confirmed_definition": (
                "Fresh original seed-list Riot ID matched uniquely to exactly one "
                "raw Match-V5 PUUID in the corresponding source; hashed player_id "
                "verified to exist in the canonical participants Parquet."
            ),
            "authoritative_definition": (
                "Alias-confirmed plus player_mastery PUUIDs from league_data.db whose "
                "hashed player_id occurs in exactly one supplied regional corpus."
            ),
            "main_analysis_cohort": "authoritative",
            "alias_confirmed_usage": "strict provenance sensitivity/robustness analysis",
            "db_distinct_seed_puuids_hashed": len(db_ids),
            "db_seed_ids_present_in_multiple_sources": len(db_multi_source),
            "sources": summary_rows,
        }

        (audit_dir / "tracking_cohort_summary.json").write_text(
            json.dumps(summary_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    finally:
        con.close()

    print("\nFINAL TRACKED COHORTS BUILT\n")
    print(summary_df.to_string(index=False))
    print(
        f"\nAlias-confirmed root:  {args.output / 'alias_confirmed'}\n"
        f"Authoritative root:   {args.output / 'authoritative'}\n"
        f"Audit summary:         {audit_dir / 'tracking_cohort_summary.csv'}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
02_extract_match_v5.py

Phase 1 data-engineering pipeline for the project.

Goal
----
Normalize raw Riot Match-V5 JSON into analysis-friendly relational tables
WITHOUT silently filtering observations. Cleaning decisions belong in the next
stage, after we audit the real distributions.

Outputs
-------
<output>/
    matches/
        part-00000.parquet ...
    participants/
        part-00000.parquet ...
    teams/
        part-00000.parquet ...
    team_bans/
        part-00000.parquet ...
    audit/
        run_summary.json
        schema_observed.json
        errors.jsonl
        seen_matches.sqlite      # when exact deduplication is enabled

Design principles
-----------------
1. Stream files: never load the full multi-GB corpus into memory.
2. One match -> one row in matches.
3. One player-match -> one row in participants.
4. One team-match -> one row in teams.
5. Do not store player names by default.
6. Create a deterministic hashed player_id from PUUID for longitudinal analysis.
7. Keep raw source files unchanged, so the extractor can be expanded later.
8. Record schema drift and parse errors instead of hiding them.
9. Deduplicate match IDs with a disk-backed SQLite index instead of a huge RAM set.
10. Extract a broad stable core + selected challenge/rune fields that are useful
    for the behavioral/performance project.

Recommended full-data format: Parquet.
For a small local smoke test, --format csv.gz is also supported.

Example
-------
python 02_extract_match_v5.py \
  --input "/project/data/Games of League of Legends/matches_raw_euw" \
          "/project/data/Games of League of Legends/matches_raw_na" \
          "/project/data/Games of League of Legends/matches_raw_kr" \
  --output "/project/data/processed/match_v5" \
  --format parquet \
  --overwrite

To smoke-test only 1000 matches:
python 02_extract_match_v5.py ... --max-matches 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

try:
    import orjson  # faster if installed
except ImportError:
    orjson = None


# ---------------------------------------------------------------------------
# Stable extraction schema
# ---------------------------------------------------------------------------

# Intentionally broad: these are inexpensive scalar fields and cover current
# performance, economy, combat, objectives, vision, surrender, pings, and
# loadout. Adding a field later only requires adding its Riot key here.
PARTICIPANT_FIELDS: List[str] = [
    # Identity within match / role
    "participantId", "teamId", "win",
    "championId", "championName", "championTransform",
    "teamPosition", "individualPosition", "lane", "role",
    "summonerLevel",

    # Core combat
    "kills", "deaths", "assists",
    "killingSprees", "doubleKills", "tripleKills", "quadraKills", "pentaKills",
    "largestKillingSpree", "largestMultiKill", "largestCriticalStrike",

    # Progress / economy
    "champExperience", "champLevel",
    "goldEarned", "goldSpent",
    "totalMinionsKilled", "neutralMinionsKilled",
    "totalAllyJungleMinionsKilled", "totalEnemyJungleMinionsKilled",

    # Damage
    "totalDamageDealt", "totalDamageDealtToChampions",
    "physicalDamageDealt", "physicalDamageDealtToChampions",
    "magicDamageDealt", "magicDamageDealtToChampions",
    "trueDamageDealt", "trueDamageDealtToChampions",
    "damageDealtToBuildings", "damageDealtToTurrets", "damageDealtToObjectives",

    # Durability / support
    "totalDamageTaken", "physicalDamageTaken", "magicDamageTaken", "trueDamageTaken",
    "damageSelfMitigated",
    "totalHeal", "totalHealsOnTeammates", "totalDamageShieldedOnTeammates",
    "totalUnitsHealed",
    "timeCCingOthers", "totalTimeCCDealt",
    "longestTimeSpentLiving", "totalTimeSpentDead",

    # Vision
    "visionScore", "wardsPlaced", "wardsKilled",
    "detectorWardsPlaced", "visionWardsBoughtInGame", "sightWardsBoughtInGame",

    # Objectives
    "turretKills", "turretTakedowns", "turretsLost",
    "inhibitorKills", "inhibitorTakedowns", "inhibitorsLost",
    "nexusKills", "nexusTakedowns", "nexusLost",
    "dragonKills", "baronKills",
    "objectivesStolen", "objectivesStolenAssists",

    # Milestones / endings
    "firstBloodKill", "firstBloodAssist",
    "firstTowerKill", "firstTowerAssist",
    "gameEndedInEarlySurrender", "gameEndedInSurrender",
    "teamEarlySurrendered", "eligibleForProgression",
    "timePlayed",

    # Inventory / purchases
    "item0", "item1", "item2", "item3", "item4", "item5", "item6",
    "itemsPurchased", "consumablesPurchased",

    # Summoner spells / ability use
    "summoner1Id", "summoner2Id",
    "summoner1Casts", "summoner2Casts",
    "spell1Casts", "spell2Casts", "spell3Casts", "spell4Casts",

    # Pings: potentially useful as observable behavior in previous matches.
    "allInPings", "assistMePings", "basicPings", "commandPings", "dangerPings",
    "enemyMissingPings", "enemyVisionPings", "getBackPings", "holdPings",
    "needVisionPings", "onMyWayPings", "pushPings", "retreatPings",
    "visionClearedPings",
]

# Selected Riot "challenges" fields. We do not extract all ~100+ challenge
# fields by default because many are sparse/niche and would bloat the main
# table. The raw JSON is preserved and schema_observed.json tells us what else
# exists, so expansion is straightforward.
CHALLENGE_FIELDS: List[str] = [
    "kda",
    "killParticipation",
    "damagePerMinute",
    "goldPerMinute",
    "visionScorePerMinute",
    "teamDamagePercentage",
    "damageTakenOnTeamPercentage",
    "laneMinionsFirst10Minutes",
    "jungleCsBefore10Minutes",
    "earlyLaningPhaseGoldExpAdvantage",
    "laningPhaseGoldExpAdvantage",
    "maxCsAdvantageOnLaneOpponent",
    "maxLevelLeadLaneOpponent",
    "soloKills",
    "takedowns",
    "takedownsFirstXMinutes",
    "turretPlatesTaken",
    "controlWardsPlaced",
    "skillshotsHit",
    "skillshotsDodged",
    "hadAfkTeammate",
]

OBJECTIVE_NAMES: List[str] = [
    "atakhan", "baron", "champion", "dragon",
    "horde", "inhibitor", "riftHerald", "tower",
]

# Nested structures that are intentionally handled separately / explicitly.
NESTED_PARTICIPANT_KEYS = {"challenges", "missions", "perks"}

# Personally identifying / unnecessary display fields are not written.
PII_PARTICIPANT_KEYS = {
    "puuid", "summonerId", "summonerName", "riotIdGameName", "riotIdTagline",
}

# Known mode-specific / placeholder fields we deliberately do not write.
IGNORED_PARTICIPANT_KEYS = {
    *(f"PlayerScore{i}" for i in range(12)),
    "placement", "subteamPlacement", "playerSubteamId",
    *(f"playerAugment{i}" for i in range(1, 7)),
    "profileIcon",
    "unrealKills",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAMEL_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def snake(name: str) -> str:
    s1 = _CAMEL_RE_1.sub(r"\1_\2", name)
    return _CAMEL_RE_2.sub(r"\1_\2", s1).lower()


def loads_json(raw: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def load_json_file(path: Path) -> Any:
    with path.open("rb") as f:
        return loads_json(f.read())


def to_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def timestamp_to_ms(value: Any) -> Optional[int]:
    x = to_int(value)
    if x is None:
        return None
    # Defensive support for second-resolution Unix timestamps.
    if x < 10_000_000_000:
        x *= 1000
    return x


def safe_per_minute(value: Any, duration_seconds: Any) -> Optional[float]:
    v = to_float(value)
    d = to_float(duration_seconds)
    if v is None or d is None or d <= 0:
        return None
    return v / (d / 60.0)


def stable_player_id(puuid: str) -> str:
    # 128 bits of SHA-256 output: compact and vastly sufficient for this corpus.
    return hashlib.sha256(puuid.encode("utf-8")).hexdigest()[:32]


def patch_major_minor(game_version: Any) -> Optional[str]:
    if not game_version:
        return None
    parts = str(game_version).split(".")
    if len(parts) < 2:
        return str(game_version)
    return ".".join(parts[:2])


def match_prefix(match_id: Any) -> str:
    if match_id is None:
        return "MISSING"
    text = str(match_id)
    return text.split("_", 1)[0] if "_" in text else "NO_PREFIX"


# ---------------------------------------------------------------------------
# Source streaming
# ---------------------------------------------------------------------------

def iter_json_files(root: Path) -> Iterator[Path]:
    """Memory-safe recursive directory traversal."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".json"):
                            yield Path(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue


def iter_source_objects(source: Path) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """
    Yield (source_container, source_member, match_dict).

    Supports:
      * directory containing JSON files
      * a single JSON file
      * ZIP containing JSON files
    """
    source = source.resolve()

    if source.is_dir():
        for path in iter_json_files(source):
            obj = load_json_file(path)
            if isinstance(obj, dict):
                yield str(source), str(path), obj
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, dict):
                        yield str(source), f"{path}#{i}", item
        return

    if source.is_file() and source.suffix.lower() == ".json":
        obj = load_json_file(source)
        if isinstance(obj, dict):
            yield str(source), str(source), obj
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    yield str(source), f"{source}#{i}", item
        return

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            for member in zf.namelist():
                if member.endswith("/") or not member.lower().endswith(".json"):
                    continue
                with zf.open(member) as f:
                    obj = loads_json(f.read())
                if isinstance(obj, dict):
                    yield str(source), member, obj
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, dict):
                            yield str(source), f"{member}#{i}", item
        return

    raise ValueError(f"Unsupported input source: {source}")


# ---------------------------------------------------------------------------
# Schema auditing
# ---------------------------------------------------------------------------

class SchemaAudit:
    def __init__(self) -> None:
        self.info_keys = Counter()
        self.participant_keys = Counter()
        self.challenge_keys = Counter()
        self.team_keys = Counter()
        self.objective_keys = Counter()
        self.perk_keys = Counter()
        self.matches_seen = 0
        self.participants_seen = 0
        self.teams_seen = 0

    def observe(self, match: Dict[str, Any]) -> None:
        self.matches_seen += 1
        info = match.get("info") or {}

        for key in info:
            self.info_keys[key] += 1

        participants = info.get("participants") or []
        for p in participants:
            if not isinstance(p, dict):
                continue
            self.participants_seen += 1
            for key in p:
                self.participant_keys[key] += 1

            challenges = p.get("challenges") or {}
            if isinstance(challenges, dict):
                for key in challenges:
                    self.challenge_keys[key] += 1

            perks = p.get("perks") or {}
            if isinstance(perks, dict):
                for key in perks:
                    self.perk_keys[key] += 1

        teams = info.get("teams") or []
        for team in teams:
            if not isinstance(team, dict):
                continue
            self.teams_seen += 1
            for key in team:
                self.team_keys[key] += 1

            objectives = team.get("objectives") or {}
            if isinstance(objectives, dict):
                for key in objectives:
                    self.objective_keys[key] += 1

    def to_dict(self) -> Dict[str, Any]:
        configured = set(PARTICIPANT_FIELDS)
        intentionally_handled = NESTED_PARTICIPANT_KEYS | PII_PARTICIPANT_KEYS | IGNORED_PARTICIPANT_KEYS
        observed = set(self.participant_keys)
        unknown = sorted(observed - configured - intentionally_handled)

        return {
            "matches_seen": self.matches_seen,
            "participants_seen": self.participants_seen,
            "teams_seen": self.teams_seen,
            "info_keys": dict(sorted(self.info_keys.items())),
            "participant_keys": dict(sorted(self.participant_keys.items())),
            "challenge_keys": dict(sorted(self.challenge_keys.items())),
            "team_keys": dict(sorted(self.team_keys.items())),
            "objective_keys": dict(sorted(self.objective_keys.items())),
            "perk_top_keys": dict(sorted(self.perk_keys.items())),
            "participant_fields_configured": PARTICIPANT_FIELDS,
            "challenge_fields_configured": CHALLENGE_FIELDS,
            "observed_unconfigured_participant_keys": unknown,
            "note": (
                "Unconfigured keys are not automatically added to the stable Parquet schema. "
                "Review them and add analytically useful fields explicitly."
            ),
        }


# ---------------------------------------------------------------------------
# Exact disk-backed deduplication
# ---------------------------------------------------------------------------

class MatchDeduper:
    def __init__(self, db_path: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.conn: Optional[sqlite3.Connection] = None
        self.pending = 0

        if enabled:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(db_path)
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS seen_matches (match_id TEXT PRIMARY KEY)"
            )
            self.conn.commit()

    def is_new(self, match_id: str) -> bool:
        if not self.enabled:
            return True
        assert self.conn is not None
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO seen_matches(match_id) VALUES (?)",
            (match_id,),
        )
        self.pending += 1
        if self.pending >= 10_000:
            self.conn.commit()
            self.pending = 0
        return cur.rowcount == 1

    def close(self) -> None:
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

def extract_runes(perks: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "rune_offense": None,
        "rune_flex": None,
        "rune_defense": None,
        "primary_style_id": None,
        "secondary_style_id": None,
        "primary_perk_1": None,
        "primary_perk_2": None,
        "primary_perk_3": None,
        "primary_perk_4": None,
        "secondary_perk_1": None,
        "secondary_perk_2": None,
    }

    if not isinstance(perks, dict):
        return row

    stat = perks.get("statPerks") or {}
    if isinstance(stat, dict):
        row["rune_offense"] = stat.get("offense")
        row["rune_flex"] = stat.get("flex")
        row["rune_defense"] = stat.get("defense")

    styles = perks.get("styles") or []
    primary_index = 0
    secondary_index = 0

    for style in styles:
        if not isinstance(style, dict):
            continue
        desc = style.get("description")
        selections = style.get("selections") or []

        if desc == "primaryStyle":
            row["primary_style_id"] = style.get("style")
            for selection in selections:
                if not isinstance(selection, dict) or primary_index >= 4:
                    continue
                primary_index += 1
                row[f"primary_perk_{primary_index}"] = selection.get("perk")

        elif desc == "subStyle":
            row["secondary_style_id"] = style.get("style")
            for selection in selections:
                if not isinstance(selection, dict) or secondary_index >= 2:
                    continue
                secondary_index += 1
                row[f"secondary_perk_{secondary_index}"] = selection.get("perk")

    return row


def flatten_match(
    match: Dict[str, Any],
    source_container: str,
    source_member: str,
    keep_puuid: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:

    metadata = match.get("metadata") or {}
    info = match.get("info") or {}

    match_id = metadata.get("matchId")
    if match_id is None:
        # Fallback only for malformed payloads; still auditable.
        match_id = info.get("gameId")
    match_id = str(match_id)

    start_ms = timestamp_to_ms(info.get("gameStartTimestamp"))
    creation_ms = timestamp_to_ms(info.get("gameCreation"))
    if start_ms is None:
        start_ms = creation_ms

    end_ms = timestamp_to_ms(info.get("gameEndTimestamp"))
    duration_s = to_int(info.get("gameDuration"))
    if end_ms is None and start_ms is not None and duration_s is not None:
        end_ms = start_ms + duration_s * 1000

    participants = info.get("participants") or []
    teams = info.get("teams") or []

    platform_id = info.get("platformId")
    queue_id = to_int(info.get("queueId"))
    game_version = info.get("gameVersion")

    match_row: Dict[str, Any] = {
        "match_id": match_id,
        "data_version": metadata.get("dataVersion"),
        "game_id": info.get("gameId"),
        "game_creation_ms": creation_ms,
        "game_start_ms": start_ms,
        "game_end_ms": end_ms,
        "game_duration_s": duration_s,
        "game_duration_min": (duration_s / 60.0) if duration_s is not None else None,
        "end_of_game_result": info.get("endOfGameResult"),
        "game_mode": info.get("gameMode"),
        "game_type": info.get("gameType"),
        "game_version": game_version,
        "patch": patch_major_minor(game_version),
        "map_id": info.get("mapId"),
        "platform_id": platform_id,
        "queue_id": queue_id,
        "participant_count": len(participants),
        "team_count": len(teams),
        "is_ranked_solo_queue_420": queue_id == 420,
        "source_container": source_container,
        "source_member": source_member,
    }

    participant_rows: List[Dict[str, Any]] = []
    for p in participants:
        if not isinstance(p, dict):
            continue

        puuid = p.get("puuid")
        if not puuid:
            # Keep row but mark player_id missing. Longitudinal cleaning will decide.
            player_id = None
        else:
            player_id = stable_player_id(str(puuid))

        row: Dict[str, Any] = {
            "match_id": match_id,
            "platform_id": platform_id,
            "queue_id": queue_id,
            "patch": patch_major_minor(game_version),
            "game_start_ms": start_ms,
            "game_end_ms": end_ms,
            "game_duration_s": duration_s,
            "player_id": player_id,
        }

        if keep_puuid:
            row["puuid"] = puuid

        for field in PARTICIPANT_FIELDS:
            row[snake(field)] = p.get(field)

        # Selected challenge fields are prefixed to make provenance explicit.
        challenges = p.get("challenges") or {}
        for field in CHALLENGE_FIELDS:
            value = challenges.get(field) if isinstance(challenges, dict) else None
            row[f"challenge_{snake(field)}"] = to_float(value)

        # Runes/perks in a compact, stable representation.
        row.update(extract_runes(p.get("perks")))

        # Derived metrics: convenient for EDA; these are outcomes from THIS match.
        # For next-match prediction, only lagged/previous-match versions may be used.
        minions = to_float(p.get("totalMinionsKilled")) or 0.0
        neutral = to_float(p.get("neutralMinionsKilled")) or 0.0
        total_cs = minions + neutral

        kills = to_float(p.get("kills")) or 0.0
        deaths = to_float(p.get("deaths")) or 0.0
        assists = to_float(p.get("assists")) or 0.0

        row["derived_total_cs"] = total_cs
        row["derived_cs_per_min"] = safe_per_minute(total_cs, duration_s)
        row["derived_gold_per_min"] = safe_per_minute(p.get("goldEarned"), duration_s)
        row["derived_damage_to_champions_per_min"] = safe_per_minute(
            p.get("totalDamageDealtToChampions"), duration_s
        )
        row["derived_vision_score_per_min"] = safe_per_minute(
            p.get("visionScore"), duration_s
        )
        row["derived_kda"] = (kills + assists) / max(1.0, deaths)
        row["derived_kills_per_10_min"] = (
            safe_per_minute(kills, duration_s) * 10.0
            if safe_per_minute(kills, duration_s) is not None else None
        )
        row["derived_deaths_per_10_min"] = (
            safe_per_minute(deaths, duration_s) * 10.0
            if safe_per_minute(deaths, duration_s) is not None else None
        )

        participant_rows.append(row)

    team_rows: List[Dict[str, Any]] = []
    ban_rows: List[Dict[str, Any]] = []

    for team in teams:
        if not isinstance(team, dict):
            continue

        team_id = team.get("teamId")
        team_row: Dict[str, Any] = {
            "match_id": match_id,
            "platform_id": platform_id,
            "team_id": team_id,
            "win": team.get("win"),
        }

        objectives = team.get("objectives") or {}
        for objective_name in OBJECTIVE_NAMES:
            obj = objectives.get(objective_name) if isinstance(objectives, dict) else None
            obj = obj if isinstance(obj, dict) else {}
            prefix = snake(objective_name)
            team_row[f"{prefix}_first"] = obj.get("first")
            team_row[f"{prefix}_kills"] = obj.get("kills")

        feats = team.get("feats") or {}
        if isinstance(feats, dict):
            for feat_name in ("EPIC_MONSTER_KILL", "FIRST_BLOOD", "FIRST_TURRET"):
                feat = feats.get(feat_name) or {}
                key = f"feat_{feat_name.lower()}_state"
                team_row[key] = feat.get("featState") if isinstance(feat, dict) else None

        bans = team.get("bans") or []
        for ban in bans:
            if not isinstance(ban, dict):
                continue
            ban_rows.append({
                "match_id": match_id,
                "platform_id": platform_id,
                "team_id": team_id,
                "pick_turn": ban.get("pickTurn"),
                "champion_id": ban.get("championId"),
            })

        team_rows.append(team_row)

    return match_row, participant_rows, team_rows, ban_rows


# ---------------------------------------------------------------------------
# Chunked writer
# ---------------------------------------------------------------------------

class ChunkWriter:
    def __init__(self, root: Path, fmt: str, compression: str = "zstd") -> None:
        self.root = root
        self.fmt = fmt
        self.compression = compression
        self.parts = defaultdict(int)

        if fmt == "parquet":
            try:
                import pyarrow  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet output requires pyarrow. Install it in the project venv: "
                    "pip install pyarrow"
                ) from exc

    def write(self, table: str, rows: List[Dict[str, Any]]) -> Optional[Path]:
        if not rows:
            return None

        table_dir = self.root / table
        table_dir.mkdir(parents=True, exist_ok=True)

        part = self.parts[table]
        self.parts[table] += 1

        df = pd.DataFrame(rows)

        if self.fmt == "parquet":
            path = table_dir / f"part-{part:05d}.parquet"
            df.to_parquet(path, index=False, compression=self.compression)
        else:
            path = table_dir / f"part-{part:05d}.csv.gz"
            df.to_csv(path, index=False, compression="gzip")

        return path


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def process(
    inputs: List[Path],
    output: Path,
    fmt: str,
    max_matches: Optional[int],
    batch_matches: int,
    keep_puuid: bool,
    deduplicate: bool,
) -> Dict[str, Any]:

    output.mkdir(parents=True, exist_ok=True)
    (output / "audit").mkdir(parents=True, exist_ok=True)

    errors_path = output / "audit" / "errors.jsonl"
    writer = ChunkWriter(output, fmt)
    schema = SchemaAudit()
    deduper = MatchDeduper(output / "audit" / "seen_matches.sqlite", deduplicate)

    summary: Dict[str, Any] = {
        "inputs": [str(p.resolve()) for p in inputs],
        "format": fmt,
        "max_matches": max_matches,
        "batch_matches": batch_matches,
        "keep_puuid": keep_puuid,
        "deduplicate": deduplicate,
        "source_json_objects_seen": 0,
        "matches_written": 0,
        "duplicates_skipped": 0,
        "parse_or_schema_errors": 0,
        "participants_written": 0,
        "teams_written": 0,
        "bans_written": 0,
        "queue_id_counts": Counter(),
        "platform_id_counts": Counter(),
        "match_prefix_counts": Counter(),
        "patch_counts": Counter(),
        "participant_count_counts": Counter(),
        "team_count_counts": Counter(),
        "game_mode_counts": Counter(),
        "end_of_game_result_counts": Counter(),
        "platform_counts_by_input": defaultdict(Counter),
    }

    match_buffer: List[Dict[str, Any]] = []
    participant_buffer: List[Dict[str, Any]] = []
    team_buffer: List[Dict[str, Any]] = []
    ban_buffer: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal match_buffer, participant_buffer, team_buffer, ban_buffer
        writer.write("matches", match_buffer)
        writer.write("participants", participant_buffer)
        writer.write("teams", team_buffer)
        writer.write("team_bans", ban_buffer)
        match_buffer = []
        participant_buffer = []
        team_buffer = []
        ban_buffer = []

    stop = False

    try:
        with errors_path.open("w", encoding="utf-8") as error_file:
            for source in inputs:
                if stop:
                    break

                try:
                    iterator = iter_source_objects(source)
                    for source_container, source_member, match in iterator:
                        summary["source_json_objects_seen"] += 1

                        try:
                            schema.observe(match)

                            metadata = match.get("metadata") or {}
                            info = match.get("info") or {}

                            raw_match_id = metadata.get("matchId")
                            if raw_match_id is None:
                                raw_match_id = info.get("gameId")
                            if raw_match_id is None:
                                raise ValueError("Match has neither metadata.matchId nor info.gameId")

                            raw_match_id = str(raw_match_id)

                            if not deduper.is_new(raw_match_id):
                                summary["duplicates_skipped"] += 1
                                continue

                            match_row, participant_rows, team_rows, ban_rows = flatten_match(
                                match=match,
                                source_container=source_container,
                                source_member=source_member,
                                keep_puuid=keep_puuid,
                            )

                            match_buffer.append(match_row)
                            participant_buffer.extend(participant_rows)
                            team_buffer.extend(team_rows)
                            ban_buffer.extend(ban_rows)

                            summary["matches_written"] += 1
                            summary["participants_written"] += len(participant_rows)
                            summary["teams_written"] += len(team_rows)
                            summary["bans_written"] += len(ban_rows)

                            q = match_row.get("queue_id")
                            platform = match_row.get("platform_id")
                            patch = match_row.get("patch")
                            participant_count = match_row.get("participant_count")
                            team_count = match_row.get("team_count")

                            summary["queue_id_counts"][str(q)] += 1
                            summary["platform_id_counts"][str(platform)] += 1
                            summary["match_prefix_counts"][match_prefix(match_row.get("match_id"))] += 1
                            summary["patch_counts"][str(patch)] += 1
                            summary["participant_count_counts"][str(participant_count)] += 1
                            summary["team_count_counts"][str(team_count)] += 1
                            summary["game_mode_counts"][str(match_row.get("game_mode"))] += 1
                            summary["end_of_game_result_counts"][str(match_row.get("end_of_game_result"))] += 1
                            summary["platform_counts_by_input"][source_container][str(platform)] += 1

                            if summary["matches_written"] % batch_matches == 0:
                                flush()
                                print(
                                    f"[progress] matches={summary['matches_written']:,} "
                                    f"participants={summary['participants_written']:,} "
                                    f"duplicates={summary['duplicates_skipped']:,}",
                                    flush=True,
                                )

                            if max_matches is not None and summary["matches_written"] >= max_matches:
                                stop = True
                                break

                        except Exception as exc:
                            summary["parse_or_schema_errors"] += 1
                            error_file.write(json.dumps({
                                "source_container": source_container,
                                "source_member": source_member,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }, ensure_ascii=False) + "\n")

                except Exception as exc:
                    summary["parse_or_schema_errors"] += 1
                    error_file.write(json.dumps({
                        "source_container": str(source),
                        "source_member": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }, ensure_ascii=False) + "\n")

        flush()

    finally:
        deduper.close()

    # JSON cannot serialize Counter/defaultdict directly in a predictable way.
    serializable_summary = dict(summary)
    for key in [
        "queue_id_counts", "platform_id_counts", "match_prefix_counts",
        "patch_counts", "participant_count_counts", "team_count_counts",
        "game_mode_counts", "end_of_game_result_counts",
    ]:
        serializable_summary[key] = dict(summary[key])

    serializable_summary["platform_counts_by_input"] = {
        k: dict(v) for k, v in summary["platform_counts_by_input"].items()
    }

    with (output / "audit" / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2, ensure_ascii=False)

    with (output / "audit" / "schema_observed.json").open("w", encoding="utf-8") as f:
        json.dump(schema.to_dict(), f, indent=2, ensure_ascii=False)

    return serializable_summary


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        has_content = any(output.iterdir())
        if has_content and not overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {output}\n"
                "Use --overwrite for a fresh reproducible run."
            )
        if has_content and overwrite:
            shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream/normalize Riot Match-V5 JSON into relational analysis tables."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="One or more directories, JSON files, or ZIP archives.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root, e.g. /project/data/processed/match_v5",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv.gz"],
        default="parquet",
        help="Use Parquet for the full dataset; csv.gz is mainly for smoke tests.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Stop after N unique matches. Useful for smoke tests.",
    )
    parser.add_argument(
        "--batch-matches",
        type=int,
        default=5_000,
        help="Number of matches per output part. Default: 5000 (~50k participant rows).",
    )
    parser.add_argument(
        "--keep-puuid",
        action="store_true",
        help="Also write raw PUUID. Off by default; hashed player_id is always written.",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable exact match_id deduplication. Not recommended until duplicates are audited.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing non-empty output directory before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_output(args.output, args.overwrite)

    summary = process(
        inputs=args.input,
        output=args.output,
        fmt=args.format,
        max_matches=args.max_matches,
        batch_matches=args.batch_matches,
        keep_puuid=args.keep_puuid,
        deduplicate=not args.no_deduplicate,
    )

    print("\nExtraction complete.")
    print(json.dumps({
        "matches_written": summary["matches_written"],
        "participants_written": summary["participants_written"],
        "teams_written": summary["teams_written"],
        "duplicates_skipped": summary["duplicates_skipped"],
        "parse_or_schema_errors": summary["parse_or_schema_errors"],
        "queue_id_counts": summary["queue_id_counts"],
        "platform_id_counts": summary["platform_id_counts"],
        "patch_counts": summary["patch_counts"],
        "participant_count_counts": summary["participant_count_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()

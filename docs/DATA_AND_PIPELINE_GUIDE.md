# Data & Pipeline Guide

This document is the detailed reference for the processed League of Legends dataset and the current project pipeline.

The short overview lives in `README.md`. This guide explains what each major dataset means, which one should be used for different analysis types, and how the processed tables relate to one another.

---

# 1. Data layers

The project has four practical data layers.

## Layer A — Raw source data

```text
data/raw/
```

Contains the original downloaded datasets, Match-V5 JSON archives, crawler-related files, seed lists, and `league_data.db`.

Use raw data only when:

- reproducing the extraction
- checking provenance
- inspecting a field that was not preserved in processed data
- debugging extraction behavior

For normal statistical analysis, do **not** work directly from the raw JSON.

---

## Layer B — Canonical processed match data

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/
```

These are the main cleaned relational datasets created from raw Match-V5 JSON.

Each regional folder contains:

```text
matches/
participants/
teams/
team_bans/
audit/
```

These tables contain **all players from every archived match**, not only tracked crawler seed players.

This layer is the correct starting point for general League-of-Legends analysis.

---

## Layer C — Tracking / longitudinal identity data

```text
data/processed/tracking/
```

This layer identifies which pseudonymous players are suitable for longitudinal analysis.

Important subfolders:

```text
authoritative/
alias_confirmed/
linked/
audit/
coverage_audit/
```

The main longitudinal cohort is `authoritative`.

---

## Layer D — Analysis audits

```text
data/processed/analysis_audit/
```

Contains summaries and diagnostics produced by `04_audit_dataset.py`.

These files describe the data; they are generally **not** the main raw input for modeling.

---

# 2. Canonical regional tables

All three regional folders (`full_na`, `full_kr`, `full_eu`) share the same basic structure.

## 2.1 `matches/`

**Unit:** one row per match.

Use it when the question is about the game itself rather than a specific participant.

Typical fields include:

- `match_id`
- `platform_id`
- `queue_id`
- `patch`
- `game_start_ms`
- `game_end_ms`
- `game_duration_s`
- `end_of_game_result`

Typical use cases:

- average / median game duration
- match counts over time
- queue distribution
- patch distribution
- match completion / short-game behavior
- regional match-level comparisons

Example:

```python
import duckdb

avg_duration = duckdb.sql("""
    SELECT AVG(game_duration_s) / 60.0 AS avg_minutes
    FROM read_parquet(
        'data/processed/full_eu/matches/*.parquet'
    )
""").df()
```

Do not calculate average match duration from `participants`, because each match appears ten times there.

---

## 2.2 `participants/`

**Unit:** one participant in one match.

Normally there are 10 participant rows per match.

This is the richest table and contains player-level game statistics.

Important groups of columns include:

### Identity / match linkage

- `match_id`
- `player_id`
- `platform_id`
- `queue_id`
- timestamps

`player_id` is a pseudonymous stable hash derived from the Riot PUUID.

### Champion / role

- champion ID / champion name
- `team_position`
- participant / team identifiers

### Outcome

- `win`

### Basic performance

- kills
- deaths
- assists
- gold
- minions / jungle CS
- damage
- vision
- objective-related statistics

### Derived metrics

The extraction also includes useful derived metrics such as:

- KDA
- total CS
- CS per minute
- gold per minute
- damage to champions per minute
- vision score per minute

### Selected Riot challenge metrics

Some Match-V5 challenge statistics are also retained.

Not every challenge field is equally complete. Use `analysis_audit/selected_missingness.csv` before relying heavily on sparse fields.

Typical use cases:

- average KDA
- champion performance
- role comparisons
- damage / gold / CS distributions
- win-rate analyses
- player-level performance models
- general participant-level machine learning

Important distinction:

`participants/` contains **tracked players plus incidental teammates/opponents**.

Therefore:

- good for general participant statistics
- not automatically appropriate for longitudinal player histories

For longitudinal work, use `tracking/linked/authoritative`.

---

## 2.3 `teams/`

**Unit:** one team in one match.

Normally there are exactly 2 rows per match.

Contains team-level results and objectives.

Typical use cases:

- team win/loss analysis
- dragon / baron / tower relationships
- objective control
- team-level predictive questions

Join with `matches` by:

```text
match_id
```

Join with participants by:

```text
match_id + team_id
```

when the relevant fields are available.

---

## 2.4 `team_bans/`

**Unit:** one team-ban record.

Contains champion ban information extracted from team data.

Typical use cases:

- most commonly banned champions
- ban differences by region / patch
- ban diversity
- team draft-related exploratory analysis

This table is optional for the current main research question but useful for teammates exploring draft/meta questions.

---

## 2.5 Regional `audit/`

Each `full_*` folder contains extraction-time metadata such as:

- `run_summary.json`
- `schema_observed.json`
- `errors.jsonl`

These are provenance / QA files.

They are useful for:

- checking what the extractor processed
- schema inspection
- debugging
- documenting reproducibility

They are not normal analysis tables.

---

# 3. Tracking data

## 3.1 Why tracking exists

The raw match corpus was built around a set of crawler seed players.

Every match contains 10 participants, but most participant identities are simply opponents or teammates who appeared incidentally.

For questions involving:

- repeated games by the same player
- sessions
- requeue timing
- streaks
- next-match performance
- longitudinal prediction

we need a defensible set of players who were actually intentionally tracked by the crawler.

---

## 3.2 `tracking/authoritative/`

**Main longitudinal cohort.**

Regional files:

```text
data/processed/tracking/authoritative/NA/tracked_players.parquet
data/processed/tracking/authoritative/KR/tracked_players.parquet
data/processed/tracking/authoritative/EU/tracked_players.parquet
```

The authoritative definition combines strong provenance evidence from:

1. fresh seed-list aliases uniquely resolved to raw PUUIDs
2. crawler seed PUUIDs recovered from `league_data.db` and assigned unambiguously to one supplied regional corpus

This cohort currently covers:

- NA: 100% of archived matches
- KR: 100%
- EU: 99.95%

Use this cohort for the main tracked-player longitudinal analyses.

---

## 3.3 `tracking/alias_confirmed/`

Stricter subset.

A player is included only when the fresh seed-list Riot ID uniquely matched a raw Match-V5 participant PUUID.

This is useful for:

- robustness checks
- sensitivity analysis
- confirming that conclusions do not depend heavily on DB-recovered seeds

It should not normally replace the authoritative cohort for the main analysis.

---

# 4. Permanent player-match linkage

## `tracking/linked/authoritative/`

This is one of the most useful datasets in the project.

Regional files:

```text
data/processed/tracking/linked/authoritative/NA/tracked_player_matches.parquet
data/processed/tracking/linked/authoritative/KR/tracked_player_matches.parquet
data/processed/tracking/linked/authoritative/EU/tracked_player_matches.parquet
```

**Unit:** one authoritative tracked player in one observed match.

The unique key is:

```text
(source, player_id, match_id)
```

The files were created by joining the authoritative tracking lookup to the canonical participant rows.

Current linked counts:

- NA: 127,679 player-match rows / 1,581 tracked players
- KR: 120,583 / 1,291
- EU: 984,321 / 8,297

No duplicate `(player_id, match_id)` rows were found.

Use this dataset when you need:

- per-player chronological sorting
- player history
- sessions
- repeated-measures analysis
- next-match questions
- player-level predictive algorithms

Example chronological query:

```python
import duckdb

history = duckdb.sql("""
    SELECT *
    FROM read_parquet(
        'data/processed/tracking/linked/authoritative/EU/tracked_player_matches.parquet'
    )
    WHERE queue_id = 420
    ORDER BY player_id, game_start_ms, match_id
""").df()
```

Do not attempt to identify tracked players again from Riot names. That provenance work has already been completed.

---

## `tracking/linked/alias_confirmed/`

Same concept, but using only the strict alias-confirmed subset.

Use for sensitivity analysis.

---

# 5. Tracking audits

## `tracking/audit/`

Produced by `02_build_tracked_players.py`.

Contains files such as:

- tracking cohort summary
- per-region seed alias resolution details
- DB seed IDs present in multiple source corpora

Use these for methodology / provenance reporting rather than day-to-day analysis.

---

## `tracking/coverage_audit/`

Produced by `03_audit_tracking_coverage.py`.

Important files include:

- `match_tracking_coverage_summary.csv` — match coverage for each cohort
- `tracked_players_per_match_distribution.csv` — 0 / 1 / 2 / 3+ tracked players per match
- `coverage_by_queue.csv` — tracking coverage by queue
- `coverage_by_platform.csv` — tracking coverage by platform
- `uncovered_matches_summary.csv` — uncovered match counts
- `uncovered_match_ids_sample.csv` — small forensic sample
- `tracking_lookup_validation.csv` — tracking lookup integrity
- `linked_player_match_summary.csv` — permanent linkage summary

These are audit / provenance files, not the usual input for analysis.

---

# 6. Final analytical audit

## `data/processed/analysis_audit/`

Produced by `04_audit_dataset.py`.

This folder answers:

> Is the authoritative tracked dataset structurally and chronologically ready for analysis?

The audit passed.

Important files:

### `dataset_overview.csv`

High-level counts by source:

- canonical matches
- tracked players
- authoritative player-match rows
- covered matches
- observation window

Use for reporting dataset size.

### `data_quality_checks.csv`

Structural integrity checks such as:

- duplicate match IDs
- invalid team counts
- duplicate tracked player-match rows
- missing IDs
- missing timestamps
- broken match links

A nonzero serious check should be investigated before modeling.

### `selected_missingness.csv`

Missingness for fields likely to matter in analysis.

Use this before choosing a feature for a model or statistical test.

Especially important for Riot `challenges` fields, where missing may not mean the same thing as `False` or `0`.

### `tracked_player_coverage.csv`

Counts of authoritative tracked players with at least:

```text
1, 2, 3, 5, 10, 20, 30, 50, 75, 100
```

observed matches.

Reported separately for:

- ranked 420 + 440 history
- Solo/Duo 420-only history

Use to decide minimum-history requirements.

### `tracked_player_match_count_quantiles.csv`

Distribution of observed match counts per tracked player.

Useful for understanding longitudinal depth and heavy tails.

### `tracked_player_history_span.csv`

Observation-span statistics per player.

Useful for understanding temporal coverage and left-censoring.

### `tracked_queue_distribution.csv`

Queue distribution among authoritative tracked player-match rows.

The project's main behavioral target is queue 420.

### `tracked_patch_distribution.csv`

Patch composition of the authoritative tracked observations.

Useful for meta-change controls / sensitivity.

### `tracked_role_distribution.csv`

Role distribution for tracked players.

Useful for role controls / stratification.

### `duration_summary.csv`

Match-duration distribution.

Use this before defining a remake / short-game policy.

Do not arbitrarily remove all matches below 15 minutes.

### `end_result_distribution.csv`

Distribution of Riot `end_of_game_result`.

Useful together with duration when deciding which matches count as normal completed games.

### `tracked_inter_match_gap_summary.csv`

Distribution of:

```text
previous match end → next observed ranked match start
```

This is the correct gap for requeue / session analysis.

### `tracked_gap_threshold_sensitivity.csv`

Counts / percentages of consecutive pairs below candidate thresholds such as:

```text
10, 15, 30, 45, 60, 90, 120 ... minutes
```

Use this to justify session boundaries.

### `tracked_queue_transitions.csv`

Observed queue transitions between consecutive ranked games.

Useful when deciding whether 440 matches should interrupt / contribute to Solo/Duo histories.

### `next_match_analysis_feasibility.csv`

How many usable chronological match pairs exist.

The final audit found more than one million Solo/Duo next-match pairs across the regions and zero negative chronological gaps.

### `shared_match_dependence.csv`

How often several tracked players appear in one physical match.

This matters statistically because player-match observations from the same match are not perfectly independent.

### `audit_summary.json`

Machine-readable summary of audit status and methodological notes.

---

# 7. Which dataset should I use?

## Average game length

Use:

```text
full_*/matches/
```

One row = one match.

## Most successful champions

Usually use:

```text
full_*/participants/
```

Define "successful" carefully: win rate, KDA, damage, gold, role-adjusted performance, etc.

## Compare NA vs KR vs EU

Use the same canonical table from each regional `full_*` folder and add a source label.

## Champion bans

Use:

```text
full_*/team_bans/
```

## Team objectives vs winning

Use:

```text
full_*/teams/
```

possibly joined to `matches`.

## Repeated games for tracked players

Use:

```text
tracking/linked/authoritative/
```

## Requeue timing

Use:

```text
tracking/linked/authoritative/
```

Sort by:

```text
source, player_id, game_start_ms, match_id
```

Then compute:

```text
current start - previous end
```

Do not use start-to-start time.

## Session analysis

Use:

```text
tracking/linked/authoritative/
```

or the derived timeline files after `05_build_player_timelines.py`.

## Next-match prediction

Best after `05`:

```text
data/analysis/timelines/solo420_targets/
```

Before `05`, it is possible from linked authoritative data, but every feature must be lagged manually and leakage prevention is easier to get wrong.

## Generic machine learning unrelated to chronology

Use the canonical table matching the unit of the target.

Examples:

- match-level target → `matches` / `teams`
- participant-level performance → `participants`
- objective outcome → `teams`

Always define when the prediction would be made and prevent target leakage.

---

# 8. Recommended joins

## Matches ↔ participants

```text
match_id
```

Relationship:

```text
1 match : 10 participant rows
```

## Matches ↔ teams

```text
match_id
```

Relationship:

```text
1 match : 2 team rows
```

## Participants ↔ tracked lookup

```text
player_id
```

This has already been materialized under:

```text
tracking/linked/
```

so normal longitudinal analysis should use the linked tables directly.

---

# 9. Analysis cautions

## Incidental players

Do not treat every participant as longitudinally sampled.

Only the tracked cohort has known crawler-based longitudinal provenance.

## Shared matches

Several tracked players can appear in one physical match.

This is correct for timeline construction, but statistical inference may need clustering / dependence handling.

## Temporal leakage

For predicting match `t`, never use information generated during or after match `t` as a predictor.

Examples of leakage:

- target-match kills
- target-match damage
- target-match final gold
- target outcome
- target duration when prediction is supposed to occur before play

`05_build_player_timelines.py` is designed to separate history features from target outcomes.

## Observation window

The data represent observed crawler histories, not complete lifetime histories.

Use wording such as:

- observed ranked sequence
- observed ranked session
- recent observed ranked volume

## Queue scope

- 420 = Ranked Solo/Duo
- 440 = Ranked Flex

For the main behavioral project, queue 420 is the primary target.

## Causality

The dataset is observational.

Prefer:

- associated with
- correlated with
- predictive of
- consistent with

Avoid causal claims unless a causal design is separately justified.

---

# 10. Current project status

Completed:

```text
raw Match-V5 data
        ↓
streaming extraction
        ↓
canonical regional Parquet
        ↓
processed-data verification
        ↓
tracked-player reconstruction
        ↓
tracking coverage validation
        ↓
permanent tracked-player ↔ match linkage
        ↓
final analytical-readiness audit
```

Current state:

```text
READY FOR GENERAL ANALYSIS
READY FOR TRACKED-PLAYER CHRONOLOGICAL ANALYSIS
```

Next step for the main behavioral research question:

```text
05_build_player_timelines.py
```

This is a derived feature-engineering layer, not another raw-data cleaning stage and not a prerequisite for teammates who want to analyze different questions from the canonical / linked processed data.

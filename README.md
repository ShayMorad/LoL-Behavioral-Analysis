# League of Legends Behavioral & Temporal Analysis

Data Science final project for **"A Needle in a Data Haystack - Introduction to Data Science"**.

This repository contains a large-scale League of Legends Match-V5 data pipeline and analysis-ready processed datasets for NA, KR, and EU.

## Current project status

The raw-data processing stage is complete.

The project currently contains:

- **497,102 unique matches**
- **4,971,020 participant rows**
- Canonical Parquet tables for NA, KR, and EU
- A validated set of **authoritative tracked players**
- Permanent tracked-player ↔ match linkage
- Structural, chronological, and tracking audits
- Analysis-ready data for:
  - general match / participant / team statistics
  - tracked-player longitudinal analysis

The final audit passed and confirmed that the data are ready for chronological timeline construction.

`05_build_player_timelines.py` is the next project-specific feature-engineering step for our main research question. It is **not required** for general statistics or for teammates who want to investigate different questions directly from the processed Parquet data.

---

## Main research direction

Our main question is:

> **How are recent ranked-game volume, session depth, and post-loss requeue timing associated with performance in a player's subsequent ranked match?**

The project is designed to support both:

1. **General League analysis**
   - average match duration
   - champion / role / queue distributions
   - player performance distributions
   - team statistics
   - patch comparisons
   - other non-longitudinal questions

2. **Tracked-player chronological analysis**
   - requeue time
   - session depth
   - previous-result effects
   - streaks
   - recent ranked volume
   - next-match performance
   - predictive modeling

---

## Project structure

```text
project/
├── code/
│   ├── 00_verify_processed_data.py
│   ├── 01_extract_match_v5.py
│   ├── 02_build_tracked_players.py
│   ├── 03_audit_tracking_coverage.py
│   ├── 04_audit_dataset.py
│   └── 05_build_player_timelines.py        # next step / project-specific
│
├── data/
│   ├── raw/                                # original datasets; not used directly for normal analysis
│   └── processed/
│       ├── full_na/
│       ├── full_kr/
│       ├── full_eu/
│       ├── tracking/
│       └── analysis_audit/
│
├── docs/
│   └── DATA_AND_PIPELINE_GUIDE.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

The large raw and processed data files may be excluded from Git through `.gitignore`.

---

## Which data should I use?

### For general statistics / plots

Use the canonical regional Parquet datasets:

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/
```

Main tables:

- `matches/` — one row per match
- `participants/` — one row per player per match
- `teams/` — one row per team per match
- `team_bans/` — team ban information

Examples:

- Average game duration → `matches`
- Average KDA / damage / gold / CS → `participants`
- Win-related team objectives → `teams`
- Champion bans → `team_bans`

---

### For tracked-player chronological analysis

Use:

```text
data/processed/tracking/linked/authoritative/
```

Each regional file contains one row for every **authoritative tracked player in every observed match**.

The unique longitudinal key is:

```text
(source, player_id, match_id)
```

This is the preferred starting point for:

- chronological player histories
- sessions
- requeue timing
- streak analysis
- next-match analysis
- longitudinal machine learning

Do **not** use all participant rows for longitudinal player conclusions, because most participants are incidental teammates/opponents rather than crawler-tracked players.

---

## Tracking cohorts

Two tracking cohorts are kept:

### `authoritative`

The main cohort used for analysis.

A player is included when there is strong crawler provenance through either:

- a fresh seed-list alias uniquely resolved to a raw PUUID, or
- a seed PUUID recovered from `league_data.db` and assigned unambiguously to one regional corpus.

Authoritative match coverage:

- NA: **100%**
- KR: **100%**
- EU: **99.95%**

### `alias_confirmed`

A stricter subset containing only players recovered through direct fresh seed-list alias matching.

Use this mainly for robustness / sensitivity checks.

---

## Pipeline

### 00 — Verify processed data

```text
code/00_verify_processed_data.py
```

Checks that the existing Parquet extraction is structurally valid and verifies that processed `player_id` values correspond to hashed raw PUUIDs.

This was used as a safety check after discovering encoding problems in an older KR seed-list copy.

### 01 — Extract Match-V5 JSON

```text
code/01_extract_match_v5.py
```

Streams the large raw Match-V5 JSON archive and converts it into efficient relational Parquet tables.

Outputs per region:

```text
matches/
participants/
teams/
team_bans/
audit/
```

This step is already complete for NA, KR, and EU.

### 02 — Build tracked players

```text
code/02_build_tracked_players.py
```

Reconstructs the crawler-tracked player population and produces:

```text
data/processed/tracking/
├── authoritative/
├── alias_confirmed/
└── audit/
```

No match-count or history-length heuristic is used to decide player identity.

### 03 — Audit tracking and create permanent linkage

```text
code/03_audit_tracking_coverage.py
```

Checks how many matches contain tracked players and permanently creates:

```text
data/processed/tracking/linked/
├── authoritative/
└── alias_confirmed/
```

This means future analysis does not need to repeat the identity-resolution process.

### 04 — Final analytical-readiness audit

```text
code/04_audit_dataset.py
```

Audits the authoritative linked data for:

- duplicates
- missing IDs / times
- chronological ordering
- queue composition
- match-duration behavior
- player-history depth
- inter-match gaps
- next-match analysis feasibility
- shared matches containing multiple tracked players

The audit passed with no negative chronological gaps and confirmed that the project is ready for timeline construction.

### 05 — Build player timelines

```text
code/05_build_player_timelines.py
```

This is the next step for **our specific research question**.

It will create strictly chronological, target-centric features such as:

- previous match result
- requeue gap
- loss/win streaks
- recent ranked volume
- recent minutes played
- session depth
- champion / role switches
- previous performance
- next-match outcomes

This step is feature engineering, not raw-data cleaning.

---

## Loading Parquet data

### pandas

```python
import pandas as pd

matches = pd.read_parquet(
    "data/processed/full_na/matches/part-00000.parquet"
)
```

For a full multi-part table, DuckDB or PyArrow is usually preferable.

### DuckDB

```python
import duckdb

df = duckdb.sql("""
    SELECT *
    FROM read_parquet(
        'data/processed/full_na/matches/*.parquet'
    )
""").df()
```

For the full participant corpus, avoid loading millions of rows into pandas unless necessary. Query only the columns / rows needed.

---

## Important methodological notes

- `player_id` is a stable pseudonymous hash derived from Riot PUUID.
- Riot display names are not needed for analytical work.
- Multiple tracked players can appear in the same match; each is a valid separate player-match observation.
- For longitudinal work, always sort within `(source, player_id)` by match time.
- Queue **420 (Ranked Solo/Duo)** is the intended primary target for our main research question.
- Queue 440 can be used in sensitivity analyses or broader ranked-history definitions.
- Session thresholds have not yet been fixed; they will be justified empirically.
- Short matches/remakes have been audited but should not be removed with an arbitrary duration cutoff without justification.
- The data are observational: associations should not be presented as causal effects.

---

## Detailed documentation

See:

```text
docs/DATA_AND_PIPELINE_GUIDE.md
```

for detailed table descriptions, use cases, recommended joins, tracking definitions, audit outputs, and common analysis workflows.

---

## Environment

Install dependencies:

```powershell
pip install -r .\requirements.txt
```

The project is designed to work well with:

- Python
- pandas
- PyArrow
- DuckDB
- orjson

---

## Current next step

For our main project question:

```text
05_build_player_timelines.py
        ↓
EDA
        ↓
session / remake policy decisions
        ↓
statistical analysis
        ↓
visualization
        ↓
predictive modeling
        ↓
evaluation and robustness checks
```

Other project members can already begin separate analyses directly from the processed canonical or authoritative linked datasets.

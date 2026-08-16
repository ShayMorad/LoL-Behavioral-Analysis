# League of Legends Behavioral & Temporal Analysis
**Course:** A Needle in a Data Haystack - Introduction to Data Science

## Project Overview
This project investigates the relationship between player behavior, temporal match dynamics, and next-match performance in *League of Legends*. Specifically, it explores how ranked-game volume, session depth, and post-loss requeue behaviors impact a player's performance trajectory, with potential extensions into churn prediction.

Currently, the project focuses on processing a massive, multi-region dataset (tens of gigabytes of raw JSON) into a highly optimized relational format for advanced analytical modeling.

## Data Sources
*   **Main Dataset:** Raw Riot Match-V5 JSON files encompassing NA, KR, and EUW/EUN1 regions. 
    *   **Scale:** Tens of gigabytes containing granular data on timestamps, roles, champions, teams, and combat/economy/vision metrics.
    *   **Privacy:** Raw PUUIDs are converted into deterministic, hashed `player_id`s to track longitudinal histories without exposing player identities.
*   **Supplementary Dataset:** A clean relational CSV dataset utilized for baseline statistics and sanity checks.
*   **Riot API:** Reserved for targeted gap-filling and retrieving recent information, bypassing bulk collection rate limits.

## Data Engineering & Pipeline
To handle the massive scale of the Match-V5 JSON dataset without memory bloating, the project utilizes a **streaming extraction pipeline**.

`Raw Riot JSON` ➔ `Streaming Extraction` ➔ `Clean Relational Tables` ➔ `Parquet Files`

**Why Parquet?**
The pipeline converts raw JSON arrays into highly compressed `.parquet` files. This preserves data types, allows for column-specific reading, and significantly accelerates processing speeds when using Pandas, PyArrow, or DuckDB for millions of rows.

### Core Extracted Tables
*   `matches`: Match-level metadata (patch, queue ID, durations).
*   `participants`: The primary analysis table (one row per player per match, including KDA, CS/min, vision/min, damage/min).
*   `teams` & `team_bans`: Team-wide performance and draft-ban information.
*   `tracked_players`: A critical table distinguishing between intentionally crawled "seed" players (who have trustworthy longitudinal histories) and incidental players.

## Repository Structure

```text
├── data/                    # 
│   ├── dataset1/            # Supplementary CSVs [Ignored in git (.gitignore)]
│   ├── Games of League of Legends/ # Raw Match-V5 JSONs [Ignored in git (.gitignore)]
│   └── processed/           # Final Parquet tables (NA, KR, EU)
│
├── code/                    # Source code for data pipeline & analysis
│   ├── 01_inspect_raw_matches.py
│   ├── 02_extract_match_v5_v2.py
│   ├── 03_audit_pilots.py
│   └── ...                  # Future EDA and ML scripts
│
├── .gitignore [Ignored in git (.gitignore)]
└── README.md
```

## Current Status & Roadmap
* [x] **Phase 1**: Pilot & Architecture: Validated schema on 1,000 matches per source; verified structural integrity of NA, KR, and EU data.

* [x] **Phase 2**: Streaming Pipeline: Built and executed the memory-safe JSON-to-Parquet extraction pipeline for all regions.

* [ ] **Phase 3**: Data Audit (In Progress):

  * Generate full-dataset statistics and tracked-player coverage.

  * Analyze duplicate/missingness and ranked inter-match gaps.

  * Define concrete cleaning rules based on the audit (avoiding "magic numbers").

* [ ] **Phase 4**: Feature Engineering: Build chronological player histories; create session, recent-volume, and streak features.

* [ ] **Phase 5**: EDA & Machine Learning: Statistical analysis and predictive modeling on next-match performance and churn.
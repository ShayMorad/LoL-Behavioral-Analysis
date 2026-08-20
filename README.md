# League of Legends Behavioral & Strategic Analysis

Data Science final project for **67978 — A Needle in a Data Haystack: Introduction to Data Science**.

## Research direction

**Question 1:** How are recent competitive volume, session depth, and post-loss requeue timing associated with performance in a player's subsequent Ranked Solo/Duo match?

The project uses a large Riot Match-V5-style corpus from NA, KR, and EU. After extraction/deduplication it contains **497,102 unique matches**, **4,971,020 participant rows**, and **994,204 team rows**. The main longitudinal cohort contains **11,169 authoritative player-region identities**.

Q1 is complete: the main within-player effects are small/non-monotonic across regions, no primary behavioral family remains significant after Holm correction, and behavioral features add only a small held-out AUC gain over history-only prediction.

## Repository layout

```text
project/
├── code/
│   ├── 01_prepare_data.py       # processed-data validation, tracking linkage, audits, timelines
│   ├── 02_q1_analysis.py        # EDA, statistics, decision tree, robustness, report figures
│   └── run_project.py           # cross-platform entry point
├── data/
│   ├── raw/                     # optional original sources; not required for normal Q1 rerun
│   ├── processed/               # canonical processed inputs + small tracked-player lookups
│   └── analysis/                # generated; safe to delete and recreate
├── docs/
│   ├── DATA_AND_PIPELINE_GUIDE.md
│   └── REPRODUCIBILITY.md
├── report/
│   └── Report.docx
├── requirements.txt
└── README.md
```

## Processed-data starting point

The normal reproduction pipeline starts from `data/processed/`.

Keep these as **inputs**:

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/
data/processed/tracking/authoritative/{NA,KR,EU}/tracked_players.parquet
data/processed/tracking/alias_confirmed/{NA,KR,EU}/tracked_players.parquet
```

The tracked-player lookup tables are treated as processed inputs because their identities were originally reconstructed from raw Riot PUUIDs, seed lists, and `league_data.db`; that identity evidence cannot be recreated from anonymized canonical Parquet alone.

The following are **generated outputs** and may be deleted before a clean rerun:

```text
data/analysis/
data/processed/analysis_audit/
data/processed/tracking/linked/
data/processed/tracking/coverage_audit/
```

## Install

Use Python **3.10+**.

```bash
python -m pip install -r requirements.txt
```

## Reproduce Q1

From the project root, on Linux/Windows/macOS:

```bash
python code/run_project.py check
python code/run_project.py q1 --strict-reference
```

The strict reference mode compares key row counts, selected tree parameters, AUCs, and the short-game robustness result with the previously validated Q1 run.

To remove only regenerated outputs first:

```bash
python code/run_project.py clean --yes
python code/run_project.py q1 --strict-reference
```

## Q1 outputs

The consolidated analysis writes to:

```text
data/analysis/q1/
├── tables/
├── figures/
│   ├── report/
│   └── supplementary/
├── predictions/
└── audit/
```

The four main report figures are:

1. inter-match-gap ECDF (session-threshold justification),
2. combined adjusted H1/H2/H3 coefficient + 95% CI figure,
3. held-out ROC + normalized confusion-matrix figure,
4. entropy-tree + feature-importance interpretation figure.

## Methodological cautions

- Target-match performance variables are never used as pre-match predictors.
- Queue 420 (Ranked Solo/Duo) is the main target; queue 420+440 are used as observed ranked history.
- Sessions use 30 minutes as the primary boundary with 45/60/90-minute sensitivity checks.
- Recent volume uses 6 hours as the primary window with 3/12/24-hour sensitivity checks.
- The analysis is observational: report associations/prediction, not causal effects of fatigue or tilt.
- Multiple observations per player and shared physical matches are handled in inference using player fixed effects and two-way clustered uncertainty.

See `docs/REPRODUCIBILITY.md` and `docs/DATA_AND_PIPELINE_GUIDE.md` for details.

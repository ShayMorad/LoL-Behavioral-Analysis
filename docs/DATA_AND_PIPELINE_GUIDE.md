# Data & Pipeline Guide

## Data layers

### Canonical processed data

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/
```

Each region contains:

- `matches/` — one row per physical match;
- `participants/` — one row per player-match;
- `teams/` — two rows per match;
- `team_bans/` — champion-ban records;
- `audit/` — extraction metadata.

Use these tables for general match/team/champion analyses and for Question 2.

### Processed tracked-player lookup

```text
data/processed/tracking/authoritative/<REGION>/tracked_players.parquet
data/processed/tracking/alias_confirmed/<REGION>/tracked_players.parquet
```

`authoritative` is the main longitudinal cohort. `alias_confirmed` is the stricter provenance subset used for robustness.

These lookup tables are **inputs** to the compact processed-stage pipeline. Do not infer tracked players from number of matches/history length.

### Generated linkage

`01_prepare_data.py` rebuilds:

```text
data/processed/tracking/linked/authoritative/<REGION>/part-*.parquet
data/processed/tracking/linked/alias_confirmed/<REGION>/part-*.parquet
```

Unit: one tracked player in one observed match. Unique key:

```text
(source, player_id, match_id)
```

### Generated Q1 timelines

`01_prepare_data.py` also rebuilds:

```text
data/analysis/timelines/ranked_history/<REGION>.parquet
data/analysis/timelines/solo420_targets/<REGION>.parquet
```

The target tables are deliberately target-centric:

- `target_*` = target-match outcome/context;
- `prev_*`, `prior_*`, recent-volume, gap and session features = strictly pre-target history.

This separation is the main leakage-control mechanism.

## Q1 design

- target queue: **420 Ranked Solo/Duo**;
- observed ranked history: **420 + 440**;
- primary target-duration rule: **≥10 minutes**;
- short-game sensitivity: **≥5 minutes**;
- primary session boundary: **30 minutes**;
- session sensitivity: **45/60/90 minutes**;
- primary recent-volume window: **6 hours**;
- volume sensitivity: **3/12/24 hours**.

## Statistical model

The primary inferential model is a linear probability model of target victory with:

- player fixed effects implemented by within-player demeaning;
- dynamic pre-target history/patch controls;
- two-way cluster-robust uncertainty by `player_id` and physical `match_id`.

The primary interpretation is percentage-point change in next-match win probability. The design is observational and is not a causal estimate of fatigue/tilt.

## Predictive model

The course-aligned model is an entropy-based decision tree (`criterion="entropy"`). Data are split chronologically within region 70%/15%/15% into train/validation/test. `max_depth` and `min_samples_leaf` are chosen using validation AUC (pre-pruning). Test evaluation includes majority and rolling historical-win-rate baselines plus accuracy, precision, recall, F1, confusion matrices, ROC and ROC-AUC.

## Q1 outputs

`02_q1_analysis.py` writes everything under:

```text
data/analysis/q1/
```

The main report uses the four files under `figures/report/`; supplementary diagnostics stay under `figures/supplementary/`.

## Question 2

Q2 should start from the canonical `full_*` tables rather than Q1 timelines unless it is itself longitudinal. Team-objective clustering would primarily use `teams` + `matches`; champion co-pick/community analysis would primarily use `participants` (+ possibly `team_bans`).

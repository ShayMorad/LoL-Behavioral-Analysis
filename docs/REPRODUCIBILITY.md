# Reproducing the Project

The submitted pipeline is designed to run from the **processed-data stage** on Linux, Windows, or macOS.

## 1. Environment

Use **Python 3.10 or newer**.

From the project root:

```bash
python -m pip install -r requirements.txt
python code/run_project.py check
```

No PowerShell-specific paths, Bash line continuations, GUI display backend, or hard-coded Windows directories are required. Plotting uses Matplotlib's non-interactive `Agg` backend for headless lab machines.

## 2. Required processed inputs

Keep:

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/
data/processed/tracking/authoritative/{NA,KR,EU}/tracked_players.parquet
data/processed/tracking/alias_confirmed/{NA,KR,EU}/tracked_players.parquet
```

The canonical `full_*` directories contain `matches/`, `participants/`, `teams/`, `team_bans/`, and extraction audit metadata.

The small tracked-player lookup tables are processed provenance inputs. They were previously created from raw seed/PUUID evidence and should not be inferred again from match counts or Riot names.

## 3. Clean rerun

To remove only outputs that are reproducible from the processed inputs:

```bash
python code/run_project.py clean --yes
```

This removes:

```text
data/analysis/
data/processed/analysis_audit/
data/processed/tracking/linked/
data/processed/tracking/coverage_audit/
```

It does **not** remove canonical `full_*` data or the authoritative/alias-confirmed tracked-player lookup tables.

Then run:

```bash
python code/run_project.py q1 --strict-reference
```

The runner executes:

```text
01_prepare_data.py
    -> validates canonical processed data
    -> validates tracked-player lookup tables
    -> rebuilds linked tracked player-match Parquet shards
    -> rebuilds compact coverage/readiness audits
    -> rebuilds target-centric Q1 timelines

02_q1_analysis.py
    -> descriptive EDA / threshold justification
    -> within-player statistical models + confidence intervals
    -> Holm multiple-testing correction / sensitivity analyses
    -> chronological entropy decision trees + baselines
    -> accuracy/precision/recall/F1/confusion-matrix/ROC-AUC evaluation
    -> tracking/short-game robustness
    -> final report-ready figures
    -> reference regression checks
```

## 4. Analysis-only rerun

If timelines already exist:

```bash
python code/run_project.py q1 --analysis-only --strict-reference
```

## 5. Expected regression checks

The Q1 script verifies important results from the previously validated run, including:

- primary ≥10-minute sample rows by region,
- best pre-pruned entropy-tree parameters,
- held-out history/behavior/combined ROC-AUC values,
- combined-minus-history AUC increment,
- stability of the ≥5-minute short-game sensitivity.

Small numeric tolerance is allowed for library-version differences. Use the generated file:

```text
data/analysis/q1/audit/reference_regression_checks.csv
```

before changing report numbers.

## 6. Raw data

`data/raw/` can be retained locally as provenance/backup, but the normal lab reproduction does not require tens of gigabytes of raw JSON. The earlier raw JSON extraction and seed/PUUID identity-reconstruction scripts are therefore not part of the compact default execution path.

If complete raw-pipeline reproducibility is required for archival purposes, retain the historical extractor/tracking scripts outside the main three-file submitted pipeline (for example under `archive/`) rather than mixing them into the normal Q1 execution path.

# Reproducing the Project

The project supports two distinct usage modes:

1. **demo / result inspection** — uses the compact result artifacts committed under `data/analysis/q1`, `q2`, and `q3`;
2. **full scientific reproduction** — starts from the large canonical processed Parquet inputs under `data/processed/`.

This distinction is intentional.

---

# 1. Environment

Use **Python 3.10 or newer**.

From the project root:

```bash
python -m pip install -r requirements.txt
```

Core dependencies include:

- NumPy
- pandas
- PyArrow
- DuckDB
- Matplotlib
- scikit-learn
- NetworkX
- SciPy
- Streamlit
- Plotly

The submitted scripts use cross-platform `pathlib` paths and Matplotlib's non-interactive backend, so they can run on Linux, Windows, or macOS.

---

# 2. Fastest way to view the project

If the compact `data/analysis/q1`, `q2`, and `q3` outputs are present, the interactive demo does **not** require the large processed dataset.

Run:

```bash
streamlit run demo/app.py
```

The app reads result CSV/JSON/PNG files already produced by the pipeline.

This is the intended mode for:

- staff browsing the project;
- recorded presentation;
- Streamlit Community Cloud deployment.

---

# 3. Required inputs for full reproduction

Keep locally:

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/
```

Each regional canonical directory contains:

```text
matches/
participants/
teams/
team_bans/
audit/
```

Also keep:

```text
data/processed/tracking/authoritative/{NA,KR,EU}/tracked_players.parquet
data/processed/tracking/alias_confirmed/{NA,KR,EU}/tracked_players.parquet
data/processed/tracking/audit/
```

The tracked-player lookup tables are processed provenance inputs. They were created earlier from Riot identity evidence and should not be inferred again from history length or names in the canonical corpus.

---

# 4. Environment/input check

Before a full rerun:

```bash
python code/run_project.py check
```

The runner checks:

- Python version;
- required packages;
- all five code files;
- canonical regional Parquet tables;
- tracked-player lookup inputs.

Problem 3 additionally checks that Problem 2 tables are available when run independently.

---

# 5. Full reproduction

From the project root:

```bash
python code/run_project.py
```

No subcommand means:

```text
prepare -> Q1 -> Q2 -> Q3
```

Equivalent explicit command:

```bash
python code/run_project.py all
```

For a validated final rerun:

```bash
python code/run_project.py all --strict-reference
```

---

# 6. Clean rerun

To remove only outputs that are reproducible from the retained processed inputs:

```bash
python code/run_project.py clean --yes
```

The runner removes:

```text
data/analysis/
data/processed/analysis_audit/
data/processed/tracking/linked/
data/processed/tracking/coverage_audit/
```

It preserves:

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/

data/processed/tracking/authoritative/
data/processed/tracking/alias_confirmed/
data/processed/tracking/audit/
```

Then run:

```bash
python code/run_project.py all --strict-reference
```

### Important GitHub/demo note

The compact Q1/Q2/Q3 analysis outputs are useful to keep in GitHub because the hosted Streamlit demo reads them directly.

Therefore, after `clean --yes`, rerun the project **before committing/pushing** if you want the repository and hosted demo to continue containing the latest result artifacts.

---

# 7. Individual runner commands

| Command | What it does |
| --- | --- |
| `python code/run_project.py check` | Check environment and required processed inputs |
| `python code/run_project.py prepare` | Rebuild linkage, audits, and Q1 timelines |
| `python code/run_project.py q1` | Run preparation and Problem 1 |
| `python code/run_project.py q1 --analysis-only` | Run Q1 from existing timelines |
| `python code/run_project.py q1 --strict-reference` | Run Q1 and verify frozen reference results |
| `python code/run_project.py q2` | Run champion-pair analysis |
| `python code/run_project.py q3` | Run network analysis from Q2 tables |
| `python code/run_project.py all` | Run preparation + all three problems |
| `python code/run_project.py clean --yes` | Delete reproducible generated outputs |

---

# 8. Resource controls

The preparation, Q1, and Q2 scripts use DuckDB.

Default runner options:

```text
--duckdb-memory-limit 4GB
--duckdb-threads min(4, available CPUs)
```

Example for a constrained lab machine:

```bash
python code/run_project.py all --duckdb-memory-limit 3GB --duckdb-threads 2
```

The runner also supports:

```text
--dry-run
--no-overwrite
```

where applicable.

---

# 9. What preparation does

`01_prepare_data.py` performs four major stages:

```text
1. validate canonical processed data and tracked-player lookups
2. rebuild authoritative and alias-confirmed player-match linkage
3. rebuild compact coverage/readiness audits
4. rebuild target-centric Q1 timelines
```

Generated preparation outputs:

```text
data/processed/tracking/linked/
data/processed/tracking/coverage_audit/
data/processed/analysis_audit/
data/analysis/timelines/
```

---

# 10. Problem 1 reproduction

`02_q1_analysis.py` reads:

```text
data/analysis/timelines/solo420_targets/{NA,KR,EU}.parquet
```

and writes:

```text
data/analysis/q1/
```

Major stages:

```text
descriptive EDA
-> within-player inference
-> Holm correction
-> chronological entropy-tree prediction
-> robustness analyses
-> report/supplementary figures
```

## Analysis-only rerun

If timelines already exist:

```bash
python code/run_project.py q1 --analysis-only --strict-reference
```

---

# 11. Q1 strict-reference checks

The runner compares regenerated Q1 outputs with the previously validated consolidated run.

Reference checks include approximately:

| Check | Reference |
| --- | ---: |
| Primary target observations | **1,146,681** |
| History tree test ROC-AUC | **0.5137815** |
| Behavior tree test ROC-AUC | **0.5122918** |
| Combined tree test ROC-AUC | **0.5185223** |
| Combined − history AUC | **0.0047408** |
| Holm-significant robustness terms | **0** |
| Max >=5m coefficient change | **0.0922165 pp** |

Reference-selected tree parameters:

| Feature set | max_depth | min_samples_leaf |
| --- | ---: | ---: |
| History | 6 | 250 |
| Behavior | 2 | 3000 |
| Combined | 6 | 3000 |

The generated audit is:

```text
data/analysis/q1/audit/reference_regression_checks.csv
```

when strict-reference checking is run.

Small numeric tolerances are allowed for library-version differences.

---

# 12. Problem 2 reproduction

Run:

```bash
python code/run_project.py q2
```

`03_q2_pairings.py` scans canonical participant Parquet across NA/KR/EU, filters valid Ranked Solo/Duo teams, and writes:

```text
data/analysis/q2/
├── figures/
├── tables/
└── summary.json
```

Key compact tables:

```text
champion_stats.csv
pair_stats.csv
role_counts.csv
```

These are also the inputs to Problem 3.

---

# 13. Problem 3 reproduction

Run after Q2:

```bash
python code/run_project.py q3
```

`04_q3_network.py` reads the compact Q2 tables rather than rescanning the full participant corpus.

It writes:

```text
data/analysis/q3/
├── figures/
├── tables/
└── summary.json
```

This stage includes Louvain, PageRank, clique analysis, Girvan-Newman, and profile-based clustering comparisons.

---

# 14. Generated-output policy

## Large/reproducible products normally ignored by Git

```text
data/analysis/timelines/
data/analysis/q1/predictions/
data/processed/tracking/linked/
data/processed/tracking/coverage_audit/
data/processed/analysis_audit/
```

## Compact result artifacts useful to keep in GitHub

```text
data/analysis/q1/audit/
data/analysis/q1/tables/
data/analysis/q1/figures/

data/analysis/q2/
data/analysis/q3/
```

These compact artifacts make the repository self-contained for **result inspection and Streamlit deployment**, even though full scientific reproduction still requires the large local processed inputs.

---

# 15. Local Streamlit demo

After results exist:

```bash
streamlit run demo/app.py
```

The app reads:

```text
data/analysis/q1/tables/
data/analysis/q1/figures/report/

data/analysis/q2/tables/

data/analysis/q3/tables/
data/analysis/q3/figures/report/
```

It does **not** require:

```text
data/analysis/timelines/
data/analysis/q1/predictions/
data/processed/full_*/
data/processed/tracking/linked/
```

at demo runtime.

---

# 16. Streamlit Community Cloud deployment

For a lightweight hosted demo:

1. commit/push `demo/app.py`, `.streamlit/config.toml`, `requirements.txt`, and the compact Q1/Q2/Q3 result artifacts;
2. connect the GitHub repository to Streamlit Community Cloud;
3. set the app entry point to:

```text
demo/app.py
```

The hosted app can then read the compact analysis outputs directly from the cloned repository.

A custom domain is not required.

Community-hosted apps may sleep when idle according to Streamlit's service policy; the deployment URL remains the shareable entry point.

---

# 17. Final report paths

The report is stored under:

```text
docs/report/
```

Main files:

```text
docs/report/Report.pdf
docs/report/Report.docx
docs/report/Report_noimages.pdf
docs/report/Report_noimages.docx
```

This location is intentionally inside `docs/` so the report, technical documentation, and analysis reference live together.

---

# 18. Troubleshooting

## `Missing canonical processed table`

The full pipeline cannot find one of:

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/
```

Restore the local canonical processed Parquet inputs.

## `Missing processed tracked-player lookup`

Restore:

```text
data/processed/tracking/authoritative/
data/processed/tracking/alias_confirmed/
```

These are inputs, not generated outputs.

## Problem 3 says Problem 2 output is missing

Run:

```bash
python code/run_project.py q2
python code/run_project.py q3
```

or simply:

```bash
python code/run_project.py all
```

## Streamlit says Q1/Q2/Q3 outputs are missing

Generate them locally:

```bash
python code/run_project.py
```

For a hosted demo, make sure the compact result artifacts were actually committed/pushed to GitHub.

## Strict-reference failure

Inspect:

```text
data/analysis/q1/tables/key_results_for_report.csv
data/analysis/q1/tables/validation_grid.csv
data/analysis/q1/audit/reference_regression_checks.csv
```

before changing report numbers.

---

# 19. Reproduction boundary

The current submitted pipeline is intentionally a **processed-stage reproduction**.

It fully reproduces the analyses from the cleaned canonical Parquet and retained tracked-player lookups.

It does not include the earlier tens-of-gigabytes raw JSON extraction/identity-resolution workflow as the normal execution path. That choice keeps the submitted project compact, understandable, and realistic to rerun on course/lab hardware.

# Reproducing the Project

The final project is designed for a **processed-data rerun** on Linux, Windows, or macOS. The original raw Riot JSON corpus is not required for normal reproduction.

The intended staff workflow is deliberately simple:

```bash
python -m pip install -r requirements.txt
python code/run_project.py
```

Running the runner with no subcommand means **run the full project**.

---

## 1. Requirements

### Python

Use **Python 3.10 or newer**.

Check:

```bash
python --version
```

### Python packages

Install from the project root:

```bash
python -m pip install -r requirements.txt
```

Required runtime packages:

- NumPy
- pandas
- PyArrow
- DuckDB
- Matplotlib
- scikit-learn
- NetworkX
- SciPy

`orjson` is **not** required by the submitted processed-stage pipeline; it was only an optional accelerator for the historical raw-JSON extraction workflow.

### Platform behavior

The submitted scripts use:
- `pathlib` paths rather than hard-coded Windows paths;
- `sys.executable` when launching child Python processes;
- no PowerShell-only or Bash-only command syntax;
- Matplotlib's non-interactive **Agg** backend, so figures can be created on headless lab machines.

---

## 2. Required local data layout

Before running the project, the following processed inputs must exist:

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/
```

Each `full_*` directory must contain Parquet tables under:

```text
matches/
participants/
teams/
team_bans/
```

Problem 1 additionally requires the processed tracked-player lookup tables:

```text
data/processed/tracking/authoritative/{NA,KR,EU}/tracked_players.parquet
data/processed/tracking/alias_confirmed/{NA,KR,EU}/tracked_players.parquet
```

These data files may be distributed separately from Git because the canonical Parquet corpus is too large for ordinary Git hosting. A code-only clone is therefore **not sufficient** until the processed inputs have been placed in the paths above.

---

## 3. Environment check

Run:

```bash
python code/run_project.py check
```

The runner checks:
- Python version;
- all required Python imports;
- presence of the five active code files;
- canonical regional Parquet inputs;
- authoritative and alias-confirmed tracking lookup tables.

A successful check ends with:

```text
Environment and required processed inputs: OK
```

---

## 4. Full reproduction

### Simplest run

```bash
python code/run_project.py
```

Equivalent to:

```bash
python code/run_project.py all
```

Execution order:

```text
01_prepare_data.py
    -> validate canonical processed inputs
    -> validate tracking lookups
    -> rebuild tracked player-match linkage
    -> rebuild coverage/readiness audits
    -> rebuild chronological Q1 timelines

02_q1_analysis.py
    -> descriptive threshold analysis
    -> within-player inference
    -> sensitivity analyses / Holm correction
    -> chronological entropy-tree prediction
    -> classification evaluation
    -> robustness analyses
    -> Q1 report/supplementary figures

03_q2_pairings.py
    -> load all valid Ranked Solo/Duo teams
    -> champion statistics
    -> pair frequency / lift / win surplus
    -> Q2 tables and report figures

04_q3_network.py
    -> load Q2 compact tables
    -> PageRank / Louvain / modularity
    -> cliques / clique percolation
    -> Girvan-Newman comparison
    -> K-Means++ / hierarchical / DBSCAN comparisons
    -> role / association heatmaps and supplementary figures
```

---

## 5. Clean validated rerun

To remove only generated outputs:

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

It **does not** remove:
- `data/processed/full_*`;
- authoritative tracked-player lookups;
- alias-confirmed tracked-player lookups;
- tracking provenance/audit inputs.

Then run the strongest final reproduction:

```bash
python code/run_project.py all --strict-reference
```

`--strict-reference` applies to the frozen Problem 1 results. Problems 2 and 3 are checked structurally through their successful output generation and summary files.

---

## 6. Individual commands

| Command | What it runs | Required existing inputs |
| --- | --- | --- |
| `python code/run_project.py check` | environment/input checks | canonical processed + tracking lookups |
| `python code/run_project.py prepare` | `01_prepare_data.py` | canonical processed + tracking lookups |
| `python code/run_project.py q1` | prepare + Q1 | canonical processed + tracking lookups |
| `python code/run_project.py q1 --analysis-only` | Q1 only | existing `data/analysis/timelines/solo420_targets/` |
| `python code/run_project.py q2` | Problem 2 | canonical processed participants |
| `python code/run_project.py q3` | Problem 3 | existing Q2 tables |
| `python code/run_project.py all` | prepare + Q1 + Q2 + Q3 | all processed inputs |
| `python code/run_project.py clean --yes` | delete reproducible outputs | — |

### Useful common options

For `prepare`, `q1`, `q2`, and `all`:

```text
--duckdb-memory-limit 4GB
--duckdb-threads 4
--no-overwrite
--dry-run
```

Examples:

```bash
python code/run_project.py all --duckdb-memory-limit 3GB --duckdb-threads 2
python code/run_project.py q2 --dry-run
python code/run_project.py q1 --analysis-only --strict-reference
```

The runner normally overwrites its own generated output folders so repeated reproduction is simple. `--no-overwrite` changes that behavior and causes a run to fail instead of replacing a non-empty output directory.

---

## 7. Problem 1 strict reference checks

Problem 1 was previously reproduced and frozen. `--strict-reference` compares the newly generated outputs with the validated results below.

### Key values

| Check | Reference |
| --- | ---: |
| Primary target observations | **1,146,681** |
| History-tree test ROC-AUC | **0.5137815** |
| Behavior-tree test ROC-AUC | **0.5122918** |
| Combined-tree test ROC-AUC | **0.5185223** |
| Combined − history ROC-AUC | **0.0047408** |
| Holm-significant robustness terms | **0** |
| Maximum >=5-minute coefficient change | **0.0922165 pp** |

### Selected entropy-tree parameters

| Model | `max_depth` | `min_samples_leaf` |
| --- | ---: | ---: |
| History tree | 6 | 250 |
| Behavior tree | 2 | 3000 |
| Combined tree | 6 | 3000 |

Small numerical tolerances are allowed for floating-point/library differences. If all checks pass, the runner writes:

```text
data/analysis/q1/audit/reference_regression_checks.csv
```

and prints:

```text
Q1 strict reference checks: PASS
```

Do not change the hard-coded reference values merely to make a failed run pass. First inspect the generated Q1 tables and determine whether the code, inputs, or software environment changed meaningfully.

---

## 8. Expected output structure

After a complete run:

```text
data/
├── processed/
│   ├── analysis_audit/
│   └── tracking/
│       ├── linked/
│       └── coverage_audit/
└── analysis/
    ├── timelines/
    ├── q1/
    ├── q2/
    └── q3/
```

Each scientific problem writes a machine-readable summary:

```text
data/analysis/q1/audit/q1_summary.json
data/analysis/q2/summary.json
data/analysis/q3/summary.json
```

The report-ready figures are kept under each problem's `figures/report/` directory; additional method/sensitivity figures are placed under `figures/supplementary/`.

---

## 9. Direct script execution

The runner is recommended, but each script can also be executed directly from the project root.

```bash
python code/01_prepare_data.py --overwrite
python code/02_q1_analysis.py --overwrite
python code/03_q2_pairings.py --overwrite
python code/04_q3_network.py --overwrite
```

Order matters:
- Q1 requires `01_prepare_data.py` unless timelines already exist;
- Q3 requires the tables written by Q2.

Direct execution is useful for development, but the runner performs clearer prerequisite checks and is the preferred staff-facing interface.

---

## 10. Lab-computer notes

### Memory / CPU

DuckDB stages default to:

```text
memory limit: 4GB
threads:      min(4, available CPUs)
```

If a lab machine is constrained, reduce them:

```bash
python code/run_project.py all --duckdb-memory-limit 2GB --duckdb-threads 2
```

This should reduce resource pressure at the cost of runtime.

### Working directory

Run commands from the **project root**, i.e. the directory containing:

```text
README.md
requirements.txt
code/
data/
```

The scripts infer the root from their own file locations, but running from the root keeps command output and environment behavior predictable.

### Headless execution

No GUI is required. All Matplotlib scripts explicitly use the `Agg` backend and save PNG files directly.

---

## 11. Troubleshooting

### `Missing canonical processed table`

The expected Parquet data is absent or placed under the wrong path. Check:

```text
data/processed/full_na/
data/processed/full_kr/
data/processed/full_eu/
```

### `Missing processed tracked-player lookup`

Problem 1 cannot rebuild longitudinal identity provenance from anonymized match data alone. Restore the small lookup files under:

```text
data/processed/tracking/authoritative/
data/processed/tracking/alias_confirmed/
```

### `Problem 3 requires Problem 2 output`

Run:

```bash
python code/run_project.py q2
python code/run_project.py q3
```

or simply:

```bash
python code/run_project.py all
```

### `Output directory is not empty`

This normally occurs only during direct script execution or when `--no-overwrite` is used. Either add `--overwrite` to the direct script or rerun without `--no-overwrite`.

### Package import error

Reinstall the locked minimum dependencies:

```bash
python -m pip install -r requirements.txt
```

### Strict reference failure

Inspect:

```text
data/analysis/q1/tables/key_results_for_report.csv
data/analysis/q1/tables/validation_grid.csv
data/analysis/q1/audit/reference_regression_checks.csv
```

A failure means the newly generated Q1 outputs differ beyond the allowed tolerance from the validated run. Treat that as a reproducibility signal, not as something to suppress.

---

## 12. Raw-data provenance

`data/raw/` may be retained locally for archival provenance, but the normal submitted pipeline does not read it. The historical raw extractor and identity-reconstruction workflow are intentionally outside the compact default execution path because:

1. the raw corpus is very large;
2. the final analytical dataset is already canonicalized and validated;
3. tracked-player identities depend on historical provenance that is represented by the retained processed lookup tables.

For scientific interpretation of the processed layers and features, see [`DATA_AND_PIPELINE_GUIDE.md`](DATA_AND_PIPELINE_GUIDE.md).

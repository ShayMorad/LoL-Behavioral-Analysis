# Reproducing the Project

This file describes the recommended way to run the submitted code on a clean
machine. The commands are intentionally platform-neutral where possible.

## Python version

Use **Python 3.10 or newer**.

The project uses modern Python type syntax and has been designed around Python
3.10+.

## Install dependencies

From the project root:

```text
python -m pip install -r requirements.txt
```

Then verify the environment and required project inputs:

```text
python run_project.py q1 --check-only
```

The runner checks:

- Python version
- required Python packages
- required stage scripts
- required processed Parquet inputs for the selected starting stage

## Recommended Question 1 reproduction

The original raw dataset is very large and is not required to reproduce the
main Question 1 analysis when the processed data are available.

To rebuild Question 1 beginning with tracked-player linkage:

```text
python run_project.py q1 --start-at 3
```

This runs:

```text
03  tracking coverage + permanent player-match linkage
04  analytical-readiness audit
05  chronological player timelines / features
06  exploratory analysis
07  statistical analysis
08  predictive modeling
09  robustness analysis
```

All stages use `sys.executable`, `pathlib.Path`, and Python subprocess calls.
The runner does not depend on PowerShell, Bash line continuations, or Windows
path separators.

## Faster analysis-only reproduction

If the timeline Parquet files under:

```text
data/analysis/timelines/solo420_targets/
```

already exist, the analytical stages can be reproduced directly:

```text
python run_project.py q1 --start-at 6
```

## Dry run

To inspect commands without executing them:

```text
python run_project.py q1 --start-at 3 --dry-run
```

## Why stages 00-02 are not the default reproduction path

Stages 00-02 document and reproduce the original raw-data verification,
Match-V5 extraction, and tracked-player reconstruction. They require the
original large raw Match-V5 archive, seed-list files, and `league_data.db`.

They are included in the submitted code because they are part of the project
implementation and provenance chain, but course staff do not need to rerun
tens of gigabytes of raw extraction to inspect or reproduce the reported
analysis.

When the original raw data are available, the individual scripts can still be
run directly using their `--help` output and documented command-line
arguments.

## Generated outputs

Question 1 writes its generated results under:

```text
data/processed/analysis_audit/
data/analysis/timelines/
data/analysis/eda/
data/analysis/statistics/
data/analysis/prediction/
data/analysis/q1_robustness/
```

These outputs are reproducible and may be safely regenerated with
`--overwrite`.

## Data availability

The original dataset source is documented in the report and README. The
processed datasets are documented in `docs/DATA_AND_PIPELINE_GUIDE.md`.

For code-only submission, the large raw corpus should not be bundled inside the
code archive. If the processed data are distributed separately (for example
through the project repository), the runner can reproduce the analysis from
stage 03 onward.

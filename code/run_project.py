#!/usr/bin/env python3
"""Cross-platform entry point for the full League of Legends project.

The normal reproduction starts from data/processed/; the 75 GB raw JSON corpus
is not required. Running with no subcommand executes the complete project:

    python code/run_project.py

Other useful commands:
    python code/run_project.py check
    python code/run_project.py q1 --strict-reference
    python code/run_project.py q2
    python code/run_project.py q3
    python code/run_project.py all --strict-reference
    python code/run_project.py clean --yes
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
DATA = ROOT / "data"

# One script per pipeline stage/research problem.
PREPARE = CODE / "01_prepare_data.py"
Q1 = CODE / "02_q1_analysis.py"
Q2 = CODE / "03_q2_pairings.py"
Q3 = CODE / "04_q3_network.py"

PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "duckdb": "duckdb",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "networkx": "networkx",
    "scipy": "scipy",
}

# Only reproducible outputs are removed by the clean command.
GENERATED = [
    DATA / "analysis",
    DATA / "processed" / "analysis_audit",
    DATA / "processed" / "tracking" / "linked",
    DATA / "processed" / "tracking" / "coverage_audit",
]

# Frozen Q1 values are regression checks only; they never affect model fitting.
Q1_REFERENCE = {
    "Primary target observations": (1_146_681.0, 0.0),
    "History tree test ROC-AUC": (0.5137815, 5e-6),
    "Behavior tree test ROC-AUC": (0.5122918, 5e-6),
    "Combined tree test ROC-AUC": (0.5185223, 5e-6),
    "Combined - history AUC": (0.0047408, 5e-6),
    "Holm-significant robustness terms": (0.0, 0.0),
    "Max >=5m coefficient change (pp)": (0.0922165, 5e-5),
}
Q1_TREE_REFERENCE = {
    "history_tree": (6, 250),
    "behavior_tree": (2, 3000),
    "combined_tree": (6, 3000),
}


def add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add shared overwrite, DuckDB resource, and dry-run options to a subcommand."""
    parser.add_argument("--no-overwrite", action="store_true", help="Do not replace existing generated outputs.")
    parser.add_argument("--duckdb-memory-limit", default="4GB")
    parser.add_argument("--duckdb-threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")


def parse_args() -> argparse.Namespace:
    # Friendly lab behavior: `python code/run_project.py` means `all`.
    """Parse runner subcommands; running without one defaults to the full project."""
    if len(sys.argv) == 1:
        sys.argv.append("all")

    p = argparse.ArgumentParser(description="Reproduce the League of Legends data-science project.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Check Python, dependencies, code files, and processed inputs.")

    clean = sub.add_parser("clean", help="Delete only generated outputs that can be rebuilt.")
    clean.add_argument("--yes", action="store_true", help="Required confirmation for deletion.")

    prepare = sub.add_parser("prepare", help="Rebuild tracking linkage, audits, and Q1 timelines.")
    add_common_options(prepare)

    q1 = sub.add_parser("q1", help="Run Problem 1: temporal behavior and next-match performance.")
    add_common_options(q1)
    q1.add_argument("--analysis-only", action="store_true", help="Use existing timelines instead of rebuilding them.")
    q1.add_argument("--strict-reference", action="store_true", help="Verify the validated Q1 reference results after the run.")

    q2 = sub.add_parser("q2", help="Run Problem 2: champion pairings and combo performance.")
    add_common_options(q2)

    q3 = sub.add_parser("q3", help="Run Problem 3: champion network structure and communities.")
    add_common_options(q3)

    all_cmd = sub.add_parser("all", help="Run preparation and all three problems.")
    add_common_options(all_cmd)
    all_cmd.add_argument("--strict-reference", action="store_true", help="Verify the validated Q1 reference results.")
    return p.parse_args()


def has_parquet(path: Path) -> bool:
    """Return whether a directory exists and contains at least one Parquet file."""
    return path.exists() and any(path.rglob("*.parquet"))


def check_environment(*, need_tracking: bool = True, need_q2_tables: bool = False) -> list[str]:
    """Validate Python, required packages, code files, processed inputs, and optional prerequisites."""
    problems: list[str] = []
    if sys.version_info < (3, 10):
        problems.append(f"Python >=3.10 required; found {sys.version.split()[0]}.")

    # Check dependencies before launching long-running analysis stages.
    for import_name, install_name in PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            problems.append(f"Missing/unusable package {install_name!r}: {exc}")

    for path in (PREPARE, Q1, Q2, Q3):
        if not path.exists():
            problems.append(f"Missing code file: {path}")

    # Canonical processed Parquet is the submitted starting point.
    for region in ("na", "kr", "eu"):
        root = DATA / "processed" / f"full_{region}"
        for table in ("matches", "participants", "teams", "team_bans"):
            if not has_parquet(root / table):
                problems.append(f"Missing canonical processed table: {root / table}")

    # Tracking lookups are required only by the longitudinal Q1 pipeline.
    if need_tracking:
        for cohort in ("authoritative", "alias_confirmed"):
            for source in ("NA", "KR", "EU"):
                path = DATA / "processed" / "tracking" / cohort / source / "tracked_players.parquet"
                if not path.exists():
                    problems.append(f"Missing processed tracked-player lookup: {path}")

    # Q3 consumes Q2 tables instead of rescanning the participant corpus.
    if need_q2_tables:
        for name in ("champion_stats.csv", "pair_stats.csv", "role_counts.csv"):
            path = DATA / "analysis" / "q2" / "tables" / name
            if not path.exists():
                problems.append(f"Problem 3 requires Problem 2 output: {path}")
    return problems


def print_check(problems: list[str]) -> int:
    """Print a concise environment-check report and return a shell-friendly status code."""
    print(f"Project root: {ROOT}")
    print(f"Python:       {sys.executable}")
    print(f"Version:      {sys.version.split()[0]}")
    if problems:
        print("\nCHECK FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nInstall dependencies with:\n  python -m pip install -r requirements.txt")
        return 1
    print("\nEnvironment and required processed inputs: OK")
    return 0


def command_common(args: argparse.Namespace) -> list[str]:
    """Build shared DuckDB resource arguments passed to analysis scripts."""
    return [
        "--duckdb-memory-limit",
        args.duckdb_memory_limit,
        "--duckdb-threads",
        str(args.duckdb_threads),
    ]


def overwrite_args(args: argparse.Namespace) -> list[str]:
    """Translate the runner overwrite choice into script command-line arguments."""
    return [] if args.no_overwrite else ["--overwrite"]


# Each stage runs in a fresh Python process, matching staff/lab execution.
def run_command(command: list[str], dry_run: bool) -> None:
    """Print and execute one child Python command from the project root."""
    print("\n$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


# Safe cleanup never touches canonical data or tracked-player provenance.
def clean_generated(confirmed: bool) -> int:
    """Delete only regenerated outputs after explicit confirmation."""
    if not confirmed:
        print("Refusing to delete anything without --yes.")
        print("The following generated folders would be removed:")
        for path in GENERATED:
            print(f"  {path}")
        return 2
    for path in GENERATED:
        if path.exists():
            print(f"Removing {path}")
            shutil.rmtree(path)
    print("Generated outputs removed; canonical processed data and tracked-player lookups were preserved.")
    return 0


def run_prepare(args: argparse.Namespace) -> None:
    """Run the processed-data preparation stage with the shared runner options."""
    run_command(
        [sys.executable, str(PREPARE), *command_common(args), *overwrite_args(args)],
        args.dry_run,
    )


def run_q1_analysis(args: argparse.Namespace) -> None:
    """Run Problem 1 using the prepared chronological timelines."""
    run_command(
        [sys.executable, str(Q1), *command_common(args), *overwrite_args(args)],
        args.dry_run,
    )


def run_q2_analysis(args: argparse.Namespace) -> None:
    """Run Problem 2 using the canonical processed participant data."""
    run_command(
        [sys.executable, str(Q2), *command_common(args), *overwrite_args(args)],
        args.dry_run,
    )


def run_q3_analysis(args: argparse.Namespace) -> None:
    # Q3 reads Q2 tables and therefore does not need DuckDB options itself.
    """Run Problem 3 using the tables generated by Problem 2."""
    run_command([sys.executable, str(Q3), *overwrite_args(args)], args.dry_run)


def _read_key_results(path: Path) -> dict[str, float]:
    """Read Q1 report metrics from the generated key-results CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def _read_best_tree_params(path: Path) -> dict[str, tuple[int, int]]:
    """Recover the validation-selected tree settings for each Q1 feature set."""
    rows: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["feature_set"], []).append(row)

    best = {}
    for feature_set, group in rows.items():
        chosen = sorted(
            group,
            key=lambda r: (
                -float(r["validation_auc"]),
                int(float(r["max_depth"])),
                -int(float(r["min_samples_leaf"])),
            ),
        )[0]
        best[feature_set] = (
            int(float(chosen["max_depth"])),
            int(float(chosen["min_samples_leaf"])),
        )
    return best


def check_q1_reference() -> None:
    """Regression-check Q1 against the previously validated consolidated run."""
    # Detect accidental code/environment drift after Q1 was already validated.
    tables = DATA / "analysis" / "q1" / "tables"
    key_file = tables / "key_results_for_report.csv"
    grid_file = tables / "validation_grid.csv"
    if not key_file.exists() or not grid_file.exists():
        raise RuntimeError("Q1 strict reference check requires completed Q1 output tables.")

    observed = _read_key_results(key_file)
    failures = []
    for metric, (expected, tolerance) in Q1_REFERENCE.items():
        actual = observed.get(metric)
        if actual is None or abs(actual - expected) > tolerance:
            failures.append(f"{metric}: expected {expected}, observed {actual}")

    best = _read_best_tree_params(grid_file)
    for model, expected in Q1_TREE_REFERENCE.items():
        if best.get(model) != expected:
            failures.append(f"{model} parameters: expected {expected}, observed {best.get(model)}")

    audit = DATA / "analysis" / "q1" / "audit" / "reference_regression_checks.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "status"])
        for metric in Q1_REFERENCE:
            writer.writerow([metric, "PASS" if not any(item.startswith(metric + ":") for item in failures) else "FAIL"])
        for model in Q1_TREE_REFERENCE:
            writer.writerow([f"{model}_best_parameters", "PASS" if not any(item.startswith(model + " parameters:") for item in failures) else "FAIL"])

    if failures:
        raise RuntimeError("Q1 strict reference check failed:\n  - " + "\n  - ".join(failures))
    print("\nQ1 strict reference checks: PASS")


def ensure_ok(**kwargs) -> int:
    """Run environment validation and return its status code."""
    problems = check_environment(**kwargs)
    return print_check(problems)


def main() -> int:
    """Run the requested repository workflow.

    Handles checks and cleanup, or executes preparation and Problems 1-3 in the
    correct dependency order; the no-command default is the full reproduction.
    """
    args = parse_args()

    # Administrative commands do not run any scientific analysis.
    if args.command == "check":
        return print_check(check_environment())
    if args.command == "clean":
        return clean_generated(args.yes)

    # Single-stage commands validate prerequisites before launching child scripts.
    if args.command == "prepare":
        if ensure_ok():
            return 1
        run_prepare(args)
        return 0

    if args.command == "q1":
        if ensure_ok():
            return 1
        if not args.analysis_only:
            run_prepare(args)
        run_q1_analysis(args)
        if args.strict_reference and not args.dry_run:
            check_q1_reference()
        return 0

    if args.command == "q2":
        if ensure_ok(need_tracking=False):
            return 1
        run_q2_analysis(args)
        return 0

    if args.command == "q3":
        if ensure_ok(need_tracking=False, need_q2_tables=True):
            return 1
        run_q3_analysis(args)
        return 0

    # Full reproduction follows the dependency order: prepare -> Q1 -> Q2 -> Q3.
    # Full reproduction follows the dependency order: prepare -> Q1 -> Q2 -> Q3.
    if args.command == "all":
        if ensure_ok():
            return 1
        run_prepare(args)
        run_q1_analysis(args)
        if args.strict_reference and not args.dry_run:
            check_q1_reference()
        run_q2_analysis(args)
        run_q3_analysis(args)
        if not args.dry_run:
            print("\nFULL PROJECT REPRODUCTION COMPLETED SUCCESSFULLY.")
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

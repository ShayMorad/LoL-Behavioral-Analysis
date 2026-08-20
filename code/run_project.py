#!/usr/bin/env python3
"""
run_project.py

Cross-platform project entry point. Place this file in code/.

Examples (run from the project root):

    python code/run_project.py check
    python code/run_project.py q1
    python code/run_project.py q1 --strict-reference
    python code/run_project.py clean --yes

The normal Q1 reproduction starts from data/processed. Raw JSON is not needed.
"""

from __future__ import annotations

import argparse
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

PREPARE = CODE / "01_prepare_data.py"
Q1 = CODE / "02_q1_analysis.py"

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "duckdb": "duckdb",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
}

# Generated products that can be safely removed and rebuilt from processed
# canonical data + tracked-player lookup tables.
GENERATED = [
    DATA / "analysis",
    DATA / "processed" / "analysis_audit",
    DATA / "processed" / "tracking" / "linked",
    DATA / "processed" / "tracking" / "coverage_audit",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-platform League project runner.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Check Python, packages, code files, and processed inputs.")

    clean = sub.add_parser("clean", help="Delete only generated outputs that are reproducible from processed data.")
    clean.add_argument("--yes", action="store_true", help="Required confirmation for deletion.")

    q1 = sub.add_parser("q1", help="Rebuild processed-stage linkage/timelines and run complete Q1 analysis.")
    q1.add_argument("--analysis-only", action="store_true", help="Skip data preparation and use existing timelines.")
    q1.add_argument("--no-overwrite", action="store_true", help="Do not pass --overwrite to child scripts.")
    q1.add_argument("--strict-reference", action="store_true", help="Fail if Q1 results differ from the prior validated reference run.")
    q1.add_argument("--duckdb-memory-limit", default="4GB")
    q1.add_argument("--duckdb-threads", type=int, default=min(4, os.cpu_count() or 1))
    q1.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def has_parquet(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def check_environment(require_timelines: bool = False) -> list[str]:
    problems: list[str] = []
    if sys.version_info < (3, 10):
        problems.append(f"Python >=3.10 required; found {sys.version.split()[0]}.")

    for import_name, install_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            problems.append(f"Missing/unusable package {install_name!r}: {exc}")

    for path in (PREPARE, Q1):
        if not path.exists():
            problems.append(f"Missing code file: {path}")

    for region in ("na", "kr", "eu"):
        root = DATA / "processed" / f"full_{region}"
        for table in ("matches", "participants", "teams", "team_bans"):
            if not has_parquet(root / table):
                problems.append(f"Missing canonical processed table: {root / table}")

    # These are deliberately retained processed inputs. Linkage is regenerated.
    for cohort in ("authoritative", "alias_confirmed"):
        for source in ("NA", "KR", "EU"):
            p = DATA / "processed" / "tracking" / cohort / source / "tracked_players.parquet"
            if not p.exists():
                problems.append(f"Missing processed tracked-player lookup: {p}")

    if require_timelines:
        for source in ("NA", "KR", "EU"):
            p = DATA / "analysis" / "timelines" / "solo420_targets" / f"{source}.parquet"
            if not p.exists():
                problems.append(f"Analysis-only run requires timeline: {p}")
    return problems


def print_check(problems: list[str]) -> int:
    print(f"Project root: {ROOT}")
    print(f"Python:       {sys.executable}")
    print(f"Version:      {sys.version.split()[0]}")
    if problems:
        print("\nCHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        print("\nInstall dependencies with:\n  python -m pip install -r requirements.txt")
        return 1
    print("\nEnvironment and processed inputs: OK")
    return 0


def run_command(command: list[str], dry_run: bool) -> None:
    print("\n$ " + shlex.join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def clean_generated(confirmed: bool) -> int:
    if not confirmed:
        print("Refusing to delete anything without --yes.")
        print("The following generated folders would be removed:")
        for p in GENERATED:
            print(f"  {p}")
        return 2
    for p in GENERATED:
        if p.exists():
            print(f"Removing {p}")
            shutil.rmtree(p)
    print("Generated outputs removed. Canonical processed data and tracked-player lookups were preserved.")
    return 0


def run_q1(args: argparse.Namespace) -> int:
    problems = check_environment(require_timelines=args.analysis_only)
    status = print_check(problems)
    if status:
        return status

    common = [
        "--duckdb-memory-limit", args.duckdb_memory_limit,
        "--duckdb-threads", str(args.duckdb_threads),
    ]
    overwrite = [] if args.no_overwrite else ["--overwrite"]

    if not args.analysis_only:
        run_command([sys.executable, str(PREPARE), *common, *overwrite], args.dry_run)

    q1_cmd = [sys.executable, str(Q1), *common, *overwrite]
    if args.strict_reference:
        q1_cmd.append("--strict-reference")
    run_command(q1_cmd, args.dry_run)

    if not args.dry_run:
        print("\nQ1 reproduction completed successfully.")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "check":
        return print_check(check_environment())
    if args.command == "clean":
        return clean_generated(args.yes)
    if args.command == "q1":
        return run_q1(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

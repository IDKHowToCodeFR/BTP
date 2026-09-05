#!/usr/bin/env python3
"""Pre-flight sanity checks: run before spending any real LLM budget.

Runs (a) the project's unit tests for the deterministic core (baselines,
preprocessing, validation, metrics -- everything that doesn't need a
network call or API key) via pytest, and (b) a few extra data-specific
diagnostics against whatever is currently in data/processed/.

This is deliberately separate from `pytest tests/` (which you should
also run, and which this script's part (a) is a subset of) -- the point
here is a single command that also sanity-checks *your actual downloaded
data*, not just the code in isolation.

Usage:
    python scripts/03_validate_baselines.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
from agentic_selection.utils import load_config, setup_logging


def run_unit_tests(project_root: Path) -> bool:
    print("== Running unit tests (baselines, preprocessing, validation, metrics) ==")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=project_root,
    )
    return result.returncode == 0


def check_processed_data(data_dir: Path) -> bool:
    print("\n== Checking processed QWS data ==")
    path = data_dir / "processed" / "qws_normalized.csv"
    if not path.exists():
        print(f"  NOT FOUND: {path}. Run scripts/02_prepare_data.py first.", file=sys.stderr)
        return False

    df = pd.read_csv(path, index_col=0)
    ok = True

    is_synthetic = bool(df.get("is_synthetic", pd.Series(dtype=bool)).any())
    print(f"  rows: {len(df)}  |  is_synthetic: {is_synthetic}")
    if is_synthetic:
        print(
            "  WARNING: this is synthetic fallback data, not the real QWS "
            "dataset. Fine for testing the pipeline, not for real results.",
            file=sys.stderr,
        )

    for col in QWS_ATTRIBUTE_COLUMNS:
        if col not in df.columns:
            print(f"  MISSING COLUMN: {col}", file=sys.stderr)
            ok = False
            continue
        cmin, cmax = df[col].min(), df[col].max()
        if cmin < -1e-9 or cmax > 1 + 1e-9:
            print(f"  OUT OF RANGE: {col} in [{cmin}, {cmax}], expected [0, 1]", file=sys.stderr)
            ok = False
    if ok:
        print("  all 9 QWS attribute columns present and within [0, 1] -- OK")

    n_nan = df[QWS_ATTRIBUTE_COLUMNS].isnull().sum().sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN cells remain in attribute columns (impute before scoring)", file=sys.stderr)
    else:
        print("  no missing values in attribute columns -- OK")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    setup_logging()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / args.config)
    data_dir = project_root / config["paths"]["data_dir"]

    tests_ok = True
    if not args.skip_tests:
        tests_ok = run_unit_tests(project_root)

    data_ok = check_processed_data(data_dir)

    print("\n== Summary ==")
    print(f"  unit tests: {'PASS' if tests_ok else 'FAIL (or skipped)'}")
    print(f"  data checks: {'PASS' if data_ok else 'FAIL'}")

    if tests_ok and data_ok:
        print("\nEverything looks sane. Safe to proceed to scripts/04_run_stable_experiment.py")
        return 0
    print("\nFix the issues above before running any real experiment.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate results/tables/{stable,drift}_results.csv into paper-ready
Markdown tables (paper §5) and PNG figures.

Refuses to produce a final report if any underlying trial was computed
on synthetic fallback data, unless --allow-synthetic is passed (see
evaluation/report.py).

Usage:
    python scripts/06_generate_report.py
    python scripts/06_generate_report.py --allow-synthetic   # preview only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from agentic_selection.evaluation import make_figures, render_markdown_tables
from agentic_selection.utils import load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    setup_logging()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / args.config)
    results_dir = project_root / config["paths"]["results_dir"]
    figures_dir = project_root / config["paths"]["figures_dir"]

    stable_path = results_dir / "stable_results.csv"
    drift_path = results_dir / "drift_results.csv"
    wsdream_path = results_dir / "wsdream_stable_results.csv"

    if not stable_path.exists():
        print(f"No stable results found at {stable_path}. Run scripts/04_run_stable_experiment.py first.", file=sys.stderr)
        return 1

    stable_df = pd.read_csv(stable_path)
    drift_df = pd.read_csv(drift_path) if drift_path.exists() else None
    wsdream_df = pd.read_csv(wsdream_path) if wsdream_path.exists() else None
    if wsdream_df is None:
        print(
            "(No WS-DREAM supplementary validation results found -- run "
            "scripts/07_run_wsdream_validation.py if you want Table 5.4 "
            "included. Not required for the main results.)"
        )

    try:
        md = render_markdown_tables(stable_df, drift_df, allow_synthetic=args.allow_synthetic, wsdream_df=wsdream_df)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    report_path = results_dir / "report_tables.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"Wrote tables to {report_path}")

    figs = make_figures(stable_df, drift_df, figures_dir)
    for f in figs:
        print(f"Wrote figure to {f}")

    print("\n--- Preview ---\n")
    print(md)
    print(
        "\nTo finish the paper: copy the tables above into "
        "paper/agentic_cloud_selection_paper.md Section 5, replacing the "
        "placeholder tables, and write the Discussion (§6) against H1-H4."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

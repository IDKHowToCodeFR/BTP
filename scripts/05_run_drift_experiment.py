#!/usr/bin/env python3
"""Run the drift experiment (paper §3.8, §4.3): tests H2 (adaptation
lag). Static baselines (global_fixed, lookup_table) are re-evaluated
only every `static_reevaluation_period` rounds; agent conditions are
re-evaluated every round -- see evaluation/metrics.py's module docstring
for why.

Usage:
    python scripts/05_run_drift_experiment.py
    python scripts/05_run_drift_experiment.py --n-trials 2   # cheap smoke test
    python scripts/05_run_drift_experiment.py --yes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from agentic_selection.agent import AgentController, build_backend_from_config
from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
from agentic_selection.evaluation import run_drift_protocol
from agentic_selection.tasks import TASK_PROFILES
from agentic_selection.utils import load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    setup_logging()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / args.config)

    data_dir = project_root / config["paths"]["data_dir"]
    norm_path = data_dir / "processed" / "qws_normalized.csv"
    if not norm_path.exists():
        print(f"Processed QWS data not found at {norm_path}. Run scripts/02_prepare_data.py first.", file=sys.stderr)
        return 1
    norm_df = pd.read_csv(norm_path, index_col=0)
    is_synthetic = bool(norm_df.get("is_synthetic", pd.Series(dtype=bool)).any())

    dcfg = config["protocol"]["drift"]
    n_trials = args.n_trials or dcfg["n_trials"]

    n_tasks = len(TASK_PROFILES)
    est_calls_per_trial = dcfg["n_rounds"] * 2  # agent_weights_only + agent_full, every round
    est_calls = n_tasks * n_trials * est_calls_per_trial
    provider = config["llm"]["provider"]

    print("== Drift experiment plan ==")
    print(f"  task profiles              : {n_tasks}")
    print(f"  trials per profile          : {n_trials}")
    print(f"  rounds per trial            : {dcfg['n_rounds']} (degrade at round {dcfg['degrade_start_round']})")
    print(f"  static re-eval period       : every {dcfg['static_reevaluation_period']} rounds")
    print(f"  LLM provider                : {provider} (model: {config['llm'].get('model')})")
    print(f"  estimated LLM calls         : ~{est_calls} "
          f"({n_tasks} profiles x {n_trials} trials x {dcfg['n_rounds']} rounds x 2 agent conditions)")
    if is_synthetic:
        print("  ** WARNING: processed data is SYNTHETIC FALLBACK, not real QWS data **")
    if provider != "mock" and est_calls > 0 and not args.yes:
        resp = input(f"\nProceed with ~{est_calls} real LLM calls via '{provider}'? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    backend = build_backend_from_config(config["llm"])
    memory_path = project_root / config["paths"]["memory_path"]
    controller = AgentController(
        backend=backend,
        attribute_cols=QWS_ATTRIBUTE_COLUMNS,
        memory_path=memory_path,
        tool_menu=tuple(config["agent"]["tool_menu"]),
        k_memory=config["agent"]["k_memory"],
    )

    results_dir = project_root / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / "drift_results.csv"

    print(f"\nRunning (resumable, writing incrementally to {output_csv}) ...")
    results = run_drift_protocol(
        norm_df,
        QWS_ATTRIBUTE_COLUMNS,
        output_csv,
        controller,
        n_trials=n_trials,
        pool_size=dcfg["pool_size"],
        n_rounds=dcfg["n_rounds"],
        degrade_start_round=dcfg["degrade_start_round"],
        degradation_profile=dcfg["degradation_profile"],
        degradation_magnitude=dcfg["degradation_magnitude"],
        static_reevaluation_period=dcfg["static_reevaluation_period"],
        base_seed=dcfg["base_seed"],
        is_synthetic_data=is_synthetic,
    )
    print(f"\nDone. {len(results)} total trial rows in {output_csv}")
    print("Next: python scripts/06_generate_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

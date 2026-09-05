#!/usr/bin/env python3
"""Run the stable-condition protocol (paper §4.3): all 4 conditions
across every task profile + held-out task, `n_pools` independent pools
each. This is the experiment that produces paper Table 5.1 and tests
H1 and H3.

This makes real LLM calls (2 per task/pool combination: agent_weights_only
and agent_full) unless llm.provider is set to "mock" in config.yaml. It
prints an estimated call count and asks for confirmation before spending
any budget -- pass --yes to skip that prompt (e.g. for a scripted/CI run).

Resumable: safe to Ctrl-C and re-run the same command; already-completed
trials are skipped (see evaluation/protocol.py).

Usage:
    python scripts/04_run_stable_experiment.py                 # full protocol per config.yaml
    python scripts/04_run_stable_experiment.py --n-pools 3      # cheap smoke test
    python scripts/04_run_stable_experiment.py --yes            # skip confirmation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from agentic_selection.agent import AgentController, build_backend_from_config
from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
from agentic_selection.evaluation import run_stable_protocol
from agentic_selection.tasks import HELD_OUT_TASKS, TASK_PROFILES
from agentic_selection.utils import load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-pools", type=int, default=None, help="Override config.yaml protocol.stable.n_pools")
    parser.add_argument("--pool-size", type=int, default=None, help="Override config.yaml protocol.stable.pool_size")
    parser.add_argument("--yes", action="store_true", help="Skip the cost-estimate confirmation prompt")
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

    n_pools = args.n_pools or config["protocol"]["stable"]["n_pools"]
    pool_size = args.pool_size or config["protocol"]["stable"]["pool_size"]
    base_seed = config["protocol"]["stable"]["base_seed"]

    n_tasks = len(TASK_PROFILES) + len(HELD_OUT_TASKS)
    est_calls = n_tasks * n_pools * 2  # agent_weights_only + agent_full
    provider = config["llm"]["provider"]

    print("== Stable-condition experiment plan ==")
    print(f"  task profiles + held-out tasks : {n_tasks}")
    print(f"  pools per task                 : {n_pools} (size {pool_size})")
    print(f"  LLM provider                   : {provider} (model: {config['llm'].get('model')})")
    print(f"  estimated LLM calls            : ~{est_calls} "
          f"({n_tasks} tasks x {n_pools} pools x 2 agent conditions)")
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
    output_csv = results_dir / "stable_results.csv"

    print(f"\nRunning (resumable, writing incrementally to {output_csv}) ...")
    results = run_stable_protocol(
        norm_df,
        QWS_ATTRIBUTE_COLUMNS,
        output_csv,
        controller,
        n_pools=n_pools,
        pool_size=pool_size,
        base_seed=base_seed,
        is_synthetic_data=is_synthetic,
    )
    print(f"\nDone. {len(results)} total trial rows in {output_csv}")
    print("Next: python scripts/05_run_drift_experiment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

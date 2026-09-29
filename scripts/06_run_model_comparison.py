#!/usr/bin/env python3
"""Run the stable-condition protocol across multiple models and reasoners.
Models: qwen2.5:1.5b, llama3.2:latest, qwen2.5:7b, llama3.1:8b
Reasoners: DirectWeightReasoner, ClassificationReasoner

Produces a model comparison table with regret+CI, weight_l1, fallback_rate, category_acc, and latency.
Flags if model ≈ uniform baseline.
"""
from __future__ import annotations

import argparse
import sys
import json
import hashlib
import uuid
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from agentic_selection.agent import AgentController, OllamaBackend, DirectWeightReasoner, ClassificationReasoner
from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
from agentic_selection.evaluation.protocol import run_stable_protocol, STABLE_RESULT_FIELDS
from agentic_selection.evaluation.storage import CsvStorage
from agentic_selection.tasks import HELD_OUT_TASKS, TASK_PROFILES
from agentic_selection.utils import load_config, setup_logging
from agentic_selection.evaluation.metrics import bootstrap_ci


def print_comparison_report(df: pd.DataFrame, model_name: str, uniform_regret: float, uniform_regret_ci_lower: float):
    print(f"\n=======================================================")
    print(f" Model: {model_name}")
    print(f"=======================================================")
    
    agent_df = df[df["condition"] == "agent_full"].copy()
    if len(agent_df) == 0:
        print("No agent_full data for this model.")
        return

    for reasoner_name, group in agent_df.groupby("reasoner"):
        # Regret + CI
        regret_ci = bootstrap_ci(group["regret"], group["pool_seed"])
        # Check against uniform baseline
        flag = ""
        if regret_ci.mean >= uniform_regret_ci_lower:
            flag = " (⚠️ ≈ Uniform Baseline)"

        weight_l1 = group["weight_l1"].mean()
        fallback_rate = group["fallback_triggered"].astype(bool).mean() * 100
        latency = group["latency_seconds"].mean()
        
        # Category Accuracy
        # (Assuming task_key is the true category. For held out tasks, it might not match exactly, but let's calculate for known profiles)
        profile_rows = group[group["task_kind"] == "profile"]
        if len(profile_rows) > 0:
            correct = (profile_rows["category"] == profile_rows["task_key"]).sum()
            category_acc = (correct / len(profile_rows)) * 100
        else:
            category_acc = float('nan')

        print(f"\n--- Reasoner: {reasoner_name} ---")
        print(f"  Regret:        {regret_ci.mean:.4f} [{regret_ci.lower:.4f}, {regret_ci.upper:.4f}]{flag}")
        print(f"  Weight L1:     {weight_l1:.4f}")
        print(f"  Fallback Rate: {fallback_rate:.1f}%")
        print(f"  Category Acc:  {category_acc:.1f}%")
        print(f"  Latency:       {latency:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-pools", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=50)
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

    base_seed = config["protocol"]["stable"]["base_seed"]

    models = ["qwen2.5:1.5b", "llama3.2:latest", "qwen2.5:7b", "llama3.1:8b"]
    reasoners = {
        "DirectWeight": DirectWeightReasoner,
        "Classification": ClassificationReasoner,
    }

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()[:8]
    except Exception:
        git_sha = "unknown"
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

    # Use a single run_id for this whole comparison so we can easily group the results
    run_id = f"compare_{uuid.uuid4().hex[:8]}"
    results_dir = project_root / config["paths"]["results_dir"] / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / "model_comparison.csv"
    
    # Let's add "reasoner" to the CSV we'll write, but protocol.py's storage expects STABLE_RESULT_FIELDS.
    # We can inject reasoner into the dataframe after run_stable_protocol finishes.
    
    all_results = []
    
    # First, let's run the uniform baseline to get the regret for comparison
    print("\nRunning Uniform Baseline...")
    storage = CsvStorage(
        csv_path=output_csv,
        key_columns=["run_id", "model", "task_key", "pool_seed", "condition"],
        fieldnames=STABLE_RESULT_FIELDS
    )
    # We need a dummy controller for baseline conditions
    dummy_backend = OllamaBackend("dummy")
    dummy_controller = AgentController(
        backend=dummy_backend,
        attribute_cols=QWS_ATTRIBUTE_COLUMNS,
        memory_path=project_root / "memory_store" / "dummy.jsonl",
        tool_menu=tuple(config["agent"]["tool_menu"]),
        k_memory=0,
        reasoning_strategy=DirectWeightReasoner(),
    )
    
    uniform_results = run_stable_protocol(
        norm_df, QWS_ATTRIBUTE_COLUMNS, storage, dummy_controller,
        run_id=run_id, config_hash=config_hash, git_sha=git_sha,
        model="baseline", n_pools=args.n_pools, pool_size=args.pool_size, base_seed=base_seed,
        is_synthetic_data=is_synthetic, conditions=("uniform",)
    )
    
    uniform_regrets = uniform_results[uniform_results["condition"] == "uniform"]["regret"]
    uniform_ci = bootstrap_ci(uniform_regrets, uniform_results[uniform_results["condition"] == "uniform"]["pool_seed"])
    print(f"Uniform Baseline Regret: {uniform_ci.mean:.4f} [{uniform_ci.lower:.4f}, {uniform_ci.upper:.4f}]")
    
    for model in models:
        model_df = []
        for r_name, r_cls in reasoners.items():
            print(f"\nEvaluating Model: {model} with Reasoner: {r_name}")
            
            # fresh cache/memory each
            memory_path = project_root / "memory_store" / f"memory_{model.replace(':', '_')}_{r_name}.jsonl"
            if memory_path.exists():
                memory_path.unlink()
                
            backend = OllamaBackend(model, options={"seed": 42, "temperature": 0.0})
            controller = AgentController(
                backend=backend,
                attribute_cols=QWS_ATTRIBUTE_COLUMNS,
                memory_path=memory_path,
                tool_menu=tuple(config["agent"]["tool_menu"]),
                k_memory=config["agent"]["k_memory"],
                reasoning_strategy=r_cls(),
            )
            
            res = run_stable_protocol(
                norm_df, QWS_ATTRIBUTE_COLUMNS, storage, controller,
                run_id=run_id, config_hash=config_hash, git_sha=git_sha,
                model=f"{model}_{r_name}", n_pools=args.n_pools, pool_size=args.pool_size, base_seed=base_seed,
                is_synthetic_data=is_synthetic, conditions=("agent_full",)
            )
            
            # Append reasoner column for local processing
            # We filter only the ones that match this run
            run_res = res[res["model"] == f"{model}_{r_name}"].copy()
            run_res["reasoner"] = r_name
            model_df.append(run_res)
            
        combined_model_df = pd.concat(model_df, ignore_index=True)
        print_comparison_report(combined_model_df, model, uniform_ci.mean, uniform_ci.lower)

    print(f"\nAll results saved to {output_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

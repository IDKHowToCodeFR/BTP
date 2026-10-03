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
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding="utf-8")
import json
import hashlib
import uuid
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from agentic_selection.agent import AgentController, OllamaBackend, DirectWeightReasoner, ClassificationReasoner, VotingClassificationReasoner
from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
from agentic_selection.evaluation.protocol import run_stable_protocol, STABLE_RESULT_FIELDS
from agentic_selection.evaluation.storage import CsvStorage
from agentic_selection.tasks import HELD_OUT_TASKS, TASK_PROFILES
from agentic_selection.utils import setup_logging
import yaml
from agentic_selection.evaluation.metrics import bootstrap_ci


def print_comparison_report(df: pd.DataFrame, model_name: str, uniform_df: pd.DataFrame):
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
        # Paired regret diff vs uniform
        joined = group.merge(uniform_df, on=["task_key", "pool_seed"], suffixes=("", "_uni"))
        if not joined.empty:
            diff = joined["regret"] - joined["regret_uni"]
            diff_ci = bootstrap_ci(diff, joined["pool_seed"])
            flag = " (⚠️ ≈ Uniform Baseline)" if diff_ci.lower <= 0 <= diff_ci.upper else ""
        else:
            diff_ci = None
            flag = ""
            
        weight_l1 = group["weight_l1"].mean()
        if weight_l1 < 0.05:
            flag += " (⚠️ Low L1 -> copy?)"

        weight_l1 = group["weight_l1"].mean()
        fallback_rate = group["fallback_triggered"].astype(bool).mean() * 100
        latency = group["latency_seconds"].mean()
        ndcg_5 = group["ndcg_at_5"].mean() if "ndcg_at_5" in group.columns else float('nan')
        kendall = group["kendall_tau"].mean() if "kendall_tau" in group.columns else float('nan')
        cache_hits = len(group) - group["api_calls"].sum()
        actual_calls = group["api_calls"].sum()
        
        # Category Accuracy
        # (Assuming task_key is the true category. For held out tasks, it might not match exactly, but let's calculate for known profiles)
        profile_rows = group[group["task_kind"] == "profile"]
        if len(profile_rows) > 0:
            correct = (profile_rows["category"] == profile_rows["task_key"]).sum()
            category_acc = (correct / len(profile_rows)) * 100
        else:
            category_acc = float('nan')

        # Additional metrics
        entropy = group["entropy"].mean() if "entropy" in group.columns else float('nan')
        # max_weight: if we had the actual weights json we could compute it, but we can compute from weights json if it's there.
        # Actually protocol.py might not have max_weight yet. But the user asked for max_weight. Let's compute it.
        import json
        max_weights = []
        for wj in group["weights_json"].dropna():
            w = json.loads(wj)
            max_weights.append(max(list(w.values()) + [0.0]))
        max_w = np.mean(max_weights) if max_weights else float('nan')

        print(f"\n--- Reasoner: {reasoner_name} ---")
        print(f"  Regret:        {regret_ci.mean:.4f} [{regret_ci.lower:.4f}, {regret_ci.upper:.4f}]{flag}")
        if diff_ci:
            print(f"  Paired Diff:   {diff_ci.mean:.4f} [{diff_ci.lower:.4f}, {diff_ci.upper:.4f}]")
        print(f"  Weight L1:     {weight_l1:.4f}")
        print(f"  Entropy:       {entropy:.4f}")
        print(f"  Max Weight:    {max_w:.4f}")
        print(f"  Fallback Rate: {fallback_rate:.1f}%")
        print(f"  Category Acc:  {category_acc:.1f}%")
        print(f"  NDCG@5:        {ndcg_5:.4f}")
        print(f"  Kendall Tau:   {kendall:.4f}")
        print(f"  Latency:       {latency:.2f}s/call (Actual API calls: {actual_calls}, Cache hits: {cache_hits})")
        
        # Add: mean weights per model for 1 task (streaming) vs ref vs uniform
        streaming_rows = group[group["task_key"] == "streaming"]
        if len(streaming_rows) > 0:
            print(f"\n  [Streaming Task Weights]")
            weights_lists = []
            for wj in streaming_rows["weights_json"].dropna():
                weights_lists.append(json.loads(wj))
            if weights_lists:
                mean_weights = pd.DataFrame(weights_lists).mean().to_dict()
                print(f"    Model Mean:   " + ", ".join(f"{k}: {v:.3f}" for k, v in mean_weights.items() if v > 0.05))
            
            # Ref and uniform
            from agentic_selection.baselines.lookup_table import TASK_LOOKUP_TABLE, GLOBAL_FIXED_WEIGHTS
            print(f"    Reference:    " + ", ".join(f"{k}: {v:.3f}" for k, v in TASK_LOOKUP_TABLE["streaming"].items() if v > 0.05))
            
            # calculate uniform l1 
            ref_w = TASK_LOOKUP_TABLE["streaming"]
            uni_w = GLOBAL_FIXED_WEIGHTS
            uni_l1 = sum(abs(ref_w.get(k, 0.0) - uni_w.get(k, 0.0)) for k in ref_w.keys() | uni_w.keys())
            print(f"    Uniform:      " + ", ".join(f"{k}: {v:.3f}" for k, v in GLOBAL_FIXED_WEIGHTS.items() if v > 0.05) + f" (L1 to ref: {uni_l1:.4f})")
            
            print(f"\n  [5 Raw Outputs for streaming]")
            for i, wj in enumerate(streaming_rows["weights_json"].dropna().head(5)):
                print(f"    {i+1}: {wj}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-pools", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--models", type=str, default="qwen2.5:1.5b,llama3.2:latest,qwen2.5:7b,llama3.1:8b")
    parser.add_argument("--reasoners", type=str, default="DirectWeight,Classification")
    args = parser.parse_args()

    setup_logging()
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(open(project_root / args.config, encoding="utf-8"))

    data_dir = project_root / config["paths"]["data_dir"]
    norm_path = data_dir / "processed" / "qws_normalized.csv"
    if not norm_path.exists():
        print(f"Processed QWS data not found at {norm_path}. Run scripts/02_prepare_data.py first.", file=sys.stderr)
        return 1
    norm_df = pd.read_csv(norm_path, index_col=0)
    is_synthetic = bool(norm_df.get("is_synthetic", pd.Series(dtype=bool)).any())

    base_seed = config["protocol"]["stable"]["base_seed"]

    models_list = [m.strip() for m in args.models.split(",")]
    
    all_reasoners_map = {
        "DirectWeight": DirectWeightReasoner,
        "Classification": ClassificationReasoner,
        "VotingClassification": VotingClassificationReasoner,
        "direct": DirectWeightReasoner,
        "classification": ClassificationReasoner,
        "voting": VotingClassificationReasoner,
    }
    
    reasoners_list = [r.strip() for r in args.reasoners.split(",")]
    reasoners = {r: all_reasoners_map[r] for r in reasoners_list}

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
    print(f"Uniform Baseline Regret: {uniform_ci.mean:.3f} [{uniform_ci.lower:.3f}, {uniform_ci.upper:.3f}]")
    
    print("\nRunning lookup_table Baseline...")
    embedding_knn_results = run_stable_protocol(
        norm_df, QWS_ATTRIBUTE_COLUMNS, storage, dummy_controller,
        run_id=run_id, config_hash=config_hash, git_sha=git_sha,
        model="baseline", n_pools=args.n_pools, pool_size=args.pool_size, base_seed=base_seed,
        is_synthetic_data=is_synthetic, conditions=("lookup_table",)
    )
    
    print("\nRunning global_fixed Baseline...")
    soft_knn_results = run_stable_protocol(
        norm_df, QWS_ATTRIBUTE_COLUMNS, storage, dummy_controller,
        run_id=run_id, config_hash=config_hash, git_sha=git_sha,
        model="baseline", n_pools=args.n_pools, pool_size=args.pool_size, base_seed=base_seed,
        is_synthetic_data=is_synthetic, conditions=("global_fixed",)
    )

    
    total_est_calls = len(models_list) * len(reasoners) * 14 * args.n_pools
    est_hours = total_est_calls * 25.0 / 3600.0
    print(f"\nTotal estimated LLM calls: {total_est_calls} (~{est_hours:.1f} hours at 25s/call)")
    import time
    start_time = time.time()
    
    total_actual_calls = 0
    total_cache_hits = 0

    for model in models_list:
        model_df = []
        for r_name, r_cls in reasoners.items():
            print(f"\nEvaluating Model: {model} with Reasoner: {r_name}")
            
            # fresh cache/memory each
            memory_path = project_root / "memory_store" / f"memory_{model.replace(':', '_')}_{r_name}.jsonl"
            if memory_path.exists():
                memory_path.unlink()
                
            backend = OllamaBackend(model, seed=42, temperature=0.0, num_predict=300, num_ctx=2048)
            controller = AgentController(
                backend=backend,
                attribute_cols=QWS_ATTRIBUTE_COLUMNS,
                memory_path=memory_path,
                tool_menu=tuple(config["agent"]["tool_menu"]),
                k_memory=config["agent"]["k_memory"],
                reasoning_strategy=r_cls(),
            )
            
            # Tuning alpha on first pools
            best_alpha = 0.0
            best_val_reg = float('inf')
            tune_pools = min(3, args.n_pools)
            print(f"  Tuning alpha on {tune_pools} val pools...")
            for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
                controller.alpha_shrinkage = a
                val_res = run_stable_protocol(
                    norm_df, QWS_ATTRIBUTE_COLUMNS, storage, controller,
                    run_id=run_id + f"_tune_{a}", config_hash=config_hash, git_sha=git_sha,
                    model=model, n_pools=tune_pools, pool_size=args.pool_size, base_seed=base_seed-5,
                    is_synthetic_data=is_synthetic, conditions=("agent_full",)
                )
                mean_r = val_res[val_res["condition"]=="agent_full"]["regret"].mean()
                if mean_r < best_val_reg:
                    best_val_reg = mean_r
                    best_alpha = a
            print(f"  Best alpha = {best_alpha}")

            controller.alpha_shrinkage = best_alpha
            controller.cache_hits = 0
            controller.cache_misses = 0

            res = run_stable_protocol(
                norm_df, QWS_ATTRIBUTE_COLUMNS, storage, controller,
                run_id=run_id, config_hash=config_hash, git_sha=git_sha,
                model=f"{model}_{r_name}", n_pools=args.n_pools, pool_size=args.pool_size, base_seed=base_seed,
                is_synthetic_data=is_synthetic, conditions=("agent_full",)
            )
            
            cache_rate = controller.cache_hits / max(1, controller.cache_hits + controller.cache_misses)
            print(f"  API Cache Hit Rate: {cache_rate*100:.1f}% ({controller.cache_hits}/{controller.cache_hits+controller.cache_misses})")
            
            # Append reasoner column for local processing
            # We filter only the ones that match this run
            run_res = res[res["model"] == f"{model}_{r_name}"].copy()
            run_res["reasoner"] = r_name
            model_df.append(run_res)
            
        combined_model_df = pd.concat(model_df, ignore_index=True)
        
        actual_calls = combined_model_df["api_calls"].sum()
        cache_hits = len(combined_model_df) - actual_calls
        total_actual_calls += actual_calls
        total_cache_hits += cache_hits
        
        print_comparison_report(combined_model_df, model, uniform_results[uniform_results["condition"] == "uniform"])

    end_time = time.time()
    
    print("\n=======================================================")
    print(" Baselines Comparison")
    print("=======================================================")
    all_baselines = pd.concat([uniform_results, embedding_knn_results, soft_knn_results])
    for condition in ["lookup_table", "global_fixed", "uniform"]:
        cond_df = all_baselines[all_baselines["condition"] == condition]
        if len(cond_df) > 0:
            regret = cond_df["regret"].mean()
            l1 = cond_df["weight_l1"].mean()
            ndcg = cond_df["ndcg_at_5"].mean()
            tau = cond_df["kendall_tau"].mean()
            
            regret_ci = bootstrap_ci(cond_df["regret"].values, cond_df["pool_seed"])
            ci_str = f"[{regret_ci.lower:.4f}, {regret_ci.upper:.4f}]" if regret_ci else ""
            
            print(f"\n--- {condition} ---")
            print(f"  Regret:        {regret:.4f} {ci_str}")
            print(f"  Weight L1:     {l1:.4f}")
            print(f"  NDCG@5:        {ndcg:.4f}")
            print(f"  Kendall Tau:   {tau:.4f}")
            
    print(f"\nAll results saved to {output_csv}")
    print(f"Total Actual LLM Calls: {total_actual_calls}")
    print(f"Total Cache Hits: {total_cache_hits}")
    print(f"Wall Time: {end_time - start_time:.2f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

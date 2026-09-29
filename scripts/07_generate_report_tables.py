#!/usr/bin/env python3
"""Generate publication-ready Markdown tables for the final report."""

import pandas as pd
from pathlib import Path

LABELS = {
    "global_fixed": "Static Baseline (Global)",
    "lookup_table": "Static Baseline (Task-Specific)",
    "global_every_round": "Static Baseline (Global)",
    "lookup_every_round": "Static Baseline (Task-Specific)",
    "agent_weights_only": "Agent (Weights Only)",
    "agent_full": "Agent (Full Autonomy)"
}

def format_condition(cond: str) -> str:
    return LABELS.get(cond, cond)

def generate_stable_table(stable: pd.DataFrame, out_path: Path):
    if stable.empty:
        return
        
    summary = stable.groupby("condition").agg(
        Mean_Regret=("regret", "mean"),
        Std_Regret=("regret", "std"),
        Latency_s=("latency_seconds", "mean"),
        API_Calls=("api_calls", "mean")
    ).reset_index()
    
    summary["condition"] = summary["condition"].apply(format_condition)
    summary.rename(columns={"condition": "Condition"}, inplace=True)
    
    # Sort logically
    order = [LABELS["global_every_round"], LABELS["lookup_every_round"], LABELS["agent_weights_only"], LABELS["agent_full"]]
    summary["sort_key"] = summary["Condition"].map({k: i for i, k in enumerate(order)})
    summary = summary.sort_values("sort_key").drop(columns=["sort_key"])
    
    # Format numbers
    summary["Mean_Regret"] = summary["Mean_Regret"].map(lambda x: f"{x:.4f}")
    summary["Std_Regret"] = summary["Std_Regret"].map(lambda x: f"{x:.4f}")
    summary["Latency_s"] = summary["Latency_s"].map(lambda x: f"{x:.2f}s")
    summary["API_Calls"] = summary["API_Calls"].map(lambda x: f"{x:.1f}")
    
    markdown = "### Table 1: Performance in Stable Environments\n\n"
    markdown += summary.to_markdown(index=False)
    
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(markdown + "\n\n")

def generate_drift_table(drift: pd.DataFrame, out_path: Path):
    if drift.empty:
        return
        
    # Exclude censored runs for lag
    clean_drift = drift[~drift["censored"].astype(bool)]
    
    summary = clean_drift.groupby("condition").agg(
        Mean_Lag_Rounds=("lag_rounds", "mean")
    ).reset_index()
    
    summary["condition"] = summary["condition"].apply(format_condition)
    summary.rename(columns={"condition": "Condition"}, inplace=True)
    
    # Format numbers
    summary["Mean_Lag_Rounds"] = summary["Mean_Lag_Rounds"].map(lambda x: f"{x:.1f}")
    
    markdown = "### Table 2: Adaptation to QoS Drift (Failover)\n\n"
    markdown += summary.to_markdown(index=False)
    
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(markdown + "\n\n")

def generate_ablation_table(stable: pd.DataFrame, out_path: Path):
    if stable.empty or "task_kind" not in stable.columns:
        return
        
    ablation = stable[stable["condition"].isin(["agent_weights_only", "agent_full"])]
    if ablation.empty:
        return
        
    summary = ablation.groupby(["task_kind", "condition"])["regret"].mean().unstack().fillna(0)
    summary = summary.rename(columns={
        "agent_weights_only": LABELS["agent_weights_only"],
        "agent_full": LABELS["agent_full"]
    })
    
    summary.index = [t.replace("_", " ").title() for t in summary.index]
    summary.index.name = "Task Domain"
    
    # Format numbers
    for col in summary.columns:
        summary[col] = summary[col].map(lambda x: f"{x:.4f}")
        
    markdown = "### Table 3: Autonomy Ablation (Regret by Domain)\n\n"
    markdown += summary.reset_index().to_markdown(index=False)
    
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(markdown + "\n\n")

def main():
    root = Path(__file__).resolve().parents[1]
    tables_dir = root / "results" / "tables"
    
    out_path = tables_dir / "report_tables.md"
    
    # Clear existing file
    if out_path.exists():
        out_path.unlink()
        
    stable = None
    if (tables_dir / "stable_results.csv").exists():
        stable = pd.read_csv(tables_dir / "stable_results.csv")
        
    drift = None
    if (tables_dir / "drift_results.csv").exists():
        drift = pd.read_csv(tables_dir / "drift_results.csv")
        
    if stable is not None:
        generate_stable_table(stable, out_path)
        
    if drift is not None:
        generate_drift_table(drift, out_path)
        
    if stable is not None:
        generate_ablation_table(stable, out_path)
        
    print(f"Generated Markdown tables at: {out_path.relative_to(root)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate extra demo/evaluation artifacts.

This script is intentionally separate from scripts/06_generate_report.py:
06 produces the paper tables, while 08 produces professor-facing visuals
that make the system behavior easier to explain in a BTP evaluation.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CONDITION_DISPLAY = {
    "global_fixed": "Global fixed",
    "lookup_table": "Lookup table",
    "agent_weights_only": "Agent weights-only",
    "agent_full": "Full agent",
}
CONDITION_ORDER = ["global_fixed", "lookup_table", "agent_weights_only", "agent_full"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing required result file: {path}")
    return pd.read_csv(path)


def _save_bar(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df[x], df[y], color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_task_kind_regret(stable: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    agg = (
        stable.groupby(["task_kind", "condition"], as_index=False)["regret"]
        .mean()
        .assign(condition_label=lambda d: d["condition"].map(CONDITION_DISPLAY))
    )
    pivot = agg.pivot(index="task_kind", columns="condition_label", values="regret")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Mean regret by task kind")
    ax.set_ylabel("Mean regret")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return agg


def _plot_latency(stable: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    agents = stable[stable["condition"].isin(["agent_weights_only", "agent_full"])].copy()
    agg = (
        agents.groupby("condition", as_index=False)["latency_seconds"]
        .mean()
        .assign(condition=lambda d: d["condition"].map(CONDITION_DISPLAY))
    )
    _save_bar(
        agg,
        "condition",
        "latency_seconds",
        out_path,
        "Mean agent decision latency",
        "Seconds",
    )
    return agg


def _read_memory_records(memory_path: Path) -> list[dict]:
    if not memory_path.exists():
        return []
    records = []
    for line in memory_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _plot_strategy_frequency(records: list[dict], out_path: Path) -> pd.DataFrame:
    if records:
        strategies = pd.DataFrame(records)["strategy"].value_counts().rename_axis("strategy").reset_index(name="count")
    else:
        strategies = pd.DataFrame({"strategy": ["no_memory_records"], "count": [0]})
    _save_bar(strategies, "strategy", "count", out_path, "Strategy choices in agent memory", "Count")
    return strategies


def _parse_trace(raw: str) -> list[str]:
    try:
        return [str(x) for x in ast.literal_eval(raw)]
    except Exception:
        return []


def _plot_drift_trace(drift: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if drift.empty:
        return pd.DataFrame()

    seed = drift["trial_seed"].iloc[0]
    task = drift["task_key"].iloc[0]
    example = drift[(drift["trial_seed"] == seed) & (drift["task_key"] == task)].copy()
    rows = []
    for _, row in example.iterrows():
        trace = _parse_trace(row["trace_json"])
        for round_idx, service_id in enumerate(trace):
            rows.append(
                {
                    "round": round_idx,
                    "condition": CONDITION_DISPLAY.get(row["condition"], row["condition"]),
                    "still_recommending_degraded": service_id == str(row["target_service_id"]),
                    "target_service_id": str(row["target_service_id"]),
                }
            )
    trace_df = pd.DataFrame(rows)
    if trace_df.empty:
        return trace_df

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for condition, group in trace_df.groupby("condition"):
        ax.step(
            group["round"],
            group["still_recommending_degraded"].astype(int),
            where="post",
            label=condition,
        )
    ax.set_title(f"Drift trace example: {task}, seed {seed}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Still recommending degraded service")
    ax.set_yticks([0, 1])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return trace_df


def _write_markdown(
    out_path: Path,
    stable_summary: pd.DataFrame,
    kind_summary: pd.DataFrame,
    latency_summary: pd.DataFrame,
    drift_summary: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    figures: list[Path],
) -> None:
    parts = [
        "# Extended BTP Analysis",
        "",
        "This file is generated by `scripts/08_generate_extended_analysis.py`.",
        "",
        "## Stable Regret Summary",
        "",
        stable_summary.to_markdown(index=False),
        "",
        "## Regret By Task Kind",
        "",
        kind_summary.to_markdown(index=False),
        "",
        "## Drift Adaptation Summary",
        "",
        drift_summary.to_markdown(index=False),
        "",
        "## Agent Latency Summary",
        "",
        latency_summary.to_markdown(index=False),
        "",
        "## Strategy Frequency From Memory",
        "",
        strategy_summary.to_markdown(index=False),
        "",
        "## Figures Generated",
        "",
    ]
    for fig in figures:
        parts.append(f"- `{fig.as_posix()}`")
    parts.append("")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--memory-path", default="memory_store/agent_memory.jsonl")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_dir = root / args.results_dir
    figures_dir = root / args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    stable = _read_csv(results_dir / "stable_results.csv")
    drift = _read_csv(results_dir / "drift_results.csv")
    records = _read_memory_records(root / args.memory_path)

    stable_summary = (
        stable.groupby("condition", as_index=False)["regret"]
        .mean()
        .assign(condition=lambda d: d["condition"].map(CONDITION_DISPLAY))
        .sort_values("regret")
    )
    drift_clean = drift[~drift["censored"].astype(bool)].copy()
    drift_clean["lag_rounds"] = drift_clean["lag_rounds"].astype(float)
    drift_summary = (
        drift_clean.groupby("condition", as_index=False)["lag_rounds"]
        .mean()
        .assign(condition=lambda d: d["condition"].map(CONDITION_DISPLAY))
        .sort_values("lag_rounds")
    )

    figures = [
        figures_dir / "regret_by_task_kind.png",
        figures_dir / "agent_latency_by_condition.png",
        figures_dir / "strategy_selection_frequency.png",
        figures_dir / "drift_trace_example.png",
    ]
    kind_summary = _plot_task_kind_regret(stable, figures[0])
    latency_summary = _plot_latency(stable, figures[1])
    strategy_summary = _plot_strategy_frequency(records, figures[2])
    _plot_drift_trace(drift, figures[3])

    out_md = results_dir / "extended_analysis.md"
    _write_markdown(
        out_md,
        stable_summary,
        kind_summary,
        latency_summary,
        drift_summary,
        strategy_summary,
        figures,
    )

    print(f"Wrote extended analysis to {out_md}")
    for fig in figures:
        print(f"Wrote figure to {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

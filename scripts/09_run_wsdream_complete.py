#!/usr/bin/env python3
"""Run the complete offline WS-DREAM QoS evaluation.

This covers the two canonical QoS releases in the official Zenodo record:
Dataset #1 as a full 30-pool static experiment and Dataset #2 as both a
30-pool static experiment and a 64-slice temporal experiment.

The agent conditions use DemoHeuristicBackend so this script can run without
API credentials. Its outputs are cross-dataset engineering evidence, not
live-LLM results.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agentic_selection.agent import AgentController, DemoHeuristicBackend
from agentic_selection.baselines import topsis
from agentic_selection.baselines.wsdream_lookup_table import (
    WSDREAM_ATTRIBUTE_COLUMNS,
    WSDREAM_GLOBAL_FIXED_WEIGHTS,
    WSDREAM_TASK_LOOKUP_TABLE,
    get_wsdream_lookup_weights,
)
from agentic_selection.data.preprocessing import normalize_benefit_oriented
from agentic_selection.data.wsdream_loader import load_wsdream_dataset2_aggregated
from agentic_selection.evaluation import run_stable_protocol
from agentic_selection.tasks.wsdream_profiles import WSDREAM_TASK_PROFILES

TEMPORAL_CONDITIONS = (
    "global_static",
    "lookup_static",
    "global_dynamic",
    "lookup_dynamic",
    "agent_weights_only",
    "agent_full",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def load_or_build_aggregates(root: Path, rebuild: bool) -> dict:
    raw_dir = root / "data" / "raw" / "wsdream2" / "dataset2"
    source_paths = [raw_dir / "rtdata.txt", raw_dir / "tpdata.txt"]
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Dataset #2 files: {missing}")

    cache_path = root / "data" / "processed" / "wsdream2_service_time.pkl"
    source_sizes = {path.name: path.stat().st_size for path in source_paths}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if cached.get("source_sizes") == source_sizes:
            print(f"Loading verified Dataset #2 aggregate cache: {cache_path}")
            return cached

    print("Aggregating Dataset #2 response time and throughput in chunks ...")
    aggregated = load_wsdream_dataset2_aggregated(raw_dir)
    aggregated["source_sizes"] = source_sizes
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(aggregated, cache_path)
    print(f"Cached service/time aggregates: {cache_path}")
    return aggregated


def weighted_service_mean(values: pd.DataFrame, counts: pd.DataFrame) -> pd.Series:
    weighted_sum = (values.fillna(0.0) * counts).sum(axis=0)
    total_count = counts.sum(axis=0)
    return weighted_sum.div(total_count.where(total_count > 0))


def normalize_temporal(
    rt: pd.DataFrame,
    tp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rt_min, rt_max = np.nanmin(rt.to_numpy()), np.nanmax(rt.to_numpy())
    tp_min, tp_max = np.nanmin(tp.to_numpy()), np.nanmax(tp.to_numpy())
    rt_norm = 1.0 - (rt - rt_min) / (rt_max - rt_min)
    tp_norm = (tp - tp_min) / (tp_max - tp_min)
    return rt_norm.fillna(rt_norm.stack().mean()), tp_norm.fillna(tp_norm.stack().mean())


def build_slice_pool(
    rt_norm: pd.DataFrame,
    tp_norm: pd.DataFrame,
    time_slice: int,
    service_ids: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "response_time": rt_norm.loc[time_slice, service_ids].to_numpy(),
            "throughput": tp_norm.loc[time_slice, service_ids].to_numpy(),
        },
        index=pd.Index(service_ids, name="service_id"),
    )


def reference_regret(pool: pd.DataFrame, chosen: object, weights: dict) -> float:
    values = pool.loc[:, WSDREAM_ATTRIBUTE_COLUMNS].to_numpy(dtype=float)
    weight_vector = np.array([weights[col] for col in WSDREAM_ATTRIBUTE_COLUMNS])
    scores = pd.Series(values @ weight_vector, index=pool.index)
    return float(scores.max() - scores.loc[chosen])


def run_temporal_protocol(
    rt_norm: pd.DataFrame,
    tp_norm: pd.DataFrame,
    controller: AgentController,
    output_csv: Path,
    n_pools: int,
    pool_size: int,
    base_seed: int,
) -> pd.DataFrame:
    eligible = np.array(sorted(set(rt_norm.columns) & set(tp_norm.columns)))
    if pool_size > len(eligible):
        raise ValueError(f"pool_size={pool_size} exceeds {len(eligible)} services")

    rows = []
    n_slices = len(rt_norm.index)
    for profile in WSDREAM_TASK_PROFILES:
        reference_weights = WSDREAM_TASK_LOOKUP_TABLE[profile.key]
        for pool_index in range(n_pools):
            seed = base_seed + pool_index
            rng = np.random.default_rng(seed)
            service_ids = np.sort(rng.choice(eligible, size=pool_size, replace=False))
            pools = [
                build_slice_pool(rt_norm, tp_norm, time_slice, service_ids)
                for time_slice in rt_norm.index
            ]

            initial = pools[0]
            initial_choices = {
                "global_static": topsis(
                    initial,
                    WSDREAM_GLOBAL_FIXED_WEIGHTS,
                    WSDREAM_ATTRIBUTE_COLUMNS,
                ).idxmax(),
                "lookup_static": topsis(
                    initial,
                    reference_weights,
                    WSDREAM_ATTRIBUTE_COLUMNS,
                ).idxmax(),
            }

            for condition in TEMPORAL_CONDITIONS:
                choices = []
                regrets = []
                latency = 0.0
                api_calls = 0
                for pool in pools:
                    if condition in initial_choices:
                        chosen = initial_choices[condition]
                    elif condition == "global_dynamic":
                        chosen = topsis(
                            pool,
                            WSDREAM_GLOBAL_FIXED_WEIGHTS,
                            WSDREAM_ATTRIBUTE_COLUMNS,
                        ).idxmax()
                    elif condition == "lookup_dynamic":
                        chosen = topsis(
                            pool,
                            reference_weights,
                            WSDREAM_ATTRIBUTE_COLUMNS,
                        ).idxmax()
                    else:
                        override = "topsis" if condition == "agent_weights_only" else None
                        decision = controller.decide(
                            profile.description,
                            pool,
                            strategy_override=override,
                            use_memory=False,
                        )
                        chosen = decision.top_service_id()
                        latency += decision.latency_seconds
                        api_calls += decision.api_calls

                    choices.append(chosen)
                    regrets.append(reference_regret(pool, chosen, reference_weights))

                rows.append(
                    {
                        "task_key": profile.key,
                        "pool_seed": seed,
                        "condition": condition,
                        "n_time_slices": n_slices,
                        "mean_regret": float(np.mean(regrets)),
                        "p95_regret": float(np.quantile(regrets, 0.95)),
                        "max_regret": float(np.max(regrets)),
                        "top1_accuracy": float(np.mean(np.asarray(regrets) <= 1e-12)),
                        "recommendation_switches": int(
                            sum(left != right for left, right in zip(choices, choices[1:]))
                        ),
                        "total_latency_seconds": latency,
                        "api_calls": api_calls,
                        "backend": "demo_heuristic",
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


def aggregate_condition(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.groupby("condition", as_index=False)[metric]
        .mean()
        .sort_values(metric)
        .reset_index(drop=True)
    )


def markdown_table(df: pd.DataFrame, digits: int = 6) -> str:
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def write_summary(
    root: Path,
    dataset1: pd.DataFrame,
    dataset2_static: pd.DataFrame,
    temporal: pd.DataFrame,
) -> None:
    ds1 = aggregate_condition(dataset1, "regret")
    ds2 = aggregate_condition(dataset2_static, "regret")
    ds1_profile = (
        dataset1.groupby(["task_key", "condition"], as_index=False)["regret"]
        .mean()
        .sort_values(["task_key", "regret"])
    )
    ds2_profile = (
        dataset2_static.groupby(["task_key", "condition"], as_index=False)["regret"]
        .mean()
        .sort_values(["task_key", "regret"])
    )
    dynamic = (
        temporal.groupby("condition", as_index=False)
        .agg(
            mean_regret=("mean_regret", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            mean_switches=("recommendation_switches", "mean"),
        )
        .sort_values("mean_regret")
    )
    dynamic_profile = (
        temporal.groupby(["task_key", "condition"], as_index=False)
        .agg(
            mean_regret=("mean_regret", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            mean_switches=("recommendation_switches", "mean"),
        )
        .sort_values(["task_key", "mean_regret"])
    )

    summary_path = root / "results" / "tables" / "wsdream_complete_summary.md"
    summary = f"""# Complete WS-DREAM QoS Results

These experiments use the deterministic offline demo controller. They validate
the complete data and evaluation pipeline, but they are not live-LLM evidence.

## Coverage

- Dataset #1: 339 users, 5,825 services, one static snapshot, 2 QoS attributes.
- Dataset #2: 142 users, 4,500 services, 64 time slices, 2 QoS attributes.
- Full static protocol: 30 pools per task, 20 services per pool, 2 task profiles.
- Temporal protocol: the same 30 pools tracked through all 64 observed slices.

## Dataset #1 Static Mean Regret

{markdown_table(ds1)}

### Dataset #1 by task profile

{markdown_table(ds1_profile)}

## Dataset #2 Static Mean Regret

{markdown_table(ds2)}

### Dataset #2 static results by task profile

{markdown_table(ds2_profile)}

## Dataset #2 Temporal Results

{markdown_table(dynamic)}

### Dataset #2 temporal results by task profile

{markdown_table(dynamic_profile)}

## Interpretation

Dynamic baselines re-rank every time slice and therefore isolate ranking quality.
Static baselines hold the initial recommendation and represent a deployment that
is not automatically re-evaluated. Agent conditions re-evaluate every slice.
Regret uses researcher-authored task weights as the reference, not external
ground truth.

## Availability Audit

The official Zenodo record contains Dataset #1 and Dataset #2. The historical
ICWS 2012 and Cloud 2013 download links currently return HTTP 404 and are not
included in the Zenodo record. Log and review datasets are separate research
modalities and do not contain the QoS matrices required by this selection method.
"""
    summary_path.write_text(summary, encoding="utf-8")

    figure_path = root / "results" / "figures" / "wsdream_complete_comparison.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, data, metric, title in (
        (axes[0], ds1, "regret", "Dataset #1 static regret"),
        (axes[1], ds2, "regret", "Dataset #2 static regret"),
        (axes[2], dynamic, "mean_regret", "Dataset #2 temporal regret"),
    ):
        axis.barh(data["condition"], data[metric], color="#287271")
        axis.set_title(title)
        axis.set_xlabel("Mean regret, lower is better")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote figure: {figure_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pools", type=positive_int, default=30)
    parser.add_argument("--pool-size", type=positive_int, default=20)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset1_path = root / "results" / "tables" / "wsdream_demo_results.csv"
    if not dataset1_path.exists():
        raise FileNotFoundError(
            "Run scripts/07_run_wsdream_validation.py --n-pools 30 "
            "--yes --demo-backend first"
        )
    dataset1 = pd.read_csv(dataset1_path)

    aggregated = load_or_build_aggregates(root, args.rebuild_cache)
    rt = aggregated["rt"]
    tp = aggregated["tp"]
    rt_counts = aggregated["rt_counts"]
    tp_counts = aggregated["tp_counts"]
    print(f"Dataset #2 aggregate shape: {rt.shape}; missing means: {int(rt.isna().sum().sum() + tp.isna().sum().sum())}")

    static_raw = pd.DataFrame(
        {
            "response_time": weighted_service_mean(rt, rt_counts),
            "throughput": weighted_service_mean(tp, tp_counts),
        }
    ).dropna()
    static_normalized = normalize_benefit_oriented(
        static_raw,
        attribute_cols=WSDREAM_ATTRIBUTE_COLUMNS,
        cost_attributes={"response_time"},
    )

    controller = AgentController(
        backend=DemoHeuristicBackend(WSDREAM_ATTRIBUTE_COLUMNS),
        attribute_cols=WSDREAM_ATTRIBUTE_COLUMNS,
        memory_path=root / "memory_store" / "agent_memory_wsdream_complete.jsonl",
        tool_menu=("weighted_sum", "topsis", "skyline_then_topsis"),
        k_memory=3,
    )
    static_output = root / "results" / "tables" / "wsdream2_demo_stable_results.csv"
    dataset2_static = run_stable_protocol(
        static_normalized,
        WSDREAM_ATTRIBUTE_COLUMNS,
        static_output,
        controller,
        n_pools=args.n_pools,
        pool_size=args.pool_size,
        base_seed=10_000,
        is_synthetic_data=False,
        task_profiles=WSDREAM_TASK_PROFILES,
        held_out_tasks=[],
        global_fixed_weights=WSDREAM_GLOBAL_FIXED_WEIGHTS,
        lookup_table=WSDREAM_TASK_LOOKUP_TABLE,
        lookup_weights_fn=get_wsdream_lookup_weights,
    )

    rt_norm, tp_norm = normalize_temporal(rt, tp)
    temporal_output = root / "results" / "tables" / "wsdream2_demo_temporal_results.csv"
    temporal = run_temporal_protocol(
        rt_norm,
        tp_norm,
        controller,
        temporal_output,
        n_pools=args.n_pools,
        pool_size=args.pool_size,
        base_seed=12_000,
    )
    write_summary(root, dataset1, dataset2_static, temporal)
    print(f"Dataset #1 rows: {len(dataset1)}")
    print(f"Dataset #2 static rows: {len(dataset2_static)}")
    print(f"Dataset #2 temporal rows: {len(temporal)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

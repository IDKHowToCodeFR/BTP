#!/usr/bin/env python3
"""Generate presentation-ready figures from completed experiment results."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from agentic_selection.evaluation.plot_style import (
    CANVAS,
    CONDITION_COLORS,
    CONDITION_LABELS,
    INK,
    MUTED,
    apply_publication_style,
    save_figure,
    style_axis,
)

QWS_ORDER = ["global_fixed", "lookup_table", "agent_weights_only", "agent_full", "rag_agent"]
TEMPORAL_ORDER = [
    "global_static",
    "lookup_static",
    "global_dynamic",
    "lookup_dynamic",
    "agent_weights_only",
    "agent_full",
    "rag_agent",
]
TASK_LABELS = {
    "streaming": "Streaming",
    "batch_processing": "Batch processing",
    "financial_transaction": "Financial transaction",
    "low_cost_prototype": "Low-cost prototype",
    "iot_telemetry_ingestion": "IoT telemetry ingestion",
    "compliance_sensitive_backend": "Compliance-sensitive backend",
}

apply_publication_style()


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return pd.read_csv(path)

def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _condition_summary(df: pd.DataFrame, metric: str, order: list[str]) -> pd.DataFrame:
    summary = df.groupby("condition", as_index=False)[metric].mean().set_index("condition").reindex(order).dropna()
    summary["label"] = [CONDITION_LABELS.get(key, key) for key in summary.index]
    summary["color"] = [CONDITION_COLORS.get(key, "#7B8794") for key in summary.index]
    return summary


def _label_horizontal_bars(ax, bars, labels: list[str], maximum: float | None = None) -> None:
    values = [bar.get_width() for bar in bars]
    peak = max(values) if values else 0.0
    padding = peak * 0.025 if peak else 0.02
    right = maximum if maximum is not None else peak * 1.2
    ax.set_xlim(0, max(right, peak + padding * 4))
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_width() + padding,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            color=INK,
            fontsize=8.5,
            fontweight="bold",
        )


def _barh_panel(ax, summary: pd.DataFrame, metric: str, title: str, xlabel: str, fmt: str) -> None:
    bars = ax.barh(summary["label"], summary[metric], color=summary["color"], height=0.6, zorder=3)
    ax.invert_yaxis()
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    _label_horizontal_bars(ax, bars, [format(value, fmt) for value in summary[metric]])
    style_axis(ax)


def plot_results_overview(
    stable: pd.DataFrame,
    drift: pd.DataFrame,
    temporal: pd.DataFrame | None,
    out_path: Path,
) -> None:
    qws = _condition_summary(stable, "regret", QWS_ORDER)
    clean_drift = drift[~drift["censored"].astype(bool)].copy()
    lag = _condition_summary(clean_drift, "lag_rounds", QWS_ORDER)

    has_temporal = temporal is not None
    if has_temporal:
        temporal_regret = _condition_summary(temporal, "mean_regret", TEMPORAL_ORDER)
        temporal_accuracy = _condition_summary(temporal, "top1_accuracy", TEMPORAL_ORDER)
        temporal_accuracy["accuracy_percent"] = temporal_accuracy["top1_accuracy"] * 100

    fig, axes = plt.subplots(2 if has_temporal else 1, 2, figsize=(15.5, 10.2 if has_temporal else 5.5))
    axes = np.atleast_2d(axes)
    
    fig.suptitle("Context-Aware Cloud Service Selection | Results Overview", x=0.055, ha="left")
    fig.text(
        0.055,
        0.938 if has_temporal else 0.88,
        "QWS decision quality and adaptation" + (", plus WS-DREAM temporal validation" if has_temporal else ""),
        color=MUTED,
        fontsize=11,
    )

    _barh_panel(axes[0, 0], qws, "regret", "A. QWS Decision Quality", "Mean regret - lower is better", ".4f")
    _barh_panel(axes[0, 1], lag, "lag_rounds", "B. Adaptation After QoS Drift", "Mean lag in rounds - lower is better", ".1f")

    if has_temporal:
        bars = axes[1, 0].barh(
            temporal_accuracy["label"],
            temporal_accuracy["accuracy_percent"],
            color=temporal_accuracy["color"],
            height=0.6,
            zorder=3,
        )
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_title("C. WS-DREAM Temporal Top-1 Accuracy", pad=10)
        axes[1, 0].set_xlabel("Optimal-service selections (%) - higher is better")
        _label_horizontal_bars(
            axes[1, 0],
            bars,
            [f"{value:.1f}%" for value in temporal_accuracy["accuracy_percent"]],
            maximum=100.0,
        )
        style_axis(axes[1, 0])

        _barh_panel(
            axes[1, 1],
            temporal_regret,
            "mean_regret",
            "D. WS-DREAM Temporal Regret",
            "Mean cumulative regret - lower is better",
            ".4f",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035 if has_temporal else -0.05), ncol=4)
    fig.tight_layout(rect=(0.03, 0.105 if has_temporal else 0.05, 0.99, 0.88 if has_temporal else 0.8), w_pad=4.2)
    save_figure(fig, out_path, dpi=240)

    fig.text(
        0.055,
        0.018,
        "WS-DREAM agent conditions use the deterministic offline controller; results validate the pipeline rather than a live LLM.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.035, 0.05, 0.99, 0.91), h_pad=3.3, w_pad=3.2)
    save_figure(fig, out_path, dpi=240)


def plot_qws_heatmap(stable: pd.DataFrame, out_path: Path) -> None:
    task_meta = stable[["task_key", "task_kind"]].drop_duplicates()
    known = sorted(task_meta.loc[task_meta["task_kind"] == "profile", "task_key"])
    held_out = sorted(task_meta.loc[task_meta["task_kind"] == "held_out", "task_key"])
    task_order = known + held_out

    pivot = stable.groupby(["task_key", "condition"])["regret"].mean().unstack().reindex(task_order)
    pivot = pivot.reindex(columns=QWS_ORDER)
    display_tasks = [TASK_LABELS.get(key, key.replace("_", " ").title()) for key in pivot.index]
    display_conditions = [CONDITION_LABELS[key] for key in pivot.columns]

    cmap = plt.get_cmap("coolwarm")
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.4, 8.8))
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0, vmax=float(np.nanmax(values)))
    ax.set_title("Where Each Method Wins and Struggles", pad=22)
    ax.text(
        0,
        1.02,
        "QWS mean regret by task - paler cells are better",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    ax.set_xticks(np.arange(len(display_conditions)), display_conditions)
    ax.set_yticks(np.arange(len(display_tasks)), display_tasks)
    ax.tick_params(axis="x", labelrotation=0, pad=10)
    ax.tick_params(axis="y", pad=8)

    threshold = float(np.nanmax(values)) * 0.58
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="#FFFFFF" if value > threshold else INK,
                fontsize=8.5,
                fontweight="bold" if value == np.nanmin(values[row]) else "normal",
            )

    ax.set_xticks(np.arange(-0.5, len(display_conditions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(display_tasks), 1), minor=True)
    ax.grid(which="minor", color=CANVAS, linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    if known:
        boundary = len(known) - 0.5
        ax.axhline(boundary, color=INK, linewidth=2.2)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.027, pad=0.025)
    colorbar.set_label("Mean regret - lower is better")
    colorbar.outline.set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=240)


def plot_qws_distribution(stable: pd.DataFrame, out_path: Path) -> None:
    data = [stable.loc[stable["condition"] == key, "regret"].to_numpy(dtype=float) for key in QWS_ORDER]
    labels = [CONDITION_LABELS[key].replace(": ", ":\n") for key in QWS_ORDER]
    colors = [CONDITION_COLORS[key] for key in QWS_ORDER]
    positions = np.arange(1, len(QWS_ORDER) + 1)

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    violin = ax.violinplot(data, positions=positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 2},
        whiskerprops={"color": MUTED, "linewidth": 1.2},
        capprops={"color": MUTED, "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        patch.set_edgecolor("none")

    rng = np.random.default_rng(42)
    for position, values, color in zip(positions, data, colors):
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(position + jitter, values, s=18, color=color, alpha=0.38, edgecolors="none", zorder=3)
        ax.scatter(
            position,
            float(np.mean(values)),
            marker="D",
            s=55,
            color=color,
            edgecolors="#FFFFFF",
            linewidth=1.2,
            zorder=5,
        )

    ax.set_title("Trial-Level Regret Distribution", pad=18)
    ax.text(
        0,
        1.02,
        "Each dot is one evaluated service pool; diamonds mark means",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    ax.set_ylabel("Regret - lower is better")
    ax.set_xticks(positions, labels)
    ax.set_ylim(bottom=0)
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    save_figure(fig, out_path, dpi=240)


def plot_wsdream_temporal_behavior(temporal: pd.DataFrame, out_path: Path) -> None:
    grouped = (
        temporal.groupby("condition", as_index=False)
        .agg(
            mean_regret=("mean_regret", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            recommendation_switches=("recommendation_switches", "mean"),
        )
        .set_index("condition")
        .reindex(TEMPORAL_ORDER)
    )
    grouped["label"] = [CONDITION_LABELS[key] for key in grouped.index]
    grouped["color"] = [CONDITION_COLORS[key] for key in grouped.index]
    grouped["top1_percent"] = grouped["top1_accuracy"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.1))
    fig.suptitle("How Methods Behave Across 64 Changing QoS Slices", x=0.045, ha="left")
    fig.text(
        0.045,
        0.91,
        "WS-DREAM Dataset #2 temporal evaluation",
        color=MUTED,
        fontsize=11,
    )

    configurations = (
        ("mean_regret", "Decision regret", "Mean regret - lower is better", ".5f", None),
        ("top1_percent", "Top-1 accuracy", "Optimal selections (%) - higher is better", ".1f", 105),
        ("recommendation_switches", "Adaptation activity", "Mean recommendation switches", ".1f", None),
    )
    for ax, (metric, title, xlabel, fmt, maximum) in zip(axes, configurations):
        bars = ax.barh(grouped["label"], grouped[metric], color=grouped["color"], height=0.6, zorder=3)
        ax.invert_yaxis()
        ax.set_title(title, pad=11)
        ax.set_xlabel(xlabel)
        labels = [format(value, fmt) + ("%" if metric == "top1_percent" else "") for value in grouped[metric]]
        _label_horizontal_bars(ax, bars, labels, maximum=maximum)
        style_axis(ax)

    fig.text(
        0.045,
        0.02,
        "Static baselines never switch. Agent conditions use the deterministic offline controller.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.025, 0.055, 0.99, 0.88), w_pad=3.3)
    save_figure(fig, out_path, dpi=240)


def plot_wsdream_profiles(dataset1: pd.DataFrame, dataset2: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.8), sharey=True)
    fig.suptitle("WS-DREAM Performance by Workload Profile", x=0.05, ha="left")
    fig.text(
        0.05,
        0.91,
        "Static evaluation on both canonical QoS datasets - lower regret is better",
        color=MUTED,
        fontsize=11,
    )

    for ax, data, title in zip(axes, (dataset1, dataset2), ("Dataset #1", "Dataset #2")):
        grouped = data.groupby(["task_key", "condition"])["regret"].mean().unstack().reindex(columns=QWS_ORDER)
        task_keys = list(grouped.index)
        centers = np.arange(len(task_keys))
        height = 0.17
        offsets = np.linspace(-0.27, 0.27, len(QWS_ORDER))
        all_bars = []
        all_values = []
        for offset, condition in zip(offsets, QWS_ORDER):
            values = grouped[condition].to_numpy(dtype=float)
            bars = ax.barh(
                centers + offset,
                values,
                height=height,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
                zorder=3,
            )
            all_bars.extend(bars)
            all_values.extend(values)
        ax.set_yticks(centers, [TASK_LABELS.get(key, key.replace("_", " ").title()) for key in task_keys])
        ax.invert_yaxis()
        ax.set_title(title, pad=11)
        ax.set_xlabel("Mean regret")
        peak = max(all_values)
        ax.set_xlim(0, peak * 1.24)
        for bar in all_bars:
            value = bar.get_width()
            ax.text(
                value + peak * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                ha="left",
                color=INK,
                fontsize=7.5,
            )
        style_axis(ax)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4)
    fig.tight_layout(rect=(0.03, 0.105, 0.99, 0.88), w_pad=4.2)
    save_figure(fig, out_path, dpi=240)


def write_figure_index(out_path: Path, figures: list[tuple[str, Path, str]]) -> None:
    lines = [
        "# Presentation Figure Set",
        "",
        "Generated by `scripts/10_generate_presentation_figures.py` from the saved experiment CSV files.",
        "",
    ]
    for title, path, caption in figures:
        lines.extend([f"## {title}", "", caption, "", f"`{path.as_posix()}`", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tables = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    stable = _read_required(tables / "stable_results.csv")
    drift = _read_required(tables / "drift_results.csv")
    dataset1 = _read_optional(tables / "wsdream_demo_results.csv")
    dataset2 = _read_optional(tables / "wsdream2_demo_stable_results.csv")
    temporal = _read_optional(tables / "wsdream2_demo_temporal_results.csv")

    figures = [
        (
            "Project Results Overview",
            figures_dir / "project_results_overview.png",
            "One-slide summary of decision quality, drift adaptation, temporal accuracy, and temporal regret.",
        ),
        (
            "QWS Task Heatmap",
            figures_dir / "qws_regret_heatmap.png",
            "Task-level regret matrix separating known profiles from held-out requests.",
        ),
        (
            "QWS Regret Distribution",
            figures_dir / "qws_regret_distribution.png",
            "Trial-level distributions showing variance, central tendency, and outliers.",
        ),
    ]

    if temporal is not None:
        figures.append((
            "WS-DREAM Temporal Behavior",
            figures_dir / "wsdream_temporal_behavior.png",
            "Joint view of temporal regret, top-1 accuracy, and recommendation switching.",
        ))

    if dataset1 is not None and dataset2 is not None:
        figures.append((
            "WS-DREAM Workload Profiles",
            figures_dir / "wsdream_profile_comparison.png",
            "Static results for both WS-DREAM datasets split by workload profile.",
        ))

    plot_results_overview(stable, drift, temporal, figures_dir / "project_results_overview.png")
    plot_qws_heatmap(stable, figures_dir / "qws_regret_heatmap.png")
    plot_qws_distribution(stable, figures_dir / "qws_regret_distribution.png")
    
    if temporal is not None:
        plot_wsdream_temporal_behavior(temporal, figures_dir / "wsdream_temporal_behavior.png")
    if dataset1 is not None and dataset2 is not None:
        plot_wsdream_profiles(dataset1, dataset2, figures_dir / "wsdream_profile_comparison.png")

    write_figure_index(tables / "presentation_figures.md", figures)

    for _, path, _ in figures:
        print(f"Wrote figure: {path}")
    print(f"Wrote figure index: {tables / 'presentation_figures.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

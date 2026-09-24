"""Report generation (paper §5): turns the raw per-trial CSVs produced by
evaluation/protocol.py into the paper's Table 5.1-5.3 and figures.

Guard rail: if ANY row in the input data is flagged is_synthetic_data
(meaning the real QWS download failed at some point and the synthetic
fallback generator was used instead -- see data/download.py), this
module's `render_paper_tables` refuses to produce a "final" report
unless explicitly overridden, and always stamps the output with a
visible SYNTHETIC-DATA warning. This is the enforcement mechanism behind
the promise made throughout this codebase: synthetic fallback data is
for smoke-testing the pipeline only, never for reporting results.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agentic_selection.evaluation.metrics import mean_confidence_interval
from agentic_selection.evaluation.plot_style import (
    CONDITION_COLORS,
    add_bar_labels,
    apply_publication_style,
    save_figure,
    style_axis,
)

CONDITION_DISPLAY_NAMES = {
    "global_fixed": "Global fixed",
    "lookup_table": "Lookup table",
    "agent_weights_only": "Agent (weights only)",
    "agent_full": "Agent (full)",
}
CONDITION_ORDER = ["global_fixed", "lookup_table", "agent_weights_only", "agent_full"]


def _check_synthetic(df: pd.DataFrame, allow_synthetic: bool) -> bool:
    has_synth = bool(df.get("is_synthetic_data", pd.Series(dtype=bool)).any())
    if has_synth and not allow_synthetic:
        raise RuntimeError(
            "Refusing to render a report: some trials were computed on "
            "SYNTHETIC fallback data (real dataset download must have "
            "failed at some point). Re-run scripts/01_download_data.py "
            "to get the real QWS dataset, then re-run the experiment "
            "scripts before generating a report. If you really want a "
            "preview anyway (e.g. for a smoke test), pass "
            "allow_synthetic=True / --allow-synthetic."
        )
    return has_synth


def regret_table(stable_df: pd.DataFrame, allow_synthetic: bool = False) -> pd.DataFrame:
    """Paper Table 5.1: mean regret +/- 95% CI, by condition and task."""
    has_synth = _check_synthetic(stable_df, allow_synthetic)

    rows = []
    for task_key, group in stable_df.groupby("task_key"):
        row = {"task": task_key}
        for cond in CONDITION_ORDER:
            sub = group[group["condition"] == cond]["regret"]
            ci = mean_confidence_interval(sub.tolist()) if len(sub) else None
            row[CONDITION_DISPLAY_NAMES[cond]] = str(ci) if ci else "n/a"
        rows.append(row)
    table = pd.DataFrame(rows).set_index("task")
    if has_synth:
        table.attrs["warning"] = "COMPUTED ON SYNTHETIC FALLBACK DATA -- NOT REAL RESULTS"
    return table


def drift_table(drift_df: pd.DataFrame, allow_synthetic: bool = False) -> pd.DataFrame:
    """Paper Table 5.2: mean adaptation lag by condition, with censored
    trials (target never stopped being recommended within the window)
    reported separately rather than silently dropped or treated as 0.
    """
    has_synth = _check_synthetic(drift_df, allow_synthetic)

    rows = []
    for cond in CONDITION_ORDER:
        sub = drift_df[drift_df["condition"] == cond]
        censored = sub["censored"].astype(bool)
        n_censored = int(censored.sum())
        observed = sub.loc[~censored, "lag_rounds"].astype(float)
        ci = mean_confidence_interval(observed.tolist()) if len(observed) else None
        rows.append(
            {
                "condition": CONDITION_DISPLAY_NAMES[cond],
                "mean_adaptation_lag": str(ci) if ci else "n/a",
                "n_trials": len(sub),
                "n_censored": n_censored,
                "censored_note": (
                    f"{n_censored}/{len(sub)} trials never adapted within the "
                    f"observation window (right-censored, excluded from the mean "
                    f"above -- report this fraction, don't just drop it silently)"
                    if n_censored > 0
                    else ""
                ),
            }
        )
    table = pd.DataFrame(rows).set_index("condition")
    if has_synth:
        table.attrs["warning"] = "COMPUTED ON SYNTHETIC FALLBACK DATA -- NOT REAL RESULTS"
    return table


def operational_table(stable_df: pd.DataFrame, allow_synthetic: bool = False) -> pd.DataFrame:
    """Paper Table 5.3: fallback trigger rate, latency, API calls -- agent
    conditions only (static baselines have no LLM call, no fallback)."""
    has_synth = _check_synthetic(stable_df, allow_synthetic)

    rows = []
    for cond in ("agent_weights_only", "agent_full"):
        sub = stable_df[stable_df["condition"] == cond]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "condition": CONDITION_DISPLAY_NAMES[cond],
                "fallback_trigger_rate": float(sub["fallback_triggered"].astype(bool).mean()),
                "mean_latency_seconds": float(sub["latency_seconds"].mean()),
                "mean_api_calls": float(sub["api_calls"].mean()),
                "n_decisions": len(sub),
            }
        )
    table = pd.DataFrame(rows).set_index("condition")
    if has_synth:
        table.attrs["warning"] = "COMPUTED ON SYNTHETIC FALLBACK DATA -- NOT REAL RESULTS"
    return table


def render_markdown_tables(
    stable_df: pd.DataFrame,
    drift_df: pd.DataFrame | None,
    allow_synthetic: bool = False,
    wsdream_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> str:
    """Render Tables 5.1-5.3 (and, if wsdream_df is given, 5.4) as
    GitHub-flavored markdown, ready to paste into
    paper/agentic_cloud_selection_paper.md replacing the Section 5
    placeholders."""
    parts = []
    
    if config:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts.append(
            f"<!--\n"
            f"Report Generated: {timestamp}\n"
            f"LLM Provider: {config.get('llm', {}).get('provider', 'unknown')}\n"
            f"LLM Model: {config.get('llm', {}).get('model', 'unknown')}\n"
            f"Note: This header captures the experimental environment that produced these results.\n"
            f"-->\n"
        )

    if bool(stable_df.get("is_synthetic_data", pd.Series(dtype=bool)).any()):
        parts.append(
            "> **WARNING: these numbers include trials computed on SYNTHETIC "
            "fallback data. Do not use this report as-is; re-run data "
            "download and the experiment scripts to get real results.**\n"
        )

    rt = regret_table(stable_df, allow_synthetic=allow_synthetic)
    parts.append("### Table 5.1: Mean regret ± 95% CI, by condition and task profile\n")
    parts.append(rt.to_markdown())
    parts.append("")

    ot = operational_table(stable_df, allow_synthetic=allow_synthetic)
    parts.append("\n### Table 5.3: Operational metrics for the agent conditions\n")
    parts.append(ot.to_markdown())
    parts.append("")

    if drift_df is not None and len(drift_df) > 0:
        dt = drift_table(drift_df, allow_synthetic=allow_synthetic)
        parts.append("\n### Table 5.2: Mean adaptation lag under drift, by condition\n")
        parts.append(dt.to_markdown())
        parts.append("")

    if wsdream_df is not None and len(wsdream_df) > 0:
        wt = regret_table(wsdream_df, allow_synthetic=allow_synthetic)
        parts.append(
            "\n### Table 5.4: Supplementary validation on WS-DREAM Dataset #1 "
            "(2-attribute schema: response_time, throughput only)\n"
        )
        parts.append(wt.to_markdown())
        parts.append("")

    return "\n".join(parts)


def _plot_bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    conditions: list[str],
    out_path: Path,
    xlabel: str,
    title: str,
    subtitle: str,
    fmt: str = ".4f",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    means, errs, labels, colors = [], [], [], []
    for cond in conditions:
        sub = df[df["condition"] == cond][metric_col]
        if len(sub) == 0:
            continue
        ci = mean_confidence_interval(sub.astype(float).tolist())
        means.append(ci.mean)
        errs.append(ci.mean - ci.lower)
        labels.append(CONDITION_DISPLAY_NAMES[cond])
        colors.append(CONDITION_COLORS[cond])
    
    if not means:
        plt.close(fig)
        return

    bars = ax.barh(labels, means, xerr=errs, capsize=4, color=colors, height=0.62, zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=15)
    ax.text(
        0, 1.02, subtitle, transform=ax.transAxes, color="#5D6878", fontsize=10,
    )
    style_axis(ax)
    add_bar_labels(ax, bars, fmt=fmt)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_drift_timeline(drift_df: pd.DataFrame, out_path: Path):
    import ast
    import matplotlib.pyplot as plt
    sub = drift_df[drift_df["condition"] == "agent_full"]
    if len(sub) == 0: return
    row = sub.iloc[0]
    trace = ast.literal_eval(row["trace_json"])
    rounds = list(range(1, len(trace) + 1))
    
    target_service_id = row["target_service_id"]
    selections = [1 if t == target_service_id else 0 for t in trace]
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.step(rounds, selections, where='mid', color="#3b82f6", linewidth=2.5, zorder=3)
    ax.fill_between(rounds, selections, step='mid', color="#3b82f6", alpha=0.1)
    
    ax.axvline(x=int(row["degrade_start_round"]), color="#ef4444", linestyle="--", linewidth=2, label="Drift Injection", zorder=2)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Alternative Service", "Target Service"])
    ax.set_xlabel("Evaluation Round")
    ax.set_title(f"Live Failover Timeline (Profile: {row['profile']})", pad=15)
    ax.text(0, 1.05, "Agent rapidly fails over to a new service when the target drops.", transform=ax.transAxes, color="#5D6878", fontsize=10)
    ax.legend(loc="lower left")
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)

def _plot_radar_chart(df: pd.DataFrame, out_path: Path):
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    sub = df[df["condition"] == "agent_full"]
    if len(sub) == 0 or "weights_json" not in df.columns: return
    
    tasks_to_plot = ["batch_processing", "streaming", "compliance_sensitive_backend"]
    data_to_plot = {}
    attributes = None
    for t in tasks_to_plot:
        t_rows = sub[sub["task_key"] == t]
        if len(t_rows) > 0:
            w = json.loads(t_rows.iloc[0]["weights_json"])
            if not attributes:
                attributes = list(w.keys())
            data_to_plot[t] = [w[a] for a in attributes]
            
    if not data_to_plot: return
    
    num_vars = len(attributes)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    colors = ["#3b82f6", "#ef4444", "#10b981"]
    
    for i, (t, w_vals) in enumerate(data_to_plot.items()):
        vals = w_vals + w_vals[:1]
        ax.plot(angles, vals, label=t, color=colors[i % len(colors)], linewidth=2)
        ax.fill(angles, vals, color=colors[i % len(colors)], alpha=0.15)
        
    ax.set_xticks(angles[:-1])
    # Replace underscores with spaces for prettier labels
    pretty_attrs = [a.replace("_", " ").title() for a in attributes]
    ax.set_xticklabels(pretty_attrs, size=9)
    ax.set_title("Agent Inferred Attribute Weights", pad=20, weight="bold")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    
    # polar plots don't play well with style_axis (it removes spines inconsistently), so we skip it
    ax.spines['polar'].set_color('#E2E8F0')
    ax.grid(color='#E2E8F0')
    
    save_figure(fig, out_path)
    plt.close(fig)


def make_figures(stable_df: pd.DataFrame, drift_df: pd.DataFrame | None, out_dir: Path | str) -> list:
    """Save a small set of matplotlib figures to out_dir. Returns the
    list of file paths written. Uses the non-interactive Agg backend so
    this runs headless (CI, remote server) without needing a display.
    """
    import matplotlib

    matplotlib.use("Agg")
    apply_publication_style()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    p1 = out_dir / "regret_by_condition.png"
    _plot_bar_chart(
        df=stable_df,
        metric_col="regret",
        conditions=CONDITION_ORDER,
        out_path=p1,
        xlabel="Mean regret (95% CI) - lower is better",
        title="Decision Quality Across Methods",
        subtitle="Regret pooled across all task profiles",
        fmt=".4f",
    )
    written.append(p1)



    agent_conditions = ["agent_weights_only", "agent_full"]
    p_lat = out_dir / "latency_by_condition.png"
    _plot_bar_chart(
        df=stable_df,
        metric_col="latency_seconds",
        conditions=agent_conditions,
        out_path=p_lat,
        xlabel="Mean latency in seconds (95% CI) - lower is better",
        title="System Response Time",
        subtitle="Impact of Exact-Match Dictionary Caching",
        fmt=".4f",
    )
    written.append(p_lat)

    if drift_df is not None and len(drift_df) > 0:
        p2 = out_dir / "adaptation_lag_by_condition.png"
        filtered_drift = drift_df[~drift_df["censored"].astype(bool)]
        _plot_bar_chart(
            df=filtered_drift,
            metric_col="lag_rounds",
            conditions=CONDITION_ORDER,
            out_path=p2,
            xlabel="Mean adaptation lag in rounds (95% CI) - lower is better",
            title="Response Speed After QoS Degradation",
            subtitle="Censored trials are excluded from the mean",
            fmt=".1f",
        )
        written.append(p2)

    if "strategy" in stable_df.columns:
        p_strat = out_dir / "strategy_selection_frequency.png"
        agent_full_df = stable_df[stable_df["condition"] == "agent_full"]
        if len(agent_full_df) > 0:
            import matplotlib.pyplot as plt
            counts = agent_full_df["strategy"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(counts.index.astype(str), counts.values, color="#3b82f6", height=0.5, zorder=3)
            ax.invert_yaxis()
            ax.set_xlabel("Number of times strategy was chosen")
            ax.set_title("Agent (full) Strategy Selection Frequency", pad=15)
            ax.text(0, 1.02, "Shows whether the agent relies exclusively on TOPSIS or explores others.", transform=ax.transAxes, color="#5D6878", fontsize=10)
            style_axis(ax)
            add_bar_labels(ax, bars, fmt=".0f")
            fig.tight_layout()
            save_figure(fig, p_strat)
            plt.close(fig)
            written.append(p_strat)


    p_radar = out_dir / "radar_chart_weights.png"
    _plot_radar_chart(stable_df, p_radar)
    written.append(p_radar)

    if drift_df is not None and len(drift_df) > 0:
        p_timeline = out_dir / "drift_failover_timeline.png"
        _plot_drift_timeline(drift_df, p_timeline)
        written.append(p_timeline)

    return written

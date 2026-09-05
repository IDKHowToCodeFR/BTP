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
from typing import Optional

import numpy as np
import pandas as pd

from agentic_selection.evaluation.metrics import mean_confidence_interval

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
    drift_df: Optional[pd.DataFrame],
    allow_synthetic: bool = False,
    wsdream_df: Optional[pd.DataFrame] = None,
) -> str:
    """Render Tables 5.1-5.3 (and, if wsdream_df is given, 5.4) as
    GitHub-flavored markdown, ready to paste into
    paper/agentic_cloud_selection_paper.md replacing the Section 5
    placeholders."""
    parts = []
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


def make_figures(stable_df: pd.DataFrame, drift_df: Optional[pd.DataFrame], out_dir: Path | str) -> list:
    """Save a small set of matplotlib figures to out_dir. Returns the
    list of file paths written. Uses the non-interactive Agg backend so
    this runs headless (CI, remote server) without needing a display.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Regret by condition (aggregated across all task profiles)
    fig, ax = plt.subplots(figsize=(7, 4))
    means, errs, labels = [], [], []
    for cond in CONDITION_ORDER:
        sub = stable_df[stable_df["condition"] == cond]["regret"]
        if len(sub) == 0:
            continue
        ci = mean_confidence_interval(sub.tolist())
        means.append(ci.mean)
        errs.append(ci.mean - ci.lower)
        labels.append(CONDITION_DISPLAY_NAMES[cond])
    ax.bar(labels, means, yerr=errs, capsize=4)
    ax.set_ylabel("Mean regret (95% CI)")
    ax.set_title("Regret by condition (all task profiles pooled)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    p1 = out_dir / "regret_by_condition.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    written.append(p1)

    if drift_df is not None and len(drift_df) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        means, errs, labels = [], [], []
        for cond in CONDITION_ORDER:
            sub = drift_df[(drift_df["condition"] == cond) & (~drift_df["censored"].astype(bool))]
            if len(sub) == 0:
                continue
            ci = mean_confidence_interval(sub["lag_rounds"].astype(float).tolist())
            means.append(ci.mean)
            errs.append(ci.mean - ci.lower)
            labels.append(CONDITION_DISPLAY_NAMES[cond])
        ax.bar(labels, means, yerr=errs, capsize=4, color="orange")
        ax.set_ylabel("Mean adaptation lag (rounds, 95% CI)")
        ax.set_title("Adaptation lag under drift, by condition (censored trials excluded)")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        p2 = out_dir / "adaptation_lag_by_condition.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        written.append(p2)

    return written

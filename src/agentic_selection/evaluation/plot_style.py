"""Shared publication style for result figures."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

INK = "#172033"
MUTED = "#5D6878"
GRID = "#DCE3EA"
CANVAS = "#F5F7FA"
WHITE = "#FFFFFF"

CONDITION_LABELS = {
    "global_fixed": "Global fixed",
    "lookup_table": "Lookup table",
    "agent_weights_only": "Agent: weights only",
    "agent_full": "Agent: full",
    "global_static": "Global baseline: static",
    "lookup_static": "Lookup baseline: static",
    "global_dynamic": "Global baseline: dynamic",
    "lookup_dynamic": "Lookup baseline: dynamic",
}

# Warm colors identify baselines; cool colors identify agent conditions.
CONDITION_COLORS = {
    "global_fixed": "#D55E5E",
    "lookup_table": "#E69F00",
    "global_static": "#C94C4C",
    "lookup_static": "#E69F00",
    "global_dynamic": "#CC79A7",
    "lookup_dynamic": "#F0B44D",
    "agent_weights_only": "#009E73",
    "agent_full": "#4C6FB1",
}

ACCENT_COLORS = ("#4C6FB1", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")


def apply_publication_style() -> None:
    """Apply a restrained, high-contrast style suitable for reports and slides."""
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": WHITE,
            "figure.facecolor": CANVAS,
            "savefig.facecolor": CANVAS,
            "axes.edgecolor": "#AAB4C0",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.85,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.titlesize": 18,
            "figure.titleweight": "bold",
        }
    )


def style_axis(ax, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB4C0")
    ax.spines["bottom"].set_color("#AAB4C0")
    ax.tick_params(length=0)


def add_bar_labels(ax, bars: Iterable, fmt: str = ".3f", horizontal: bool = True) -> None:
    """Add compact labels while reserving enough axis room for them."""
    bars = list(bars)
    if not bars:
        return

    values = [bar.get_width() if horizontal else bar.get_height() for bar in bars]
    maximum = max(values) if values else 0.0
    padding = maximum * 0.025 if maximum > 0 else 0.02

    if horizontal:
        left, right = ax.get_xlim()
        ax.set_xlim(left, max(right, maximum * 1.18 if maximum > 0 else 1.0))
        for bar, value in zip(bars, values):
            ax.text(
                value + padding,
                bar.get_y() + bar.get_height() / 2,
                format(value, fmt),
                va="center",
                ha="left",
                color=INK,
                fontsize=9,
                fontweight="bold",
            )
    else:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, max(top, maximum * 1.16 if maximum > 0 else 1.0))
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + padding,
                format(value, fmt),
                va="bottom",
                ha="center",
                color=INK,
                fontsize=8.5,
                fontweight="bold",
            )


def save_figure(fig, path: Path | str, dpi: int = 220) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

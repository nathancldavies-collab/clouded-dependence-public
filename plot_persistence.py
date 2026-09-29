#!/usr/bin/env python3
"""
Bar Charts: Supplier Persistence by View, and Against the Shuffle Null
======================================================================
Two panels from `run_persistence_analysis.py`:

  - Left, the raw rates: the share of an office's suppliers kept the next
    year, by contractor (cA, naive view) and by platform (cB, unclouded view).
    A bracket gives their ratio PR = cB / cA with its paired interval.
  - Right, the inference: PR against PR_0, the median PR when platforms are
    shuffled across records within each year. PR_0 is how much more often a
    platform than a contractor is kept from there being fewer platforms
    alone. A bracket gives PR / PR_0, the persistence beyond what coarsening
    explains, with the one-sided permutation p-value.

Error bars: cA, cB and PR have 95% BCa intervals over offices; PR_0 has the
2.5-97.5th percentiles of the null distribution. cA and cB are paired, so
overlapping bars on the left do not imply PR's interval spans 1.

Run from project root (after `just persistence`):
    uv run plot_persistence.py [--scope all|attributed] [--file-type pdf]

Design notes:
  - Styling is imported from plot_concentration so the figures match: the
    baseline (naive view, null) is grey, the finding (unclouded view,
    observed PR) is blue.
  - The left axis starts at zero, so bar heights compare honestly. The right
    starts at 1x: platforms merge contractors, so PR cannot fall below 1, and
    1x (platforms kept exactly as often as contractors) is the floor.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from clouded_deps.directories import OUTPUTS_DIR
from clouded_deps.plot_style import (
    NAIVE_COLOR,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    UNCLOUDED_COLOR,
)
from plot_concentration import LABEL_FONT_SIZE, TICK_FONT_SIZE, _style_axis
from run_persistence_analysis import SCOPES, summary_path

BAR_WIDTH = 0.5
# (statistic in the summary, tick label, colour), left to right, per panel.
RATE_BARS = [
    ("cA", "Naive view\n(by contractor)", NAIVE_COLOR),
    ("cB", "Unclouded view\n(by platform)", UNCLOUDED_COLOR),
]
RATIO_BARS = [
    ("pr_null", "Shuffle null (PR$_0$)", NAIVE_COLOR),
    ("pr", "Observed (PR)", UNCLOUDED_COLOR),
]


def format_p(p: float, n_perm: int) -> str:
    """The p-value, as a bound when no permutation reached the observed PR."""
    # With no null draw >= PR, p = 1 / (n_perm + 1): all we know is p < 1 / n_perm.
    if p <= 1 / (n_perm + 1) * 1.000001:
        return f"p < {1 / n_perm:.2g}"
    return f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.1g}"


def _draw_bars(
    ax: plt.Axes,
    stats: pd.DataFrame,
    bars: list[tuple[str, str, str]],
    label: Callable[[float], str],
    base: float = 0.0,
) -> float:
    """
    Bars rising from `base`, with their intervals and value labels; returns the
    highest whisker.
    """
    tops = []
    for x, (stat, _, color) in enumerate(bars):
        row = stats.loc[stat]
        ax.bar(
            x,
            row["estimate"] - base,
            bottom=base,
            width=BAR_WIDTH,
            color=color,
            zorder=2,
        )
        ax.errorbar(
            x,
            row["estimate"],
            yerr=[
                [row["estimate"] - row["ci_low"]],
                [row["ci_high"] - row["estimate"]],
            ],
            fmt="none",
            ecolor=TEXT_PRIMARY,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=3,
        )
        # Inside the bar, just below the error bar's lower cap.
        ax.annotate(
            label(row["estimate"]),
            xy=(x, row["ci_low"]),
            xytext=(0, -8),
            textcoords="offset points",
            va="top",
            ha="center",
            fontsize=TICK_FONT_SIZE,
            color=SURFACE,
            zorder=4,
        )
        tops.append(row["ci_high"])
    ax.set_xlim(-0.6, len(bars) - 0.4)
    return max(tops)


def _draw_bracket(ax: plt.Axes, top: float, label: str, base: float = 0.0) -> None:
    """
    A labelled bracket over both bars; sets the y-limits, from `base`, to make
    room for it.
    """
    y = base + (top - base) * 1.06
    tick = (top - base) * 0.025
    ax.plot(
        [0, 0, 1, 1],
        [y - tick, y, y, y - tick],
        color=TEXT_SECONDARY,
        linewidth=1.2,
        zorder=3,
    )
    ax.text(
        0.5,
        y + tick,
        label,
        ha="center",
        va="bottom",
        fontsize=LABEL_FONT_SIZE - 2,
        color=TEXT_PRIMARY,
    )
    ax.set_ylim(base, base + (y - base) * 1.14)


def _draw_rates(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Left panel: the share of suppliers kept the next year, in each view."""
    top = _draw_bars(ax, stats, RATE_BARS, lambda value: f"{value:.0%}")
    _style_axis(
        ax,
        [label for _, label, _ in RATE_BARS],
        "Persistence by view",
        "Suppliers kept the next year",
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    pr = stats.loc["pr"]
    _draw_bracket(
        ax,
        top,
        f"PR = {pr['estimate']:.2f}×   [{pr['ci_low']:.2f}–{pr['ci_high']:.2f}]",
    )


def _draw_ratio(ax: plt.Axes, stats: pd.DataFrame, n_perm: int) -> None:
    """Right panel: the observed PR against the shuffle null."""
    # PR cannot fall below 1 (platforms merge contractors), so bars rise from 1x.
    top = _draw_bars(ax, stats, RATIO_BARS, lambda value: f"{value:.2f}×", base=1)
    _style_axis(
        ax,
        [label for _, label, _ in RATIO_BARS],
        "Beyond aggregation",
        "Platform ÷ contractor\npersistence",
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:g}×")
    excess = stats.loc["pr_excess", "estimate"]
    p_label = format_p(stats.loc["p_value", "estimate"], n_perm)
    _draw_bracket(ax, top, f"PR / PR$_0$ = {excess:.2f}×   ({p_label})", base=1)


def plot_persistence(summary: pd.DataFrame, output_path: Path) -> Path:
    """Draw the rates and the ratio side by side and save the figure."""
    stats = summary.set_index("statistic")
    n_perm = int(summary["n_perm"].iloc[0])

    fig, (left, right) = plt.subplots(ncols=2, figsize=(11, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(wspace=0.35)
    _draw_rates(left, stats)
    _draw_ratio(right, stats, n_perm)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="all",
        help="Which run_persistence_analysis.py scope to plot (default: all)",
    )
    parser.add_argument(
        "--file-type",
        choices=["png", "pdf"],
        default="png",
        help="Output file format (default: png)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = summary_path(OUTPUTS_DIR / "persistence", args.scope)
    if not summary.exists():
        raise SystemExit(
            f"Missing {summary}\nRun `just persistence --scope {args.scope}` first."
        )
    name = summary.stem.replace("persistence_summary", "persistence_ratio")
    saved = plot_persistence(
        pd.read_csv(summary), summary.parent / f"{name}.{args.file_type}"
    )
    print(f"  saved: {saved}")

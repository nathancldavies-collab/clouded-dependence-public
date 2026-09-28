#!/usr/bin/env python3
"""
Bar Chart: Observed Platform Persistence vs the Shuffle Null
============================================================
Two bars from `run_persistence_analysis.py`:

  - PR_0: the median PR when platforms are shuffled across records within
    each year. This is how much more often a platform than a contractor is
    kept, from there being fewer platforms alone.
  - PR: the observed ratio of platform to contractor retention.

A bracket over the pair gives PR / PR_0, the persistence beyond what
coarsening explains, with the one-sided permutation p-value.

Error bars: PR has its 95% BCa interval over offices; PR_0 has the
2.5-97.5th percentiles of the null distribution.

Run from project root (after `just persistence`):
    uv run plot_persistence.py [--file-type pdf]

Design notes:
  - Styling is imported from plot_concentration so the figures match: the null
    is the grey baseline, the observed value carries the finding in blue.
  - The axis starts at zero, so bar heights compare honestly, and a dashed line
    marks 1x, where platforms are kept exactly as often as contractors.
"""

import argparse
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

BAR_WIDTH = 0.5
# (statistic in the summary, tick label, colour), left to right.
BARS = [
    ("pr_null", "Shuffle null (PR$_0$)", NAIVE_COLOR),
    ("pr", "Observed (PR)", UNCLOUDED_COLOR),
]


def format_p(p: float, n_perm: int) -> str:
    """The p-value, as a bound when no permutation reached the observed PR."""
    # With no null draw >= PR, p = 1 / (n_perm + 1): all we know is p < 1 / n_perm.
    if p <= 1 / (n_perm + 1) * 1.000001:
        return f"p < {1 / n_perm:.2g}"
    return f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.1e}"


def plot_persistence(summary: pd.DataFrame, output_path: Path) -> Path:
    """Draw the two bars with the ratio bracket and save the figure."""
    stats = summary.set_index("statistic")
    n_perm = int(summary["n_perm"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    tops = []
    for x, (stat, _, color) in enumerate(BARS):
        row = stats.loc[stat]
        ax.bar(x, row["estimate"], width=BAR_WIDTH, color=color, zorder=2)
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
            f"{row['estimate']:.2f}×",
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

    _style_axis(
        ax,
        [label for _, label, _ in BARS],
        "",
        "Platform ÷ contractor\npersistence",
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:g}×")
    ax.set_xlim(-0.6, len(BARS) - 0.4)
    ax.axhline(1, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (2, 2)), zorder=1)

    # Bracket over both bars, labelled with the ratio and its p-value.
    y = max(tops) * 1.06
    tick = max(tops) * 0.025
    ax.plot(
        [0, 0, 1, 1],
        [y - tick, y, y, y - tick],
        color=TEXT_SECONDARY,
        linewidth=1.2,
        zorder=3,
    )
    excess = stats.loc["pr_excess", "estimate"]
    p_label = format_p(stats.loc["p_value", "estimate"], n_perm)
    ax.text(
        0.5,
        y + tick,
        f"PR / PR$_0$ = {excess:.2f}×   ({p_label})",
        ha="center",
        va="bottom",
        fontsize=LABEL_FONT_SIZE - 2,
        color=TEXT_PRIMARY,
    )
    ax.set_ylim(0, y * 1.14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=OUTPUTS_DIR / "persistence" / "persistence_summary.csv",
        help="Summary table from run_persistence_analysis.py",
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
    if not args.summary.exists():
        raise SystemExit(f"Missing {args.summary}\nRun `just persistence` first.")
    saved = plot_persistence(
        pd.read_csv(args.summary),
        args.summary.parent / f"persistence_ratio.{args.file_type}",
    )
    print(f"  saved: {saved}")

#!/usr/bin/env python3
"""
Stacked Bar: How Much Cloud Spending the Pipeline Uncovers
==========================================================
One horizontal bar of all procurement spending (FY2017-2024), split by the first
classification step that identifies each record as cloud:

  - Naive: the record directly mentions a hyperscaler (AWS, Azure, Google
    Cloud) -- what is directly labelled as cloud.
  - Stage 1 (RegEx): brand, service and contractor-name patterns.
  - Stage 2 (LLM): contextual classification of the description.
  - Rest: spending the pipeline leaves as non-cloud.

Each record counts once, at the first step that flags it, and only if the final
synthesis keeps it as cloud, so the cloud layers sum to the pipeline's cloud
total. Synthesis only drops records (RegEx hits the LLM rejects), so it adds no
layer of its own. The stages together are the spending the pipeline uncovers
beyond the naive view.

Run from project root:
    uv run plot_cloud_discovery.py [--file-type pdf] [--table]

Design notes:
  - Naive is the repo's baseline grey; the uncovered stages are blues that
    darken with each step, ending on the repo's dark blue. The rest of
    spending is a near-surface grey so the cloud share reads against it.
  - A legend below the x-axis names the cloud layers; the rest of spending is
    labelled inside its own segment.
  - Dollar values sit inside their segments, in ink or white by the fill's
    luminance. Where a segment is too narrow, the value moves above it on a
    vertical leader line. Earlier segments take higher tiers, so each value
    runs right over the shorter leaders of the segments after it.
  - The bracket below the bar marks the uncovered spending, the finding.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from clouded_deps.directories import DATA_DIR, OUTPUTS_DIR
from clouded_deps.pipeline.views import FIRST_FY, LAST_FY
from clouded_deps.plot_style import (
    NAIVE_COLOR,
    SPINE,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    UNCLOUDED_COLOR,
)

# --- Palette -----------------------------------------------------------------
# Validated against NAIVE_COLOR on SURFACE: normal-vision ΔE 19, CVD ΔE 18. It
# falls below 3:1 contrast against the surface, which the direct labels relieve.
STAGE1_COLOR = "#8cc0f0"
REST_COLOR = "#ebeae5"

LABEL_FONT_SIZE = 16
VALUE_FONT_SIZE = 16
TICK_FONT_SIZE = 14
# Horizontal padding, in points, an in-bar value needs on each side to fit.
VALUE_PAD_PT = 4
# Vertical step between tiers of values above the bar, in bar-height units.
TIER_STEP = 0.32

# (key, label, colour); order is the order in the bar.
LAYERS = [
    ("naive", "Naive: direct mention", NAIVE_COLOR),
    ("stage1", "Stage 1: RegEx patterns", STAGE1_COLOR),
    ("stage2", "Stage 2: LLM classification", UNCLOUDED_COLOR),
    ("rest", "Rest of procurement (non-cloud)", REST_COLOR),
]
UNCOVERED = ["stage1", "stage2"]


def discovery_records(path: Path) -> pd.DataFrame:
    """
    Records in scope, each with its discovery layer.

    A cloud record's layer is the first step that flags it; non-cloud is "rest".
    """
    if not path.exists():
        raise SystemExit(f"Missing {path}\nRun `just pipeline` first.")
    df = pd.read_csv(
        path,
        usecols=[
            "dollars",
            "fiscal_year",
            "is_cloud",
            "explicit_cloud_mention",
            "regex_is_cloud",
        ],
    )
    # Same scope as the views: positive dollars, the paper's fiscal years.
    df = df[(df["dollars"] > 0) & df["fiscal_year"].between(FIRST_FY, LAST_FY)]

    layer = pd.Series("rest", index=df.index)
    layer[df["is_cloud"]] = "stage2"
    layer[df["is_cloud"] & df["regex_is_cloud"]] = "stage1"
    layer[df["is_cloud"] & df["explicit_cloud_mention"]] = "naive"
    return df.assign(layer=layer)


def discovery_layers(records: pd.DataFrame) -> pd.Series:
    """Dollars per layer, in bar order."""
    return records.groupby("layer")["dollars"].sum().reindex([k for k, _, _ in LAYERS])


def _billions(value: float) -> str:
    return f"${value / 1e9:,.1f}B"


def _value_color(fill: str) -> str:
    """White on dark fills, ink on light ones, so in-bar text clears contrast."""
    r, g, b = to_rgb(fill)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.4 else TEXT_PRIMARY


def plot_discovery(layers: pd.Series, output_path: Path) -> Path:
    """Render the stacked bar with its legend and bracket, and save the figure."""
    total = layers.sum()
    # Dollar amounts contain "$", which mathtext would read as math delimiters.
    plt.rcParams["text.parse_math"] = False
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    renderer = fig.canvas.get_renderer()
    pad_px = VALUE_PAD_PT * fig.dpi / 72

    bar_half = 0.5
    # Limits first: the fit test below measures segments in display space.
    ax.set_xlim(0, total)
    ax.set_ylim(-bar_half - 0.6, bar_half + 0.8)
    starts = layers.cumsum() - layers
    above = []
    for key, label, color in LAYERS:
        ax.barh(
            0,
            layers[key],
            left=starts[key],
            height=2 * bar_half,
            color=color,
            # The surface-coloured edge is the gap between stacked segments.
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=2,
        )
        value = _billions(layers[key])
        # The rest has no legend entry, so its label travels with its value.
        text = f"{label}  {value}" if key == "rest" else value
        value_text = ax.text(
            starts[key] + layers[key] / 2,
            0,
            text,
            ha="center",
            va="center",
            fontsize=VALUE_FONT_SIZE,
            fontweight="normal" if key == "rest" else "bold",
            color=TEXT_SECONDARY if key == "rest" else _value_color(color),
            zorder=3,
        )
        left_px, right_px = ax.transData.transform(
            [(starts[key], 0), (starts[key] + layers[key], 0)]
        )[:, 0]
        if value_text.get_window_extent(renderer).width + 2 * pad_px > (
            right_px - left_px
        ):
            above.append((key, value_text))

    # Values that did not fit: on a vertical leader above their segment, with
    # the value just right of the leader's top. Earlier segments go higher.
    for tier, (key, value_text) in enumerate(reversed(above)):
        x = starts[key] + layers[key] / 2
        y = bar_half + 0.2 + TIER_STEP * tier
        ax.plot([x, x], [bar_half + 0.03, y], color=TEXT_SECONDARY, lw=1, zorder=1)
        value_text.set(
            x=x + total * 0.004, y=y, ha="left", va="center", color=TEXT_PRIMARY
        )

    # Bracket under the uncovered stages: the finding, labelled beneath it.
    uncovered = layers[UNCOVERED].sum()
    left, right = starts[UNCOVERED[0]], starts["rest"]
    top, bottom = -bar_half - 0.08, -bar_half - 0.2
    ax.plot(
        [left, left, right, right],
        [top, bottom, bottom, top],
        color=TEXT_PRIMARY,
        lw=1.2,
        zorder=1,
    )
    ax.text(
        left,
        bottom - 0.1,
        f"Uncovered by the pipeline: +{_billions(uncovered)}"
        f"  ({uncovered / layers['naive']:.0f}× the naive view)",
        ha="left",
        va="top",
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )

    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda value, _: f"${value / 1e9:,.0f}B")
    ax.set_xlabel(
        f"Procurement spending, FY{FIRST_FY}–{LAST_FY}",
        fontsize=LABEL_FONT_SIZE,
        color=TEXT_SECONDARY,
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=TICK_FONT_SIZE, pad=6)

    # Legend under the x-axis label, centred on the axes.
    label_bottom = ax.xaxis.label.get_window_extent(renderer).y0
    fig.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for key, label, color in LAYERS
            if key != "rest"
        ],
        loc="upper center",
        bbox_to_anchor=(
            (ax.get_position().x0 + ax.get_position().x1) / 2,
            (label_bottom - 3 * pad_px) / fig.bbox.height,
        ),
        ncols=3,
        frameon=False,
        borderaxespad=0,
        handlelength=1.2,
        handleheight=1.2,
        handletextpad=0.5,
        columnspacing=1.2,
        fontsize=LABEL_FONT_SIZE,
        labelcolor=TEXT_SECONDARY,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_table(records: pd.DataFrame) -> None:
    """Print the underlying numbers, so the chart is never the only view."""
    layers = discovery_layers(records)
    total = layers.sum()
    print(f"\n  Procurement spending by discovery layer, FY{FIRST_FY}–{LAST_FY}")
    print(f"    {'layer':<34s} {'dollars':>10s} {'share':>7s}")
    print(f"    {'-' * 53}")
    for key, label, _ in LAYERS:
        print(
            f"    {label:<34s} {_billions(layers[key]):>10s}"
            f" {layers[key] / total:>7.1%}"
        )
    print(f"    {'-' * 53}")
    print(f"    {'Total':<34s} {_billions(total):>10s}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "02_processed" / "03_classified" / "attributed_dataset.csv",
        help="Attributed dataset from the pipeline",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUTS_DIR / "discovery",
        help="Directory for the output figure",
    )
    parser.add_argument(
        "--file-type",
        choices=["png", "pdf"],
        default="png",
        help="Output file format (default: png)",
    )
    parser.add_argument(
        "--table", action="store_true", help="Also print the numbers as a table"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records = discovery_records(args.data)
    layers = discovery_layers(records)
    if args.table:
        print_table(records)
    saved = plot_discovery(layers, args.out_dir / f"cloud_discovery.{args.file_type}")
    print(f"  saved: {saved}")

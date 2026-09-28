#!/usr/bin/env python3
"""
Line Plot: Naive vs Unclouded Cloud Concentration Over Time
===========================================================
Plots HHI of federal cloud spending over three-year rolling fiscal-year windows,
in two views:

  - Naive view: spending grouped by the contractor on the record (resolved UEI).
  - Unclouded view: the same spending, with every platform-attributed record
    reassigned from its contractor to its platform. Unattributed cloud records
    keep their contractor.

The two views therefore differ ONLY in attribution: same records, same dollars,
same market. The left panel shows both views; the right panel shows their
ratio (unclouded / naive), i.e. how many times more concentrated the market is
than the contracting record suggests.

Error bars are 95% BCa (bias-corrected and accelerated) intervals (Efron 1987;
DiCiccio & Efron 1996) from a cluster bootstrap: within each window, prime
awards are resampled with replacement, each carrying its subawards, since a
prime and its subawards are not independent draws.

  - Bias correction z0 comes from the share of bootstrap draws below the point
    estimate. It matters here: the plug-in HHI is biased upward under
    resampling (a large award drawn twice has its share squared), so plain
    percentile intervals sit lopsidedly above the estimate.
  - Acceleration a comes from the leave-one-prime-award-out jackknife, the same
    unit the bootstrap resamples. The jackknife HHI is computed in closed form,
    not by refitting once per award.

Both views are computed on the same resamples and jackknife replicates, so the
ratio has its own paired interval -- overlapping error bars in the left panel do
NOT imply the ratio spans one. The intervals capture sampling variability in
which awards were made, not classification or attribution error.

References:
  Efron, B. (1987). Better Bootstrap Confidence Intervals. JASA 82, 171-185.
  DiCiccio, T. J. & Efron, B. (1996). Bootstrap Confidence Intervals.
    Statistical Science 11, 189-228.

Run from project root:
    uv run plot_concentration.py [--file-type pdf] [--table] [--n-boot 20000]
                                 [--no-cache]

By default the bootstrap results are cached in rolling_hhi.csv and reused on
later runs, so restyling the figure does not re-run the bootstrap. The cache is
ignored when --n-boot or --seed differ from the cached run, or when the dataset
is newer than the cache; --no-cache forces a recompute.

Design notes:
  - The naive view is grey because it is the baseline the reader already has;
    the unclouded view carries the finding and takes the repo's dark blue. The
    grey deliberately fails the chroma floor (it should read as grey); CVD
    separation from the blue and contrast against the surface both pass.
  - Series are also told apart by line style (dashed vs. solid) and marker
    shape (circle vs. star), so identity never rests on colour alone.
  - The two series are nudged apart horizontally so their error bars do not
    overlap where the lines run close together.
  - The ratio is a derived quantity, not either view, so it is drawn in
    neutral ink with square markers rather than borrowing a view's identity.
  - Both panels start at zero, so heights compare honestly; the ratio panel
    marks 1x, where attribution would reveal nothing hidden.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from clouded_deps.directories import DATA_DIR, OUTPUTS_DIR
from clouded_deps.pipeline.baseline_merged_hhi import calculate_hhi
from clouded_deps.pipeline.views import FIRST_FY, LAST_FY, load_records
from clouded_deps.stats import bca_interval

# --- Palette -----------------------------------------------------------------
NAIVE_COLOR = "#8a8880"
UNCLOUDED_COLOR = "#17538f"
RATIO_COLOR = "#3d3c38"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"
SPINE = "#d8d7d1"

NAIVE_MARKER = "o"
UNCLOUDED_MARKER = "*"
RATIO_MARKER = "s"
# A star's ink sits inside its bounding box, so it needs a larger size to read
# at the same weight as the circle. Matched by eye, as in plot_exposure_barbell.
NAIVE_SIZE = 7
UNCLOUDED_SIZE = 12
RATIO_SIZE = 6
LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 12.5
LEGEND_FONT_SIZE = 12

NAIVE_LINESTYLE = (0, (5, 3))
UNCLOUDED_LINESTYLE = "solid"
# Horizontal nudge, in window units, so the two views' error bars sit side by side.
DODGE = 0.07

VIEWS = {
    "naive": (
        "Naive view",
        NAIVE_COLOR,
        NAIVE_MARKER,
        NAIVE_SIZE,
        NAIVE_LINESTYLE,
        -DODGE,
    ),
    "unclouded": (
        "Unclouded view",
        UNCLOUDED_COLOR,
        UNCLOUDED_MARKER,
        UNCLOUDED_SIZE,
        UNCLOUDED_LINESTYLE,
        DODGE,
    ),
}

WINDOW = 3
CI_LEVEL = 0.95


def hhi(df: pd.DataFrame, key: str) -> float:
    """HHI (0-10,000) of dollars grouped by `key`."""
    spending = df.groupby(key)["dollars"].sum()
    return float(calculate_hhi(spending / spending.sum() * 100))


def bootstrap_hhi(
    df: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """
    Bootstrap HHI for each view by resampling prime awards with replacement.

    Every view is computed on the same resampled awards, so the draws are paired
    across views and their ratio has a valid bootstrap distribution too.
    """
    prime_codes, primes = pd.factorize(df["original_prime_id"])
    key_codes = {view: pd.factorize(df[view])[0] for view in VIEWS}
    dollars = df["dollars"].to_numpy()
    n_primes = len(primes)

    draws = {view: np.empty(n_boot) for view in VIEWS}
    for b in range(n_boot):
        # How many times each prime award was drawn, applied to all its rows.
        counts = np.bincount(rng.integers(0, n_primes, n_primes), minlength=n_primes)
        weights = counts[prime_codes] * dollars
        for view, codes in key_codes.items():
            spending = np.bincount(codes, weights=weights)
            draws[view][b] = calculate_hhi(spending / spending.sum() * 100)
    return draws


def jackknife_hhi(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Leave-one-prime-award-out HHI for each view, in closed form.

    Dropping award p removes its dollars d_pk from each key k. With shares taken
    over the full window (T_k key totals, S their sum, D_p award p's total):

        HHI_(-p) = sum_k (T_k - d_pk)^2 / (S - D_p)^2
                 = (sum_k T_k^2 - 2 sum_k T_k d_pk + sum_k d_pk^2) / (S - D_p)^2

    so every replicate costs one pass over the award-key totals rather than a
    refit per award. Everything is scaled by S first to keep the terms near 1.
    """
    prime_codes, primes = pd.factorize(df["original_prime_id"])
    n_primes = len(primes)
    share = df["dollars"].to_numpy() / df["dollars"].sum()
    award_total = np.bincount(prime_codes, weights=share, minlength=n_primes)

    replicates = {}
    for view in VIEWS:
        key_codes = pd.factorize(df[view])[0]
        # Collapse to one row per (award, key) so d_pk is a single number.
        cells = (
            pd.DataFrame({"prime": prime_codes, "key": key_codes, "share": share})
            .groupby(["prime", "key"], sort=False)["share"]
            .sum()
            .reset_index()
        )
        key_total = np.bincount(cells["key"], weights=cells["share"])
        d = cells["share"].to_numpy()
        cross = np.bincount(
            cells["prime"], weights=key_total[cells["key"]] * d, minlength=n_primes
        )
        own = np.bincount(cells["prime"], weights=d**2, minlength=n_primes)
        numerator = (key_total**2).sum() - 2 * cross + own
        replicates[view] = numerator / (1 - award_total) ** 2 * 10_000
    return replicates


def rolling_hhi(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """
    HHI per view, and their ratio, for every three-year fiscal-year window,
    each with a BCa bootstrap interval.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for start in range(FIRST_FY, LAST_FY - WINDOW + 2):
        end = start + WINDOW - 1
        records = df[df["fiscal_year"].between(start, end)]
        estimates = {view: hhi(records, view) for view in VIEWS}
        draws = bootstrap_hhi(records, n_boot, rng)
        jackknife = jackknife_hhi(records)
        # The ratio is resampled and jackknifed on the same paired replicates.
        estimates["ratio"] = estimates["unclouded"] / estimates["naive"]
        draws["ratio"] = draws["unclouded"] / draws["naive"]
        jackknife["ratio"] = jackknife["unclouded"] / jackknife["naive"]
        for series, estimate in estimates.items():
            low, high = bca_interval(
                estimate, draws[series], jackknife[series], CI_LEVEL
            )
            rows.append(
                {
                    "window": f"{start}–{str(end)[-2:]}",
                    "start": start,
                    "series": series,
                    "hhi": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "dollars": records["dollars"].sum(),
                }
            )
    return pd.DataFrame(rows)


def _style_axis(ax: plt.Axes, windows: list[str], title: str, ylabel: str) -> None:
    """Shared axis furniture: window ticks, zero-based y, recessive grid."""
    ax.set_title(title, loc="left", fontsize=16, color=TEXT_PRIMARY, pad=10)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows)
    ax.set_xlim(-0.3, len(windows) - 1 + 0.3)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, color=TEXT_SECONDARY)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")

    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=TICK_FONT_SIZE)
    # Drop the window labels clear of the zero tick, which they meet at the corner.
    ax.tick_params(axis="x", pad=9)


def _draw_series(
    ax: plt.Axes,
    series: pd.DataFrame,
    color: str,
    marker: str,
    size: float,
    linestyle: str | tuple,
    dodge: float = 0.0,
) -> None:
    """One line with its bootstrap error bars."""
    x = [position + dodge for position in range(len(series))]
    ax.errorbar(
        x,
        series["hhi"],
        yerr=[series["hhi"] - series["ci_low"], series["ci_high"] - series["hhi"]],
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        zorder=2,
    )
    ax.plot(
        x,
        series["hhi"],
        color=color,
        linewidth=2,
        linestyle=linestyle,
        marker=marker,
        markersize=size,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        zorder=3,
    )


def _draw_views(ax: plt.Axes, data: pd.DataFrame, windows: list[str]) -> None:
    """Left panel: both views, with a legend."""
    views = data[data["series"].isin(VIEWS)]
    for view, (_, color, marker, size, linestyle, dodge) in VIEWS.items():
        series = views[views["series"] == view].sort_values("start")
        _draw_series(ax, series, color, marker, size, linestyle, dodge)

    _style_axis(
        ax, windows, "Concentration by view", "Herfindahl–Hirschman Index (HHI)"
    )
    ax.set_ylim(0, views["ci_high"].max() * 1.08)

    handles = [
        Line2D(
            [],
            [],
            color=color,
            linewidth=2,
            linestyle=linestyle,
            marker=marker,
            markersize=size,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=label,
        )
        for label, color, marker, size, linestyle in [
            (
                "Naive view (contracting record only)",
                NAIVE_COLOR,
                NAIVE_MARKER,
                NAIVE_SIZE,
                NAIVE_LINESTYLE,
            ),
            (
                "Unclouded view (with platform attribution)",
                UNCLOUDED_COLOR,
                UNCLOUDED_MARKER,
                UNCLOUDED_SIZE,
                UNCLOUDED_LINESTYLE,
            ),
        ]
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        fontsize=LEGEND_FONT_SIZE,
        labelcolor=TEXT_SECONDARY,
    )


def _draw_ratio(ax: plt.Axes, data: pd.DataFrame, windows: list[str]) -> None:
    """Right panel: unclouded over naive HHI, with its paired interval."""
    series = data[data["series"] == "ratio"].sort_values("start")
    _draw_series(ax, series, RATIO_COLOR, RATIO_MARKER, RATIO_SIZE, "solid")
    _style_axis(ax, windows, "Hidden concentration", "Ratio of HHI (unclouded ÷ naive)")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:g}×")
    ax.set_ylim(0, series["ci_high"].max() * 1.08)
    # 1x is "attribution reveals nothing": the reference the ratio is read against.
    ax.axhline(1, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (2, 2)), zorder=1)


def plot_concentration(data: pd.DataFrame, output_path: Path) -> Path:
    """Render the views and their difference side by side and save the figure."""
    fig, (left, right) = plt.subplots(ncols=2, figsize=(14, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2, wspace=0.22)

    windows = data.drop_duplicates("start").sort_values("start")["window"].tolist()
    _draw_views(left, data, windows)
    _draw_ratio(right, data, windows)

    fig.supxlabel(
        "Three-year rolling window (fiscal years)",
        y=0.04,
        fontsize=LABEL_FONT_SIZE,
        color=TEXT_SECONDARY,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_table(data: pd.DataFrame) -> None:
    """Print the underlying numbers, so the chart is never the only view."""
    wide = data.pivot_table(
        index=["start", "window"], columns="series", values=["hhi", "ci_low", "ci_high"]
    ).reset_index()

    print(f"\n  Cloud spending HHI  ({CI_LEVEL:.0%} BCa bootstrap intervals)")
    print(
        f"    {'window':<9s} {'naive':>6s} {'':<13s}"
        f" {'unclouded':>9s} {'':<13s} {'ratio':>6s}"
    )
    print(f"    {'-' * 76}")
    for _, row in wide.sort_values("start").iterrows():
        cells = [
            f"{row[('hhi', series)]:>{width}{fmt}}"
            f" [{row[('ci_low', series)]:{fmt}}–{row[('ci_high', series)]:{fmt}}]".ljust(
                width + 14
            )
            for series, width, fmt in [
                ("naive", 6, ",.0f"),
                ("unclouded", 9, ",.0f"),
                ("ratio", 6, ".2f"),
            ]
        ]
        print(f"    {row[('window', '')]:<9s} " + " ".join(cells))


def load_cache(
    cache_path: Path, data_path: Path, n_boot: int, seed: int
) -> pd.DataFrame | None:
    """Cached results, or None if missing or computed with different inputs."""
    if not cache_path.exists():
        return None
    if data_path.stat().st_mtime > cache_path.stat().st_mtime:
        print(f"  cache: {data_path.name} is newer than the cache; recomputing")
        return None
    cached = pd.read_csv(cache_path)
    if {"n_boot", "seed"} - set(cached.columns) or (
        (cached["n_boot"] != n_boot).any() or (cached["seed"] != seed).any()
    ):
        print("  cache: computed with a different --n-boot or --seed; recomputing")
        return None
    print(f"  cache: using {cache_path}")
    return cached


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
        default=OUTPUTS_DIR / "concentration",
        help="Directory for the output figure and data",
    )
    parser.add_argument(
        "--file-type",
        choices=["png", "pdf"],
        default="png",
        help="Output file format (default: png)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=20_000,
        # BCa reads the draws at adjusted percentiles. DiCiccio & Efron suggest
        # ~2000 as a floor, but the HHI's bias correction here is large (|z0| up
        # to ~1), pushing endpoints past the 0.1st/99.9th percentile, so more
        # draws are needed for them to rest on more than a few resamples.
        help="Bootstrap resamples per window (default: 20000)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260928, help="Bootstrap random seed"
    )
    parser.add_argument(
        "--table", action="store_true", help="Also print the numbers as a table"
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse bootstrap results from rolling_hhi.csv when they match (default)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cache_path = args.out_dir / "rolling_hhi.csv"

    data = (
        load_cache(cache_path, args.data, args.n_boot, args.seed)
        if args.cache
        else None
    )
    if data is None:
        data = rolling_hhi(load_records(args.data), args.n_boot, args.seed).assign(
            n_boot=args.n_boot, seed=args.seed
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path, index=False)
    if args.table:
        print_table(data)
    saved = plot_concentration(
        data, args.out_dir / f"concentration_rolling_hhi.{args.file_type}"
    )
    print(f"  saved: {saved}")

#!/usr/bin/env python3
"""
Power Analysis for Krippendorff's Alpha
=======================================
Simulates two-reviewer nominal coding to show how precisely the observed
Krippendorff's alpha estimates the true alpha as the validation sample grows.

Run from project root:
    uv run validation/power_analysis.py --n-categories 3

Model: for each item the two reviewers agree with probability equal to the
target ("true") alpha; otherwise each draws a category independently and
uniformly. Under uniform category prevalence this makes the expected observed
alpha equal the target, so the simulated lines are directly interpretable.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clouded_deps.directories import OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_ALPHAS = (0.5, 0.7, 0.8, 0.9)
DEFAULT_N_SIMS = 2000
DEFAULT_MIN_SAMPLES = 10
DEFAULT_MAX_SAMPLES = 300
DEFAULT_N_POINTS = 15
DEFAULT_SEED = 20260908
CI_LEVEL = 0.95
ACCEPTABLE_ALPHA = 0.667  # Krippendorff's conventional reliability threshold

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# Krippendorff's alpha (nominal, two coders, no missing values)
# ---------------------------------------------------------------------------
def krippendorff_alpha_nominal(
    coder_a: np.ndarray, coder_b: np.ndarray, n_categories: int
) -> np.ndarray:
    """Vectorised nominal alpha for many replicates at once.

    Args:
        coder_a: (n_replicates, n_items) integer category codes from reviewer 1.
        coder_b: (n_replicates, n_items) integer category codes from reviewer 2.
        n_categories: number of possible categories.

    Returns:
        (n_replicates,) array of alpha values; NaN where alpha is undefined
        (every value falls in a single category, so expected disagreement is 0).
    """
    n_replicates, n_items = coder_a.shape
    n_values = 2 * n_items

    # Observed disagreement: each disagreeing item contributes 2 ordered pairs.
    observed = 2.0 * (coder_a != coder_b).sum(axis=1)

    # Marginal counts per replicate, via one offset bincount.
    offsets = np.arange(n_replicates)[:, None] * n_categories
    flat = np.concatenate([coder_a + offsets, coder_b + offsets], axis=1).ravel()
    counts = np.bincount(flat, minlength=n_replicates * n_categories).reshape(
        n_replicates, n_categories
    )

    # Expected disagreement: all ordered pairs of unlike values.
    expected = float(n_values) ** 2 - (counts.astype(np.float64) ** 2).sum(axis=1)

    alpha = np.full(n_replicates, np.nan)
    valid = expected > 0
    alpha[valid] = 1.0 - (n_values - 1) * observed[valid] / expected[valid]
    return alpha


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_alphas(
    true_alpha: float,
    n_items: int,
    n_categories: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw `n_sims` two-reviewer codings and return their observed alphas."""
    agrees = rng.random((n_sims, n_items)) < true_alpha
    coder_a = rng.integers(0, n_categories, (n_sims, n_items))
    independent = rng.integers(0, n_categories, (n_sims, n_items))
    coder_b = np.where(agrees, coder_a, independent)
    return krippendorff_alpha_nominal(coder_a, coder_b, n_categories)


def sample_grid(min_samples: int, max_samples: int, n_points: int) -> np.ndarray:
    """Log-spaced sample sizes from `min_samples` to `max_samples`."""
    grid = np.geomspace(min_samples, max_samples, n_points)
    return np.unique(np.round(grid).astype(int))


def run_power_analysis(
    n_categories: int,
    max_samples: int,
    alphas: list[float],
    n_sims: int,
    min_samples: int,
    n_points: int,
    seed: int,
) -> tuple[np.ndarray, dict[float, dict[str, np.ndarray]]]:
    """Simulate observed alpha across the sample-size grid for each target alpha."""
    rng = np.random.default_rng(seed)
    sizes = sample_grid(min_samples, max_samples, n_points)
    lower_q = (1 - CI_LEVEL) / 2 * 100
    upper_q = 100 - lower_q

    results: dict[float, dict[str, np.ndarray]] = {}
    for true_alpha in alphas:
        means, lows, highs = [], [], []
        for n_items in sizes:
            observed = simulate_alphas(
                true_alpha, int(n_items), n_categories, n_sims, rng
            )
            means.append(np.nanmean(observed))
            lows.append(np.nanpercentile(observed, lower_q))
            highs.append(np.nanpercentile(observed, upper_q))
        results[true_alpha] = {
            "mean": np.array(means),
            "low": np.array(lows),
            "high": np.array(highs),
        }
    return sizes, results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_power_analysis(
    sizes: np.ndarray,
    results: dict[float, dict[str, np.ndarray]],
    n_categories: int,
    n_sims: int,
    output_path: Path,
) -> Path:
    """Render the power-analysis figure and write it to `output_path`."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(
        ACCEPTABLE_ALPHA,
        color=TEXT_MUTED,
        linewidth=1,
        linestyle=(0, (4, 4)),
        zorder=1,
    )
    ax.text(
        sizes[-1],
        ACCEPTABLE_ALPHA - 0.02,
        f"conventional threshold α = {ACCEPTABLE_ALPHA:.3f}",
        color=TEXT_MUTED,
        fontsize=8,
        ha="right",
        va="top",
    )

    for index, (true_alpha, series) in enumerate(results.items()):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        label = f"true α = {true_alpha:g}"
        ax.fill_between(
            sizes,
            series["low"],
            series["high"],
            color=color,
            alpha=0.11,
            linewidth=0,
            zorder=2,
        )
        for edge in ("low", "high"):
            ax.plot(
                sizes, series[edge], color=color, linewidth=0.8, alpha=0.45, zorder=2
            )
        ax.plot(sizes, series["mean"], color=color, linewidth=2, label=label, zorder=3)
        ax.annotate(
            label,
            xy=(sizes[-1], series["mean"][-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=9,
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel(
        "Double-coded items in the validation sample", color=TEXT_SECONDARY, fontsize=10
    )
    ax.set_ylabel("Observed Krippendorff's α", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(
        f"Precision of Krippendorff's α with two reviewers and {n_categories} categories",
        color=TEXT_PRIMARY,
        fontsize=13,
        pad=32,
        loc="left",
    )
    ax.text(
        0,
        1.005,
        f"Line = mean of {n_sims:,} simulations; band = central {CI_LEVEL:.0%} of simulated α",
        transform=ax.transAxes,
        color=TEXT_MUTED,
        fontsize=9,
        va="bottom",
    )

    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d1")
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=TEXT_SECONDARY)
    ax.margins(x=0.08)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-categories",
        type=int,
        required=True,
        help="Number of categories in the coding level",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Largest double-coded sample size to simulate",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Smallest sample size to simulate",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help="True alphas to simulate",
    )
    parser.add_argument(
        "--n-sims", type=int, default=DEFAULT_N_SIMS, help="Simulations per sample size"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=DEFAULT_N_POINTS,
        help="Sample sizes on the x-axis",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--output", type=Path, default=None, help="Output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_categories < 2:
        raise SystemExit("--n-categories must be at least 2")
    if args.max_samples <= args.min_samples:
        raise SystemExit("--max-samples must be greater than --min-samples")
    if not all(0 <= alpha <= 1 for alpha in args.alphas):
        raise SystemExit("--alphas must lie in [0, 1]")

    sizes, results = run_power_analysis(
        n_categories=args.n_categories,
        max_samples=args.max_samples,
        alphas=args.alphas,
        n_sims=args.n_sims,
        min_samples=args.min_samples,
        n_points=args.n_points,
        seed=args.seed,
    )

    output_path = (
        args.output
        or OUTPUTS_DIR / "validation" / f"power_analysis_k{args.n_categories}.png"
    )
    plot_power_analysis(sizes, results, args.n_categories, args.n_sims, output_path)

    print(
        f"Krippendorff α power analysis — {args.n_categories} categories, 2 reviewers, {args.n_sims:,} sims"
    )
    for true_alpha, series in results.items():
        print(f"\n  true α = {true_alpha:g}")
        for index, n_items in enumerate(sizes):
            low, high = series["low"][index], series["high"][index]
            print(
                f"    n={n_items:>5}  mean={series['mean'][index]:.3f}  {CI_LEVEL:.0%} CI [{low:.3f}, {high:.3f}]  width={high - low:.3f}"
            )
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()

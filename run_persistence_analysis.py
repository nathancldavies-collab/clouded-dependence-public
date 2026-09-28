#!/usr/bin/env python3
"""
Supplier Persistence: Naive vs Unclouded View
=============================================
Tests whether platforms persist where contractors churn. For each awarding
office and fiscal year, how often does a cloud supplier used this year get
used again next year? It is computed twice:

  - cA: suppliers keyed by contractor (naive view)
  - cB: suppliers keyed by platform   (unclouded view)
  - kappa = cB / cA

The scope is platform-attributed cloud records only: unattributed records have
the same key in both views and would only pull kappa toward 1.

Since platforms are fewer than contractors, kappa > 1 almost by construction.
The test is therefore against a label-shuffle null (platforms permuted across
records within each year), reported as its median kappa_0, the excess
kappa / kappa_0, and a one-sided permutation p-value.

Intervals are 95% BCa from a cluster bootstrap over offices, with cA and cB on
the same resamples so that kappa's interval is paired.

Run from project root:
    uv run run_persistence_analysis.py [--n-boot 20000] [--n-perm 10000]
                                       [--table]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clouded_deps.directories import DATA_DIR, OUTPUTS_DIR
from clouded_deps.pipeline.persistence import attach_buyer, persistence
from clouded_deps.pipeline.views import ATTRIBUTED_METHODS, load_records
from clouded_deps.stats import bca_interval

CI_LEVEL = 0.95


def load_attributed(data_path: Path, primes_path: Path) -> pd.DataFrame:
    """Platform-attributed cloud records, each with its buyer."""
    records = load_records(data_path)
    records = records[records["attribution_method"].isin(ATTRIBUTED_METHODS)]
    primes = pd.read_csv(
        primes_path,
        usecols=["Award ID", "Awarding Department/Agency", "Awarding Office"],
    )
    records = attach_buyer(records, primes)
    missing = records["buyer"].isna()
    if missing.any():
        print(f"  dropped {missing.sum():,} records with no matching prime award")
    return records[~missing]


def summarise(result: dict, n_boot: int, n_perm: int, seed: int) -> pd.DataFrame:
    """One row per statistic, with BCa intervals where they apply."""
    rows = []
    for stat in ("cA", "cB", "kappa"):
        low, high = bca_interval(
            result[stat], result[f"{stat}_boot"], result[f"{stat}_jack"], CI_LEVEL
        )
        rows.append(
            {
                "statistic": stat,
                "estimate": result[stat],
                "ci_low": low,
                "ci_high": high,
            }
        )

    null = result["kappa_null"]
    null_low, null_median, null_high = np.percentile(null, [2.5, 50, 97.5])
    p_value = (1 + np.sum(null >= result["kappa"])) / (1 + len(null))
    rows += [
        {
            "statistic": "kappa_null",
            "estimate": null_median,
            "ci_low": null_low,
            "ci_high": null_high,
        },
        {"statistic": "kappa_excess", "estimate": result["kappa"] / null_median},
        {"statistic": "p_value", "estimate": p_value},
    ]
    return pd.DataFrame(rows).assign(
        n_buyers=result["n_buyers"],
        n_records=result["n_records"],
        n_pairs_cA=result["n_pairs_cA"],
        n_pairs_cB=result["n_pairs_cB"],
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )


def print_table(summary: pd.DataFrame) -> None:
    """Print the summary as a readable table."""
    first = summary.iloc[0]
    print(
        f"\n  Supplier persistence  ({first['n_records']:,} attributed records,"
        f" {first['n_buyers']:,} offices;"
        f" {first['n_pairs_cA']:,} contractor pairs, {first['n_pairs_cB']:,} platform"
        " pairs at risk)"
    )
    labels = {
        "cA": "cA  contractor retained",
        "cB": "cB  platform retained",
        "kappa": "κ   cB ÷ cA",
        "kappa_null": "κ₀  shuffle-null median",
        "kappa_excess": "κ ÷ κ₀",
        "p_value": "p   (one-sided, κ ≥ κ_null)",
    }
    print(f"    {'-' * 62}")
    for _, row in summary.iterrows():
        interval = (
            f"[{row['ci_low']:.3f}–{row['ci_high']:.3f}]"
            if pd.notna(row["ci_low"])
            else ""
        )
        print(
            f"    {labels[row['statistic']]:<30s} {row['estimate']:>7.4f}  {interval}"
        )
    print(f"    {'-' * 62}")
    print("    intervals: 95% BCa over offices; κ₀ interval: null 2.5–97.5th pct")


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
        "--primes",
        type=Path,
        default=DATA_DIR
        / "02_processed"
        / "01_filtered"
        / "prime_services_filtered.csv",
        help="Filtered prime awards, for the awarding office",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUTS_DIR / "persistence",
        help="Directory for the output table",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=20_000,
        help="Bootstrap resamples over offices (default: 20000)",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=10_000,
        help="Label-shuffle permutations for the null (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260928, help="Random seed for both"
    )
    parser.add_argument(
        "--table", action="store_true", help="Also print the numbers as a table"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records = load_attributed(args.data, args.primes)
    result = persistence(records, args.n_boot, args.n_perm, args.seed)
    summary = summarise(result, args.n_boot, args.n_perm, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "persistence_summary.csv"
    summary.to_csv(out_path, index=False)
    if args.table:
        print_table(summary)
    print(f"  saved: {out_path}")

#!/usr/bin/env python3
"""
Baseline HHI Analysis on Merged Dataset
========================================
Calculates contractor-level market concentration (HHI) using the merged dataset
where primes with subawards have been decomposed into constituent flows.

This is the BASELINE view: "How concentrated is spending across contractors?"
No cloud classification or platform attribution - purely contractor-level.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Any


def calculate_hhi(market_shares: pd.Series) -> float:
    """
    Calculate Herfindahl-Hirschman Index.

    Args:
        market_shares: Series of market shares as percentages (0-100)

    Returns:
        HHI value (0-10,000)
    """
    return (market_shares ** 2).sum()


def classify_hhi(hhi: float) -> str:
    """Classify market concentration based on HHI."""
    if hhi < 1500:
        return "Unconcentrated (Competitive)"
    elif hhi < 2500:
        return "Moderately Concentrated"
    else:
        return "Highly Concentrated"


def calculate_gini(values: pd.Series) -> float:
    """
    Calculate Gini coefficient for inequality measurement.

    Args:
        values: Series of positive values (e.g., dollar amounts)

    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    values = values.sort_values().values
    n = len(values)
    if n == 0:
        return 0.0
    cumulative = np.cumsum(values)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    gini = (2.0 * np.sum((np.arange(1, n + 1) * values)) / (n * total)) - (n + 1) / n
    return gini


def calculate_contractor_hhi(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate baseline HHI on merged dataset grouped by contractor.

    This replaces the old approach of calculating HHI on primes grouped by
    Contractor Name. Now uses the merged dataset where subaward flows are
    separate records.

    Args:
        merged_df: Merged dataset from Stage 0C

    Returns:
        Dictionary with HHI, C4, C10, Gini, market shares, etc.
    """
    print("\n" + "=" * 80)
    print("BASELINE ANALYSIS: CONTRACTOR-LEVEL HHI (MERGED DATASET)")
    print("=" * 80)

    df = merged_df.copy()

    # Filter out zero/negative dollars (de-obligations can distort HHI/Gini)
    nonpositive_mask = df['dollars'] <= 0
    if nonpositive_mask.any():
        excluded_count = nonpositive_mask.sum()
        excluded_dollars = df.loc[nonpositive_mask, 'dollars'].sum()
        print(f"\n  Excluding non-positive dollars: {excluded_count:,} records, "
              f"${excluded_dollars:,.2f}")
        df = df.loc[~nonpositive_mask].copy()

    # Group by contractor entity (resolved UEI)
    contractor_spending = df.groupby('contractor').agg(
        dollars=('dollars', 'sum'),
        records=('dollars', 'count'),
        name=('contractor_name', 'first')
    ).sort_values('dollars', ascending=False)

    total_spending = contractor_spending['dollars'].sum()
    n_contractors = len(contractor_spending)

    if total_spending <= 0 or n_contractors == 0:
        print("\n  WARNING: No positive spending records available for HHI calculation.")
        return {
            'hhi': 0.0,
            'classification': 'Unconcentrated (Competitive)',
            'c4': 0.0,
            'c10': 0.0,
            'n_equivalent_firms': 0.0,
            'gini': 0.0,
            'n_contractors': 0,
            'total_spending': 0.0,
            'market_shares': pd.Series(dtype=float),
            'contractor_spending': contractor_spending,
        }

    # Market shares as percentages
    market_shares = (contractor_spending['dollars'] / total_spending * 100)

    # HHI
    hhi = calculate_hhi(market_shares)

    # Concentration ratios
    c4 = market_shares.nlargest(4).sum()
    c10 = market_shares.nlargest(10).sum()

    # Equivalent firms
    n_equivalent = 10000 / hhi if hhi > 0 else 0

    # Gini coefficient
    gini = calculate_gini(contractor_spending['dollars'])

    result = {
        'hhi': hhi,
        'classification': classify_hhi(hhi),
        'c4': c4,
        'c10': c10,
        'n_equivalent_firms': n_equivalent,
        'gini': gini,
        'n_contractors': n_contractors,
        'total_spending': total_spending,
        'market_shares': market_shares,
        'contractor_spending': contractor_spending,
    }

    # Print report
    print(f"\n  Total spending: ${total_spending:,.2f}")
    print(f"  Total contractors: {n_contractors:,}")
    print(f"\n  HHI: {hhi:,.1f}")
    print(f"  Classification: {classify_hhi(hhi)}")
    print(f"  C4 (top 4): {c4:.2f}%")
    print(f"  C10 (top 10): {c10:.2f}%")
    print(f"  Equivalent firms: {n_equivalent:.1f}")
    print(f"  Gini coefficient: {gini:.4f}")

    print(f"\n  Top 20 contractors:")
    for i, (entity, row) in enumerate(contractor_spending.head(20).iterrows(), 1):
        share = row['dollars'] / total_spending * 100
        print(f"    {i:2d}. {row['name'][:45]:45s} ${row['dollars']:>14,.2f} ({share:5.2f}%) [{row['records']:,} records]")

    return result


def compare_baseline_approaches(primes_df: pd.DataFrame, merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare HHI between old approach (primes only) and new approach (merged dataset).

    Args:
        primes_df: Original primes dataframe (must have resolved_entity and Contractor Name)
        merged_df: Merged dataset from Stage 0C

    Returns:
        Dictionary with comparison metrics
    """
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON: OLD (PRIMES ONLY) vs NEW (MERGED)")
    print("=" * 80)

    # Old approach: group primes by contractor
    print("\n--- Old approach: Primes grouped by contractor ---")
    primes_df = primes_df.copy()
    primes_df = primes_df[primes_df['Total Dollars Obligated'] > 0]
    old_spending = primes_df.groupby('resolved_entity').agg(
        dollars=('Total Dollars Obligated', 'sum'),
        name=('Contractor Name', 'first')
    )
    old_total = old_spending['dollars'].sum()
    old_shares = old_spending['dollars'] / old_total * 100
    old_hhi = calculate_hhi(old_shares)
    old_c4 = old_shares.nlargest(4).sum()
    old_n = len(old_spending)

    print(f"  Contractors: {old_n:,}")
    print(f"  HHI: {old_hhi:,.1f}")
    print(f"  C4: {old_c4:.2f}%")

    # New approach: group merged by contractor
    print("\n--- New approach: Merged dataset grouped by contractor ---")
    new_result = calculate_contractor_hhi(merged_df)

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    hhi_change = new_result['hhi'] - old_hhi
    c4_change = new_result['c4'] - old_c4
    n_change = new_result['n_contractors'] - old_n

    print(f"\n  {'Metric':<30s} {'Old (Primes)':>15s} {'New (Merged)':>15s} {'Change':>15s}")
    print(f"  {'-'*75}")
    print(f"  {'HHI':<30s} {old_hhi:>15,.1f} {new_result['hhi']:>15,.1f} {hhi_change:>+15,.1f}")
    print(f"  {'C4 (%)':<30s} {old_c4:>15.2f} {new_result['c4']:>15.2f} {c4_change:>+15.2f}")
    print(f"  {'N contractors':<30s} {old_n:>15,} {new_result['n_contractors']:>15,} {n_change:>+15,}")
    print(f"  {'Classification':<30s} {classify_hhi(old_hhi):>15s} {new_result['classification']:>15s}")

    if hhi_change < 0:
        print(f"\n  Baseline HHI DECREASED by {abs(hhi_change):,.1f} points.")
        print(f"  Market appears LESS concentrated when subaward flows are visible.")
        print(f"  This is because {n_change:+,} additional entities appear in the merged data.")
    elif hhi_change > 0:
        print(f"\n  Baseline HHI INCREASED by {hhi_change:,.1f} points.")
    else:
        print(f"\n  Baseline HHI unchanged.")

    return {
        'old_hhi': old_hhi,
        'new_hhi': new_result['hhi'],
        'old_c4': old_c4,
        'new_c4': new_result['c4'],
        'old_n_contractors': old_n,
        'new_n_contractors': new_result['n_contractors'],
        'hhi_change': hhi_change,
        'new_result': new_result,
    }


# --- Main Execution ---

def main():
    """Run baseline HHI analysis on merged dataset."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    merged_path = os.path.join(project_root, 'data', '02_processed', '02_merged', 'merged_dataset.csv')

    if not os.path.exists(merged_path):
        print(f"ERROR: Merged dataset not found at {merged_path}")
        print("Run create_merged_dataset.py first (Stage 0C)")
        return

    print("Loading merged dataset...")
    merged_df = pd.read_csv(merged_path)
    print(f"  Loaded {len(merged_df):,} records")

    result = calculate_contractor_hhi(merged_df)
    return result


if __name__ == '__main__':
    main()
    

#!/usr/bin/env python3
"""
Filter Prime Contracts: Hardware vs Services/Software
======================================================
Filters IT contracts to services/software only, excluding hardware purchases
that aren't relevant to cloud platform consolidation analysis.

Filters applied (in order):
1. Date range: FY2017-2024 (Effective Date between 2017-01-01 and 2024-12-31)
2. PSC code: Exclude known hardware PSC prefixes
3. Keyword: Exclude contracts strongly flagged as hardware by description

Adapted from 02_data_cleaning_and_filtering.ipynb.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple

# --- Hardware PSC codes to EXCLUDE ---
HARDWARE_PSC = [
    '7010',  # ADP Equipment (mainframes, servers, storage)
    '7020',  # ADP Peripheral Equipment (printers, scanners, monitors)
    '7021',  # ADP Office Machines
    '7025',  # ADP System Configuration (complete systems)
    '7035',  # ADP Support Equipment
    '5895',  # Miscellaneous Communication Equipment
    '5998',  # Electrical and Electronic Equipment Components
]

# --- Keyword lists ---
HARDWARE_KEYWORDS = [
    'desktop', 'laptop', 'computer', 'workstation', 'pc', 'personal computer',
    'printer', 'monitor', 'keyboard', 'mouse', 'display',
    'server hardware', 'rack mount', 'blade server', 'physical server',
    'network switch', 'router', 'firewall appliance', 'firewall hardware',
    'storage array', 'disk drive', 'hard drive', 'tape backup',
    'equipment procurement', 'hardware refresh', 'hardware upgrade',
    'ups', 'uninterruptible power', 'power supply',
]

SERVICE_KEYWORDS = [
    # Cloud
    'cloud', 'aws', 'azure', 'google cloud', 'gcp', 'oracle cloud',
    'iaas', 'paas', 'saas', 'cloud migration', 'cloud hosting',
    'serverless', 'lambda', 'elastic compute', 'ec2', 's3',
    # Software
    'software', 'application', 'platform', 'database', 'software license',
    'software development', 'custom software', 'software as a service',
    'erp', 'crm', 'enterprise software',
    # Services
    'consulting', 'integration', 'managed services', 'support services',
    'implementation', 'maintenance', 'technical support', 'help desk',
    'development services', 'engineering services', 'professional services',
    'cybersecurity services', 'security operations', 'security services',
    'devops', 'devsecops', 'agile development', 'systems integration',
]


def classify_by_keywords(description: str) -> str:
    """
    Classify a contract description as hardware, service, or unknown.

    Returns: 'hardware', 'hardware_weak', 'service', or 'unknown'
    """
    if pd.isna(description):
        return 'unknown'

    desc_lower = str(description).lower()

    hardware_count = sum(1 for kw in HARDWARE_KEYWORDS if kw in desc_lower)
    service_count = sum(1 for kw in SERVICE_KEYWORDS if kw in desc_lower)

    if hardware_count >= 2 and service_count == 0:
        return 'hardware'
    elif service_count > 0:
        return 'service'
    elif hardware_count > 0:
        return 'hardware_weak'
    else:
        return 'unknown'


def filter_primes(primes_df: pd.DataFrame,
                  year_min: int = 2017,
                  year_max: int = 2024) -> Tuple[pd.DataFrame, dict]:
    """
    Apply date, PSC, and keyword filters to prime contracts.

    Args:
        primes_df: Raw primes DataFrame (must have 'Effective Date' already parsed,
                   'Total Dollars Obligated' already numeric, 'PSC Code' column)
        year_min: Start of date range (inclusive)
        year_max: End of date range (inclusive)

    Returns:
        (filtered_df, summary_dict)
    """
    print("\n" + "=" * 80)
    print("FILTERING PRIME CONTRACTS")
    print("=" * 80)

    baseline_count = len(primes_df)
    baseline_dollars = primes_df['Total Dollars Obligated'].sum()
    print(f"\n  Baseline: {baseline_count:,} contracts, ${baseline_dollars / 1e9:.1f}B")

    # --- Step 1: Date filter ---
    primes_df = primes_df.copy()
    year_col = primes_df['Effective Date'].dt.year
    date_mask = year_col.between(year_min, year_max)
    df = primes_df[date_mask].copy()

    after_date_count = len(df)
    after_date_dollars = df['Total Dollars Obligated'].sum()
    print(f"\n  After date filter ({year_min}-{year_max}):")
    print(f"    {after_date_count:,} contracts ({after_date_count / baseline_count * 100:.1f}%), "
          f"${after_date_dollars / 1e9:.1f}B ({after_date_dollars / baseline_dollars * 100:.1f}%)")

    # --- Step 2: PSC code filter ---
    df['psc_prefix'] = df['PSC Code'].astype(str).str[:4]
    df['is_hardware_psc'] = df['psc_prefix'].isin(HARDWARE_PSC)
    df_psc = df[~df['is_hardware_psc']].copy()

    after_psc_count = len(df_psc)
    after_psc_dollars = df_psc['Total Dollars Obligated'].sum()
    psc_excluded = after_date_count - after_psc_count
    print(f"\n  After PSC filter (exclude {len(HARDWARE_PSC)} hardware codes):")
    print(f"    {after_psc_count:,} contracts ({after_psc_count / baseline_count * 100:.1f}%), "
          f"${after_psc_dollars / 1e9:.1f}B")
    print(f"    Excluded: {psc_excluded:,} hardware contracts")

    # --- Step 3: Keyword filter ---
    df_psc['keyword_classification'] = df_psc['Description'].apply(classify_by_keywords)
    keep_classifications = ['service', 'unknown', 'hardware_weak']
    df_final = df_psc[df_psc['keyword_classification'].isin(keep_classifications)].copy()

    after_kw_count = len(df_final)
    after_kw_dollars = df_final['Total Dollars Obligated'].sum()
    kw_excluded = after_psc_count - after_kw_count
    print(f"\n  After keyword filter (exclude strong hardware matches):")
    print(f"    {after_kw_count:,} contracts ({after_kw_count / baseline_count * 100:.1f}%), "
          f"${after_kw_dollars / 1e9:.1f}B")
    print(f"    Excluded: {kw_excluded:,} keyword-flagged hardware contracts")

    # --- Summary ---
    total_excluded = baseline_count - after_kw_count
    total_dollars_excluded = baseline_dollars - after_kw_dollars
    print(f"\n  Final filtered dataset:")
    print(f"    {after_kw_count:,} contracts, ${after_kw_dollars / 1e9:.1f}B")
    print(f"    Total excluded: {total_excluded:,} contracts ({total_excluded / baseline_count * 100:.1f}%), "
          f"${total_dollars_excluded / 1e9:.1f}B ({total_dollars_excluded / baseline_dollars * 100:.1f}%)")

    # Drop temporary columns
    df_final = df_final.drop(columns=['psc_prefix', 'is_hardware_psc', 'keyword_classification'],
                              errors='ignore')

    summary = {
        'baseline_count': baseline_count,
        'baseline_dollars': baseline_dollars,
        'after_date_count': after_date_count,
        'after_date_dollars': after_date_dollars,
        'after_psc_count': after_psc_count,
        'after_psc_dollars': after_psc_dollars,
        'after_keyword_count': after_kw_count,
        'after_keyword_dollars': after_kw_dollars,
        'total_excluded_count': total_excluded,
        'total_excluded_dollars': total_dollars_excluded,
    }

    return df_final, summary

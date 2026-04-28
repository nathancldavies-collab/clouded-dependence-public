#!/usr/bin/env python3
"""
Stage 0C: Create Merged Dataset
================================
Merges prime contracts with their subawards into a unified analysis dataset.

Methodology:
- For primes WITH subawards (~17%): Replace with constituent flows
  * Retained portion: prime_dollars - sum(sub_dollars)
  * Individual subcontracts: each sub becomes separate record
- For primes WITHOUT subawards (~83%): Keep as-is

Critical invariant: total dollars in merged == total dollars in primes (no double-counting)
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, Optional, Union


# --- Data Loading ---

def compute_fiscal_year(dates: pd.Series) -> pd.Series:
    """Compute federal fiscal year (Oct-Sep) from a datetime series."""
    fy = dates.dt.year
    fy = fy.where(dates.dt.month < 10, fy + 1)
    return fy


def load_primes(filepath: str) -> pd.DataFrame:
    """Load prime awards CSV with standard cleaning."""
    print(f"Loading primes from: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # Clean dollar columns (handle string types from pandas 3.0+)
    for col in ['Total Dollars Obligated', 'Current Value of Award', 'Potential Value of Award']:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col].astype(str).str.replace('$', '', regex=False)
                       .str.replace(',', '', regex=False)
                       .astype(float))

    # Derive fiscal year from Effective Date
    df['Effective Date'] = pd.to_datetime(df['Effective Date'], errors='coerce')
    df['fiscal_year'] = compute_fiscal_year(df['Effective Date'])

    # Entity resolution: use Parent UEI when available, else UEI
    df['resolved_entity'] = df['Contractor Parent UEI'].fillna(df['Contractor UEI'])

    print(f"  Loaded {len(df):,} prime contracts")
    print(f"  Total dollars: ${df['Total Dollars Obligated'].sum():,.2f}")
    print(f"  Unique Award IDs: {df['Award ID'].nunique():,}")
    print(f"  Unique contractors (by resolved entity): {df['resolved_entity'].nunique():,}")
    print(f"  Primes with subs (Subaward Count > 0): {(df['Subaward Count'] > 0).sum():,}")

    return df


def dedupe_primes_by_award(primes_df: pd.DataFrame,
                           dollars_agg: str = "max") -> pd.DataFrame:
    """
    Deduplicate primes by Award ID to avoid double-counting.

    Args:
        primes_df: Prime awards dataframe (post-filtering).
        dollars_agg: Aggregation for dollar fields ("max" or "sum").

    Returns:
        Deduplicated DataFrame with one row per Award ID.
    """
    if dollars_agg not in ("max", "sum"):
        raise ValueError("dollars_agg must be 'max' or 'sum'")

    df = primes_df.copy()

    def _first_non_null(s: pd.Series) -> Optional[str]:
        s = s.dropna()
        return s.iloc[0] if len(s) > 0 else None

    def _longest_text(s: pd.Series) -> Optional[str]:
        s = s.dropna().astype(str)
        if len(s) == 0:
            return None
        return s.iloc[s.str.len().idxmax()]

    dollars_cols = ['Total Dollars Obligated', 'Current Value of Award', 'Potential Value of Award']
    agg_map = {c: dollars_agg for c in dollars_cols if c in df.columns}

    # Dates: use latest available
    for col in ['Effective Date', 'Completion Date', 'Potential End Date']:
        if col in df.columns:
            agg_map[col] = 'max'

    # Counts: take max
    if 'Subaward Count' in df.columns:
        agg_map['Subaward Count'] = 'max'

    # IDs / names / descriptions
    text_cols_long = ['Description']
    text_cols_first = [
        'Parent Award ID', 'PIID', 'Contractor Name', 'Contractor UEI',
        'Contractor Parent UEI', 'Award Type', 'NAICS Code', 'PSC Code',
        'Funding Department/Agency', 'Funding Parent Office', 'Funding Office',
        'Funding Office City', 'Funding Office State',
        'Awarding Department/Agency', 'Awarding Parent Office', 'Awarding Office',
        'Price Type', 'Type of Set Aside', 'Extent Competed',
        'Clinger-Cohen Act Compliant'
    ]

    for col in text_cols_long:
        if col in df.columns:
            agg_map[col] = _longest_text
    for col in text_cols_first:
        if col in df.columns:
            agg_map[col] = _first_non_null

    before = len(df)
    deduped = df.groupby('Award ID', as_index=False).agg(agg_map)

    # Recompute resolved entity and fiscal year (in case of mixed rows)
    deduped['Contractor UEI'] = deduped.get('Contractor UEI')
    deduped['Contractor Parent UEI'] = deduped.get('Contractor Parent UEI')
    deduped['resolved_entity'] = deduped['Contractor Parent UEI'].fillna(deduped['Contractor UEI'])
    if 'Effective Date' in deduped.columns:
        deduped['Effective Date'] = pd.to_datetime(deduped['Effective Date'], errors='coerce')
        deduped['fiscal_year'] = compute_fiscal_year(deduped['Effective Date'])

    print(f"\nDeduped primes by Award ID: {before:,} -> {len(deduped):,} "
          f"(dollars_agg={dollars_agg})")
    return deduped


def load_subs(filepath: str) -> pd.DataFrame:
    """Load subawards CSV with standard cleaning."""
    print(f"\nLoading subawards from: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # Clean dollar column (handle string types from pandas 3.0+)
    if not pd.api.types.is_numeric_dtype(df['Total Amount Obligated']):
        df['Total Amount Obligated'] = (df['Total Amount Obligated'].astype(str)
                                        .str.replace('$', '', regex=False)
                                        .str.replace(',', '', regex=False)
                                        .astype(float))

    print(f"  Loaded {len(df):,} subaward records")
    print(f"  Total sub dollars: ${df['Total Amount Obligated'].sum():,.2f}")
    print(f"  Unique subcontractors (by UEI): {df['Subcontractor UEI'].nunique():,}")

    return df


# --- Prime Award ID Extraction ---

def extract_prime_award_id(subaward_id: str, prime_id_set: set) -> Optional[str]:
    """
    Extract the prime Award ID from a Subaward ID.

    Strategy:
    1. Try first 6 underscore-separated parts (standard format)
    2. Longest prefix match against known prime Award IDs
    3. Return None if no match
    """
    if not isinstance(subaward_id, str) or not subaward_id:
        return None

    parts = subaward_id.split('_')

    # Standard: CONT_AWD_XXXX_XXXX_XXXX_XXXX (6 parts)
    if len(parts) >= 6:
        candidate = '_'.join(parts[:6])
        if candidate in prime_id_set:
            return candidate

    # Longest prefix match
    for n in range(len(parts), 1, -1):
        candidate = '_'.join(parts[:n])
        if candidate in prime_id_set:
            return candidate

    return None


# --- Core Merge Function ---

def create_merged_dataset(primes_df: pd.DataFrame,
                          subs_df: pd.DataFrame,
                          return_unlinked: bool = False
                          ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Merge primes and subs into a unified dataset for analysis.

    For primes WITH subawards:
      - Retained portion = prime_dollars - sum(sub_dollars)
      - Each subaward becomes a separate record

    For primes WITHOUT subawards:
      - Keep as-is

    Returns:
        If return_unlinked=False:
            DataFrame with columns: contractor, contractor_name, dollars, description,
            record_type, original_prime_id, fiscal_year, prime_contractor
        If return_unlinked=True:
            (merged_df, unlinked_subs_df)
    """
    print("\n" + "=" * 80)
    print("STAGE 0C: CREATING MERGED DATASET")
    print("=" * 80)

    # Build UEI -> Parent UEI cross-reference from primes
    # Subcontractor data lacks Parent UEI, but if a sub's UEI appears as a
    # Contractor UEI in primes, we can look up that prime's Parent UEI.
    _has_parent = primes_df.dropna(subset=['Contractor UEI', 'Contractor Parent UEI'])
    _has_parent = _has_parent[_has_parent['Contractor UEI'] != _has_parent['Contractor Parent UEI']]
    uei_to_parent = (_has_parent.drop_duplicates('Contractor UEI')
                     .set_index('Contractor UEI')['Contractor Parent UEI'].to_dict())

    print(f"\n  Sub entity resolution: {len(uei_to_parent):,} UEI->Parent mappings from primes")

    # Build prime ID lookup set
    prime_id_set = set(primes_df['Award ID'].unique())

    # Extract prime Award ID for each subaward
    print("\nLinking subawards to prime contracts...")
    subs_df = subs_df.copy()
    subs_df['prime_award_id'] = subs_df['Subaward ID'].apply(
        lambda sid: extract_prime_award_id(sid, prime_id_set)
    )

    linked = subs_df['prime_award_id'].notna().sum()
    total_subs = len(subs_df)
    print(f"  Linked: {linked:,} / {total_subs:,} ({linked/total_subs*100:.1f}%)")

    # Preserve unlinked subs for reporting (but exclude from merged to keep prime totals)
    unlinked_df = subs_df[subs_df['prime_award_id'].isna()].copy()
    if len(unlinked_df) > 0:
        unlinked = len(unlinked_df)
        unlinked_dollars = unlinked_df['Total Amount Obligated'].sum()
        print(f"  WARNING: {unlinked:,} subawards could not be linked to a prime "
              f"(${unlinked_dollars:,.2f})")
        # Drop unlinked subs from merged dataset
        subs_df = subs_df[subs_df['prime_award_id'].notna()].copy()
    else:
        unlinked_df = unlinked_df.iloc[0:0]

    # Group subs by prime award ID
    subs_by_prime = subs_df.groupby('prime_award_id')

    # Build lookup for prime info (fiscal year, etc.)
    # Handle duplicate Award IDs by taking the first occurrence
    # Track which primes have subs
    primes_with_subs = set(subs_by_prime.groups.keys())
    print(f"\nPrimes with linked subawards: {len(primes_with_subs):,}")
    print(f"Primes without subawards: {len(prime_id_set) - len(primes_with_subs):,}")

    # --- Build merged records ---
    merged_records = []
    scaled_count = 0  # Primes where sub total exceeded prime amount

    print("\nBuilding merged records...")
    primes_processed = 0

    for _, prime_row in primes_df.iterrows():
        award_id = prime_row['Award ID']
        prime_dollars = prime_row['Total Dollars Obligated']
        prime_entity = prime_row['resolved_entity']
        prime_name = prime_row['Contractor Name']
        prime_desc = prime_row['Description']
        prime_fy = prime_row['fiscal_year']

        if award_id in primes_with_subs:
            # This prime has subawards - decompose it
            sub_group = subs_by_prime.get_group(award_id)
            sub_total = sub_group['Total Amount Obligated'].sum()

            # Handle case where sub totals exceed prime amount
            # (data inconsistency in USAspending reporting)
            if sub_total > prime_dollars and sub_total > 0:
                # Cap subaward amounts proportionally to match prime total
                scale_factor = prime_dollars / sub_total
                scaled_count += 1
            else:
                scale_factor = 1.0

            # Retained portion (what the prime contractor keeps)
            scaled_sub_total = sub_total * scale_factor
            retained = prime_dollars - scaled_sub_total
            if retained > 0.01:  # threshold to avoid floating point dust
                merged_records.append({
                    'contractor': prime_entity,
                    'contractor_name': prime_name,
                    'dollars': retained,
                    'description': prime_desc,
                    'record_type': 'prime_retained',
                    'original_prime_id': award_id,
                    'fiscal_year': prime_fy,
                    'prime_contractor': prime_name,
                })

            # Each subaward becomes its own record
            for _, sub_row in sub_group.iterrows():
                # For subs, cross-reference UEI against primes to find parent
                sub_uei = sub_row['Subcontractor UEI']
                sub_entity = uei_to_parent.get(sub_uei, sub_uei) if pd.notna(sub_uei) else sub_uei
                sub_name = sub_row['Subcontractor']
                sub_dollars = sub_row['Total Amount Obligated'] * scale_factor
                sub_desc = sub_row['Subaward Description']

                merged_records.append({
                    'contractor': sub_entity if pd.notna(sub_entity) else sub_name,
                    'contractor_name': sub_name,
                    'dollars': sub_dollars,
                    'description': sub_desc if pd.notna(sub_desc) else prime_desc,
                    'record_type': 'subcontract',
                    'original_prime_id': award_id,
                    'fiscal_year': prime_fy,
                    'prime_contractor': prime_name,
                })
        else:
            # No subawards - keep as-is
            merged_records.append({
                'contractor': prime_entity,
                'contractor_name': prime_name,
                'dollars': prime_dollars,
                'description': prime_desc,
                'record_type': 'prime_no_subs',
                'original_prime_id': award_id,
                'fiscal_year': prime_fy,
                'prime_contractor': None,
            })

        primes_processed += 1
        if primes_processed % 50000 == 0:
            print(f"  Processed {primes_processed:,} primes...")

    merged_df = pd.DataFrame(merged_records)
    print(f"  Processed {primes_processed:,} primes total")
    if scaled_count > 0:
        print(f"  NOTE: {scaled_count:,} primes had sub totals exceeding prime amount "
              f"(sub dollars scaled down proportionally)")

    if return_unlinked:
        return merged_df, unlinked_df
    return merged_df


# --- Verification ---

def verify_merged_dataset(merged_df: pd.DataFrame, primes_df: pd.DataFrame,
                          subs_df: pd.DataFrame) -> bool:
    """
    Verify the merged dataset maintains dollar integrity.

    Critical check: merged total == prime total (no double-counting)
    """
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    prime_total = primes_df['Total Dollars Obligated'].sum()
    merged_total = merged_df['dollars'].sum()
    difference = abs(merged_total - prime_total)

    print(f"\n  Prime total:  ${prime_total:,.2f}")
    print(f"  Merged total: ${merged_total:,.2f}")
    print(f"  Difference:   ${difference:,.2f}")

    # Record type breakdown
    print(f"\n  Record type breakdown:")
    for rtype, group in merged_df.groupby('record_type'):
        count = len(group)
        dollars = group['dollars'].sum()
        print(f"    {rtype:20s}: {count:>8,} records, ${dollars:>16,.2f}")

    print(f"\n  Total merged records: {len(merged_df):,}")
    print(f"  Unique contractors (merged): {merged_df['contractor'].nunique():,}")

    # Check for negative retained amounts (sub dollars > prime dollars)
    retained = merged_df[merged_df['record_type'] == 'prime_retained']
    if (retained['dollars'] < 0).any():
        neg_count = (retained['dollars'] < 0).sum()
        neg_total = retained.loc[retained['dollars'] < 0, 'dollars'].sum()
        print(f"\n  WARNING: {neg_count:,} retained records have negative dollars (${neg_total:,.2f})")
        print(f"  This means subaward totals exceed prime amounts for those contracts")

    # Dollar integrity check
    passed = difference < 1.0
    if passed:
        print(f"\n  PASSED: Dollar totals match (diff < $1.00)")
    else:
        print(f"\n  FAILED: Dollar totals do not match! Difference: ${difference:,.2f}")

    return passed


def print_merge_summary(merged_df: pd.DataFrame):
    """Print summary statistics of the merged dataset."""
    print("\n" + "=" * 80)
    print("MERGED DATASET SUMMARY")
    print("=" * 80)

    total_records = len(merged_df)
    total_dollars = merged_df['dollars'].sum()

    # By record type
    print("\nBy record type:")
    for rtype in ['prime_no_subs', 'prime_retained', 'subcontract']:
        subset = merged_df[merged_df['record_type'] == rtype]
        count = len(subset)
        dollars = subset['dollars'].sum()
        pct_records = count / total_records * 100
        pct_dollars = dollars / total_dollars * 100
        print(f"  {rtype:20s}: {count:>8,} records ({pct_records:5.1f}%), "
              f"${dollars:>16,.2f} ({pct_dollars:5.1f}%)")

    # Top contractors by dollars
    print("\nTop 15 contractors (by dollars):")
    contractor_totals = merged_df.groupby(['contractor', 'contractor_name']).agg(
        dollars=('dollars', 'sum'),
        records=('dollars', 'count')
    ).sort_values('dollars', ascending=False)

    for i, ((entity, name), row) in enumerate(contractor_totals.head(15).iterrows(), 1):
        pct = row['dollars'] / total_dollars * 100
        print(f"  {i:2d}. {name[:45]:45s} ${row['dollars']:>14,.2f} ({pct:5.2f}%) [{row['records']:,} records]")

    # Fiscal year breakdown
    print("\nBy fiscal year:")
    fy_totals = merged_df.groupby('fiscal_year').agg(
        dollars=('dollars', 'sum'),
        records=('dollars', 'count')
    ).sort_index()
    for fy, row in fy_totals.iterrows():
        fy_display = int(fy) if pd.notna(fy) else 'Unknown'
        print(f"  FY{fy_display}: {row['records']:>8,} records, ${row['dollars']:>14,.2f}")


# --- Main Execution ---

def main():
    """Run Stage 0C: Create Merged Dataset."""
    # Paths (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(project_root, 'data', '01_raw')
    output_dir = os.path.join(project_root, 'data', '02_processed', '02_merged')

    prime_path = os.path.join(raw_dir, 'prime_awards.csv')
    sub_path = os.path.join(raw_dir, 'subawards.csv')
    output_path = os.path.join(output_dir, 'merged_dataset.csv')

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    primes_df = load_primes(prime_path)
    subs_df = load_subs(sub_path)

    # Create merged dataset
    merged_df = create_merged_dataset(primes_df, subs_df)

    # Verify integrity
    passed = verify_merged_dataset(merged_df, primes_df, subs_df)
    if not passed:
        print("\nERROR: Verification failed. Check merge logic.")
        sys.exit(1)

    # Print summary
    print_merge_summary(merged_df)

    # Save
    print(f"\nSaving merged dataset to: {output_path}")
    merged_df.to_csv(output_path, index=False)
    print(f"  Saved {len(merged_df):,} records")

    return merged_df


if __name__ == '__main__':
    main()

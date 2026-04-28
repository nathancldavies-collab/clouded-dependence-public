#!/usr/bin/env python3
"""
Stage 2: Platform Attribution (Phases 1-3 + Final Synthesis)
==============================================================
Attributes cloud spending to specific platform entities.

Operates on the classified dataset from Stage 1, where each record
already has: is_cloud, cloud_classification, platform_stage1.

Attribution phases (cloud records only):
  Phase 1: Direct entity — contractor IS a platform vendor
  Phase 2: Description-based — use Stage 1 platform identification
  Phase 3: Pattern matching — learn contractor platform specialisations

Final synthesis assigns final_platform to every record:
  - Non-cloud: final_platform = contractor_name
  - Attributed cloud: final_platform = platform name
  - Unattributed cloud: final_platform = contractor_name
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Optional, Any


# =============================================================================
# PLATFORM ENTITY LIST (for Phase 1: Direct Attribution)
# =============================================================================

PLATFORM_ENTITIES = {
    # AWS
    'AMAZON WEB SERVICES, INC.': 'AWS',
    'AMAZON WEB SERVICES LLC': 'AWS',
    'AMAZON WEB SERVICES INC.': 'AWS',
    'AMAZON WEB SERVICES INC': 'AWS',
    'AMAZON.COM SERVICES LLC': 'AWS',
    'AMAZON.COM, INC.': 'AWS',
    # Azure / Microsoft
    'MICROSOFT CORPORATION': 'Azure',
    'MICROSOFT CORP.': 'Azure',
    'MICROSOFT CORP': 'Azure',
    # Google Cloud
    'GOOGLE LLC': 'Google Cloud',
    'GOOGLE INC.': 'Google Cloud',
    'ALPHABET INC.': 'Google Cloud',
    # Salesforce
    'SALESFORCE.COM, INC.': 'Salesforce',
    'SALESFORCE, INC.': 'Salesforce',
    'SALESFORCE.COM INC.': 'Salesforce',
    'SALESFORCE INC.': 'Salesforce',
    'SALESFORCE.COM, INC': 'Salesforce',
    # Oracle Cloud
    'ORACLE AMERICA, INC.': 'Oracle Cloud',
    'ORACLE CORPORATION': 'Oracle Cloud',
    'ORACLE AMERICA INC.': 'Oracle Cloud',
    # IBM Cloud
    'IBM CORPORATION': 'IBM Cloud',
    'INTERNATIONAL BUSINESS MACHINES CORPORATION': 'IBM Cloud',
    'IBM CORP.': 'IBM Cloud',
    # ServiceNow
    'SERVICENOW, INC.': 'ServiceNow',
    'SERVICENOW INC.': 'ServiceNow',
    'SERVICENOW INC': 'ServiceNow',
    # Workday
    'WORKDAY, INC.': 'Workday',
    'WORKDAY INC.': 'Workday',
    'WORKDAY INC': 'Workday',
}

_PLATFORM_ENTITIES_UPPER = {k.upper(): v for k, v in PLATFORM_ENTITIES.items()}

# Contractor name patterns for secondary matching (from PLATFORM_PATTERNS)
_CONTRACTOR_NAME_PATTERNS = {
    'AWS': ['amazon', 'aws', 'amazon web services'],
    'Azure': ['microsoft', 'azure'],
    'Google Cloud': ['google', 'alphabet'],
    'Oracle Cloud': ['oracle'],
    'IBM Cloud': ['ibm', 'international business machines'],
    'Salesforce': ['salesforce'],
    'ServiceNow': ['servicenow'],
    'Workday': ['workday'],
}

# Concrete platform providers (for multi-platform splitting)
CONCRETE_PLATFORMS = {
    'AWS', 'Azure', 'Google Cloud', 'Oracle Cloud', 'IBM Cloud',
    'Salesforce', 'ServiceNow', 'Workday'
}


def _parse_platform_mentions(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.startswith('[') and v.endswith(']'):
            try:
                import json
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return []
    return []


def expand_platform_mentions(df: pd.DataFrame,
                             mentions_col: str = 'platform_mentions',
                             base_platform_col: str = 'final_platform') -> pd.DataFrame:
    """
    Expand rows with multiple platform mentions by splitting dollars equally.

    Returns a new DataFrame with:
      - platform_for_analysis: platform used for HHI/shares
      - dollars_split: dollars after split (sums to original total)
    """
    rows = []
    for _, row in df.iterrows():
        mentions = _parse_platform_mentions(row.get(mentions_col))
        mentions = [m for m in mentions if m in CONCRETE_PLATFORMS]
        base_platform = row.get(base_platform_col)
        dollars = row.get('dollars', 0)

        if mentions and len(mentions) >= 2:
            split = dollars / len(mentions) if len(mentions) > 0 else 0
            for plat in mentions:
                new_row = row.copy()
                new_row['platform_for_analysis'] = plat
                new_row['dollars_split'] = split
                new_row['attribution_method'] = 'multi_platform_split'
                rows.append(new_row)
        elif mentions and len(mentions) == 1:
            new_row = row.copy()
            new_row['platform_for_analysis'] = mentions[0]
            new_row['dollars_split'] = dollars
            rows.append(new_row)
        else:
            new_row = row.copy()
            new_row['platform_for_analysis'] = base_platform
            new_row['dollars_split'] = dollars
            rows.append(new_row)

    return pd.DataFrame(rows)


# =============================================================================
# PHASE 1: DIRECT ENTITY ATTRIBUTION
# =============================================================================

def attribute_direct_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: Check if contractor IS a platform entity.

    Catches BOTH:
    - Direct prime contracts to AWS/Azure/etc.
    - Subcontracts flowing to platform vendors (KEY finding)
    """
    print("\n" + "-" * 60)
    print("Phase 1: Direct Entity Attribution")
    print("-" * 60)

    df = df.copy()
    df['platform_phase1'] = None
    df['attribution_method'] = None

    def _match_platform_entity(name):
        if pd.isna(name):
            return None
        name_upper = str(name).strip().upper()
        if name_upper in _PLATFORM_ENTITIES_UPPER:
            return _PLATFORM_ENTITIES_UPPER[name_upper]
        for entity_name, platform in _PLATFORM_ENTITIES_UPPER.items():
            if entity_name in name_upper:
                return platform
        return None

    df['platform_phase1'] = df['contractor_name'].apply(_match_platform_entity)

    # Secondary: check contractor name patterns
    import re as _re
    def _check_name_affiliation(name):
        if pd.isna(name):
            return None
        name_lower = str(name).lower()
        for platform, patterns in _CONTRACTOR_NAME_PATTERNS.items():
            for pattern_name in patterns:
                if _re.search(r'\b' + _re.escape(pattern_name) + r'\b', name_lower):
                    return platform
        return None

    name_affiliation = df['contractor_name'].apply(_check_name_affiliation)
    mask_no_phase1 = df['platform_phase1'].isna()
    df.loc[mask_no_phase1, 'platform_phase1'] = name_affiliation[mask_no_phase1]

    df.loc[df['platform_phase1'].notna(), 'attribution_method'] = 'phase1_direct'

    direct_count = df['platform_phase1'].notna().sum()
    direct_dollars = df.loc[df['platform_phase1'].notna(), 'dollars'].sum()

    print(f"  Direct entity matches: {direct_count:,}")
    print(f"  Direct entity dollars: ${direct_dollars:,.2f}")

    for rtype in ['prime_no_subs', 'prime_retained', 'subcontract']:
        subset = df[(df['record_type'] == rtype) & (df['platform_phase1'].notna())]
        if len(subset) > 0:
            print(f"    {rtype:20s}: {len(subset):,} records, ${subset['dollars'].sum():,.2f}")

    if direct_count > 0:
        print(f"\n  Direct entity breakdown:")
        for platform, group in df[df['platform_phase1'].notna()].groupby('platform_phase1'):
            print(f"    {platform:20s}: {len(group):,} records, ${group['dollars'].sum():,.2f}")

    return df


# =============================================================================
# PHASE 2: DESCRIPTION-BASED ATTRIBUTION
# =============================================================================

def attribute_from_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: Use Stage 1 platform identification (platform_stage1).

    KEY UPDATE from old code: reads platform_stage1 (which includes LLM results)
    instead of the old cloud_platform_from_desc (RegEx-only).

    Only applies to cloud records not yet attributed in Phase 1.
    """
    print("\n" + "-" * 60)
    print("Phase 2: Description-Based Attribution")
    print("-" * 60)

    df = df.copy()
    df['platform_phase2'] = df['platform_phase1'].copy()

    # Apply Stage 1 platform to unattributed cloud records
    unattributed_cloud = (
        df['is_cloud'] &
        df['platform_phase2'].isna() &
        df['platform_stage1'].notna() &
        ~df['platform_stage1'].isin(['Unspecified', 'N/A', 'Multi-cloud'])
    )

    df.loc[unattributed_cloud, 'platform_phase2'] = df.loc[unattributed_cloud, 'platform_stage1']
    df.loc[unattributed_cloud, 'attribution_method'] = 'phase2_description'

    phase2_count = unattributed_cloud.sum()
    phase2_dollars = df.loc[unattributed_cloud, 'dollars'].sum()

    print(f"  Description-attributed: {phase2_count:,}")
    print(f"  Description-attributed dollars: ${phase2_dollars:,.2f}")

    # Count remaining unattributed cloud
    cloud_unattributed = df['is_cloud'] & df['platform_phase2'].isna()
    unspec_count = cloud_unattributed.sum()
    print(f"  Remaining unattributed cloud: {unspec_count:,}")

    return df


# =============================================================================
# PHASE 3: PATTERN MATCHING
# =============================================================================

def build_contractor_patterns(df: pd.DataFrame,
                              min_evidence: int = 5,
                              consistency_threshold: float = 0.90,
                              use_dollars: bool = False) -> pd.DataFrame:
    """
    Build platform patterns from contractor behavior.

    For each contractor: look at all their attributed records.
    If >= min_evidence pieces and >= consistency_threshold consistent,
    assign that platform to their unattributed cloud records.
    """
    print("\n" + "-" * 60)
    metric = 'dollars' if use_dollars else 'count'
    print(f"Phase 3: Pattern Matching (min_evidence={min_evidence}, "
          f"consistency={consistency_threshold:.0%}, metric={metric})")
    print("-" * 60)

    # Only look at cloud records with a specific platform attribution
    attributed = df[
        df['is_cloud'] &
        df['platform_phase2'].notna() &
        (df['dollars'] > 0)
    ].copy()

    print(f"  Attributed records for pattern building: {len(attributed):,}")

    if len(attributed) == 0:
        return pd.DataFrame()

    contractor_platforms = attributed.groupby(['contractor', 'platform_phase2']).agg(
        evidence_count=('dollars', 'count'),
        evidence_dollars=('dollars', 'sum')
    ).reset_index()

    patterns = []
    for contractor in contractor_platforms['contractor'].unique():
        cdata = contractor_platforms[contractor_platforms['contractor'] == contractor]
        if use_dollars:
            total_evidence = cdata['evidence_dollars'].sum()
        else:
            total_evidence = cdata['evidence_count'].sum()

        if total_evidence < min_evidence:
            continue

        if use_dollars:
            dominant = cdata.loc[cdata['evidence_dollars'].idxmax()]
        else:
            dominant = cdata.loc[cdata['evidence_count'].idxmax()]
        dominant_platform = dominant['platform_phase2']
        dominant_count = dominant['evidence_count']
        dominant_value = dominant['evidence_dollars'] if use_dollars else dominant_count
        consistency = dominant_value / total_evidence

        if consistency >= consistency_threshold:
            contractor_name = df.loc[df['contractor'] == contractor, 'contractor_name'].iloc[0]
            patterns.append({
                'contractor': contractor,
                'contractor_name': contractor_name,
                'learned_platform': dominant_platform,
                'evidence_count': total_evidence,
                'consistency': consistency,
                'dominant_count': dominant_count,
                'dominant_value': dominant_value,
            })

    patterns_df = pd.DataFrame(patterns)
    print(f"  Contractors with clear patterns: {len(patterns_df):,}")

    if len(patterns_df) > 0:
        print(f"\n  Pattern breakdown:")
        for platform, group in patterns_df.groupby('learned_platform'):
            print(f"    {platform:20s}: {len(group):,} contractors")

    return patterns_df


def apply_patterns(df: pd.DataFrame, patterns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply learned patterns to unattributed cloud records.
    """
    print("\n  Applying patterns to unattributed records...")

    df = df.copy()

    if len(patterns_df) == 0:
        print("  No patterns to apply.")
        return df

    pattern_lookup = patterns_df.set_index('contractor')['learned_platform'].to_dict()

    # Find unattributed cloud records whose contractor has a pattern
    unattributed_mask = (
        df['is_cloud'] &
        df['platform_phase2'].isna() &
        df['contractor'].isin(pattern_lookup.keys())
    )

    df.loc[unattributed_mask, 'platform_phase2'] = (
        df.loc[unattributed_mask, 'contractor'].map(pattern_lookup)
    )
    df.loc[unattributed_mask, 'attribution_method'] = 'phase3_pattern'

    reattributed = unattributed_mask.sum()
    reattributed_dollars = df.loc[unattributed_mask, 'dollars'].sum()

    print(f"  Re-attributed: {reattributed:,} records, ${reattributed_dollars:,.2f}")

    return df


# =============================================================================
# FINAL SYNTHESIS
# =============================================================================

def synthesize_final_platform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all attribution phases into final_platform column.

    Rules:
    - Non-cloud: final_platform = contractor_name
    - Phase 1 attributed: final_platform = platform name
    - Phase 2 attributed: final_platform = platform name
    - Phase 3 attributed: final_platform = platform name
    - Unattributed cloud: final_platform = contractor_name
    """
    print("\n" + "-" * 60)
    print("Final Synthesis: Creating final_platform")
    print("-" * 60)

    df = df.copy()
    df['final_platform'] = None
    df['confidence'] = None

    # Fill missing contractor_name with contractor (UEI) as fallback
    name_fallback = df['contractor_name'].fillna(df['contractor'])

    # Non-cloud records → contractor name
    non_cloud = ~df['is_cloud']
    df.loc[non_cloud, 'final_platform'] = name_fallback[non_cloud]
    df.loc[non_cloud, 'attribution_method'] = 'non_cloud_contractor'
    df.loc[non_cloud, 'confidence'] = 'N/A'

    # Cloud records with Phase 1 attribution
    p1 = df['is_cloud'] & df['platform_phase1'].notna()
    df.loc[p1, 'final_platform'] = df.loc[p1, 'platform_phase1']
    df.loc[p1, 'confidence'] = 'conclusive'

    # Cloud records with Phase 2 attribution (not already Phase 1)
    p2 = df['is_cloud'] & df['platform_phase1'].isna() & df['platform_phase2'].notna()
    df.loc[p2, 'final_platform'] = df.loc[p2, 'platform_phase2']
    # Confidence from classification stage
    if 'classification_confidence' in df.columns:
        df.loc[p2, 'confidence'] = df.loc[p2, 'classification_confidence']
    else:
        df.loc[p2, 'confidence'] = 'medium'

    # Check for Phase 3 records (attribution_method == 'phase3_pattern')
    p3 = df['is_cloud'] & (df['attribution_method'] == 'phase3_pattern')
    df.loc[p3, 'final_platform'] = df.loc[p3, 'platform_phase2']
    df.loc[p3, 'confidence'] = 'pattern_inferred'

    # Remaining unattributed cloud → contractor name
    unattributed = df['is_cloud'] & df['final_platform'].isna()
    df.loc[unattributed, 'final_platform'] = name_fallback[unattributed]
    df.loc[unattributed, 'attribution_method'] = 'cloud_unattributed_contractor'
    df.loc[unattributed, 'confidence'] = 'unattributed'

    # Report
    cloud_df = df[df['is_cloud']]
    total_cloud = len(cloud_df)
    total_cloud_dollars = cloud_df['dollars'].sum()

    # Count records attributed to known platforms vs contractor names
    known_platforms = {'AWS', 'Azure', 'Google Cloud', 'Salesforce', 'ServiceNow',
                       'Workday', 'Oracle Cloud', 'IBM Cloud', 'Multi-cloud'}
    platform_attributed = cloud_df[cloud_df['final_platform'].isin(known_platforms)]
    contractor_attributed = cloud_df[~cloud_df['final_platform'].isin(known_platforms)]

    print(f"\n  Cloud records: {total_cloud:,}")
    print(f"  Attributed to platform: {len(platform_attributed):,} "
          f"({len(platform_attributed)/total_cloud*100:.1f}%), "
          f"${platform_attributed['dollars'].sum()/1e6:.1f}M")
    print(f"  At contractor level: {len(contractor_attributed):,} "
          f"({len(contractor_attributed)/total_cloud*100:.1f}%), "
          f"${contractor_attributed['dollars'].sum()/1e6:.1f}M")

    print(f"\n  Attribution method breakdown:")
    for method, group in df[df['is_cloud']].groupby('attribution_method'):
        n = len(group)
        dollars = group['dollars'].sum()
        print(f"    {str(method):35s}: {n:>7,} records, ${dollars/1e6:>9.1f}M")

    return df


# =============================================================================
# PLATFORM-LEVEL HHI
# =============================================================================

def calculate_platform_hhi(df: pd.DataFrame,
                           platform_column: str = 'final_platform',
                           dollars_column: str = 'dollars') -> Dict[str, Any]:
    """
    Calculate platform-level HHI on the merged dataset.

    Groups by platform_column for cloud records.
    If multi-platform mentions are expanded, use dollars_column='dollars_split'
    and platform_column='platform_for_analysis'.
    """
    print("\n" + "=" * 80)
    print("PLATFORM-LEVEL HHI (CLOUD SPENDING)")
    print("=" * 80)

    cloud_df = df[df['is_cloud'] == True].copy()

    # Filter non-positive dollars
    nonpositive_mask = cloud_df[dollars_column] <= 0
    if nonpositive_mask.any():
        excluded_count = nonpositive_mask.sum()
        excluded_dollars = cloud_df.loc[nonpositive_mask, dollars_column].sum()
        print(f"\n  Excluding non-positive dollars: {excluded_count:,} records, "
              f"${excluded_dollars:,.2f}")
        cloud_df = cloud_df.loc[~nonpositive_mask].copy()

    total_cloud_dollars = cloud_df[dollars_column].sum()

    platform_spending = cloud_df.groupby(platform_column)[dollars_column].sum().sort_values(ascending=False)
    if total_cloud_dollars <= 0:
        print("\n  WARNING: No positive cloud dollars available for platform HHI.")
        return {
            'hhi': 0.0,
            'classification': "Unconcentrated (Competitive)",
            'c4': 0.0,
            'n_equivalent_firms': 0.0,
            'n_contractors': 0,
            'platform_spending': platform_spending,
            'platform_shares': pd.Series(dtype=float),
            'total_cloud_dollars': 0.0,
        }

    platform_shares = platform_spending / total_cloud_dollars * 100

    hhi = (platform_shares ** 2).sum()
    c4 = platform_shares.nlargest(4).sum()
    n_equivalent = 10000 / hhi if hhi > 0 else 0

    n_contractors = len(platform_spending)

    print(f"\n  Cloud spending: ${total_cloud_dollars:,.2f}")
    print(f"  Cloud records: {len(cloud_df):,}")
    print(f"  Entities: {n_contractors:,}")

    print(f"\n  HHI: {hhi:,.1f}")
    classification = ("Unconcentrated (Competitive)" if hhi < 1500
                      else "Moderately Concentrated" if hhi < 2500
                      else "Highly Concentrated")
    print(f"  Classification: {classification}")
    print(f"  C4: {c4:.2f}%")
    print(f"  Equivalent firms: {n_equivalent:.1f}")

    print(f"\n  Top 15 platform market shares:")
    for platform, share in platform_shares.head(15).items():
        dollars = platform_spending[platform]
        count = len(cloud_df[cloud_df[platform_column] == platform])
        print(f"    {str(platform)[:30]:30s}: {share:6.2f}% (${dollars:>14,.2f}, {count:,} records)")

    # Attribution method breakdown
    if 'attribution_method' in cloud_df.columns:
        print(f"\n  Attribution method breakdown:")
        method_summary = cloud_df.groupby('attribution_method').agg(
            records=(dollars_column, 'count'),
            dollars=(dollars_column, 'sum')
        ).sort_values('dollars', ascending=False)
        for method, row in method_summary.iterrows():
            pct = row['dollars'] / total_cloud_dollars * 100
            print(f"    {str(method):35s}: {row['records']:>7,} records, "
                  f"${row['dollars']:>14,.2f} ({pct:.1f}%)")

    return {
        'hhi': hhi,
        'classification': classification,
        'c4': c4,
        'n_equivalent_firms': n_equivalent,
        'n_contractors': n_contractors,
        'platform_spending': platform_spending,
        'platform_shares': platform_shares,
        'total_cloud_dollars': total_cloud_dollars,
    }


# =============================================================================
# ORCHESTRATORS
# =============================================================================

def run_stage2(df: pd.DataFrame,
               min_evidence: int = 5,
               consistency_threshold: float = 0.90) -> Tuple[pd.DataFrame, Dict]:
    """
    Run Stage 2: Platform Attribution pipeline.

    Requires Stage 1 columns: is_cloud, platform_stage1

    Returns:
        (attributed_df, platform_hhi_result)
    Note:
        Platform HHI uses multi-platform splitting when platform_mentions are present.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: PLATFORM ATTRIBUTION")
    print("=" * 80)

    cloud_count = df['is_cloud'].sum()
    cloud_dollars = df.loc[df['is_cloud'], 'dollars'].sum()
    print(f"\n  Input: {len(df):,} records, {cloud_count:,} cloud (${cloud_dollars/1e9:.2f}B)")

    # Phase 1: Direct entity
    df = attribute_direct_entities(df)

    # Phase 2: Description-based (uses platform_stage1 from Stage 1)
    df = attribute_from_descriptions(df)

    # Phase 3: Pattern matching
    contractor_patterns = build_contractor_patterns(
        df, min_evidence=min_evidence, consistency_threshold=consistency_threshold, use_dollars=False
    )
    contractor_patterns_dollars = build_contractor_patterns(
        df, min_evidence=min_evidence, consistency_threshold=consistency_threshold, use_dollars=True
    )

    # Sensitivity check: count-based vs dollars-based patterns
    if len(contractor_patterns) > 0 and len(contractor_patterns_dollars) > 0:
        merged = contractor_patterns[['contractor', 'learned_platform']].merge(
            contractor_patterns_dollars[['contractor', 'learned_platform']],
            on='contractor',
            how='inner',
            suffixes=('_count', '_dollars')
        )
        diff = merged[merged['learned_platform_count'] != merged['learned_platform_dollars']]
        if len(diff) > 0:
            print(f"\n  Pattern sensitivity: {len(diff):,} contractors differ between "
                  f"count-based and dollars-based patterns.")
        else:
            print(f"\n  Pattern sensitivity: count-based and dollars-based patterns align.")

    df = apply_patterns(df, contractor_patterns)

    # Final synthesis
    df = synthesize_final_platform(df)

    # Platform HHI (multi-platform mentions split across providers)
    expanded_df = expand_platform_mentions(df)
    platform_result = calculate_platform_hhi(
        expanded_df,
        platform_column='platform_for_analysis',
        dollars_column='dollars_split'
    )

    return df, platform_result


def run_attribution_pipeline(merged_df: pd.DataFrame,
                             api_key: Optional[str] = None,
                             cache_dir: Optional[str] = None,
                             run_llm: bool = False,
                             min_evidence: int = 5,
                             consistency_threshold: float = 0.90,
                             skip_confirmation: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Backward-compatible wrapper: runs Stage 1 + Stage 2.

    This replaces the old run_attribution_pipeline() from revised_platform_attribution.py.
    """
    # Import Stage 1 from sibling module
    import importlib.util
    cls_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cloud_classification.py')
    spec = importlib.util.spec_from_file_location('cloud_classification', cls_path)
    cls_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cls_mod)

    # Stage 1
    classified_df = cls_mod.run_stage1(
        merged_df, api_key=api_key, cache_dir=cache_dir,
        run_llm=run_llm, skip_confirmation=skip_confirmation
    )

    # Stage 2
    attributed_df, platform_result = run_stage2(
        classified_df,
        min_evidence=min_evidence,
        consistency_threshold=consistency_threshold
    )

    return attributed_df, platform_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Stage 2: Platform Attribution on existing classified dataset."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    classified_path = os.path.join(project_root, 'data', '02_processed', '03_classified',
                                    'classified_dataset.csv')
    merged_path = os.path.join(project_root, 'data', '02_processed', '02_merged', 'merged_dataset.csv')
    output_dir = os.path.join(project_root, 'data', '02_processed', '03_classified')

    # Prefer classified dataset; fall back to merged
    if os.path.exists(classified_path):
        print(f"Loading classified dataset: {classified_path}")
        df = pd.read_csv(classified_path)
    elif os.path.exists(merged_path):
        print(f"WARNING: No classified dataset found. Using merged dataset (RegEx-only mode).")
        df = pd.read_csv(merged_path)
        # Run quick Stage 1 with RegEx only
        import importlib.util
        cls_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cloud_classification.py')
        spec = importlib.util.spec_from_file_location('cloud_classification', cls_path)
        cls_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls_mod)
        df = cls_mod.run_stage1(df)
    else:
        print(f"ERROR: No dataset found.")
        return

    print(f"  Loaded {len(df):,} records")

    attributed_df, platform_result = run_stage2(df)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'attributed_dataset.csv')
    attributed_df.to_csv(output_path, index=False)
    print(f"\nSaved attributed dataset: {output_path} ({len(attributed_df):,} rows)")

    return attributed_df, platform_result


if __name__ == '__main__':
    main()

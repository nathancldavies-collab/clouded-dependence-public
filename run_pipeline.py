#!/usr/bin/env python3
"""
Revised Analysis Pipeline - Main Runner
=========================================
Implements the corrected methodology:
0. FILTER primes (date range, PSC codes, hardware keywords)
1. MERGE filtered primes + subs into unified dataset (Stage 0C)
2. Baseline HHI on merged dataset (contractor-level)
3. Stage 1: Cloud classification (RegEx + optional LLM)
4. Stage 2: Platform attribution (Phases 1-3)
5. Platform-level HHI + concentration gap analysis

LLM classification:
  - If ANTHROPIC_API_KEY is set, the pipeline will auto-run LLM classification
    (with a cost confirmation prompt) unless cached results already exist.
  - If no API key is set, the pipeline runs in RegEx-only mode.
  - LLM results are cached in data/02_processed/03_classified/llm_cache/
    and will be reused on subsequent runs.
  - If an LLM cache exists, the pipeline will by default reuse existing
    filtered/merged datasets to preserve alignment. Set FORCE_REMERGE=1 to rebuild.

Run from project root:
    python3 run_pipeline.py
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import importlib.util

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# If LLM cache exists, avoid rebuilding merged dataset to preserve alignment
FREEZE_MERGED_IF_LLM_CACHE = True


def _import_module(name: str, filepath: str):
    """Import a module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import modules from directories with numeric prefixes (not valid Python packages)
_merge_mod = _import_module(
    'create_merged_dataset',
    os.path.join(PROJECT_ROOT, 'pipeline', '01_prep', 'create_merged_dataset.py')
)
_filter_mod = _import_module(
    'filter_primes',
    os.path.join(PROJECT_ROOT, 'pipeline', '01_prep', 'filter_primes.py')
)
_baseline_mod = _import_module(
    'baseline_merged_hhi',
    os.path.join(PROJECT_ROOT, 'pipeline', '02_base_analysis', 'baseline_merged_hhi.py')
)
_classification_mod = _import_module(
    'cloud_classification',
    os.path.join(PROJECT_ROOT, 'pipeline', '03_classification', 'cloud_classification.py')
)
_attribution_mod = _import_module(
    'platform_attribution',
    os.path.join(PROJECT_ROOT, 'pipeline', '04_attribution', 'platform_attribution.py')
)

# Re-export functions
load_primes = _merge_mod.load_primes
load_subs = _merge_mod.load_subs
create_merged_dataset = _merge_mod.create_merged_dataset
verify_merged_dataset = _merge_mod.verify_merged_dataset
print_merge_summary = _merge_mod.print_merge_summary
dedupe_primes_by_award = _merge_mod.dedupe_primes_by_award

filter_primes = _filter_mod.filter_primes

calculate_contractor_hhi = _baseline_mod.calculate_contractor_hhi
compare_baseline_approaches = _baseline_mod.compare_baseline_approaches
calculate_hhi = _baseline_mod.calculate_hhi
classify_hhi = _baseline_mod.classify_hhi

run_stage1 = _classification_mod.run_stage1
run_stage2 = _attribution_mod.run_stage2
calculate_platform_hhi = _attribution_mod.calculate_platform_hhi
expand_platform_mentions = _attribution_mod.expand_platform_mentions


def run_full_pipeline():
    """Execute the complete revised analysis pipeline."""
    start_time = time.time()

    def _calc_hhi_from_group(df: pd.DataFrame, group_col: str, dollars_col: str = 'dollars'):
        subset = df[df[dollars_col] > 0].copy()
        total = subset[dollars_col].sum()
        if total <= 0 or len(subset) == 0:
            return 0.0, pd.Series(dtype=float)
        shares = subset.groupby(group_col)[dollars_col].sum() / total * 100
        return calculate_hhi(shares), shares

    print("=" * 80)
    print("REVISED PLATFORM CONCENTRATION ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Methodology: Filter -> Merge primes+subs -> Classify -> Attribute -> HHI")

    # --- Paths ---
    raw_dir = os.path.join(PROJECT_ROOT, 'data', '01_raw')
    filtered_dir = os.path.join(PROJECT_ROOT, 'data', '02_processed', '01_filtered')
    merged_dir = os.path.join(PROJECT_ROOT, 'data', '02_processed', '02_merged')
    classified_dir = os.path.join(PROJECT_ROOT, 'data', '02_processed', '03_classified')
    llm_cache_dir = os.path.join(classified_dir, 'llm_cache')

    prime_path = os.path.join(raw_dir, 'prime_awards.csv')
    sub_path = os.path.join(raw_dir, 'subawards.csv')
    filtered_path = os.path.join(filtered_dir, 'prime_services_filtered.csv')
    merged_path = os.path.join(merged_dir, 'merged_dataset.csv')
    classified_path = os.path.join(classified_dir, 'classified_dataset.csv')
    attributed_path = os.path.join(classified_dir, 'attributed_dataset.csv')

    for d in [filtered_dir, merged_dir, classified_dir]:
        os.makedirs(d, exist_ok=True)

    # --- LLM configuration ---
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    has_llm_cache = os.path.exists(os.path.join(llm_cache_dir, 'llm_progress.json'))
    force_remerge = os.environ.get('FORCE_REMERGE', '').lower() in ('1', 'true', 'yes')
    freeze_merged = FREEZE_MERGED_IF_LLM_CACHE and has_llm_cache and not force_remerge

    if api_key:
        print(f"\n  ANTHROPIC_API_KEY: set")
        if has_llm_cache:
            print(f"  LLM cache: found (will reuse cached results)")
        else:
            print(f"  LLM cache: not found (will run LLM classification)")
    else:
        print(f"\n  ANTHROPIC_API_KEY: not set (RegEx-only mode)")
    if freeze_merged:
        print(f"  Freeze merged dataset: enabled (to preserve LLM alignment)")

    # =========================================================================
    # STAGE 0A-0C: LOAD/FILTER/MERGE (optionally skipped when freezing)
    # =========================================================================
    if freeze_merged and os.path.exists(merged_path) and os.path.exists(filtered_path):
        print("\n\n" + "#" * 80)
        print("# STAGE 0A-0C: USING EXISTING FILTERED + MERGED DATASETS")
        print("#" * 80)
        print("  NOTE: Freeze enabled to preserve LLM alignment.")

        primes_df = _classification_mod.safe_read_csv(filtered_path)
        merged_df = _classification_mod.safe_read_csv(merged_path)
        filter_summary = {
            'baseline_count': len(primes_df),
            'after_keyword_count': len(primes_df),
        }
    else:
        # =========================================================================
        # STAGE 0A: LOAD RAW DATA
        # =========================================================================
        print("\n\n" + "#" * 80)
        print("# STAGE 0A: LOAD RAW DATA")
        print("#" * 80)

        primes_raw = load_primes(prime_path)
        subs_df = load_subs(sub_path)

        # =========================================================================
        # STAGE 0B: FILTER PRIMES (date, PSC, keywords)
        # =========================================================================
        print("\n\n" + "#" * 80)
        print("# STAGE 0B: FILTER PRIMES (Hardware Exclusion)")
        print("#" * 80)

        primes_df, filter_summary = filter_primes(primes_raw, year_min=2017, year_max=2024)
        primes_df = dedupe_primes_by_award(primes_df, dollars_agg="max")
        filter_summary['after_keyword_count'] = len(primes_df)
        filter_summary['after_keyword_dollars'] = primes_df['Total Dollars Obligated'].sum()

        # Save filtered primes
        print(f"\n  Saving filtered primes to: {filtered_path}")
        primes_df.to_csv(filtered_path, index=False)
        print(f"  Saved {len(primes_df):,} records")

        # =========================================================================
        # STAGE 0C: CREATE MERGED DATASET
        # =========================================================================
        print("\n\n" + "#" * 80)
        print("# STAGE 0C: CREATE MERGED DATASET")
        print("#" * 80)

        merged_df, unlinked_df = create_merged_dataset(primes_df, subs_df, return_unlinked=True)

        passed = verify_merged_dataset(merged_df, primes_df, subs_df)
        if not passed:
            print("\nFATAL: Merge verification failed. Aborting.")
            sys.exit(1)

        print_merge_summary(merged_df)

        # Save merged dataset
        print(f"\nSaving merged dataset to: {merged_path}")
        merged_df.to_csv(merged_path, index=False)
        print(f"  Saved {len(merged_df):,} records")

        # Save unlinked subawards for sensitivity
        if len(unlinked_df) > 0:
            unlinked_path = os.path.join(merged_dir, 'unlinked_subawards.csv')
            unlinked_df.to_csv(unlinked_path, index=False)
            print(f"\nSaved unlinked subawards to: {unlinked_path} ({len(unlinked_df):,} rows)")

    # =========================================================================
    # BASELINE ANALYSIS: CONTRACTOR-LEVEL HHI
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# BASELINE ANALYSIS: CONTRACTOR-LEVEL HHI")
    print("#" * 80)

    # Compare old vs new baseline
    baseline_comparison = compare_baseline_approaches(primes_df, merged_df)

    # =========================================================================
    # STAGE 1: CLOUD CLASSIFICATION (RegEx + LLM)
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# STAGE 1: CLOUD CLASSIFICATION")
    print("#" * 80)

    # Determine whether to run LLM
    run_llm = api_key is not None

    classified_df = run_stage1(
        merged_df,
        api_key=api_key,
        cache_dir=llm_cache_dir,
        run_llm=run_llm,
        skip_confirmation=False,  # Always ask before spending money
        conflict_mode="explicit_override",
        noncloud_override_confidence="high"
    )

    # Save classified dataset
    print(f"\nSaving classified dataset to: {classified_path}")
    classified_df.to_csv(classified_path, index=False)
    print(f"  Saved {len(classified_df):,} records")

    # Sensitivity: strict LLM conflict resolution (no explicit override)
    has_llm = ('llm_classification' in classified_df.columns and
               classified_df['llm_classification'].notna().any())
    classified_df_strict = None
    if has_llm:
        classified_df_strict = _classification_mod.synthesize_classification(
            classified_df.copy(),
            conflict_mode="llm_strict",
            noncloud_override_confidence="high"
        )

    # =========================================================================
    # STAGE 2: PLATFORM ATTRIBUTION
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# STAGE 2: PLATFORM ATTRIBUTION")
    print("#" * 80)

    attributed_df, platform_result = run_stage2(
        classified_df,
        min_evidence=5,
        consistency_threshold=0.90
    )

    attributed_df_strict = None
    platform_result_strict = None
    if classified_df_strict is not None:
        print("\n\n" + "#" * 80)
        print("# SENSITIVITY RUN: LLM-STRICT CONFLICT RESOLUTION")
        print("#" * 80)
        attributed_df_strict, platform_result_strict = run_stage2(
            classified_df_strict,
            min_evidence=5,
            consistency_threshold=0.90
        )

    # Save attributed dataset
    print(f"\nSaving attributed dataset to: {attributed_path}")
    attributed_df.to_csv(attributed_path, index=False)
    print(f"  Saved {len(attributed_df):,} records")

    # =========================================================================
    # BEFORE/AFTER COMPARISON: THE KEY FINDING
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# BEFORE/AFTER COMPARISON: CONTRACTOR vs PLATFORM VIEW")
    print("#" * 80)

    baseline_hhi = baseline_comparison['new_hhi']
    platform_hhi = platform_result['hhi']
    concentration_gap = platform_hhi / baseline_hhi if baseline_hhi > 0 else 0

    # Apples-to-apples: cloud-only contractor vs platform HHI
    expanded_df = expand_platform_mentions(attributed_df)
    cloud_df = attributed_df[attributed_df['is_cloud'] == True]
    cloud_contractor_hhi, _ = _calc_hhi_from_group(cloud_df, 'contractor')
    cloud_platform_hhi, _ = _calc_hhi_from_group(
        expanded_df[expanded_df['is_cloud'] == True],
        'platform_for_analysis',
        dollars_col='dollars_split'
    )

    # Infrastructure-only HHI (cloud-infrastructure)
    infra_df = attributed_df[attributed_df['cloud_classification'] == 'cloud-infrastructure']
    infra_expanded = expand_platform_mentions(infra_df)
    infra_contractor_hhi, _ = _calc_hhi_from_group(infra_df, 'contractor')
    infra_platform_hhi, _ = _calc_hhi_from_group(
        infra_expanded[infra_expanded['is_cloud'] == True],
        'platform_for_analysis',
        dollars_col='dollars_split'
    )

    print(f"\n  {'Metric':<35s} {'Baseline':>15s} {'Platform':>15s}")
    print(f"  {'(Contractor)':35s} {'(Merged)':>15s} {'(Attributed)':>15s}")
    print(f"  {'-'*65}")
    print(f"  {'HHI':<35s} {baseline_hhi:>15,.1f} {platform_hhi:>15,.1f}")
    print(f"  {'Classification':<35s} {classify_hhi(baseline_hhi):>15s} "
          f"{platform_result['classification']:>15s}")
    print(f"  {'C4 (%)':<35s} {baseline_comparison['new_c4']:>15.2f} "
          f"{platform_result['c4']:>15.2f}")

    print(f"\n  Concentration Gap: {concentration_gap:.1f}x")
    print(f"  (Platform HHI is {concentration_gap:.1f}x the baseline contractor HHI)")

    print(f"\n  Apples-to-apples (cloud-only):")
    print(f"    Contractor HHI (cloud-only): {cloud_contractor_hhi:,.1f}")
    print(f"    Platform HHI  (cloud-only): {cloud_platform_hhi:,.1f}")
    if cloud_contractor_hhi > 0:
        print(f"    Gap (cloud-only): {cloud_platform_hhi / cloud_contractor_hhi:.1f}x")

    print(f"\n  Infrastructure-only (cloud-infrastructure):")
    print(f"    Contractor HHI (infra-only): {infra_contractor_hhi:,.1f}")
    print(f"    Platform HHI  (infra-only): {infra_platform_hhi:,.1f}")

    if platform_result_strict is not None:
        strict_hhi = platform_result_strict['hhi']
        print(f"\n  Sensitivity (LLM-strict conflict resolution):")
        print(f"    Platform HHI (strict): {strict_hhi:,.1f}")
        if platform_hhi > 0:
            delta_pct = (strict_hhi - platform_hhi) / platform_hhi * 100
            print(f"    Change vs explicit-override: {delta_pct:+.1f}%")

    # =========================================================================
    # COMPARISON WITH OLD METHODOLOGY
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# COMPARISON: OLD vs NEW METHODOLOGY")
    print("#" * 80)

    old_baseline_hhi = baseline_comparison['old_hhi']

    print(f"\n  OLD METHODOLOGY (primes only, subs as evidence):")
    print(f"    Baseline HHI (filtered primes by contractor): {old_baseline_hhi:,.1f}")

    print(f"\n  NEW METHODOLOGY (filter -> merge -> classify -> attribute):")
    print(f"    Baseline HHI (merged by contractor): {baseline_hhi:,.1f}")
    print(f"    Platform HHI (cloud-attributed):     {platform_hhi:,.1f}")
    print(f"    Concentration gap:                   {concentration_gap:.1f}x")

    hhi_baseline_change = baseline_hhi - old_baseline_hhi
    print(f"\n  Baseline change: {hhi_baseline_change:+,.1f} "
          f"({'lower' if hhi_baseline_change < 0 else 'higher'} after merge)")

    if hhi_baseline_change < 0:
        print(f"  -> More entities visible after merge -> baseline DECREASES")
        print(f"  -> Gap between baseline and platform view is LARGER")
        print(f"  -> Finding is STRONGER with correct methodology")

    # =========================================================================
    # HIDDEN CONCENTRATION: HYPERSCALER CONTRACTOR vs PLATFORM SHARE
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# HIDDEN CONCENTRATION: CONTRACTOR SHARE vs PLATFORM SHARE")
    print("#" * 80)

    total_dollars = attributed_df['dollars'].sum()
    cloud_df = attributed_df[attributed_df['is_cloud'] == True]
    cloud_total = expanded_df[expanded_df['is_cloud'] == True]['dollars_split'].sum()

    # Define hyperscaler/platform provider name patterns for contractor matching
    hyperscaler_patterns = {
        'Azure':        ['MICROSOFT'],
        'AWS':          ['AMAZON WEB SERVICES', 'AMAZON.COM SERVICES', 'AMAZON'],
        'Google Cloud': ['GOOGLE'],
        'Oracle Cloud': ['ORACLE'],
        'Salesforce':   ['SALESFORCE'],
        'ServiceNow':   ['SERVICENOW'],
    }

    print(f"\n  Total IT spending:    ${total_dollars/1e9:.1f}B")
    print(f"  Cloud spending:       ${cloud_total/1e9:.1f}B")
    print(f"\n  {'Platform':<15s} {'As Contractor':>15s} {'As Platform':>15s} {'Change':>10s}")
    print(f"  {'':15s} {'(all spending)':>15s} {'(cloud spend)':>15s} {'':>10s}")
    print(f"  {'-'*58}")

    for platform, patterns in hyperscaler_patterns.items():
        # Baseline: share as a contractor across ALL spending
        contractor_mask = attributed_df['contractor_name'].str.upper().str.contains(
            '|'.join(p.upper() for p in patterns), na=False)
        baseline_dollars = attributed_df.loc[contractor_mask, 'dollars'].sum()
        baseline_pct = baseline_dollars / total_dollars * 100

        # Platform: share of cloud spending via final_platform
        platform_mask = expanded_df['platform_for_analysis'] == platform
        platform_dollars = expanded_df.loc[platform_mask, 'dollars_split'].sum()
        platform_pct = platform_dollars / cloud_total * 100

        # Percentage change
        if baseline_pct >= 0.005:
            pct_increase = ((platform_pct - baseline_pct) / baseline_pct) * 100
            if pct_increase >= 0:
                change_str = f"+{pct_increase:,.0f}%"
            else:
                change_str = f"{pct_increase:,.0f}%"
        elif platform_pct > 0:
            change_str = "new"
        else:
            change_str = "-"

        print(f"  {platform:<15s} {baseline_pct:>14.2f}% {platform_pct:>14.2f}% {change_str:>10s}")

    print(f"\n  'As Contractor' = share of total IT market as a direct vendor")
    print(f"  'As Platform'   = share of cloud market where spending depends on their platform")
    print(f"  'Change'        = percentage increase from contractor to platform share")

    # =========================================================================
    # TEMPORAL ANALYSIS: CONCENTRATION TRENDS (3-Year Rolling Windows)
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# TEMPORAL ANALYSIS: CONCENTRATION TRENDS (3-Year Rolling Windows)")
    print("#" * 80)

    WINDOW = 3
    temporal_df = attributed_df[attributed_df['fiscal_year'].between(2017, 2024)].copy()
    temporal_cloud = temporal_df[temporal_df['is_cloud'] == True]
    temporal_expanded = expanded_df[expanded_df['fiscal_year'].between(2017, 2024)].copy()
    temporal_cloud_expanded = temporal_expanded[temporal_expanded['is_cloud'] == True]

    windows = []
    for start_fy in range(2017, 2025 - WINDOW + 1):
        end_fy = start_fy + WINDOW - 1
        label = f'FY{start_fy}-{str(end_fy)[2:]}'

        w_all = temporal_df[temporal_df['fiscal_year'].between(start_fy, end_fy)]
        w_cloud = temporal_cloud[temporal_cloud['fiscal_year'].between(start_fy, end_fy)]
        w_cloud_expanded = temporal_cloud_expanded[
            temporal_cloud_expanded['fiscal_year'].between(start_fy, end_fy)
        ]

        total_d = w_all['dollars'].sum()
        cloud_d = w_cloud['dollars'].sum()
        cloud_pct = cloud_d / total_d * 100

        contractor_shares = w_all.groupby('contractor')['dollars'].sum() / total_d * 100
        baseline_hhi = (contractor_shares ** 2).sum()

        platform_shares = (
            w_cloud_expanded.groupby('platform_for_analysis')['dollars_split'].sum() / cloud_d * 100
        )
        platform_hhi = (platform_shares ** 2).sum()

        gap = platform_hhi / baseline_hhi if baseline_hhi > 0 else 0
        c4 = platform_shares.nlargest(4).sum()

        windows.append({
            'label': label, 'start': start_fy, 'end': end_fy,
            'total_d': total_d, 'cloud_d': cloud_d, 'cloud_pct': cloud_pct,
            'baseline_hhi': baseline_hhi, 'platform_hhi': platform_hhi,
            'gap': gap, 'c4': c4, 'platform_shares': platform_shares
        })

    # 1. Headline concentration trends
    print(f"\n  {'Window':>12s} {'Total $B':>10s} {'Cloud $B':>10s} {'Cloud %':>8s} "
          f"{'Base HHI':>10s} {'Plat HHI':>10s} {'Gap':>8s} {'C4':>8s}")
    print(f"  {'-'*80}")
    for w in windows:
        print(f"  {w['label']:>12s} {w['total_d']/1e9:>10.1f} {w['cloud_d']/1e9:>10.1f} "
              f"{w['cloud_pct']:>7.1f}% {w['baseline_hhi']:>10.1f} "
              f"{w['platform_hhi']:>10.1f} {w['gap']:>7.1f}x {w['c4']:>7.1f}%")

    # 2. Platform share trends
    key_platforms = ['Azure', 'AWS', 'Salesforce', 'Oracle Cloud', 'ServiceNow', 'Google Cloud']
    print(f"\n  Platform share trends (% of cloud spending per window):")
    header = f"  {'Window':>12s}"
    for p in key_platforms:
        header += f" {p:>12s}"
    print(header)
    print(f"  {'-'*88}")
    for w in windows:
        row = f"  {w['label']:>12s}"
        for p in key_platforms:
            share = w['platform_shares'].get(p, 0)
            row += f" {share:>11.1f}%"
        print(row)

    # 3. First vs last window comparison
    first_w = windows[0]
    last_w = windows[-1]
    print(f"\n  Platform share: {first_w['label']} vs {last_w['label']}:")
    print(f"  {'Platform':>15s} {first_w['label']:>14s} {last_w['label']:>14s} {'Change':>12s}")
    print(f"  {'-'*58}")
    for p in key_platforms:
        first_share = first_w['platform_shares'].get(p, 0)
        last_share = last_w['platform_shares'].get(p, 0)
        if first_share >= 0.1:
            delta = last_share - first_share
            print(f"  {p:>15s} {first_share:>13.1f}% {last_share:>13.1f}% {delta:>+11.1f}pp")
        elif last_share > 0:
            print(f"  {p:>15s} {first_share:>13.1f}% {last_share:>13.1f}% {'emerging':>12s}")

    # Summary
    print(f"\n  Key trends:")
    print(f"    Cloud penetration: {first_w['cloud_pct']:.1f}% -> {last_w['cloud_pct']:.1f}% "
          f"of IT spending (+{last_w['cloud_pct'] - first_w['cloud_pct']:.1f}pp)")
    print(f"    Concentration gap: {first_w['gap']:.1f}x -> {last_w['gap']:.1f}x "
          f"({'narrowing' if last_w['gap'] < first_w['gap'] else 'widening'})")
    first_azure = first_w['platform_shares'].get('Azure', 0)
    last_azure = last_w['platform_shares'].get('Azure', 0)
    print(f"    Azure share: {first_azure:.1f}% -> {last_azure:.1f}% "
          f"({'declining' if last_azure < first_azure else 'growing'}, "
          f"but still dominant)")
    combined_others = sum(last_w['platform_shares'].get(p, 0)
                          for p in ['AWS', 'Salesforce', 'Oracle Cloud', 'ServiceNow'])
    combined_first = sum(first_w['platform_shares'].get(p, 0)
                         for p in ['AWS', 'Salesforce', 'Oracle Cloud', 'ServiceNow'])
    print(f"    Challenger platforms (AWS+Salesforce+Oracle+ServiceNow): "
          f"{combined_first:.1f}% -> {combined_others:.1f}% "
          f"(+{combined_others - combined_first:.1f}pp)")

    # =========================================================================
    # VERIFICATION CHECKLIST
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# VERIFICATION CHECKLIST")
    print("#" * 80)

    prime_total = primes_df['Total Dollars Obligated'].sum()
    merged_total = merged_df['dollars'].sum()
    dollar_match = abs(merged_total - prime_total) < 1.0

    cloud_records = attributed_df['is_cloud'].sum() if 'is_cloud' in attributed_df.columns else 0
    all_classified = cloud_records > 0

    null_platforms = attributed_df['final_platform'].isna().sum()
    no_nulls = null_platforms == 0

    checks = [
        ("Merged total = Filtered prime total (no double-counting)", dollar_match,
         f"${merged_total:,.2f} vs ${prime_total:,.2f}"),
        ("Baseline HHI calculated on merged dataset", baseline_hhi > 0,
         f"HHI = {baseline_hhi:,.1f}"),
        ("Cloud classification ran on ALL merged records", all_classified,
         f"{cloud_records:,} cloud records found"),
        ("Platform HHI shows higher concentration than baseline",
         platform_hhi > baseline_hhi,
         f"{platform_hhi:,.1f} > {baseline_hhi:,.1f}"),
        ("No null final_platform values", no_nulls,
         f"{null_platforms} null values"),
        ("Concentration gap >= 2x", concentration_gap >= 2.0,
         f"Gap = {concentration_gap:.1f}x"),
    ]

    all_passed = True
    for description, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {description}")
        print(f"         {detail}")

    if all_passed:
        print(f"\n  All checks PASSED.")
    else:
        print(f"\n  Some checks FAILED - review results above.")

    # =========================================================================
    # LIMITATIONS
    # =========================================================================
    print("\n\n" + "#" * 80)
    print("# LIMITATIONS")
    print("#" * 80)
    print("  - Subaward fiscal year inherits prime fiscal year (subaward dates unavailable).")
    print("  - Filtering primes before merge excludes subawards tied to filtered primes.")
    print("  - Unlinked subawards are excluded from merged dataset but saved separately.")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time

    print("\n\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n  Filtering: {filter_summary['baseline_count']:,} raw -> "
          f"{filter_summary['after_keyword_count']:,} filtered primes")
    print(f"  Merging:   {len(merged_df):,} records (primes + sub flows)")
    print(f"  Spending:  ${merged_total:,.2f}")

    # LLM status
    has_llm = ('classification_source' in attributed_df.columns and
               attributed_df['classification_source'].str.contains('llm', na=False).any())
    print(f"\n  Classification: {'RegEx + LLM' if has_llm else 'RegEx-only'}")

    print(f"\n  Key results:")
    print(f"    Baseline HHI (contractor-level): {baseline_hhi:,.1f} - {classify_hhi(baseline_hhi)}")
    print(f"    Platform HHI (cloud-attributed): {platform_hhi:,.1f} - {platform_result['classification']}")
    print(f"    Concentration gap:               {concentration_gap:.1f}x")
    print(f"\n  Output files:")
    print(f"    Filtered primes:    {filtered_path}")
    print(f"    Merged dataset:     {merged_path}")
    print(f"    Classified dataset: {classified_path}")
    print(f"    Attributed dataset: {attributed_path}")
    unlinked_path = os.path.join(merged_dir, 'unlinked_subawards.csv')
    if os.path.exists(unlinked_path):
        print(f"    Unlinked subawards: {unlinked_path}")
    print(f"\n  Elapsed: {elapsed:.1f} seconds")


if __name__ == '__main__':
    run_full_pipeline()
    

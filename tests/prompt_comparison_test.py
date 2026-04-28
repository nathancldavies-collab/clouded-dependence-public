#!/usr/bin/env python3
"""
Prompt Comparison Test
======================
Tests two LLM prompt designs on a strategic sample of 200 records
to compare false positive/negative rates before committing to a full run.

Run from project root:
    python3 tests/prompt_comparison_test.py

Estimated cost: ~$0.15 total (200 records × 2 prompts)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import re
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 300
SAMPLE_SIZE = 200  # Total test records

# ---------------------------------------------------------------------------
# Prompt A: Concise (current implementation, ~180 input tokens)
# ---------------------------------------------------------------------------
def prompt_concise(description: str, contractor_name: str) -> str:
    desc = str(description) if pd.notna(description) else "No description provided"
    name = str(contractor_name) if pd.notna(contractor_name) else "Unknown"
    return f"""Classify this federal IT contract.

Description: {desc}
Contractor: {name}

Return ONLY JSON (no other text):
{{"classification":"non-cloud|cloud-dependent|cloud-infrastructure","platform":"AWS|Azure|Google Cloud|Oracle Cloud|IBM Cloud|Salesforce|Multi-cloud|Unspecified|N/A","confidence":"high|medium|low","evidence":"brief reason"}}

Definitions:
- non-cloud: Traditional IT, on-premise, hardware, legacy software
- cloud-dependent: Services requiring cloud infrastructure (SaaS, web apps, managed services, analytics)
- cloud-infrastructure: Direct cloud IaaS/PaaS purchase (VMs, storage, compute, platform services)"""


# ---------------------------------------------------------------------------
# Prompt B: Extended (with examples and explicit FP/FN guards, ~500 tokens)
# ---------------------------------------------------------------------------
def prompt_extended(description: str, contractor_name: str) -> str:
    desc = str(description) if pd.notna(description) else "No description provided"
    name = str(contractor_name) if pd.notna(contractor_name) else "Unknown"
    return f"""Classify this federal IT contract as cloud or non-cloud.

Description: {desc}
Contractor: {name}

Return ONLY valid JSON:
{{"classification":"non-cloud|cloud-dependent|cloud-infrastructure","platform":"AWS|Azure|Google Cloud|Oracle Cloud|IBM Cloud|Salesforce|Multi-cloud|Unspecified|N/A","confidence":"high|medium|low","evidence":"brief reason"}}

CLASSIFICATION RULES:
- cloud-infrastructure: Direct IaaS/PaaS purchases — VMs, storage, compute, cloud hosting credits, reserved instances
- cloud-dependent: Services built ON cloud platforms — SaaS products (Salesforce, ServiceNow, Workday), cloud-hosted applications, cloud migration, cloud managed services, cloud engineering
- non-cloud: Traditional IT with NO cloud dependency — on-premise servers, desktop software licenses, hardware, consulting/labor not tied to cloud, physical data centers, legacy system maintenance

KEY DISTINCTIONS:
- "Managed hosting" or "hosting services" WITHOUT cloud mention → likely non-cloud (on-prem)
- "Hosting" WITH cloud/AWS/Azure mention → cloud-infrastructure
- SaaS products are ALWAYS cloud-dependent, even without "cloud" in description
- Microsoft consulting/support labor is NOT cloud unless specifically for Azure
- Large IT services contracts that merely MENTION cloud as a small component → non-cloud
- Adobe Creative Cloud, Salesforce, ServiceNow → cloud-dependent (SaaS)
- "Subscription" alone is NOT enough — must be for a cloud-based product

EXAMPLES:
1. "AMAZON WEB SERVICES CLOUD HOSTING" → cloud-infrastructure, AWS
2. "SERVICENOW SAAS LICENSE ENTITLEMENTS" → cloud-dependent, Salesforce
3. "MICROSOFT CONSULTING SERVICES - LABOR" → non-cloud, N/A
4. "ENTERPRISE CLOUD MIGRATION AND O&M" → cloud-dependent, Unspecified
5. "IT INFRASTRUCTURE SUPPORT AND MAINTENANCE" → non-cloud, N/A"""


# ---------------------------------------------------------------------------
# Parsing (shared)
# ---------------------------------------------------------------------------
VALID_CLASSIFICATIONS = ['non-cloud', 'cloud-dependent', 'cloud-infrastructure']
VALID_PLATFORMS = ['AWS', 'Azure', 'Google Cloud', 'Oracle Cloud', 'IBM Cloud',
                   'Salesforce', 'Multi-cloud', 'Unspecified', 'N/A']

def parse_response(text: str) -> Dict:
    """Parse LLM JSON response with fallback."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        if text.startswith('json'):
            text = text[4:].strip()
    try:
        result = json.loads(text)
        classification = result.get('classification', 'error')
        if classification not in VALID_CLASSIFICATIONS:
            classification = 'error'
        platform = result.get('platform', 'Unspecified')
        if platform not in VALID_PLATFORMS:
            plat_upper = str(platform).upper()
            if 'AWS' in plat_upper or 'AMAZON' in plat_upper: platform = 'AWS'
            elif 'AZURE' in plat_upper or 'MICROSOFT' in plat_upper: platform = 'Azure'
            elif 'GOOGLE' in plat_upper or 'GCP' in plat_upper: platform = 'Google Cloud'
            elif 'ORACLE' in plat_upper: platform = 'Oracle Cloud'
            elif 'IBM' in plat_upper: platform = 'IBM Cloud'
            elif 'SALESFORCE' in plat_upper: platform = 'Salesforce'
            else: platform = 'Unspecified'
        return {
            'classification': classification,
            'platform': platform,
            'confidence': result.get('confidence', 'low'),
            'evidence': str(result.get('evidence', ''))[:200],
        }
    except (json.JSONDecodeError, AttributeError):
        cls_match = re.search(r'"classification"\s*:\s*"(non-cloud|cloud-dependent|cloud-infrastructure)"', text)
        plat_match = re.search(r'"platform"\s*:\s*"([^"]+)"', text)
        return {
            'classification': cls_match.group(1) if cls_match else 'error',
            'platform': plat_match.group(1) if plat_match else 'error',
            'confidence': 'low',
            'evidence': 'parse_error',
        }


# ---------------------------------------------------------------------------
# Strategic sample builder
# ---------------------------------------------------------------------------
def build_test_sample(merged_path: str, classified_path: str) -> pd.DataFrame:
    """
    Build a 200-record strategic sample covering:
    - 50 clear cloud (RegEx caught, platform identified) — should stay cloud
    - 50 unspecified-platform cloud (RegEx caught, no platform) — LLM should ID platform
    - 50 borderline (SaaS/hosting/managed-svc mentions, RegEx missed) — FN test
    - 50 clear non-cloud (no cloud terms at all) — FP test
    """
    merged = pd.read_csv(merged_path)
    classified = pd.read_csv(classified_path)

    np.random.seed(42)
    samples = []

    # Category 1: Clear cloud (RegEx caught + has platform)
    clear_cloud = classified[
        (classified['is_cloud'] == True) &
        (classified['platform_stage1'].notna()) &
        (~classified['platform_stage1'].isin(['Unspecified', 'N/A']))
    ]
    cat1 = clear_cloud.sample(min(50, len(clear_cloud)))
    cat1 = cat1.copy()
    cat1['test_category'] = 'clear_cloud'
    cat1['expected_cloud'] = True
    samples.append(cat1)
    print(f"  Category 1 (clear cloud): {len(cat1)} records")

    # Category 2: Unspecified platform (RegEx caught cloud, no specific platform)
    unspec_cloud = classified[
        (classified['is_cloud'] == True) &
        (classified['platform_stage1'].isin(['Unspecified']))
    ]
    cat2 = unspec_cloud.sample(min(50, len(unspec_cloud)))
    cat2 = cat2.copy()
    cat2['test_category'] = 'unspecified_platform'
    cat2['expected_cloud'] = True
    samples.append(cat2)
    print(f"  Category 2 (unspecified platform): {len(cat2)} records")

    # Category 3: Borderline — non-cloud records with cloud-adjacent terms
    noncloud = classified[classified['is_cloud'] == False]
    borderline_patterns = r'(?i)\b(hosting|managed\s+service|saas|paas|iaas|virtuali|data\s*cent|subscription.*(?:cloud|platform))\b'
    borderline = noncloud[noncloud['description'].str.contains(borderline_patterns, na=False, regex=True)]
    # Weight by log-dollars so we bias toward larger records without extreme skew
    weights = np.log1p(borderline['dollars'].clip(lower=1))
    weights = weights / weights.sum()
    cat3 = borderline.sample(min(50, len(borderline)), weights=weights)
    cat3 = cat3.copy()
    cat3['test_category'] = 'borderline'
    cat3['expected_cloud'] = None  # Unknown — this is what we're testing
    samples.append(cat3)
    print(f"  Category 3 (borderline): {len(cat3)} records")

    # Category 4: Clear non-cloud (no cloud terms at all)
    no_cloud_terms = r'(?i)\b(cloud|saas|paas|iaas|hosting|aws|azure|google\s+cloud|salesforce|oracle\s+cloud|ibm\s+cloud)\b'
    clear_noncloud = noncloud[~noncloud['description'].str.contains(no_cloud_terms, na=False, regex=True)]
    cat4 = clear_noncloud.sample(min(50, len(clear_noncloud)))
    cat4 = cat4.copy()
    cat4['test_category'] = 'clear_noncloud'
    cat4['expected_cloud'] = False
    samples.append(cat4)
    print(f"  Category 4 (clear non-cloud): {len(cat4)} records")

    result = pd.concat(samples, ignore_index=True)
    print(f"\n  Total test sample: {len(result)} records")
    return result


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_prompt_test(sample_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Run both prompts on the sample and collect results."""
    try:
        import anthropic
        from tqdm import tqdm
    except ImportError:
        print("ERROR: Install anthropic and tqdm: pip3 install anthropic tqdm")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    n = len(sample_df)
    prompts = {'concise': prompt_concise, 'extended': prompt_extended}

    results = {name: [] for name in prompts}

    for prompt_name, prompt_fn in prompts.items():
        print(f"\n  Running prompt '{prompt_name}' on {n} records...")
        errors = 0

        for i in tqdm(range(n), desc=f"  {prompt_name}"):
            row = sample_df.iloc[i]
            prompt_text = prompt_fn(row['description'], row['contractor_name'])

            parsed = None
            for attempt in range(3):
                try:
                    response = client.messages.create(
                        model=LLM_MODEL,
                        max_tokens=LLM_MAX_TOKENS,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt_text}]
                    )
                    parsed = parse_response(response.content[0].text)
                    break
                except Exception as e:
                    if 'RateLimit' in type(e).__name__ or '529' in str(e):
                        time.sleep(2 * (2 ** attempt))
                    else:
                        errors += 1
                        parsed = {'classification': 'error', 'platform': 'error',
                                  'confidence': 'low', 'evidence': str(e)[:100]}
                        break

            if parsed is None:
                errors += 1
                parsed = {'classification': 'error', 'platform': 'error',
                          'confidence': 'low', 'evidence': 'max_retries'}
            results[prompt_name].append(parsed)

        print(f"    Errors: {errors}")

    # Add results to DataFrame
    df = sample_df.copy()
    for prompt_name, result_list in results.items():
        prefix = f'{prompt_name}_'
        for key in ['classification', 'platform', 'confidence', 'evidence']:
            df[f'{prefix}{key}'] = [r[key] for r in result_list]
        df[f'{prefix}is_cloud'] = df[f'{prefix}classification'].isin(
            ['cloud-dependent', 'cloud-infrastructure'])

    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_results(df: pd.DataFrame):
    """Compare prompt A vs prompt B across all test categories."""
    print("\n" + "=" * 80)
    print("PROMPT COMPARISON RESULTS")
    print("=" * 80)

    for prompt_name in ['concise', 'extended']:
        prefix = f'{prompt_name}_'
        is_cloud_col = f'{prefix}is_cloud'
        cls_col = f'{prefix}classification'
        plat_col = f'{prefix}platform'
        conf_col = f'{prefix}confidence'

        print(f"\n{'─' * 40}")
        print(f"  PROMPT: {prompt_name.upper()}")
        print(f"{'─' * 40}")

        # Overall
        cloud_count = df[is_cloud_col].sum()
        print(f"  Overall: {cloud_count}/{len(df)} classified as cloud ({cloud_count/len(df)*100:.1f}%)")

        # Classification distribution
        print(f"  Classification distribution:")
        for cls, count in df[cls_col].value_counts().items():
            print(f"    {cls:30s}: {count:>4}")

        # By test category
        for cat in ['clear_cloud', 'unspecified_platform', 'borderline', 'clear_noncloud']:
            subset = df[df['test_category'] == cat]
            if len(subset) == 0:
                continue
            cloud = subset[is_cloud_col].sum()
            noncloud = len(subset) - cloud

            print(f"\n  Category: {cat} ({len(subset)} records)")
            print(f"    Cloud: {cloud} | Non-cloud: {noncloud}")

            if cat == 'clear_cloud':
                # Should all be cloud — misses are false negatives
                fn = noncloud
                print(f"    False negatives: {fn} ({fn/len(subset)*100:.1f}%)")
                if fn > 0:
                    misses = subset[~subset[is_cloud_col]]
                    for _, r in misses.head(5).iterrows():
                        print(f"      FN: ${r['dollars']/1e6:.1f}M | {str(r['description'])[:80]}")
                        print(f"          Evidence: {r[f'{prefix}evidence']}")

            elif cat == 'clear_noncloud':
                # Should all be non-cloud — hits are false positives
                fp = cloud
                print(f"    False positives: {fp} ({fp/len(subset)*100:.1f}%)")
                if fp > 0:
                    fps = subset[subset[is_cloud_col]]
                    for _, r in fps.head(5).iterrows():
                        print(f"      FP: ${r['dollars']/1e6:.1f}M | {str(r['description'])[:80]}")
                        print(f"          Evidence: {r[f'{prefix}evidence']}")

            elif cat == 'unspecified_platform':
                # Check how many got a specific platform
                has_platform = subset[~subset[plat_col].isin(['Unspecified', 'N/A', 'error'])]
                print(f"    Platform identified: {len(has_platform)}/{len(subset)}")
                if len(has_platform) > 0:
                    print(f"    Platforms found:")
                    for plat, count in has_platform[plat_col].value_counts().items():
                        print(f"      {plat:20s}: {count}")

            elif cat == 'borderline':
                # These are the interesting ones — how does LLM classify?
                print(f"    Classification breakdown:")
                for cls, count in subset[cls_col].value_counts().items():
                    dollars = subset.loc[subset[cls_col] == cls, 'dollars'].sum()
                    print(f"      {cls:30s}: {count:>3} records, ${dollars/1e6:>8.1f}M")

        # Confidence distribution
        print(f"\n  Confidence (cloud records only):")
        cloud_mask = df[is_cloud_col]
        if cloud_mask.any():
            for c, count in df.loc[cloud_mask, conf_col].value_counts().items():
                print(f"    {c:15s}: {count:>4}")

        # Parse errors
        error_count = (df[cls_col] == 'error').sum()
        print(f"\n  Parse errors: {error_count}")

    # Head-to-head comparison
    print(f"\n{'=' * 80}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 80}")

    agree = (df['concise_is_cloud'] == df['extended_is_cloud']).sum()
    disagree = len(df) - agree
    print(f"\n  Agreement: {agree}/{len(df)} ({agree/len(df)*100:.1f}%)")
    print(f"  Disagreement: {disagree}/{len(df)} ({disagree/len(df)*100:.1f}%)")

    if disagree > 0:
        conflicts = df[df['concise_is_cloud'] != df['extended_is_cloud']]
        print(f"\n  Disagreement details:")
        for _, r in conflicts.iterrows():
            desc = str(r['description'])[:70]
            c_cls = r['concise_classification']
            e_cls = r['extended_classification']
            print(f"    [{r['test_category']:20s}] ${r['dollars']/1e6:>7.1f}M | {desc}")
            print(f"      Concise: {c_cls:25s} | Extended: {e_cls}")
            print(f"      Concise evidence: {r['concise_evidence'][:80]}")
            print(f"      Extended evidence: {r['extended_evidence'][:80]}")

    # Platform agreement on cloud records
    both_cloud = df[df['concise_is_cloud'] & df['extended_is_cloud']]
    if len(both_cloud) > 0:
        plat_agree = (both_cloud['concise_platform'] == both_cloud['extended_platform']).sum()
        print(f"\n  Platform agreement (where both say cloud): {plat_agree}/{len(both_cloud)} ({plat_agree/len(both_cloud)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    merged_path = os.path.join(PROJECT_ROOT, 'data', '02_processed', '02_merged', 'merged_dataset.csv')
    classified_path = os.path.join(PROJECT_ROOT, 'data', '02_processed', '03_classified', 'classified_dataset.csv')

    # Build sample
    print("=" * 80)
    print("BUILDING STRATEGIC TEST SAMPLE")
    print("=" * 80)
    sample_df = build_test_sample(merged_path, classified_path)

    # Estimate cost
    n = len(sample_df)
    tokens_per_record_concise = 180 + 80  # input + output
    tokens_per_record_extended = 500 + 80
    cost_concise = n * 180 * 0.80 / 1e6 + n * 80 * 4.00 / 1e6
    cost_extended = n * 500 * 0.80 / 1e6 + n * 80 * 4.00 / 1e6
    total_cost = cost_concise + cost_extended
    print(f"\n  Estimated cost: ${total_cost:.2f} ({n} records × 2 prompts)")
    print(f"    Concise: ${cost_concise:.3f}")
    print(f"    Extended: ${cost_extended:.3f}")

    confirm = input("\n  Proceed? (yes/no): ")
    if confirm.lower() != 'yes':
        print("  Cancelled.")
        return

    # Run test
    print("\n" + "=" * 80)
    print("RUNNING PROMPT COMPARISON")
    print("=" * 80)

    results_df = run_prompt_test(sample_df, api_key)

    # Save results
    output_dir = os.path.join(PROJECT_ROOT, 'tests', 'prompt_test_results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'prompt_comparison_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")

    # Analyze
    analyze_results(results_df)

    return results_df


if __name__ == '__main__':
    main()
    

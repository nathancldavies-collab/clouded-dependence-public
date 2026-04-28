#!/usr/bin/env python3
"""
Build Human-in-the-Loop Validation Sample
==========================================
Generates two stratified CSVs for manual review:

  Part A: Classification validation (~175 records)
    Stratified by: cloud_classification, classification_confidence,
    and RegEx/LLM agreement (classification_source)

  Part B: Attribution validation (~100 records)
    Stratified by: attribution_method (phase), final_platform

Output files (in outputs_results_v2/validation/):
  part_a_classification_sample.csv
  part_b_attribution_sample.csv
  validation_summary_template.md

Usage:
  python notebooks/04_validation/build_validation_sample.py
"""

import os
import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data", "02_processed", "03_classified", "attributed_dataset.csv"
)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs_results_v2", "validation")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
ID_COLS = [
    "original_prime_id",
    "record_type",
    "fiscal_year",
    "dollars",
]
CONTRACTOR_COLS = [
    "contractor_name",
    "prime_contractor",   # only meaningful for subcontracts
]
TEXT_COL = "description"

PIPELINE_COLS = [
    "cloud_classification",       # FINAL: non-cloud | cloud-dependent | cloud-infrastructure
    "is_cloud",
    "final_platform",             # FINAL platform (or contractor name if unattributed)
    "attribution_method",         # phase1_direct | phase2_description | phase3_pattern | ...
    "confidence",                 # conclusive | high | medium | pattern_inferred | unattributed | N/A
]
DIAGNOSTIC_COLS = [
    "regex_is_cloud",
    "regex_platforms",
    "regex_confidence",
    "llm_classification",
    "llm_platform",
    "llm_confidence",
    "classification_source",      # regex+llm_agree | llm_only | regex_only | conflict
    "classification_confidence",  # high | medium | low | conflict
]

# Reviewer-filled columns
REVIEW_COLS_A = [
    "manual_classification",   # non-cloud | cloud-dependent | cloud-infrastructure
    "correct_classification",  # Y / N / ?
    "notes",
]
REVIEW_COLS_B = [
    "manual_platform",         # platform name | 'cannot determine' | 'unattributable'
    "correct_attribution",     # Y / N / ?
    "notes",
]

ALL_SOURCE_COLS = ID_COLS + CONTRACTOR_COLS + [TEXT_COL] + PIPELINE_COLS + DIAGNOSTIC_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_sample(df: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
    """Sample up to n rows; warn if fewer available."""
    actual = min(n, len(df))
    if actual < n:
        print(f"  [warn] '{label}': requested {n}, only {actual} available")
    if actual == 0:
        return pd.DataFrame(columns=df.columns)
    return df.sample(n=actual, random_state=RANDOM_SEED)


def load_data() -> pd.DataFrame:
    print(f"Loading {DATA_FILE} …")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"  {len(df):,} records, {df.shape[1]} columns")

    if "classification_source" not in df.columns:
        df["classification_source"] = "unknown"
    if "classification_confidence" not in df.columns:
        df["classification_confidence"] = "unknown"

    return df


def already_selected(frames: list) -> pd.Series:
    """Return set of prime IDs already in sampled frames."""
    if not frames:
        return pd.Series(dtype=str)
    return pd.concat(frames)["original_prime_id"]


# ---------------------------------------------------------------------------
# Part A: Classification validation sample (~175 records)
# ---------------------------------------------------------------------------

def build_part_a(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified sample designed to surface the cases most likely to be wrong:

    1. RegEx / LLM conflict             ~30  (hardest; method disagrees)
    2. LLM-only (no regex support)      ~25  (no corroboration)
    3. Low-confidence cloud             ~25  (pipeline itself flagged doubt)
    4. cloud-infrastructure             ~25  (minority class, distinct meaning)
    5. cloud-dependent, high confidence ~25  (easy positives — baseline check)
    6. non-cloud baseline               ~45  (easy negatives — false-positive check)
    """
    print("\n=== Part A: Classification validation sample ===")
    frames = []

    # Stratum 1: conflict
    conflict = df[df["classification_source"] == "conflict"]
    s = safe_sample(conflict, 30, "conflict"); s["_stratum"] = "1_conflict_regex_llm"; frames.append(s)
    print(f"  Conflict (regex vs LLM disagree):   {len(conflict):,} pool → {len(s)} sampled")

    # Stratum 2: LLM-only
    llm_only = df[df["classification_source"] == "llm_only"]
    llm_only = llm_only[~llm_only["original_prime_id"].isin(already_selected(frames))]
    s = safe_sample(llm_only, 25, "llm_only"); s["_stratum"] = "2_llm_only"; frames.append(s)
    print(f"  LLM-only (no regex signal):          {len(llm_only):,} pool → {len(s)} sampled")

    # Stratum 3: low-confidence cloud
    low_conf = df[
        df["is_cloud"].fillna(False).astype(bool) &
        df["classification_confidence"].isin(["low", "conflict"])
    ]
    low_conf = low_conf[~low_conf["original_prime_id"].isin(already_selected(frames))]
    s = safe_sample(low_conf, 25, "low_conf_cloud"); s["_stratum"] = "3_low_confidence_cloud"; frames.append(s)
    print(f"  Low-confidence cloud:                {len(low_conf):,} pool → {len(s)} sampled")

    # Stratum 4: cloud-infrastructure (minority class)
    infra = df[df["cloud_classification"] == "cloud-infrastructure"]
    infra = infra[~infra["original_prime_id"].isin(already_selected(frames))]
    s = safe_sample(infra, 25, "cloud-infrastructure"); s["_stratum"] = "4_cloud_infrastructure"; frames.append(s)
    print(f"  cloud-infrastructure:                {len(infra):,} pool → {len(s)} sampled")

    # Stratum 5: cloud-dependent, high confidence (easy positives)
    cdep_high = df[
        (df["cloud_classification"] == "cloud-dependent") &
        (df["classification_confidence"] == "high") &
        (df["classification_source"] == "regex+llm_agree")
    ]
    cdep_high = cdep_high[~cdep_high["original_prime_id"].isin(already_selected(frames))]
    s = safe_sample(cdep_high, 25, "cloud_dep_high"); s["_stratum"] = "5_cloud_dependent_high_conf"; frames.append(s)
    print(f"  cloud-dependent high-conf:           {len(cdep_high):,} pool → {len(s)} sampled")

    # Stratum 6: non-cloud baseline
    non_cloud = df[df["cloud_classification"] == "non-cloud"]
    non_cloud = non_cloud[~non_cloud["original_prime_id"].isin(already_selected(frames))]
    s = safe_sample(non_cloud, 45, "non_cloud"); s["_stratum"] = "6_non_cloud_baseline"; frames.append(s)
    print(f"  non-cloud baseline:                  {len(non_cloud):,} pool → {len(s)} sampled")

    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["original_prime_id"])
    print(f"\n  Total Part A: {len(result)} records")

    out_cols = [c for c in ALL_SOURCE_COLS + ["_stratum"] if c in result.columns]
    result = result[out_cols].copy()
    for col in REVIEW_COLS_A:
        result[col] = ""

    return result.sort_values("_stratum").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part B: Attribution validation sample (~100 records)
# ---------------------------------------------------------------------------

def build_part_b(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cloud records only, stratified by attribution phase and platform coverage:

    1. Phase 1 – direct entity          ~15  (should be highly accurate)
    2. Phase 2 – description, high/med  ~15  (description-driven attribution)
    3. Phase 2 – description, low/conf  ~10  (riskier description attribution)
    4. Phase 3 – pattern inferred       ~20  (most assumptions; highest risk)
    5. Unattributed cloud               ~20  (check: are they truly unattributable?)
    + Platform top-ups to ensure ≥3 records per major platform
    """
    print("\n=== Part B: Attribution validation sample ===")
    cloud = df[df["is_cloud"].fillna(False).astype(bool)].copy()
    print(f"  Cloud records: {len(cloud):,}")
    frames = []

    # Phase 1
    p1 = cloud[cloud["attribution_method"] == "phase1_direct"]
    s = safe_sample(p1, 15, "phase1"); s["_stratum"] = "1_phase1_direct_entity"; frames.append(s)
    print(f"  Phase 1 (direct entity):             {len(p1):,} pool → {len(s)} sampled")

    # Phase 2 – high/medium confidence
    p2h = cloud[
        (cloud["attribution_method"] == "phase2_description") &
        (cloud["classification_confidence"].isin(["high", "medium"]))
    ]
    p2h = p2h[~p2h["original_prime_id"].isin(already_selected(frames))]
    s = safe_sample(p2h, 15, "phase2_high"); s["_stratum"] = "2_phase2_description_highmed"; frames.append(s)

    # Phase 2 – low/conflict confidence
    p2l = cloud[
        (cloud["attribution_method"] == "phase2_description") &
        (cloud["classification_confidence"].isin(["low", "conflict"]))
    ]
    p2l = p2l[~p2l["original_prime_id"].isin(already_selected(frames))]
    s = safe_sample(p2l, 10, "phase2_low"); s["_stratum"] = "3_phase2_description_lowconf"; frames.append(s)
    print(f"  Phase 2 (description):               {len(p2h)+len(p2l):,} pool → {15+len(s)} sampled")

    # Phase 3
    p3 = cloud[cloud["attribution_method"] == "phase3_pattern"]
    p3 = p3[~p3["original_prime_id"].isin(already_selected(frames))]
    s = safe_sample(p3, 20, "phase3"); s["_stratum"] = "4_phase3_pattern_inferred"; frames.append(s)
    print(f"  Phase 3 (pattern inferred):          {len(p3):,} pool → {len(s)} sampled")

    # Unattributed cloud
    unattr = cloud[cloud["attribution_method"] == "cloud_unattributed_contractor"]
    unattr = unattr[~unattr["original_prime_id"].isin(already_selected(frames))]
    s = safe_sample(unattr, 20, "unattributed"); s["_stratum"] = "5_unattributed_cloud"; frames.append(s)
    print(f"  Unattributed cloud:                  {len(unattr):,} pool → {len(s)} sampled")

    # Platform top-ups: ensure ≥3 records per major platform
    MAJOR_PLATFORMS = [
        "AWS", "Azure", "Google Cloud", "Oracle Cloud",
        "IBM Cloud", "Salesforce", "ServiceNow", "Workday"
    ]
    for platform in MAJOR_PLATFORMS:
        so_far = pd.concat(frames)
        have = len(so_far[so_far["final_platform"] == platform])
        if have < 3:
            pool = cloud[
                (cloud["final_platform"] == platform) &
                (~cloud["original_prime_id"].isin(so_far["original_prime_id"]))
            ]
            s = safe_sample(pool, 3 - have, f"topup_{platform}")
            if len(s):
                s["_stratum"] = f"6_platform_topup_{platform}"
                frames.append(s)
                print(f"  Platform top-up {platform}: added {len(s)}")

    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["original_prime_id"])
    print(f"\n  Total Part B: {len(result)} records")

    out_cols = [c for c in ALL_SOURCE_COLS + ["_stratum"] if c in result.columns]
    result = result[out_cols].copy()
    for col in REVIEW_COLS_B:
        result[col] = ""

    return result.sort_values("_stratum").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary template
# ---------------------------------------------------------------------------

def write_summary_template(path: str, n_a: int, n_b: int):
    text = f"""# Human Validation Summary — Clouded Dependence

**Date of review:** _______________
**Reviewer:** _______________
**Records reviewed:** Part A: ___ / {n_a}   |   Part B: ___ / {n_b}

---

## Part A: Classification Accuracy

### By stratum
| Stratum | N reviewed | Correct (Y) | Incorrect (N) | Uncertain (?) | Accuracy |
|---------|-----------|------------|--------------|--------------|---------|
| 1. Conflict (RegEx vs LLM disagree) | | | | | |
| 2. LLM-only (no regex support) | | | | | |
| 3. Low-confidence cloud | | | | | |
| 4. Cloud-infrastructure | | | | | |
| 5. Cloud-dependent (high confidence) | | | | | |
| 6. Non-cloud baseline | | | | | |
| **TOTAL** | | | | | |

### By classification category
| Category | N reviewed | Correct | Accuracy |
|----------|-----------|---------|---------|
| non-cloud | | | |
| cloud-dependent | | | |
| cloud-infrastructure | | | |
| **TOTAL** | | | |

### Misclassification patterns observed
*(Describe any systematic errors — e.g., "IT security contracts mentioning cloud storage
were classified as cloud-dependent despite describing on-premise systems.")*

_______________________________________________________________________________
_______________________________________________________________________________

---

## Part B: Attribution Accuracy

### By attribution phase
| Phase | N reviewed | Correct (Y) | Incorrect (N) | Uncertain (?) | Accuracy |
|-------|-----------|------------|--------------|--------------|---------|
| Phase 1 – direct entity | | | | | |
| Phase 2 – description (high/med conf) | | | | | |
| Phase 2 – description (low/conf) | | | | | |
| Phase 3 – pattern inferred | | | | | |
| Unattributed (check if truly unattributable) | | | | | |
| **TOTAL** | | | | | |

### By platform
| Platform | N reviewed | Correct | Accuracy |
|----------|-----------|---------|---------|
| AWS | | | |
| Azure | | | |
| Google Cloud | | | |
| Oracle Cloud | | | |
| IBM Cloud | | | |
| Salesforce | | | |
| ServiceNow | | | |
| Workday | | | |
| Unattributed | | | |

### Percentage of unattributed records judged genuinely unattributable
Of the {{}}/20 unattributed cloud records reviewed:
- Truly unattributable (no platform signal in text): _____ (____%)
- Could have been attributed (platform identifiable from text): _____ (____%)
- Ambiguous: _____ (____%)

### Attribution error patterns observed
*(Describe systematic errors — e.g., "Phase 3 incorrectly inferred AWS for Booz Allen
Hamilton records based on a small number of early AWS-related contracts.")*

_______________________________________________________________________________
_______________________________________________________________________________

---

## Summary Statistics for Paper (Section 3.7)

**Classification accuracy (Part A, N={n_a}):**
- Overall: ____% (___ / ___ records)
- cloud-dependent: ____%
- cloud-infrastructure: ____%
- non-cloud: ____%
- Lowest-confidence records: ____%
- RegEx/LLM conflict cases: ____%

**Attribution accuracy (Part B, N={n_b} cloud records):**
- Overall: ____% (___ / ___ records)
- Phase 1 (direct entity match): ____%
- Phase 2 (description-based): ____%
- Phase 3 (pattern-inferred): ____%
- Unattributed records confirmed genuinely unattributable: ____% of sample

---

## Suggested paper text (Section 3.7)

> "We manually validated {n_a} records stratified by classification category (non-cloud,
> cloud-dependent, cloud-infrastructure), LLM confidence level, and method agreement
> between the RegEx and LLM classifiers, plus {n_b} cloud records stratified by attribution
> phase. Overall classification accuracy was ____% (cloud-dependent: __%, cloud-infrastructure:
> __%). Classification accuracy for records where RegEx and LLM disagreed was ___%, compared
> to ___% where both methods agreed. Attribution accuracy was ____% for Phase 1 (direct
> entity matching), ____% for Phase 2 (description-based), and ____% for Phase 3
> (pattern-inferred). Of the unattributed cloud records reviewed, ____% were confirmed to
> be genuinely unattributable from available contract text."

---
*Generated by notebooks/04_validation/build_validation_sample.py*
"""
    with open(path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()

    part_a = build_part_a(df)
    part_b = build_part_b(df)

    path_a = os.path.join(OUT_DIR, "part_a_classification_sample.csv")
    path_b = os.path.join(OUT_DIR, "part_b_attribution_sample.csv")
    path_tmpl = os.path.join(OUT_DIR, "validation_summary_template.md")

    part_a.to_csv(path_a, index=False)
    part_b.to_csv(path_b, index=False)
    write_summary_template(path_tmpl, len(part_a), len(part_b))

    print(f"\n=== Outputs written ===")
    print(f"  {path_a}  ({len(part_a)} records, {part_a.shape[1]} cols)")
    print(f"  {path_b}  ({len(part_b)} records, {part_b.shape[1]} cols)")
    print(f"  {path_tmpl}")

    print("\n--- Part A strata ---")
    print(part_a["_stratum"].value_counts().sort_index().to_string())
    print("\n--- Part B strata ---")
    print(part_b["_stratum"].value_counts().sort_index().to_string())
    print("\n--- Part B platform coverage ---")
    print(part_b["final_platform"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()

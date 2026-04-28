# Data Requirements

This document describes the input data the pipeline expects and the schema of all intermediate and final outputs.

No data is included in this repository. See the README for sources.

---

## Input Files

Place both files in `data/01_raw/` before running the pipeline.

### `prime_awards.csv` — FPDS-NG Prime Contract Awards

Expected encoding: UTF-8 with BOM (`utf-8-sig`).

| Column | Type | Description |
|--------|------|-------------|
| `Award ID` | string | Unique award identifier (PIID + agency codes) |
| `Recipient Name` | string | Contractor name |
| `Recipient UEI` | string | Unique Entity Identifier |
| `Recipient Parent UEI` | string | Parent organisation UEI |
| `Recipient Parent Name` | string | Parent organisation name |
| `Total Dollars Obligated` | float | Total obligated dollar amount |
| `Award Description` | string | Free-text contract description |
| `Awarding Agency Name` | string | Federal agency name |
| `NAICS Code` | string | Industry classification |
| `Product or Service Code` | string | PSC code |
| `Period of Performance Start Date` | date | Contract start |
| `Period of Performance Current End Date` | date | Contract end |

The pipeline filters to:
- **Date range:** FY 2017–2024 (based on Period of Performance Start Date)
- **Services only:** PSC codes in ranges D (ADP services), R (support services), and selected others — hardware PSC codes are excluded
- **Deduplication:** one row per Award ID (max dollars)

### `subawards.csv` — FPDS-NG Subawards

Expected encoding: UTF-8.

| Column | Type | Description |
|--------|------|-------------|
| `Subaward Number` | string | Subaward identifier |
| `Prime Award ID` | string | Parent prime award ID |
| `Sub-Awardee Name` | string | Subcontractor name |
| `Sub-Awardee UEI` | string | Subcontractor UEI |
| `Subaward Amount` | float | Dollar amount |
| `Subaward Description` | string | Free-text description |

**Note:** The FPDS-NG subaward file does not include Subcontractor Parent UEI. The pipeline resolves parent entities for subcontractors by cross-referencing their UEI against the `Recipient UEI` → `Recipient Parent UEI` mapping derived from the prime awards file.

---

## Intermediate Outputs

All intermediate files are written to `data/02_processed/` (gitignored).

### `data/02_processed/01_filtered/prime_services_filtered.csv`
Filtered prime awards. Same schema as input `prime_awards.csv`, fewer rows.
- Rows: ~189,000 (from 206,000 raw)
- Dollars: ~$237B (from ~$295B raw)

### `data/02_processed/02_merged/merged_dataset.csv`
Unified prime + subaward dataset. Core working file for all downstream stages.

| Column | Type | Description |
|--------|------|-------------|
| `contractor` | string | Resolved entity identifier (Parent UEI or UEI) |
| `contractor_name` | string | Resolved contractor name |
| `dollars` | float | Dollar amount for this record |
| `description` | string | Free-text description |
| `record_type` | string | `prime_no_subs` / `prime_retained` / `subcontract` |
| `original_prime_id` | string | Source prime Award ID |
| `fiscal_year` | float | Fiscal year (Oct–Sep) |
| `prime_contractor` | string | Prime contractor name (subcontract rows only) |

- Rows: ~216,600
- `prime_no_subs`: primes with no associated subawards (dollar amount = prime amount)
- `prime_retained`: primes with subawards (dollar amount = prime minus sub total, or 0 if subs exceed prime)
- `subcontract`: individual subaward flows

### `data/02_processed/03_classified/classified_dataset.csv`
Merged dataset with Stage 1 classification columns appended.

Additional columns beyond merged dataset:

| Column | Type | Description |
|--------|------|-------------|
| `regex_is_cloud` | bool | RegEx detected cloud signal |
| `regex_platforms` | string | Platform(s) detected by RegEx |
| `regex_confidence` | string | `High` / `Medium` / `Low` |
| `explicit_hyperscaler_mentions` | list | Explicit platform names found |
| `explicit_cloud_mention` | bool | Whether a hyperscaler was explicitly named |
| `llm_classification` | string | `non-cloud` / `cloud-dependent` / `cloud-infrastructure` |
| `llm_platform` | string | Platform identified by LLM |
| `llm_confidence` | string | LLM confidence score |
| `llm_evidence` | string | LLM chain-of-thought reasoning |
| `is_cloud` | bool | **Final** cloud flag (synthesised) |
| `cloud_classification` | string | **Final** classification label |
| `platform_stage1` | string | Best platform from Stage 1 |
| `classification_confidence` | string | `high` / `medium` / `low` / `conflict` |
| `classification_source` | string | `regex+llm_agree` / `llm_only` / `regex_only` / `conflict` |
| `platform_mentions` | list | Concrete platform mentions (for multi-platform splitting) |

---

## Final Output

### `data/02_processed/03_classified/attributed_dataset.csv`
Full dataset with all classification and attribution columns. This is the primary analysis file.

All columns from `classified_dataset.csv`, plus:

| Column | Type | Description |
|--------|------|-------------|
| `platform_phase1` | string | Phase 1 attribution (direct entity match) |
| `attribution_method` | string | `phase1_direct` / `phase2_description` / `phase3_pattern` / `cloud_unattributed_contractor` / `non_cloud_contractor` |
| `platform_phase2` | string | Phase 2/3 platform assignment |
| `final_platform` | string | **Final** platform name, or contractor name if unattributed |
| `confidence` | string | `conclusive` / `high` / `medium` / `pattern_inferred` / `unattributed` / `N/A` |

**Attribution method values:**

| Value | Meaning |
|-------|---------|
| `phase1_direct` | Contractor IS a platform vendor (e.g. Microsoft → Azure) |
| `phase2_description` | Platform identified in description-level classification |
| `phase3_pattern` | Platform inferred from contractor's historical behaviour (≥90% concentration threshold) |
| `cloud_unattributed_contractor` | Cloud record with no platform attribution — `final_platform` = contractor name |
| `non_cloud_contractor` | Non-cloud record — `final_platform` = contractor name |

**Note on Phase 3:** Phase 3 attribution is not amenable to record-level validation. The evidence is the contractor's aggregate portfolio, not the individual description. Results excluding Phase 3 attribution can be computed by treating `attribution_method == 'phase3_pattern'` rows as `cloud_unattributed_contractor`.

---

## LLM Cache

`data/02_processed/03_classified/llm_cache/` (gitignored) stores:

| File | Description |
|------|-------------|
| `llm_results_final.csv` | Merged LLM results for all records |
| `llm_progress.json` | Progress tracker across batch chunks |
| `batch_manifest.json` | Batch API chunk metadata |
| `chunk_*.jsonl` | Raw batch input files |
| `chunk_*_results.jsonl` | Raw batch output files |

The cache is reused automatically on re-runs. Delete `llm_progress.json` to force re-classification.

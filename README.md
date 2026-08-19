# Clouded Dependence — Code Repository

Code for the multi-step attribution pipeline described in:

> Davies, N. & Rystrøm, J. (2026). *Clouded Dependence: Supplier Diversity and Hidden Platform Concentration in Digital Government*. [DG.O (journal TBD)]

The paper identifies and measures hidden platform concentration in US federal IT procurement by tracing cloud platform dependencies embedded within contractor relationships. Across 216,604 contract records (FY 2017–2024, $237B), the pipeline finds that platform-level concentration (HHI 793) is 7.6× the contractor-level baseline (HHI 105) — a gap that is invisible when using standard contractor-level market analysis.

---

## Pipeline Overview

```
Raw FPDS-NG data
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 0: Data Preparation                           │
│  0A. Load prime awards + subawards                  │
│  0B. Filter: FY2017–2024, services only (PSC codes) │
│  0C. Merge: primes + subs into unified dataset      │
│      Entity resolution via Contractor Parent UEI    │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: Cloud Classification                       │
│  RegEx: ~200 compiled patterns across brand names,  │
│         specific services, contractor name signals  │
│  LLM:   Claude Haiku 4.5 (chain-of-thought prompt) │
│         classifies each record independently        │
│  Synthesis: RegEx + LLM agreement → final label    │
│  Output: non-cloud / cloud-dependent /             │
│          cloud-infrastructure                       │
└─────────────────────────┬───────────────────────────┘
                          │ (cloud records only)
                          ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: Platform Attribution                       │
│  Phase 1: Direct entity matching                    │
│           (contractor IS a platform vendor)         │
│  Phase 2: Description-based attribution             │
│           (platform identified in Stage 1)          │
│  Phase 3: Behavioural pattern matching              │
│           (contractor ≥90% concentrated on one      │
│            platform across attributed records)      │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: Analysis                                   │
│  Contractor-level HHI (baseline)                    │
│  Platform-level HHI                                 │
│  Concentration gap                                  │
│  Temporal trends (3-year rolling windows)           │
│  Platform share decomposition                       │
└─────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
clouded-dependence-public/
├── run_pipeline.py                  # Main runner — executes all stages in order
├── batch_manager.py                 # LLM batch job management (chunked Batch API)
├── pyproject.toml
├── justfile
│
├── pipeline/
│   ├── 01_prep/
│   │   ├── filter_primes.py         # Date/PSC/hardware filtering
│   │   ├── create_merged_dataset.py # Prime–sub merge + entity resolution
│   │   └── 01_filter_and_merge.ipynb
│   │
│   ├── 02_base_analysis/
│   │   ├── baseline_merged_hhi.py   # Contractor-level HHI
│   │   └── 02_baseline_hhi.ipynb
│   │
│   ├── 03_classification/
│   │   ├── cloud_classification.py  # RegEx patterns + LLM classification
│   │   └── 03a_cloud_classification.ipynb
│   │
│   ├── 04_attribution/
│   │   ├── platform_attribution.py  # Phase 1–3 attribution + platform HHI
│   │   └── 03b_platform_attribution.ipynb
│   │
│   └── 05_analysis/
│       ├── refresh_outputs_calendar_year.py  # Calendar-year figures + tables
│       └── 04_platform_hhi.ipynb
│
└── tests/
    ├── prompt_comparison_test.py    # LLM prompt A/B test on small sample
    └── build_validation_sample.py  # Stratified human-validation sampler
```

---

## Data Requirements

See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for full schema details.

The pipeline expects two CSV files in `data/01_raw/`:

| File | Description |
|------|-------------|
| `prime_awards.csv` | FPDS-NG prime contract awards |
| `subawards.csv` | FPDS-NG subaward data |

FPDS-NG data is publicly available via [USAspending.gov](https://www.usaspending.gov/download_center/award_data_archive). The specific extract used in the paper was provided by Leadership Connect and covers FY 2017–2024.

---

## Setup

Install [`uv`](https://docs.astral.sh/uv/), then use it to get [`just`](https://just.systems):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install rust-just
```

```bash
git clone https://github.com/[your-username]/clouded-dependence-public
cd clouded-dependence-public
just install
```

Place your FPDS-NG data files in `data/01_raw/` (this directory is gitignored).

---

## Usage

### RegEx-only mode (no API key required)

```bash
python run_pipeline.py
```

This runs Stages 0–2 using only the RegEx classifier. Results will differ from the paper, which uses RegEx + LLM classification.

### Full pipeline with LLM classification

Set your Anthropic API key and run:

```bash
export ANTHROPIC_API_KEY=your_key_here
python run_pipeline.py
```

The pipeline will prompt for confirmation before submitting LLM batches. LLM results are cached in `data/02_processed/03_classified/llm_cache/` and reused on subsequent runs.

**Cost note:** Running LLM classification on the full dataset (216,604 records) costs approximately $195 using Claude Haiku 4.5 via the Batch API. The pipeline processes records in 50,000-record chunks due to the 256MB Batch API limit. Use `batch_manager.py` to manage multi-session batch runs.

### Force rebuild

To rebuild from raw data even if a cache exists:

```bash
FORCE_REMERGE=1 python run_pipeline.py
```

---

## LLM Classification

Stage 1 uses **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) via the [Anthropic Batch API](https://docs.anthropic.com/en/docs/build-with-claude/message-batches).

The classification prompt (`C_cot`, chain-of-thought) asks the model to identify four sequential signals before reaching a classification:

1. Does the description name a specific cloud platform or cloud-native product?
2. Does the description describe a service that is fundamentally hosted/delivered via cloud infrastructure?
3. Does the contractor name suggest a cloud platform provider or reseller?
4. Is there any evidence this could be on-premise or hardware-based (negating signals)?

The prompt is defined in `pipeline/03_classification/cloud_classification.py` in the `build_llm_prompt()` function.

---

## Classification Scheme

| Label | Definition |
|-------|-----------|
| `non-cloud` | IT services, consulting, hardware, or software with no meaningful cloud dependency |
| `cloud-dependent` | Software or services that fundamentally require a cloud platform to function (SaaS, managed cloud services, cloud-hosted systems) |
| `cloud-infrastructure` | Direct procurement of cloud compute, storage, or infrastructure — the contractor is the cloud provider or is reselling raw cloud capacity (IaaS) |

---

## Platform Taxonomy

The pipeline attributes cloud records to eight platforms:

| Platform | Vendor |
|----------|--------|
| AWS | Amazon Web Services |
| Azure | Microsoft |
| Google Cloud | Google / Alphabet |
| Oracle Cloud | Oracle America |
| IBM Cloud | IBM / International Business Machines |
| Salesforce | Salesforce |
| ServiceNow | ServiceNow |
| Workday | Workday |

**Note on subsidiaries:** Phase 1 entity matching maps parent company name variants only. Subsidiary brand names (e.g., Heroku → Salesforce, Tableau Online → Salesforce) are captured via description-level patterns in Phase 2. Cerner Corporation, acquired by Oracle in 2022, is attributed as a standalone entity in the baseline results; a sensitivity specification mapping post-acquisition Cerner contracts to Oracle Cloud is available via `data/02_processed/03_classified/attributed_dataset_oracle_cerner.csv`.


---

## License

MIT License. See [LICENSE](LICENSE).

The code is provided for reproducibility. No underlying procurement data is included in this repository. FPDS-NG data is published by the US federal government and available via USAspending.gov.

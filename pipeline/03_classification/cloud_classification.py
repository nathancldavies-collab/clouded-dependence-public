#!/usr/bin/env python3
"""
Stage 1: Cloud Classification (RegEx + LLM + Synthesis)
=========================================================
Classifies each record in the merged dataset as cloud or non-cloud,
and extracts platform mentions where possible.

Two parallel classification methods:
  Step 1A: RegEx — fast, free, precise but limited recall
  Step 1B: LLM  — comprehensive, contextual, costs money
  Step 1C: Synthesis — combine both with confidence scoring

Output columns:
  is_cloud: bool (final determination)
  cloud_classification: non-cloud / cloud-dependent / cloud-infrastructure
  platform_stage1: best platform from classification (or Unspecified/N/A)
  platform_mentions: list of concrete platform mentions (for multi-platform split)
  classification_confidence: high / medium / low / conflict
  classification_source: regex+llm_agree / llm_only / regex_only / conflict / etc.
"""

import pandas as pd
import numpy as np
import re
import os
import json
import time
from typing import Dict, Tuple, Optional, List, Any

# =============================================================================
# ENHANCED REGEX PATTERNS (based on Luitse 2024 + expanded coverage)
# =============================================================================

PLATFORM_PATTERNS = {
    'AWS': {
        'brand_names': [
            r'\bAWS\b',
            r'Amazon\s+Web\s+Services',
            r'Amazon\s+Cloud',
        ],
        'specific_services': [
            # IaaS - Compute
            r'\bEC2\b',
            r'\bLambda\b(?!\s+(function|expression|calculus|county|school))',
            r'\bFargate\b',
            r'Elastic\s+Beanstalk\b',
            r'\bLightsail\b',
            r'\bInferentia\b',  # Custom chip
            r'\bTrainium\b',   # Custom chip
            
            # IaaS - Storage
            r'\bS3\b(?!\s+(compatible|protocol|class))',
            r'\bEBS\b(?=\s|,|\.)',
            r'\bEFS\b(?=\s|,|\.)',
            r'\bGlacier\b(?=\s+(storage|archive|vault|deep))',
            r'AWS\s+(?:Storage|Backup)',
            
            # Database
            r'\bRDS\b(?=\s|,|\.)',
            r'\bDynamoDB\b',
            r'\bAurora\b(?=\s+(database|DB|MySQL|PostgreSQL))',
            r'\bRedshift\b(?!.*\s+shift)',
            r'\bNeptune\b(?=\s+(database|graph|DB))',
            r'DocumentDB',
            r'ElastiCache',
            
            # PaaS - ML/AI
            r'\bSageMaker\b',
            r'SageMaker\s+(?:Studio|Ground\s+Truth|Neo|Autopilot|Clarify|Pipelines|JumpStart)',
            r'\bAWS\s+Neuron\b',
            r'Deep\s+Learning\s+AMI',
            r'Amazon\s+(?:Rekognition|Comprehend|Lex|Polly|Textract|Translate)',
            
            # Networking
            r'\bVPC\b(?=\s|,|\.)(?!.*\s+(virtual\s+private\s+connection))',
            r'Route\s*53',
            r'\bCloudFront\b',
            r'Direct\s+Connect',
            r'AWS\s+(?:VPC|Transit\s+Gateway|VPN)',
            
            # Management
            r'\bCloudFormation\b',
            r'\bCloudWatch\b',
            r'\bCloudTrail\b',
            r'AWS\s+(?:Config|Systems\s+Manager|OpsWorks)',
            
            # Security
            r'\bIAM\b(?=\s+(role|policy|user|permission|authentication))',
            r'\bCognito\b',
            r'AWS\s+(?:IAM|WAF|Shield|GuardDuty|Inspector|Macie)',
            r'\bKMS\b(?=\s+(key|encryption|decrypt))',
            
            # Integration
            r'\bSNS\b(?=\s|,|\.)',
            r'\bSQS\b(?=\s|,|\.)',
            r'Step\s*Functions',
            r'AWS\s+(?:SNS|SQS|EventBridge|AppSync)',
            
            # Analytics
            r'\bAthena\b(?=\s+(query|database|SQL))',
            r'\bEMR\b(?=\s|,|\.)',
            r'\bKinesis\b',
            r'\bQuickSight\b',
            r'AWS\s+(?:Athena|EMR|Kinesis|Glue)',
            
            # Containers
            r'\bECS\b(?=\s|,|\.)',
            r'\bEKS\b(?=\s|,|\.)',
            r'AWS\s+(?:ECS|EKS|Fargate)',
            
            # Specialized
            r'Amazon\s+(?:Monitron|Lookout|Panorama|HealthLake)',
            r'Amazon\s+A2I\b',
            r'\bMechanical\s+Turk\b',
            
            # GovCloud
            r'\bAWS\s+GovCloud\b',
            r'GovCloud\s+(?:East|West)',
        ],
        'contractor_names': ['amazon', 'aws', 'amazon web services']
    },
    
    'Azure': {
        'brand_names': [
            r'\bAzure\b',
            r'Microsoft\s+Azure',
            r'MS\s+Azure',
        ],
        'specific_services': [
            # IaaS - Compute
            r'Azure\s+(?:Virtual\s+)?Machines?',
            r'Azure\s+Functions',
            r'Azure\s+App\s+Service',
            r'Azure\s+Container\s+(?:Instances|Apps)',
            r'Azure\s+Batch',
            
            # IaaS - Storage
            r'Azure\s+(?:Blob\s+)?Storage',
            r'Azure\s+Files',
            r'Azure\s+Disk',
            r'Azure\s+Data\s+Lake',
            
            # Database
            r'Azure\s+SQL',
            r'\bCosmos\s*DB\b',
            r'Azure\s+Database(?:\s+for\s+(?:MySQL|PostgreSQL|MariaDB))?',
            r'Azure\s+Synapse',
            r'Azure\s+Cache\s+for\s+Redis',
            
            # PaaS - ML/AI
            r'Azure\s+Machine\s+Learning(?:\s+Designer)?',
            r'Azure\s+(?:ML|MLOps)\b',
            r'Azure\s+Cognitive\s+(?:Services|Toolkit)',
            r'Microsoft\s+Cognitive\s+Toolkit\b',
            r'Azure\s+(?:Percept|Bot\s+Services)',
            r'\bResponsible\s+AI\b(?=.*Azure)',
            
            # Networking
            r'Azure\s+Virtual\s+Network',
            r'Azure\s+Load\s+Balancer',
            r'Azure\s+Front\s+Door',
            r'Azure\s+CDN',
            r'Azure\s+(?:VPN|ExpressRoute|Firewall)',
            
            # Management
            r'Azure\s+Resource\s+Manager',
            r'\bARM\s+template',
            r'Azure\s+Monitor',
            r'Azure\s+DevOps',
            r'Azure\s+(?:Automation|Advisor|Policy)',
            
            # Security
            r'Azure\s+Active\s+Directory',
            r'\bAAD\b(?=\s|,|\.)',
            r'Azure\s+Key\s+Vault',
            r'Azure\s+Sentinel',
            r'Azure\s+Security\s+Center',
            r'Azure\s+(?:AD|B2C|B2B)',
            
            # Integration
            r'Azure\s+Logic\s+Apps',
            r'Azure\s+Service\s+Bus',
            r'Azure\s+Event\s+(?:Grid|Hubs)',
            r'Azure\s+API\s+Management',
            
            # Analytics
            r'Azure\s+Synapse',
            r'Azure\s+Data\s+(?:Lake|Factory|Explorer)',
            r'\bPower\s*BI\b',
            r'Azure\s+(?:Stream\s+Analytics|HDInsight)',
            
            # Containers
            r'\bAKS\b(?=\s|,|\.)',
            r'Azure\s+Kubernetes',
            r'Azure\s+Container\s+Registry',
            
            # Microsoft 365 (cloud-dependent)
            r'\bM365\b',
            r'Microsoft\s+365',
            r'Office\s+365',
            r'\bO365\b',
            r'\bSharePoint\s+Online\b',
            r'\bOneDrive\b(?=.*business)',
            r'\bDynamics\s+365\b',
            r'\bIntune\b',
            r'\bTeams\b(?=.*Microsoft)',
            r'Exchange\s+Online',
        ],
        'contractor_names': ['microsoft', 'azure']
    },
    
    'Google Cloud': {
        'brand_names': [
            r'\bGCP\b',
            r'Google\s+Cloud',
            r'Google\s+Cloud\s+Platform',
        ],
        'specific_services': [
            # IaaS - Compute
            r'Compute\s+Engine',
            r'\bGCE\b(?=\s|,|\.)',
            r'Cloud\s+Functions(?!\s+\(AWS)',
            r'App\s+Engine',
            r'Cloud\s+Run',
            r'\bTPU\b',
            r'Tensor\s+Processing\s+Unit',
            
            # IaaS - Storage
            r'Cloud\s+Storage(?!\s+\(AWS)',
            r'Persistent\s+Disk',
            r'Filestore',
            
            # Database
            r'\bBigQuery\b',
            r'Cloud\s+SQL',
            r'Cloud\s+Spanner',
            r'\bFirestore\b',
            r'Cloud\s+Bigtable',
            r'Firebase(?:\s+Realtime\s+Database)?',
            
            # PaaS - ML/AI
            r'(?:Google\s+)?Vertex\s+AI\b',
            r'(?:Google\s+)?AI\s+Platform(?:\s+Notebooks)?',
            r'Cloud\s+AI\s+Platform\b',
            r'AutoML\s+(?:Vision|Natural\s+Language|Translation|Tables|Video)',
            r'MED-PaLM',
            r'\bAutoML\b',
            r'Cloud\s+(?:Vision|Natural\s+Language|Translation|Speech)',
            r'AI\s+Platform',
            r'What-If\s+Tool\b',
            r'Data\s+Labeling\s+Service\b',
            r'\bVizier\b(?=.*(?:Google|GCP))',
            r'Neural\s+Architecture\s+Search\b',
            r'Deep\s+Learning\s+(?:Containers|VM\s+Image)',
            
            # Networking
            r'Cloud\s+VPC',
            r'Cloud\s+CDN',
            r'Cloud\s+Load\s+Balancing',
            r'Cloud\s+(?:VPN|Interconnect|NAT)',
            
            # Management
            r'Cloud\s+Deployment\s+Manager',
            r'\bStackdriver\b',
            r'Cloud\s+Monitoring',
            r'Cloud\s+Logging',
            r'Cloud\s+(?:Build|Source\s+Repositories)',
            
            # Security
            r'Cloud\s+IAM',
            r'Cloud\s+KMS',
            r'Cloud\s+(?:Identity|Armor|Security\s+Scanner)',
            
            # Analytics
            r'\bBigQuery\b',
            r'\bDataflow\b(?=\s|,|\.)',
            r'\bDataproc\b',
            r'Pub/Sub',
            r'Cloud\s+(?:Dataflow|Dataproc|Composer|Data\s+Fusion)',
            
            # Containers
            r'\bGKE\b(?=\s|,|\.)',
            r'Google\s+Kubernetes\s+Engine',
            r'Artifact\s+Registry',
            r'Container\s+Registry',
            
            # Google Workspace (cloud-dependent)
            r'G\s+Suite\b',
            r'Google\s+Workspace',
            r'\bGmail\b(?=.*(?:business|enterprise|government))',
            r'Google\s+Drive\b(?=.*(?:business|enterprise))',
            r'Google\s+Meet\b',
            r'Google\s+Calendar\b(?=.*enterprise)',
            r'Google\s+Docs\b(?=.*business)',
        ],
        'contractor_names': ['google', 'alphabet']
    },
    
    'Oracle Cloud': {
        'brand_names': [
            r'\bOracle\s+Cloud\b',
            r'\bOCI\b(?=.*(oracle|cloud|compute|storage|network|database))',
        ],
        'specific_services': [
            r'Oracle\s+Cloud\s+Infrastructure',
            r'Oracle\s+Autonomous\s+(?:Database|Data\s+Warehouse)',
            r'Oracle\s+Cloud\s+(?:Compute|Storage|Network|Database)',
            r'OCI\s+(?:Compute|Block\s+Volume|Object\s+Storage)',
            r'Oracle\s+(?:Cloud|OCI)\s+(?:ERP|HCM|SCM)',
        ],
        'contractor_names': ['oracle']
    },
    
    'IBM Cloud': {
        'brand_names': [
            r'\bIBM\s+Cloud\b',
            r'IBM\s+Bluemix',
        ],
        'specific_services': [
            r'Watson\s+(?:AI|Studio|Discovery|Assistant|Machine\s+Learning)',
            r'IBM\s+Cloud\s+(?:Foundry|Kubernetes|Object\s+Storage)',
            r'IBM\s+Cloud\s+(?:Functions|Databases|Pak)',
            r'Red\s+Hat\s+OpenShift\s+on\s+IBM',
        ],
        'contractor_names': ['ibm', 'international business machines']
    },
    
    'Salesforce': {
        'brand_names': [
            r'\bSalesforce\b',
            r'Force\.com',
            r'\bSFDC\b',
        ],
        'specific_services': [
            r'Sales\s+Cloud',
            r'Service\s+Cloud',
            r'Marketing\s+Cloud',
            r'Commerce\s+Cloud',
            r'Community\s+Cloud',
            r'Analytics\s+Cloud',
            r'\bHeroku\b',
            r'Tableau\s+(?:Online|CRM|Cloud)',
            r'Einstein\s+(?:AI|Analytics|Discovery)',
            r'Lightning\s+(?:Platform|Experience|Components)',
            r'Apex\s+(?:code|class|trigger)',
            r'Visualforce',
            r'Force\.com\s+(?:Platform|IDE)',
            r'Salesforce\s+(?:CRM|Platform|Integration)',
        ],
        'contractor_names': ['salesforce']
    },
    
    'ServiceNow': {
        'brand_names': [
            r'\bServiceNow\b',
            r'Service\s+Now',
        ],
        'specific_services': [
            r'ServiceNow\s+(?:ITSM|ITOM|ITBM|CSM|FSM)',
            r'ServiceNow\s+Platform',
            r'Now\s+Platform',
            r'ServiceNow\s+(?:Instance|Portal|Integration)',
        ],
        'contractor_names': ['servicenow']
    },
    
    'Workday': {
        'brand_names': [
            r'\bWorkday\b(?!\s+(work\s+day|working\s+day))',
        ],
        'specific_services': [
            r'Workday\s+(?:HCM|Financial|Planning|Recruiting)',
            r'Workday\s+(?:Human\s+Capital|Enterprise)',
        ],
        'contractor_names': ['workday']
    },
    
    'Multi-cloud': {
        'keywords': [
            r'\bmulti[\s-]?cloud\b',
            r'\bhybrid\s+cloud\b(?=.*(?:aws|azure|google))',
            r'\bcross[\s-]?cloud\b',
        ],
        'contractor_names': []
    },
}

# Generic cloud keywords - these indicate cloud without specific platform
GENERIC_CLOUD_KEYWORDS = [
    # Cloud terms
    r'\bcloud[- ]computing\b',
    r'\bcloud[- ]based\b',
    r'\bcloud[- ]native\b',
    r'\bcloud[- ]service',
    r'\bcloud[- ]hosting\b',
    r'\bcloud[- ]infrastructure\b',
    r'\bcloud[- ]migration\b',
    r'\bcloud[- ]deployment\b',
    r'\bcloud[- ]environment\b',
    r'\bcloud[- ]platform\b',
    r'\bcloud[- ]solution\b',
    r'\bcloud[- ]managed\b',
    r'\bcloud[- ]enabled\b',
    r'\bpublic\s+cloud\b',
    r'\bprivate\s+cloud\b',
    r'\bhybrid\s+cloud\b',
    r'\bmulti[- ]cloud\b',
    r'\bcloud\b(?=\s+(storage|compute|database|network|security))',
    
    # Service models
    r'\bIaaS\b', r'\bPaaS\b', r'\bSaaS\b',
    r'\bsoftware[- ]as[- ]a[- ]service\b',
    r'\bplatform[- ]as[- ]a[- ]service\b',
    r'\binfrastructure[- ]as[- ]a[- ]service\b',
    
    # Cloud architecture
    r'\bserverless\b',
    r'\bcloud[- ]native\b',
    r'\bmicroservices\b',
    r'\bcontainerized\b',
    r'\bcontainer[- ]based\b',
    
    # Platform-agnostic tools
    r'\bKubernetes\b(?!\s+Engine|\s+Service)',
    r'\bDocker\b(?!\s+Hub)',
    r'\bTerraform\b',
    r'\bAnsible\b',
]

# LLM configuration
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_SAVE_EVERY = 1000
LLM_MAX_RETRIES = 5
LLM_RETRY_BASE_DELAY = 2.0
LLM_MAX_TOKENS = 300
LLM_INPUT_TOKENS_EST = 500
LLM_OUTPUT_TOKENS_EST = 80
# Haiku 4.5 pricing (per 1M tokens)
LLM_INPUT_PRICE_PER_M = 0.80
LLM_OUTPUT_PRICE_PER_M = 4.00

VALID_CLASSIFICATIONS = ['non-cloud', 'cloud-dependent', 'cloud-infrastructure']
VALID_PLATFORMS = ['AWS', 'Azure', 'Google Cloud', 'Oracle Cloud', 'IBM Cloud',
                   'Salesforce', 'ServiceNow', 'Workday', 'Multi-cloud', 'Unspecified', 'N/A']

# Concrete platform providers (used for multi-platform splitting)
CONCRETE_PLATFORMS = [
    'AWS', 'Azure', 'Google Cloud', 'Oracle Cloud', 'IBM Cloud',
    'Salesforce', 'ServiceNow', 'Workday'
]

# Explicit hyperscaler mentions (description or contractor name)
EXPLICIT_HYPERSCALER_PATTERNS = {
    'AWS': [
        r'\bAWS\b',
        r'Amazon\s+Web\s+Services',
        r'Amazon\s+Cloud',
    ],
    'Azure': [
        r'\bAzure\b',
        r'Microsoft\s+Azure',
        r'\bMS\s+Azure\b',
    ],
    'Google Cloud': [
        r'\bGoogle\s+Cloud\b',
        r'\bGCP\b',
        r'Google\s+Cloud\s+Platform',
    ],
}


def safe_read_csv(path: str,
                  usecols: Optional[List[str]] = None,
                  chunksize: int = 200_000) -> pd.DataFrame:
    """
    Read a CSV with fallbacks to avoid parser timeouts.
    """
    # Fast path
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False, memory_map=True)
    except Exception:
        pass

    # Try pyarrow engine if available
    try:
        return pd.read_csv(path, usecols=usecols, engine="pyarrow")
    except Exception:
        pass

    # Fallback: chunked read
    chunks = []
    try:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunks.append(chunk)
        if not chunks:
            return pd.DataFrame(columns=usecols or [])
        return pd.concat(chunks, ignore_index=True)
    except Exception as e:
        print(f"  ERROR: failed to read CSV {path}: {type(e).__name__} - {e}")
        return pd.DataFrame(columns=usecols or [])


# =============================================================================
# STEP 1A: REGEX CLASSIFICATION
# =============================================================================

def compile_patterns() -> Dict:
    """Pre-compile all regex patterns for performance."""
    compiled = {
        'platforms': {},
        'generic_cloud': [],
        'explicit_hyperscaler': {}
    }
    for platform, data in PLATFORM_PATTERNS.items():
        # Support both old format (keywords) and new format (brand_names + specific_services)
        if 'keywords' in data:
            all_patterns = data['keywords']
        else:
            all_patterns = data.get('brand_names', []) + data.get('specific_services', [])
        compiled['platforms'][platform] = {
            'keywords': [re.compile(p, re.IGNORECASE) for p in all_patterns],
            'contractor_names': data.get('contractor_names', [])
        }
    compiled['generic_cloud'] = [re.compile(p, re.IGNORECASE) for p in GENERIC_CLOUD_KEYWORDS]
    compiled['explicit_hyperscaler'] = {
        platform: [re.compile(p, re.IGNORECASE) for p in pats]
        for platform, pats in EXPLICIT_HYPERSCALER_PATTERNS.items()
    }
    return compiled


def classify_cloud_regex(merged_df: pd.DataFrame, patterns: Dict) -> pd.DataFrame:
    """
    Step 1A: RegEx-based cloud classification.

    Scans each record's description against cloud and platform patterns.

    Adds columns:
      regex_is_cloud: bool
      regex_platforms: str or None (detected platform or Multi-cloud)
      regex_platforms_list: list of concrete platform mentions
      regex_confidence: High / Medium / Low
      explicit_hyperscaler_mentions: list of explicit AWS/Azure/GCP mentions
      explicit_cloud_mention: bool (explicit hyperscaler mention)
    """
    print("\n" + "-" * 60)
    print("Step 1A: RegEx Classification (all merged records)")
    print("-" * 60)

    df = merged_df.copy()
    n = len(df)

    def _explicit_hyperscaler_mentions(text_a, text_b=None) -> List[str]:
        mentions = set()
        for platform, pats in patterns['explicit_hyperscaler'].items():
            for pat in pats:
                if text_a and pat.search(text_a):
                    mentions.add(platform)
                    break
                if text_b and pat.search(text_b):
                    mentions.add(platform)
                    break
        return list(mentions)

    def _is_cloud(description, contractor_name):
        desc = str(description).lower() if pd.notna(description) else ""
        name = str(contractor_name).lower() if pd.notna(contractor_name) else ""

        # Explicit hyperscaler mention in description or contractor name
        if _explicit_hyperscaler_mentions(desc, name):
            return True

        # Generic cloud / service patterns in description only
        for pat in patterns['generic_cloud']:
            if pat.search(desc):
                return True
        for platform_data in patterns['platforms'].values():
            for pat in platform_data['keywords']:
                if pat.search(desc):
                    return True
        return False

    def _detect_platforms(description, contractor_name):
        desc = str(description).lower() if pd.notna(description) else ""
        name = str(contractor_name).lower() if pd.notna(contractor_name) else ""

        platform_matches = {}
        for platform, data in patterns['platforms'].items():
            matches = sum(1 for pat in data['keywords'] if pat.search(desc))
            if matches > 0:
                platform_matches[platform] = matches

        explicit_mentions = _explicit_hyperscaler_mentions(desc, name)
        for plat in explicit_mentions:
            platform_matches[plat] = platform_matches.get(plat, 0) + 2

        if not platform_matches:
            return [], None, 'Low'

        concrete = [p for p in platform_matches.keys() if p in CONCRETE_PLATFORMS]
        if len(concrete) >= 2:
            return concrete, 'Multi-cloud', 'Medium'

        # Single concrete platform or only Multi-cloud
        if len(concrete) == 1:
            platform = concrete[0]
            match_count = platform_matches.get(platform, 1)
            if match_count >= 3:
                return [platform], platform, 'High'
            elif match_count >= 2:
                return [platform], platform, 'Medium'
            return [platform], platform, 'Low'

        # Only Multi-cloud keyword without specific providers
        if 'Multi-cloud' in platform_matches:
            return [], 'Multi-cloud', 'Low'

        return [], None, 'Low'

    print(f"  Classifying {n:,} records...")

    df['regex_is_cloud'] = df.apply(
        lambda r: _is_cloud(r.get('description'), r.get('contractor_name')), axis=1
    )
    platform_results = df.apply(
        lambda r: _detect_platforms(r.get('description'), r.get('contractor_name')), axis=1
    )
    df['regex_platforms_list'] = platform_results.apply(lambda x: x[0])
    df['regex_platforms'] = platform_results.apply(lambda x: x[1])
    df['regex_confidence'] = platform_results.apply(lambda x: x[2])
    df['explicit_hyperscaler_mentions'] = df.apply(
        lambda r: _explicit_hyperscaler_mentions(
            str(r.get('description')).lower() if pd.notna(r.get('description')) else "",
            str(r.get('contractor_name')).lower() if pd.notna(r.get('contractor_name')) else ""
        ),
        axis=1
    )
    df['explicit_cloud_mention'] = df['explicit_hyperscaler_mentions'].apply(lambda x: bool(x))

    cloud_count = df['regex_is_cloud'].sum()
    cloud_dollars = df.loc[df['regex_is_cloud'], 'dollars'].sum()
    total_dollars = df['dollars'].sum()

    print(f"  Cloud records: {cloud_count:,} / {n:,} ({cloud_count/n*100:.1f}%)")
    print(f"  Cloud dollars: ${cloud_dollars:,.2f} ({cloud_dollars/total_dollars*100:.1f}%)")

    for rtype in ['prime_no_subs', 'prime_retained', 'subcontract']:
        subset = df[df['record_type'] == rtype]
        if len(subset) > 0:
            cloud_sub = subset['regex_is_cloud'].sum()
            print(f"    {rtype:20s}: {cloud_sub:,} cloud / {len(subset):,} total "
                  f"({cloud_sub/len(subset)*100:.1f}%)")

    return df


# =============================================================================
# STEP 1B: LLM CLASSIFICATION
# =============================================================================

def estimate_llm_cost(n_records: int) -> Dict[str, Any]:
    """Estimate cost and time for LLM classification."""
    input_tokens = n_records * LLM_INPUT_TOKENS_EST
    output_tokens = n_records * LLM_OUTPUT_TOKENS_EST
    input_cost = input_tokens / 1_000_000 * LLM_INPUT_PRICE_PER_M
    output_cost = output_tokens / 1_000_000 * LLM_OUTPUT_PRICE_PER_M
    total_cost = input_cost + output_cost
    # Estimate ~60-120 records/min depending on rate limits
    est_minutes = n_records / 90

    return {
        'records': n_records,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'estimated_cost_usd': total_cost,
        'estimated_minutes': est_minutes,
        'model': LLM_MODEL,
    }


def _build_llm_prompt(description: str, contractor_name: str) -> str:
    """Build the C_cot (chain-of-thought) classification prompt."""
    desc = str(description) if pd.notna(description) else "No description provided"
    name = str(contractor_name) if pd.notna(contractor_name) else "Unknown"

    return f"""Classify this federal IT contract for cloud dependency.

Description: {desc}
Contractor: {name}

First, identify key signals:
1. Does it mention cloud platforms (AWS/Azure/GCP) or services (EC2/Lambda/etc)?
2. Does it mention SaaS products (Salesforce/ServiceNow/Microsoft 365/Workday)?
3. Does it describe cloud-native work (web apps/APIs/microservices/cloud migration)?
4. Or is it traditional IT (on-prem/hardware/consulting/desktop software)?

Then classify:
- cloud-infrastructure: Direct cloud IaaS/PaaS purchases
- cloud-dependent: SaaS products or services requiring cloud infrastructure
- non-cloud: Traditional on-premise IT, no cloud dependency

Return JSON:
{{"classification":"non-cloud|cloud-dependent|cloud-infrastructure","platform":"AWS|Azure|Google Cloud|Oracle Cloud|IBM Cloud|Salesforce|ServiceNow|Workday|Multi-cloud|Unspecified|N/A","confidence":"high|medium|low","evidence":"brief reason (max 10 words)"}}

Critical distinctions:
- "Hosting" alone → non-cloud (assume on-prem unless cloud mentioned)
- Known SaaS products → cloud-dependent (even if no "cloud" keyword)
- Vendor name alone insufficient (Microsoft ≠ cloud unless specifically Azure/365)
- Explicit AWS/Azure/GCP mentions imply cloud

Platform: Name specific cloud provider/vendor if identifiable, else "Unspecified" (cloud but unclear which) or "N/A" (non-cloud)."""


def _parse_llm_response(text: str) -> Dict[str, str]:
    """Parse LLM JSON response with fallback for malformed output."""
    text = text.strip()

    # Remove markdown fences if present
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        if text.startswith('json'):
            text = text[4:].strip()

    try:
        result = json.loads(text)
        # Validate fields
        classification = result.get('classification', 'error')
        if classification not in VALID_CLASSIFICATIONS:
            classification = 'error'
        platform = result.get('platform', 'error')
        if platform not in VALID_PLATFORMS:
            # Try to normalize common variations
            plat_upper = str(platform).upper()
            if 'AWS' in plat_upper or 'AMAZON' in plat_upper:
                platform = 'AWS'
            elif 'AZURE' in plat_upper or 'MICROSOFT' in plat_upper:
                platform = 'Azure'
            elif 'GOOGLE' in plat_upper or 'GCP' in plat_upper:
                platform = 'Google Cloud'
            elif 'ORACLE' in plat_upper:
                platform = 'Oracle Cloud'
            elif 'IBM' in plat_upper:
                platform = 'IBM Cloud'
            elif 'SERVICENOW' in plat_upper or 'SERVICE NOW' in plat_upper:
                platform = 'ServiceNow'
            elif 'WORKDAY' in plat_upper:
                platform = 'Workday'
            elif 'SALESFORCE' in plat_upper:
                platform = 'Salesforce'
            elif platform is None or str(platform).lower() in ('null', 'none', ''):
                platform = 'N/A' if classification == 'non-cloud' else 'Unspecified'
            else:
                platform = 'Unspecified'
        confidence = result.get('confidence', 'low')
        if confidence not in ('high', 'medium', 'low'):
            confidence = 'low'
        evidence = str(result.get('evidence', ''))[:200]

        return {
            'classification': classification,
            'platform': platform,
            'confidence': confidence,
            'evidence': evidence,
        }
    except (json.JSONDecodeError, AttributeError):
        # Regex fallback: try to extract fields
        cls_match = re.search(r'"classification"\s*:\s*"(non-cloud|cloud-dependent|cloud-infrastructure)"', text)
        plat_match = re.search(r'"platform"\s*:\s*"([^"]+)"', text)
        return {
            'classification': cls_match.group(1) if cls_match else 'error',
            'platform': plat_match.group(1) if plat_match else 'error',
            'confidence': 'low',
            'evidence': 'parse_error',
        }


def _load_llm_checkpoint(cache_dir: str) -> Tuple[List[Dict], int]:
    """Load existing LLM checkpoint if available. Returns (results_list, last_completed_index)."""
    progress_path = os.path.join(cache_dir, 'llm_progress.json')
    if not os.path.exists(progress_path):
        return [], 0

    with open(progress_path) as f:
        progress = json.load(f)

    if progress.get('status') == 'complete':
        # Full results available
        final_path = os.path.join(cache_dir, 'llm_results_final.csv')
        if os.path.exists(final_path):
            df = pd.read_csv(final_path)
            return df.to_dict('records'), len(df)

    # Load batch files up to last completed index
    last_completed = progress.get('last_completed', 0)
    results = []
    batch_num = 1
    while True:
        batch_path = os.path.join(cache_dir, f'llm_results_batch_{batch_num:03d}.csv')
        if not os.path.exists(batch_path):
            break
        batch_df = pd.read_csv(batch_path)
        results.extend(batch_df.to_dict('records'))
        batch_num += 1

    # Trim to last_completed
    results = results[:last_completed]
    return results, last_completed


def _save_llm_checkpoint(results: List[Dict], cache_dir: str, batch_label: str,
                         total_records: int):
    """Save intermediate LLM results."""
    os.makedirs(cache_dir, exist_ok=True)

    # Save batch file
    batch_path = os.path.join(cache_dir, f'llm_results_batch_{batch_label}.csv')
    pd.DataFrame(results).to_csv(batch_path, index=False)

    # Update progress file
    progress = {
        'last_completed': len(results),
        'total': total_records,
        'status': 'in_progress',
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(cache_dir, 'llm_progress.json'), 'w') as f:
        json.dump(progress, f, indent=2)


def classify_cloud_llm(df: pd.DataFrame, api_key: str, cache_dir: str,
                        skip_confirmation: bool = False) -> pd.DataFrame:
    """
    Step 1B: LLM-based cloud classification using Anthropic API.

    Classifies every record with:
    - Cloud classification (3-way)
    - Platform extraction
    - Confidence level
    - Evidence/reasoning

    Features:
    - Cost/time estimate printed before starting
    - Confirmation prompt (unless skip_confirmation=True)
    - tqdm progress bar
    - Intermediate saves every LLM_SAVE_EVERY records
    - Crash recovery from checkpoints

    Adds columns: llm_classification, llm_platform, llm_confidence, llm_evidence
    """
    # Lazy imports — these packages are only needed for the LLM step
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for LLM classification.\n"
            "Install it with: pip3 install anthropic"
        )
    try:
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            "The 'tqdm' package is required for progress bars.\n"
            "Install it with: pip3 install tqdm"
        )

    print("\n" + "-" * 60)
    print("Step 1B: LLM Classification (Anthropic API)")
    print("-" * 60)

    n = len(df)
    os.makedirs(cache_dir, exist_ok=True)

    # Check for existing checkpoint
    existing_results, start_idx = _load_llm_checkpoint(cache_dir)
    if start_idx >= n:
        print(f"  LLM classification already complete ({start_idx:,} records).")
        print(f"  Loading cached results.")
        llm_df = pd.DataFrame(existing_results)
        df = df.copy()
        for col in ['llm_classification', 'llm_platform', 'llm_confidence', 'llm_evidence']:
            df[col] = llm_df[col].values[:n]
        return df

    if start_idx > 0:
        print(f"  Resuming from checkpoint: {start_idx:,}/{n:,} records completed")

    # Cost estimate for remaining records
    remaining = n - start_idx
    estimate = estimate_llm_cost(remaining)

    print(f"\n  Model: {LLM_MODEL}")
    print(f"  Records remaining: {remaining:,}")
    print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.2f}")
    print(f"  Estimated time: {estimate['estimated_minutes']:.0f} minutes "
          f"({estimate['estimated_minutes']/60:.1f} hours)")

    if not skip_confirmation:
        confirm = input("\n  Proceed with LLM classification? (yes/no): ")
        if confirm.lower() != 'yes':
            print("  LLM classification cancelled.")
            df = df.copy()
            df['llm_classification'] = None
            df['llm_platform'] = None
            df['llm_confidence'] = None
            df['llm_evidence'] = None
            return df

    # Initialize API client
    client = anthropic.Anthropic(api_key=api_key)
    results = list(existing_results)  # start with checkpoint data
    errors = 0
    start_time = time.time()

    descriptions = df['description'].tolist()
    contractor_names = df['contractor_name'].tolist()

    batch_num = (start_idx // LLM_SAVE_EVERY) + 1

    for i in tqdm(range(start_idx, n), desc="LLM Classification", initial=start_idx, total=n):
        prompt = _build_llm_prompt(descriptions[i], contractor_names[i])

        parsed = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = client.messages.create(
                    model=LLM_MODEL,
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
                parsed = _parse_llm_response(response.content[0].text)
                break
            except Exception as e:
                error_name = type(e).__name__
                if 'RateLimit' in error_name or 'Overloaded' in error_name or '529' in str(e):
                    delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                else:
                    errors += 1
                    parsed = {
                        'classification': 'error',
                        'platform': 'error',
                        'confidence': 'low',
                        'evidence': f'{error_name}: {str(e)[:80]}',
                    }
                    break

        if parsed is None:
            errors += 1
            parsed = {
                'classification': 'error',
                'platform': 'error',
                'confidence': 'low',
                'evidence': 'max_retries_exceeded',
            }

        results.append(parsed)

        # Intermediate save
        if (i + 1) % LLM_SAVE_EVERY == 0:
            _save_llm_checkpoint(results, cache_dir, f'{batch_num:03d}', n)
            batch_num += 1

    elapsed = time.time() - start_time

    # Final save
    final_path = os.path.join(cache_dir, 'llm_results_final.csv')
    pd.DataFrame(results).to_csv(final_path, index=False)
    progress = {
        'last_completed': len(results),
        'total': n,
        'status': 'complete',
        'elapsed_seconds': elapsed,
        'errors': errors,
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(cache_dir, 'llm_progress.json'), 'w') as f:
        json.dump(progress, f, indent=2)

    print(f"\n  LLM classification complete.")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Errors: {errors:,} ({errors/n*100:.2f}%)")

    # Merge results into dataframe
    llm_df = pd.DataFrame(results)
    df = df.copy()
    df['llm_classification'] = llm_df['classification'].values
    df['llm_platform'] = llm_df['platform'].values
    df['llm_confidence'] = llm_df['confidence'].values
    df['llm_evidence'] = llm_df['evidence'].values

    # Summary
    print(f"\n  LLM classification distribution:")
    for cls, count in df['llm_classification'].value_counts().items():
        print(f"    {cls:25s}: {count:>8,} ({count/n*100:.1f}%)")

    return df


def load_llm_results(cache_dir: str) -> Optional[pd.DataFrame]:
    """
    Load completed LLM results from cache.

    Returns DataFrame with llm_classification, llm_platform, llm_confidence,
    llm_evidence columns, or None if no completed results exist.
    """
    if cache_dir is None:
        return None
    progress_path = os.path.join(cache_dir, 'llm_progress.json')
    if not os.path.exists(progress_path):
        return None

    with open(progress_path) as f:
        progress = json.load(f)

    if progress.get('status') != 'complete':
        return None

    final_path = os.path.join(cache_dir, 'llm_results_final.csv')
    if not os.path.exists(final_path):
        return None

    try:
        return safe_read_csv(final_path, usecols=['classification', 'platform', 'confidence', 'evidence'])
    except Exception as e:
        print(f"  WARNING: Failed to load cached LLM results: {type(e).__name__} - {e}")
        return None


# =============================================================================
# STEP 1B-BATCH: BATCH API CLASSIFICATION
# =============================================================================

def _build_batch_requests(df: pd.DataFrame, prompt_fn=None) -> List[Dict]:
    """
    Build a list of Message Batch API request objects.

    Each request is a dict with:
      custom_id: row index as string
      params: {model, max_tokens, temperature, messages}

    Args:
        df: DataFrame with 'description' and 'contractor_name' columns
        prompt_fn: Optional custom prompt function(desc, contractor) -> str.
                   Defaults to _build_llm_prompt.
    """
    if prompt_fn is None:
        prompt_fn = _build_llm_prompt

    requests = []
    for i in range(len(df)):
        prompt = prompt_fn(df.iloc[i]['description'], df.iloc[i]['contractor_name'])
        requests.append({
            "custom_id": str(i),
            "params": {
                "model": LLM_MODEL,
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        })
    return requests


def submit_llm_batch(df: pd.DataFrame, api_key: str, cache_dir: str,
                     prompt_fn=None, skip_confirmation: bool = False) -> Optional[str]:
    """
    Submit a Message Batch for async LLM classification.

    Uses the Anthropic Message Batches API (50% discount, ≤24h processing).

    Args:
        df: DataFrame to classify
        api_key: Anthropic API key
        cache_dir: Directory to save batch metadata
        prompt_fn: Optional custom prompt function
        skip_confirmation: Skip cost confirmation

    Returns:
        batch_id string, or None if cancelled/error
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("The 'anthropic' package is required. Install: pip3 install anthropic")

    n = len(df)
    os.makedirs(cache_dir, exist_ok=True)

    # Cost estimate (50% discount for batch API)
    estimate = estimate_llm_cost(n)
    batch_cost = estimate['estimated_cost_usd'] * 0.5

    print(f"\n  Batch API Classification")
    print(f"  Model: {LLM_MODEL}")
    print(f"  Records: {n:,}")
    print(f"  Estimated cost: ${batch_cost:.2f} (50% batch discount)")
    print(f"  Processing time: ≤24 hours")

    if not skip_confirmation:
        confirm = input(f"\n  Submit batch for ${batch_cost:.2f}? (yes/no): ")
        if confirm.lower() != 'yes':
            print("  Batch submission cancelled.")
            return None

    # Build requests
    print(f"  Building {n:,} requests...")
    requests = _build_batch_requests(df, prompt_fn)

    # Submit batch
    print(f"  Submitting batch...")
    client = anthropic.Anthropic(api_key=api_key)
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    # Save batch metadata
    metadata = {
        'batch_id': batch_id,
        'n_records': n,
        'model': LLM_MODEL,
        'estimated_cost': batch_cost,
        'submitted_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'submitted',
    }
    metadata_path = os.path.join(cache_dir, f'batch_{batch_id}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Batch submitted: {batch_id}")
    print(f"  Metadata saved: {metadata_path}")
    print(f"  Use check_batch_status('{batch_id}', ...) to poll for results.")

    return batch_id


def check_batch_status(batch_id: str, api_key: str, cache_dir: str = None) -> Dict:
    """
    Check status of a submitted batch.

    Returns dict with status, counts, and result if complete.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("The 'anthropic' package is required.")

    client = anthropic.Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)

    status = {
        'batch_id': batch_id,
        'processing_status': batch.processing_status,
        'request_counts': {
            'processing': batch.request_counts.processing,
            'succeeded': batch.request_counts.succeeded,
            'errored': batch.request_counts.errored,
            'canceled': batch.request_counts.canceled,
            'expired': batch.request_counts.expired,
        },
    }

    total = sum(status['request_counts'].values())
    succeeded = status['request_counts']['succeeded']
    print(f"  Batch {batch_id}: {batch.processing_status}")
    print(f"  Progress: {succeeded:,}/{total:,} succeeded, "
          f"{status['request_counts']['errored']} errored, "
          f"{status['request_counts']['processing']} processing")

    # Update metadata file if cache_dir provided
    if cache_dir:
        metadata_path = os.path.join(cache_dir, f'batch_{batch_id}.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            metadata['status'] = batch.processing_status
            metadata['request_counts'] = status['request_counts']
            metadata['checked_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    return status


def download_batch_results(batch_id: str, api_key: str, cache_dir: str,
                           n_records: int) -> Optional[pd.DataFrame]:
    """
    Download results from a completed batch and save to cache.

    Args:
        batch_id: Batch ID from submit_llm_batch
        api_key: Anthropic API key
        cache_dir: Directory to save results
        n_records: Expected number of records (for validation)

    Returns:
        DataFrame with classification/platform/confidence/evidence columns,
        or None if batch not yet complete.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("The 'anthropic' package is required.")

    client = anthropic.Anthropic(api_key=api_key)

    # Check status first
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != 'ended':
        print(f"  Batch {batch_id} not yet complete: {batch.processing_status}")
        print(f"  Succeeded: {batch.request_counts.succeeded}, "
              f"Processing: {batch.request_counts.processing}")
        return None

    print(f"  Downloading results for batch {batch_id}...")
    os.makedirs(cache_dir, exist_ok=True)

    # Download and parse results
    results = [None] * n_records
    errors = 0

    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id)
        if result.result.type == 'succeeded':
            response_text = result.result.message.content[0].text
            parsed = _parse_llm_response(response_text)
        else:
            errors += 1
            error_type = result.result.type
            parsed = {
                'classification': 'error',
                'platform': 'error',
                'confidence': 'low',
                'evidence': f'batch_{error_type}',
            }
        results[idx] = parsed

    # Fill any missing results
    for i in range(n_records):
        if results[i] is None:
            errors += 1
            results[i] = {
                'classification': 'error',
                'platform': 'error',
                'confidence': 'low',
                'evidence': 'missing_from_batch',
            }

    # Save as final results (compatible with sequential cache format)
    results_df = pd.DataFrame(results)
    final_path = os.path.join(cache_dir, 'llm_results_final.csv')
    results_df.to_csv(final_path, index=False)

    progress = {
        'last_completed': n_records,
        'total': n_records,
        'status': 'complete',
        'source': f'batch_{batch_id}',
        'errors': errors,
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(cache_dir, 'llm_progress.json'), 'w') as f:
        json.dump(progress, f, indent=2)

    print(f"  Downloaded {n_records:,} results ({errors} errors)")
    print(f"  Saved to: {final_path}")

    # Summary
    print(f"\n  Classification distribution:")
    for cls, count in results_df['classification'].value_counts().items():
        print(f"    {cls:25s}: {count:>8,} ({count/n_records*100:.1f}%)")

    return results_df


# =============================================================================
# STEP 1C: SYNTHESIS
# =============================================================================

def synthesize_classification(df: pd.DataFrame,
                              conflict_mode: str = "explicit_override",
                              noncloud_override_confidence: str = "high") -> pd.DataFrame:
    """
    Step 1C: Combine RegEx + LLM results into final classification.

    When LLM columns are missing, falls back to RegEx-only classification.

    conflict_mode:
      - "explicit_override": explicit hyperscaler mentions override LLM non-cloud
        unless LLM confidence >= noncloud_override_confidence
      - "llm_strict": always trust LLM when conflict

    Adds columns:
      is_cloud: bool (final determination)
      cloud_classification: non-cloud / cloud-dependent / cloud-infrastructure
      platform_stage1: best platform from Stage 1
      classification_confidence: high / medium / low / conflict
      classification_source: regex+llm_agree / llm_only / regex_only / conflict / etc.
    """
    print("\n" + "-" * 60)
    print("Step 1C: Synthesis (RegEx + LLM)")
    print("-" * 60)

    df = df.copy()
    if conflict_mode not in ("explicit_override", "llm_strict"):
        raise ValueError("conflict_mode must be 'explicit_override' or 'llm_strict'")

    conf_rank = {'low': 1, 'medium': 2, 'high': 3}
    override_min = conf_rank.get(noncloud_override_confidence, 3)
    has_llm = 'llm_classification' in df.columns and df['llm_classification'].notna().any()

    if not has_llm:
        print("  LLM results not available — using RegEx-only classification.")
        # RegEx-only fallback
        df['is_cloud'] = df['regex_is_cloud']
        # RegEx doesn't distinguish cloud-dependent vs cloud-infrastructure
        df['cloud_classification'] = 'non-cloud'
        df.loc[df['is_cloud'], 'cloud_classification'] = 'cloud-infrastructure'
        df['platform_stage1'] = df['regex_platforms']
        df.loc[df['is_cloud'] & df['platform_stage1'].isna(), 'platform_stage1'] = 'Unspecified'
        df.loc[~df['is_cloud'], 'platform_stage1'] = 'N/A'
        df['classification_confidence'] = 'medium'
        df.loc[df['regex_confidence'] == 'High', 'classification_confidence'] = 'high'
        df['classification_source'] = 'regex_only'
        df['platform_mentions'] = df['regex_platforms_list'] if 'regex_platforms_list' in df.columns else [[]] * len(df)
    else:
        print("  Synthesizing RegEx + LLM results...")
        n = len(df)

        is_cloud = []
        cloud_cls = []
        plat_s1 = []
        conf = []
        source = []
        plat_mentions = []

        for i in range(n):
            regex_cloud = df.iloc[i]['regex_is_cloud']
            regex_plat = df.iloc[i]['regex_platforms']
            regex_list = df.iloc[i].get('regex_platforms_list', []) or []
            explicit_cloud = df.iloc[i].get('explicit_cloud_mention', False)
            llm_cls = df.iloc[i]['llm_classification']
            llm_plat = df.iloc[i]['llm_platform']
            llm_conf = df.iloc[i].get('llm_confidence', 'low')

            llm_is_cloud = llm_cls in ('cloud-dependent', 'cloud-infrastructure')
            llm_has_error = llm_cls == 'error'

            # Build platform mentions (concrete platforms only)
            mentions = list(regex_list) if isinstance(regex_list, list) else []
            if llm_plat in CONCRETE_PLATFORMS and llm_plat not in mentions:
                mentions.append(llm_plat)

            if llm_has_error:
                # LLM failed — use RegEx only
                is_cloud.append(regex_cloud)
                cloud_cls.append('cloud-infrastructure' if regex_cloud else 'non-cloud')
                plat_s1.append(regex_plat if regex_cloud and regex_plat else
                               ('Unspecified' if regex_cloud else 'N/A'))
                conf.append('low')
                source.append('regex_only_llm_error')
                plat_mentions.append(mentions)
                continue

            # Case 1: Both agree cloud
            if regex_cloud and llm_is_cloud:
                # Check platform agreement
                if regex_plat and llm_plat == regex_plat:
                    conf.append('high')
                    source.append('regex+llm_agree')
                elif llm_plat not in ('Unspecified', 'N/A', None, 'error'):
                    conf.append('medium')
                    source.append('llm_platform')
                else:
                    conf.append('medium')
                    source.append('both_cloud_no_platform')

                is_cloud.append(True)
                cloud_cls.append(llm_cls)
                best_plat = llm_plat if llm_plat not in ('N/A', None, 'error') else regex_plat
                plat_s1.append(best_plat if best_plat else 'Unspecified')
                plat_mentions.append(mentions)
                continue

            # Case 2: LLM found cloud, RegEx didn't
            if not regex_cloud and llm_is_cloud:
                is_cloud.append(True)
                cloud_cls.append(llm_cls)
                plat_s1.append(llm_plat if llm_plat not in ('N/A', None, 'error') else 'Unspecified')
                conf.append('medium')
                source.append('llm_only')
                plat_mentions.append(mentions)
                continue

            # Case 3: Both agree non-cloud
            if not regex_cloud and not llm_is_cloud:
                is_cloud.append(False)
                cloud_cls.append('non-cloud')
                plat_s1.append('N/A')
                conf.append('high')
                source.append('both_noncloud')
                plat_mentions.append(mentions)
                continue

            # Case 4: Conflict — RegEx found cloud, LLM says non-cloud
            if regex_cloud and not llm_is_cloud:
                if conflict_mode == "llm_strict":
                    is_cloud.append(False)
                    cloud_cls.append('non-cloud')
                    plat_s1.append('N/A')
                    conf.append('low')
                    source.append('conflict_resolved_noncloud')
                else:
                    # Explicit override unless LLM is high-confidence non-cloud
                    llm_conf_rank = conf_rank.get(llm_conf, 1)
                    if explicit_cloud and llm_conf_rank < override_min:
                        is_cloud.append(True)
                        cloud_cls.append('cloud-infrastructure')
                        plat_s1.append(regex_plat if regex_plat else 'Unspecified')
                        conf.append('medium')
                        source.append('conflict_resolved_cloud')
                    else:
                        is_cloud.append(False)
                        cloud_cls.append('non-cloud')
                        plat_s1.append('N/A')
                        conf.append('low')
                        source.append('conflict_resolved_noncloud')
                plat_mentions.append(mentions)
                continue

            # Default: trust LLM
            is_cloud.append(llm_is_cloud)
            cloud_cls.append(llm_cls if llm_is_cloud else 'non-cloud')
            plat_s1.append(llm_plat if llm_is_cloud else 'N/A')
            conf.append(llm_conf)
            source.append('default_llm')
            plat_mentions.append(mentions)

        df['is_cloud'] = is_cloud
        df['cloud_classification'] = cloud_cls
        df['platform_stage1'] = plat_s1
        df['classification_confidence'] = conf
        df['classification_source'] = source
        df['platform_mentions'] = plat_mentions

    # Report
    n = len(df)
    cloud_count = df['is_cloud'].sum()
    cloud_dollars = df.loc[df['is_cloud'], 'dollars'].sum()
    total_dollars = df['dollars'].sum()

    print(f"\n  Cloud records: {cloud_count:,} / {n:,} ({cloud_count/n*100:.1f}%)")
    print(f"  Cloud dollars: ${cloud_dollars/1e9:.2f}B ({cloud_dollars/total_dollars*100:.1f}%)")

    print(f"\n  Cloud classification:")
    for cls, count in df['cloud_classification'].value_counts().items():
        dollars = df.loc[df['cloud_classification'] == cls, 'dollars'].sum()
        print(f"    {cls:25s}: {count:>8,} records, ${dollars/1e9:.2f}B")

    if has_llm:
        print(f"\n  Classification source:")
        for src, count in df['classification_source'].value_counts().items():
            print(f"    {src:35s}: {count:>8,}")

        print(f"\n  Confidence distribution (cloud records):")
        cloud_df = df[df['is_cloud']]
        for c, count in cloud_df['classification_confidence'].value_counts().items():
            print(f"    {c:15s}: {count:>8,}")

    # Platform Stage 1 summary (cloud records only)
    cloud_plats = df.loc[df['is_cloud'], 'platform_stage1']
    print(f"\n  Platform Stage 1 (cloud records):")
    for plat, count in cloud_plats.value_counts().items():
        dollars = df.loc[(df['is_cloud']) & (df['platform_stage1'] == plat), 'dollars'].sum()
        print(f"    {str(plat):25s}: {count:>6,} records, ${dollars/1e6:>9.1f}M")

    return df


def validate_llm_conflicts(df: pd.DataFrame) -> None:
    """
    Lightweight validation: flag cases where explicit hyperscaler mentions
    are classified as non-cloud by the LLM.
    """
    if 'llm_classification' not in df.columns or 'explicit_cloud_mention' not in df.columns:
        return
    subset = df[df['explicit_cloud_mention'] == True]
    if len(subset) == 0:
        return
    noncloud = subset[~subset['llm_classification'].isin(['cloud-dependent', 'cloud-infrastructure'])]
    if len(noncloud) == 0:
        print("\n  LLM validation: No explicit hyperscaler mentions classified as non-cloud.")
        return
    dollars = noncloud['dollars'].sum()
    print(f"\n  LLM validation: {len(noncloud):,} explicit hyperscaler mentions "
          f"classified as non-cloud (${dollars:,.2f}).")


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def run_stage1(merged_df: pd.DataFrame,
               api_key: Optional[str] = None,
               cache_dir: Optional[str] = None,
               run_llm: bool = False,
               skip_confirmation: bool = False,
               conflict_mode: str = "explicit_override",
               noncloud_override_confidence: str = "high") -> pd.DataFrame:
    """
    Run Stage 1: Cloud Classification pipeline.

    1. Always runs RegEx classification
    2. If run_llm=True and api_key provided, runs LLM classification
    3. If LLM was previously run, loads cached results
    4. Runs synthesis to combine results

    Args:
        merged_df: Merged dataset from Stage 0C
        api_key: Anthropic API key (required for LLM)
        cache_dir: Directory for LLM checkpoint files
        run_llm: Whether to run LLM classification if no cache exists
        skip_confirmation: Skip cost confirmation prompt
        conflict_mode: How to resolve regex vs LLM conflicts
        noncloud_override_confidence: Minimum LLM confidence to override explicit cloud mention

    Returns:
        DataFrame with classification columns added
    """
    print("\n" + "=" * 80)
    print("STAGE 1: CLOUD CLASSIFICATION")
    print("=" * 80)
    print(f"\n  Input: {len(merged_df):,} records, ${merged_df['dollars'].sum()/1e9:.1f}B")

    # Step 1A: RegEx (always)
    patterns = compile_patterns()
    df = classify_cloud_regex(merged_df, patterns)

    # Step 1B: LLM (conditional)
    if cache_dir:
        cached_llm = load_llm_results(cache_dir)
    else:
        cached_llm = None

    if cached_llm is not None:
        print(f"\n  Loading cached LLM results ({len(cached_llm):,} records)")
        if len(cached_llm) != len(df):
            print("  WARNING: Cached LLM results length does not match current dataset.")
            print("  Falling back to RegEx-only mode to avoid misalignment.")
        else:
            df['llm_classification'] = cached_llm['classification'].values[:len(df)]
            df['llm_platform'] = cached_llm['platform'].values[:len(df)]
            df['llm_confidence'] = cached_llm['confidence'].values[:len(df)]
            df['llm_evidence'] = cached_llm['evidence'].values[:len(df)]
    elif run_llm and api_key:
        df = classify_cloud_llm(df, api_key, cache_dir,
                                 skip_confirmation=skip_confirmation)
    else:
        if run_llm and not api_key:
            print("\n  WARNING: run_llm=True but no API key provided.")
            print("  Set ANTHROPIC_API_KEY environment variable.")
        print("  Skipping LLM classification (RegEx-only mode).")

    validate_llm_conflicts(df)

    # Step 1C: Synthesis
    df = synthesize_classification(
        df,
        conflict_mode=conflict_mode,
        noncloud_override_confidence=noncloud_override_confidence
    )

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Stage 1: Cloud Classification."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    merged_path = os.path.join(project_root, 'data', '02_processed', '02_merged', 'merged_dataset.csv')
    cache_dir = os.path.join(project_root, 'data', '02_processed', '03_classified', 'llm_cache')

    if not os.path.exists(merged_path):
        print(f"ERROR: Merged dataset not found at {merged_path}")
        return

    print("Loading merged dataset...")
    merged_df = pd.read_csv(merged_path)
    print(f"  Loaded {len(merged_df):,} records")

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    run_llm = api_key is not None

    df = run_stage1(merged_df, api_key=api_key, cache_dir=cache_dir, run_llm=run_llm)

    # Save
    output_dir = os.path.join(project_root, 'data', '02_processed', '03_classified')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'classified_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved classified dataset: {output_path} ({len(df):,} rows)")

    return df


if __name__ == '__main__':
    main()

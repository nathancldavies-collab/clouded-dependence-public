import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
DATA_ATTR = ROOT / 'data' / '02_processed' / '03_classified' / 'attributed_dataset.csv'
DATA_PRIME = ROOT / 'data' / '02_processed' / '01_filtered' / 'prime_services_filtered.csv'
OUT_TABLES = ROOT / 'outputs_results_v2' / 'tables'
OUT_FIGS = ROOT / 'outputs_results_v2' / 'figures'
IMG_DIR = ROOT / 'images'

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
attr_cols = [
    'contractor', 'contractor_name', 'dollars', 'original_prime_id',
    'is_cloud', 'cloud_classification', 'final_platform', 'attribution_method'
]
attr = pd.read_csv(DATA_ATTR, usecols=attr_cols)
primes = pd.read_csv(DATA_PRIME, usecols=['Award ID', 'Effective Date'])

primes['Award ID'] = primes['Award ID'].astype(str)
attr['original_prime_id'] = attr['original_prime_id'].astype(str)

# Merge effective date
merged = attr.merge(primes, left_on='original_prime_id', right_on='Award ID', how='left')
merged['Effective Date'] = pd.to_datetime(merged['Effective Date'], errors='coerce')
merged['calendar_year'] = merged['Effective Date'].dt.year

# Filter to calendar years
merged = merged[merged['calendar_year'].between(2017, 2024)]

# Recode Cerner to Oracle Cloud for calendar_year >= 2022
mask_cerner = merged['final_platform'].str.contains('CERNER', case=False, na=False) & (merged['calendar_year'] >= 2022)
merged.loc[mask_cerner, 'final_platform'] = 'Oracle Cloud'

# Helpers

def hhi_from_series(series: pd.Series) -> float:
    total = series.sum()
    if total == 0:
        return 0.0
    shares = series / total
    return float(np.sum((shares * 100) ** 2))


def format_billions(x: float) -> str:
    return f"${x/1e9:,.2f}B"


def format_pct(x: float) -> str:
    return f"{x:.1f}%"

# ---- Table A: Baseline summary (2017 vs 2024) ----

def baseline_metrics(year: int):
    dfy = merged[merged['calendar_year'] == year]
    total = dfy['dollars'].sum()
    by_contractor = dfy.groupby('contractor')['dollars'].sum().sort_values(ascending=False)
    hhi = hhi_from_series(by_contractor)
    top5 = by_contractor.head(5).sum() / total * 100
    top10 = by_contractor.head(10).sum() / total * 100
    supplier_count = by_contractor.shape[0]
    median_contract = dfy['dollars'].median()
    return hhi, supplier_count, top5, top10, median_contract

m2017 = baseline_metrics(2017)
m2024 = baseline_metrics(2024)

# Compute percent change
change_pct = [
    (m2024[0] - m2017[0]) / m2017[0] * 100,
    (m2024[1] - m2017[1]) / m2017[1] * 100,
    (m2024[2] - m2017[2]) / m2017[2] * 100,
    (m2024[3] - m2017[3]) / m2017[3] * 100,
    (m2024[4] - m2017[4]) / m2017[4] * 100,
]

rows_a = [
    ("Contractor HHI", f"{m2017[0]:.1f}", f"{m2024[0]:.1f}", f"{change_pct[0]:+.1f}%"),
    ("Supplier count", f"{m2017[1]:,.1f}", f"{m2024[1]:,.1f}", f"{change_pct[1]:+.1f}%"),
    ("Top-5 share (%)", f"{m2017[2]:.1f}", f"{m2024[2]:.1f}", f"{change_pct[2]:+.1f}%"),
    ("Top-10 share (%)", f"{m2017[3]:.1f}", f"{m2024[3]:.1f}", f"{change_pct[3]:+.1f}%"),
    ("Median contract ($)", f"{m2017[4]:,.1f}", f"{m2024[4]:,.1f}", f"{change_pct[4]:+.1f}%"),
]

table_a = pd.DataFrame(rows_a, columns=["Metric", "2017", "2024", "Change (%)"])
table_a.to_csv(OUT_TABLES / 'table_A_baseline_summary.csv', index=False)

# ---- Table B: Cloud identification ----
cloud = merged[merged['is_cloud'] == True]
cloud_total = cloud['dollars'].sum()
it_total = merged['dollars'].sum()
cloud_infra = cloud[cloud['cloud_classification'] == 'cloud-infrastructure']['dollars'].sum()

phase1 = cloud[cloud['attribution_method'] == 'phase1_direct']['dollars'].sum()
phase2 = cloud[cloud['attribution_method'] == 'phase2_description']['dollars'].sum()
phase3 = cloud[cloud['attribution_method'] == 'phase3_pattern']['dollars'].sum()
unattributed = cloud_total - (phase1 + phase2 + phase3)

rows_b = [
    ("Total IT spend", format_billions(it_total), format_pct(100.0), ""),
    ("Cloud total (multi-step)", format_billions(cloud_total), format_pct(cloud_total/it_total*100), format_pct(100.0)),
    ("Cloud infrastructure", format_billions(cloud_infra), format_pct(cloud_infra/it_total*100), format_pct(cloud_infra/cloud_total*100)),
    ("Attributed: Phase 1 (direct)", format_billions(phase1), format_pct(phase1/it_total*100), format_pct(phase1/cloud_total*100)),
    ("Attributed: Phase 2 (description)", format_billions(phase2), format_pct(phase2/it_total*100), format_pct(phase2/cloud_total*100)),
    ("Attributed: Phase 3 (pattern)", format_billions(phase3), format_pct(phase3/it_total*100), format_pct(phase3/cloud_total*100)),
    ("Unattributed cloud", format_billions(unattributed), format_pct(unattributed/it_total*100), format_pct(unattributed/cloud_total*100)),
]

table_b = pd.DataFrame(rows_b, columns=["Item", "Dollars", "% of total IT", "% of cloud"])
table_b.to_csv(OUT_TABLES / 'table_B_cloud_identification.csv', index=False)

# ---- Table C: Platform shares by layer (compact) ----
cloud = merged[merged['is_cloud'] == True]
infra = cloud[cloud['cloud_classification'] == 'cloud-infrastructure']
dep = cloud[cloud['cloud_classification'] == 'cloud-dependent']

key_platforms = ['AWS', 'Azure', 'Google Cloud', 'Oracle Cloud', 'Salesforce', 'IBM Cloud', 'ServiceNow']


def layer_table(df: pd.DataFrame, total: float):
    totals = df.groupby('final_platform')['dollars'].sum()
    rows = []
    for p in key_platforms:
        val = totals.get(p, 0.0)
        rows.append((p, val, val / total * 100))
    other = total - sum(val for _, val, _ in rows)
    rows.insert(0, ('Unattributed/Other', other, other / total * 100))
    return rows

infra_total = infra['dollars'].sum()
dep_total = dep['dollars'].sum()
infra_rows = layer_table(infra, infra_total)
dep_rows = layer_table(dep, dep_total)

rows_c = []
for (platform, infra_val, infra_pct), (_, dep_val, dep_pct) in zip(infra_rows, dep_rows):
    rows_c.append({
        'Platform': platform,
        'Infra $': format_billions(infra_val),
        'Infra %': format_pct(infra_pct),
        'Dependent $': format_billions(dep_val),
        'Dependent %': format_pct(dep_pct),
    })

table_c = pd.DataFrame(rows_c)
table_c.to_csv(OUT_TABLES / 'table_C_platform_shares_by_layer.csv', index=False)

# ---- Figure 2: Surface vs Multi-step + Attribution Stack ----

def build_fig2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # Left: surface vs multi-step scale
    surface = phase1
    multistep = cloud_total
    axes[0].bar([0], [surface / 1e9], color='#59A14F', width=0.6)
    axes[0].bar([1], [multistep / 1e9], color='#4C78A8', width=0.6)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Surface view\n(direct platform)', 'Multi-step view\n(cloud total)'])
    axes[0].set_ylabel('Dollars (Billions)')
    axes[0].set_title('Surface vs Multi-Step (Scale)')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].text(0, surface / 1e9 + 0.3, f"${surface/1e9:.1f}B", ha='center', va='bottom', fontsize=9)
    axes[0].text(1, multistep / 1e9 + 0.3, f"${multistep/1e9:.1f}B", ha='center', va='bottom', fontsize=9)

    # Right: attribution stack
    stack_vals = [phase1, phase2, phase3, unattributed]
    stack_labels = ['Phase 1 (direct)', 'Phase 2 (desc.)', 'Phase 3 (pattern)', 'Unattributed']
    stack_colors = ['#2E7D32', '#8CD17D', '#A6D5A8', '#D0D0D0']
    bottom = 0
    for val, label, color in zip(stack_vals, stack_labels, stack_colors):
        axes[1].bar([0], [val / 1e9], bottom=bottom / 1e9, color=color, width=0.6, label=label)
        bottom += val
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(['Cloud total'])
    axes[1].set_ylabel('Dollars (Billions)')
    axes[1].set_title('Multi-Step Attribution of Cloud Spending')
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].legend(loc='upper right', frameon=False)

    fig.tight_layout()
    fig2_path = OUT_FIGS / 'fig2_cloud_identification_attribution.png'
    fig.savefig(fig2_path, bbox_inches='tight')
    plt.close(fig)
    return fig2_path


# ---- Figure 3: Dependency matrix (calendar-year) ----

def build_fig3():
    # Define hyperscaler contractors by name substring
    hyperscaler_names = ['AMAZON WEB SERVICES', 'AMAZON.COM', 'MICROSOFT', 'GOOGLE', 'ALPHABET', 'ORACLE']
    name_upper = merged['contractor_name'].fillna('').str.upper()
    is_hyperscaler = pd.Series(False, index=merged.index)
    for pat in hyperscaler_names:
        is_hyperscaler |= name_upper.str.contains(pat, na=False)

    contractor_cloud = merged.groupby('contractor')['is_cloud'].any()

    def col_assign(row):
        if is_hyperscaler.loc[row.name]:
            return 'Hyperscaler'
        if contractor_cloud.get(row['contractor'], False):
            return 'Cloud-dependent'
        return 'Traditional IT firm'

    df = merged.copy()
    df['col'] = df.apply(col_assign, axis=1)
    row_map = {
        'non-cloud': 'Non-Cloud/Unattributed',
        'cloud-dependent': 'Cloud-Dependent',
        'cloud-infrastructure': 'Cloud (Infrastructure)'
    }
    df['row'] = df['cloud_classification'].map(row_map)

    matrix = df.groupby(['row', 'col'])['dollars'].sum().unstack(fill_value=0)
    total = df['dollars'].sum()

    row_order = ['Non-Cloud/Unattributed', 'Cloud-Dependent', 'Cloud (Infrastructure)']
    col_order = ['Traditional IT firm', 'Cloud-dependent', 'Hyperscaler']
    matrix = matrix.reindex(index=row_order, columns=col_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(matrix.values / 1e9, cmap='Blues')

    # annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            pct = val / total * 100 if total > 0 else 0
            ax.text(j, i, f"${val/1e9:.1f}B\n({pct:.1f}%)", ha='center', va='center', fontsize=9)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=15, ha='right')
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_title('Dependency Matrix')

    # highlight cloud-dependent contractor delivering cloud services
    rect = patches.Rectangle((1-0.5, 1-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Dollars (Billions)')

    fig.text(0.02, 0.02, "Note: Non-Cloud/Unattributed includes ambiguous cases; assignments are conservative.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    fig3_path = OUT_FIGS / 'fig3_dependency_matrix.png'
    fig.savefig(fig3_path, bbox_inches='tight')
    plt.close(fig)
    return fig3_path


# ---- Figure 4: Divergence over time (calendar-year windows) ----
windows = []
for start in range(2017, 2023):
    end = start + 2
    df_window = merged[merged['calendar_year'].between(start, end)]
    contractor_totals = df_window.groupby('contractor')['dollars'].sum()
    contractor_hhi = hhi_from_series(contractor_totals)
    cloud_window = df_window[df_window['is_cloud'] == True]
    platform_totals = cloud_window.groupby('final_platform')['dollars'].sum()
    platform_hhi = hhi_from_series(platform_totals)
    infra_window = cloud_window[cloud_window['cloud_classification'] == 'cloud-infrastructure']
    infra_totals = infra_window.groupby('final_platform')['dollars'].sum()
    infra_hhi = hhi_from_series(infra_totals)
    windows.append({
        'center': start + 1,
        'contractor_hhi': contractor_hhi,
        'cloud_hhi': platform_hhi,
        'infra_hhi': infra_hhi,
    })

wdf = pd.DataFrame(windows)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

# Left panel: baseline contractor HHI
axes[0].plot(wdf['center'], wdf['contractor_hhi'], color='#7f7f7f', marker='o')
axes[0].set_title('Baseline View (Contractor HHI)')
axes[0].set_xlabel('Calendar Year')
axes[0].set_ylabel('HHI (3-year rolling)')
axes[0].set_xticks(wdf['center'])
axes[0].grid(alpha=0.25, axis='y')

# Right panel: platform HHI
axes[1].plot(wdf['center'], wdf['cloud_hhi'], color='#4C78A8', marker='o', label='All Cloud (platform)')
axes[1].plot(wdf['center'], wdf['infra_hhi'], color='#F58518', marker='o', label='Infra Only (platform)')
axes[1].plot(wdf['center'], wdf['contractor_hhi'], color='#7f7f7f', linestyle='--', marker='o', label='Total IT (contractor)')
axes[1].set_title('Attributed View (Platform HHI)')
axes[1].set_xlabel('Calendar Year')
axes[1].set_xticks(wdf['center'])
axes[1].grid(alpha=0.25, axis='y')
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.tight_layout(rect=(0, 0, 0.85, 1))

fig4_path = OUT_FIGS / 'fig4_divergence_over_time.png'
fig.savefig(fig4_path, bbox_inches='tight')
plt.close(fig)

# ---- Figure 5: Hyperscaler visibility across layers ----

# ---- Figure 5: Hyperscaler visibility across layers ----

# Define hyperscaler palette
colors = {
    'AWS': '#FF9900',
    'Azure': '#0078D4',
    'Google Cloud': '#34A853',
    'Oracle Cloud': '#C74634',
    'Other/Unattributed': '#D0D0D0'
}

hypers = ['AWS', 'Azure', 'Google Cloud', 'Oracle Cloud']

# Baseline shares: direct entity only
it_total = merged['dollars'].sum()
baseline = merged[merged['attribution_method'] == 'phase1_direct']
base_totals = baseline.groupby('final_platform')['dollars'].sum()
base_shares = {h: base_totals.get(h, 0.0) / it_total * 100 for h in hypers}

# Cloud total shares (within cloud)
cloud_total = cloud['dollars'].sum()
cloud_totals = cloud.groupby('final_platform')['dollars'].sum()
cloud_shares = {h: cloud_totals.get(h, 0.0) / cloud_total * 100 for h in hypers}

# Infra shares (within infra)
infra_totals = infra.groupby('final_platform')['dollars'].sum()
infra_total = infra['dollars'].sum()
infra_shares = {h: infra_totals.get(h, 0.0) / infra_total * 100 for h in hypers}

# Build stacked bars
labels = [
    'Baseline (all IT)',
    'Non-cloud IT',
    'Cloud total\n(cloud + dependent)',
    'Cloud infrastructure\n(IaaS/PaaS)'
]

# For non-cloud IT, show all as Other/Unattributed
bars = [base_shares, None, cloud_shares, infra_shares]

fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

for i, label in enumerate(labels):
    left = 0
    if i == 1:
        # Non-cloud IT: all other
        ax.barh(i, 100, color=colors['Other/Unattributed'], edgecolor='white', height=0.5)
        continue
    shares = bars[i]
    # plot hyperscalers
    for h in hypers:
        val = shares.get(h, 0.0)
        if val <= 0:
            continue
        ax.barh(i, val, left=left, color=colors[h], edgecolor='white', height=0.5)
        if val >= 8:
            ax.text(left + val/2, i, f"{h} {val:.1f}%", va='center', ha='center', color='white', fontsize=9)
        left += val
    # Other/Unattributed
    other = 100 - sum(shares.values())
    ax.barh(i, other, left=left, color=colors['Other/Unattributed'], edgecolor='white', height=0.5)
    if other >= 12:
        ax.text(left + other/2, i, f"Other {other:.1f}%", va='center', ha='center', color='#333333', fontsize=9)

# Baseline shares annotation box
baseline_text = "Baseline shares\n" + "\n".join([
    f"{h}: {base_shares[h]:.2f}%" for h in hypers
])
ax.text(78, 0.15, baseline_text, fontsize=8.5, va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#dddddd'))

# Divider line
ax.axhline(0.5, color='#888888', linestyle='--', linewidth=1)
ax.text(50, 0.65, 'Multi-step view', ha='center', va='bottom', fontsize=9, color='#555555')

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel('Share of spending within category (%)')
ax.set_title('Hyperscaler Visibility Across Layers')
ax.grid(axis='x', alpha=0.2)

# Legend
handles = [plt.Rectangle((0,0),1,1, color=colors[h]) for h in hypers]
handles.append(plt.Rectangle((0,0),1,1, color=colors['Other/Unattributed']))
labels_leg = hypers + ['Other/Unattributed']
ax.legend(handles, labels_leg, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=5, frameon=False)

fig.tight_layout()
fig5_path = OUT_FIGS / 'fig5_hyperscaler_visibility.png'
fig.savefig(fig5_path, bbox_inches='tight')
plt.close(fig)

# ---- Figure 6: Infrastructure surface vs multi-step ----

def build_fig6():
    # Infrastructure-only subset
    infra = cloud[cloud['cloud_classification'] == 'cloud-infrastructure']
    infra_total = infra['dollars'].sum()

    # Base-level (contractor view, all infra)
    contractor_totals = infra.groupby('contractor_name')['dollars'].sum()
    hhi_base = hhi_from_series(contractor_totals)

    # Multi-step (platform view, all infra)
    multi_totals = infra.groupby('final_platform')['dollars'].sum()
    hhi_multi = hhi_from_series(multi_totals)

    # Direct-only platform totals (for mechanism panel)
    direct = infra[infra['attribution_method'] == 'phase1_direct']
    direct_totals = direct.groupby('final_platform')['dollars'].sum()

    # Save Table D
    table_d = pd.DataFrame({
        'Metric': ['Infrastructure HHI (base-level contractor)', 'Infrastructure HHI (multi-step platform)'],
        'Value': [f"{hhi_base:.0f}", f"{hhi_multi:.0f}"]
    })
    table_d.to_csv(OUT_TABLES / 'table_D_infra_surface_vs_multistep.csv', index=False)

    # Shares
    hypers = ['AWS', 'Azure', 'Google Cloud', 'Oracle Cloud']
    direct_shares = {h: direct_totals.get(h, 0.0) / infra_total * 100 for h in hypers}
    multi_shares = {h: multi_totals.get(h, 0.0) / infra_total * 100 for h in hypers}

    colors = {
        'AWS': '#FF9900',
        'Azure': '#0078D4',
        'Google Cloud': '#34A853',
        'Oracle Cloud': '#C74634',
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # Left: HHI bars
    axes[0].bar(['Base-level', 'Multi-step'], [hhi_base, hhi_multi], color=['#7f7f7f', '#4C78A8'])
    axes[0].set_title('Infrastructure concentration (HHI)')
    axes[0].set_ylabel('HHI')
    for i, v in enumerate([hhi_base, hhi_multi]):
        axes[0].text(i, v + 30, f"{v:.0f}", ha='center', va='bottom', fontsize=9)
    axes[0].grid(axis='y', alpha=0.25)

    # Right: hyperscaler shares
    x = np.arange(len(hypers))
    width = 0.35
    for i, h in enumerate(hypers):
        axes[1].bar(x[i] - width/2, direct_shares[h], width, color=colors[h], alpha=0.4, hatch='//', edgecolor=colors[h])
        axes[1].bar(x[i] + width/2, multi_shares[h], width, color=colors[h])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(hypers, rotation=15)
    axes[1].set_ylabel('% of infrastructure spend')
    axes[1].set_title('Hyperscaler shares within infrastructure')
    axes[1].grid(axis='y', alpha=0.25)

    # Legend
    legend_handles = [
        patches.Patch(facecolor='#7f7f7f', alpha=0.4, hatch='//', label='Direct-only'),
        patches.Patch(facecolor='#7f7f7f', label='Multi-step')
    ]
    axes[1].legend(handles=legend_handles, loc='upper right', frameon=False)

    fig.tight_layout()
    fig6_path = OUT_FIGS / 'fig6_infra_surface_vs_multistep.png'
    fig.savefig(fig6_path, bbox_inches='tight')
    plt.close(fig)
    return fig6_path

fig2_path = build_fig2()
fig3_path = build_fig3()
fig6_path = build_fig6()

# Copy updated figures into images/ for LaTeX
for src in [fig2_path, fig3_path, fig4_path, fig5_path, fig6_path]:
    dest = IMG_DIR / src.name
    dest.write_bytes(src.read_bytes())

print('Updated tables and figures saved.')

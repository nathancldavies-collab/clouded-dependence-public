#!/usr/bin/env python3
"""
Nth-Order Dependency Exposure Analysis - Main Runner
=====================================================
Compares dependency exposure between:
  (a) the NORMAL view  - observable contractual relationships only (prime -> subaward)
  (b) the CLOUDED view - the above plus platform dependency edges from attribution

Framing: this reports *exposure* only - which contracts stand in a dependency
relation to a node, i.e. which could be affected. No outage is posited and no
failure is predicted.

Reads the existing pipeline outputs, so it does NOT re-run classification or spend
LLM budget. Run `run_pipeline.py` first.

Run from project root:
    uv run run_exposure_analysis.py [--k 2] [--top 15]
"""

import argparse
import sys
import time

import pandas as pd

from clouded_deps.directories import DATA_DIR
from clouded_deps.pipeline import exposure_network as ex
from clouded_deps.pipeline.create_merged_dataset import load_primes

CONTRACT_SPECS: tuple[ex.Spec, ...] = ("conservative", "permissive")


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load filtered primes, the attributed dataset, and raw subawards."""
    filtered_path = (
        DATA_DIR / "02_processed" / "01_filtered" / "prime_services_filtered.csv"
    )
    attributed_path = (
        DATA_DIR / "02_processed" / "03_classified" / "attributed_dataset.csv"
    )
    subs_path = DATA_DIR / "01_raw" / "subawards.csv"

    for path in (filtered_path, attributed_path, subs_path):
        if not path.exists():
            print(f"FATAL: missing input {path}")
            print("       Run `just pipeline` first.")
            sys.exit(1)

    print(f"  Loading filtered primes:    {filtered_path.name}")
    primes_df = load_primes(filtered_path)
    print(f"  Loading attributed dataset: {attributed_path.name}")
    attributed_df = pd.read_csv(attributed_path, low_memory=False)
    print(f"  Loading raw subawards:      {subs_path.name}")
    subs_raw = pd.read_csv(subs_path, encoding="utf-8-sig", low_memory=False)
    return primes_df, attributed_df, subs_raw


def _print_leaderboard(board: pd.DataFrame, population_size: int, unit: str) -> None:
    """Print the two views' rankings side by side."""
    normal = board[board["view"] == "normal"].reset_index(drop=True)
    clouded = board[board["view"] == "clouded"].reset_index(drop=True)

    print(f"\n  {'':4s} {'NORMAL VIEW':44s}  {'CLOUDED VIEW':44s}")
    print(f"  {'-' * 94}")
    for i in range(max(len(normal), len(clouded))):
        left = right = " " * 44
        if i < len(normal):
            row = normal.iloc[i]
            count = round(row["exposure"] * population_size)
            left = f"{row['label'][:28]:28s} {count:7,} ({row['exposure'] * 100:5.2f}%)"
        if i < len(clouded):
            row = clouded.iloc[i]
            marker = "*" if row["is_platform"] else " "
            count = round(row["exposure"] * population_size)
            right = (
                f"{marker}{row['label'][:27]:27s} {count:7,} "
                f"({row['exposure'] * 100:5.2f}%)"
            )
        print(f"  {i + 1:2d}.  {left}  {right}")
    print(f"\n  * = cloud platform.  Counts are {unit}.")


def _print_platform_table(table: pd.DataFrame) -> None:
    """Print per-platform exposure in both views."""
    print(
        f"\n  {'Platform':<15s} {'normal':>10s} {'clouded':>10s} "
        f"{'normal %':>10s} {'clouded %':>10s} {'hidden':>10s} {'ratio':>8s}"
    )
    print(f"  {'-' * 78}")
    for _, row in table.iterrows():
        ratio = f"{row['ratio']:.1f}x" if pd.notna(row["ratio"]) else "n/a"
        print(
            f"  {row['platform']:<15s} {row['normal_count']:>10,} "
            f"{row['clouded_count']:>10,} {row['normal_pct']:>9.3f}% "
            f"{row['clouded_pct']:>9.3f}% {row['hidden_pp']:>+9.3f}pp {ratio:>8s}"
        )


def _print_summary(label: str, summary: dict) -> None:
    """Print the PDF section 4 robustness summary."""
    print(
        f"    {label:<26s} max={summary['max'] * 100:6.2f}%  "
        f"p95={summary['p95'] * 100:6.3f}%  mean={summary['mean'] * 100:6.3f}%  "
        f"gini={summary['gini']:.3f}  n>10%={summary['n_above_10pct']:,}  "
        f"scored={summary['n_scored']:,}"
    )


def _check_labels(board: pd.DataFrame) -> None:
    """Warn if any leaderboard row is an unlabelled bare UEI (plan NQ2)."""
    unlabelled = board[(board["label"] == board["node"]) & ~board["is_platform"]]
    if len(unlabelled) > 0:
        print(
            f"\n  WARNING: {len(unlabelled)} leaderboard node(s) have no resolved name: "
            f"{', '.join(unlabelled['node'].unique())}"
        )


def run_exposure_analysis(k: int = 2, top: int = 15, analysis: str = "both") -> None:
    """Execute the exposure analysis and write outputs."""
    start = time.time()

    print("=" * 80)
    print("NTH-ORDER DEPENDENCY EXPOSURE: NORMAL vs CLOUDED VIEW")
    print("=" * 80)
    print(f"\n  Tier depth k = {k}   (distance counted in dependency tiers, not edges)")
    print("  Reporting exposure only - which contracts could be affected.")
    print("  No outage is posited and no failure is predicted.\n")

    primes_df, attributed_df, subs_raw = _load_inputs()

    # --- Graph construction -------------------------------------------------
    print("\n" + "#" * 80)
    print("# GRAPH CONSTRUCTION")
    print("#" * 80)

    alias = ex.build_platform_alias(primes_df, subs_raw)
    labels = ex.build_entity_labels(primes_df, subs_raw, alias)
    population = ex.build_contract_population(primes_df, alias)
    edges = ex.resolve_subaward_endpoints(subs_raw, alias)
    deps = ex.build_contract_deps(attributed_df, alias)
    nodes = ex.all_nodes(population, edges, deps)

    g_normal = ex.build_entity_graph(nodes, edges)
    g_clouded = ex.add_cloud_edges(g_normal, attributed_df, alias)

    print(f"\n  Contract population:  {len(population):,} awards")
    print(f"  Entity nodes:         {len(nodes):,} (identical across both views)")
    print(f"  Platform aliases:     {len(alias):,} UEIs fused into platform nodes")
    print(f"  Resolved labels:      {len(labels):,}")
    print(f"  Normal-view edges:    {g_normal.number_of_edges():,}")
    print(
        f"  Clouded-view edges:   {g_clouded.number_of_edges():,} "
        f"(+{g_clouded.number_of_edges() - g_normal.number_of_edges():,} platform edges)"
    )
    print(
        f"  Contract deps:        {len(deps):,} "
        f"(sub={int((deps['kind'] == 'sub').sum()):,}, "
        f"cloud={int((deps['kind'] == 'cloud').sum()):,})"
    )

    out_dir = DATA_DIR.parent / "outputs_results_v2" / "exposure"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    summaries: list[dict] = []

    # --- Analysis 2: contract exposure --------------------------------------
    if analysis in ("contract", "both"):
        print("\n\n" + "#" * 80)
        print("# ANALYSIS: CONTRACT EXPOSURE")
        print("#" * 80)
        print(f"\n  Population: {len(population):,} prime awards.")
        print("  I_k(v) = share of OTHER contracts within k dependency tiers of v.")

        for spec in CONTRACT_SPECS:
            primary = " (PRIMARY)" if spec == "conservative" else " (upper bound)"
            print(f"\n\n  {'=' * 74}")
            print(f"  SPEC: {spec.upper()}{primary}")
            print(f"  {'=' * 74}")

            graph_n = ex.select_graph("normal", spec, g_normal, g_clouded)
            graph_c = ex.select_graph("clouded", spec, g_normal, g_clouded)

            normal = ex.contract_exposure(
                graph_n, nodes, population, deps, k, spec, include_cloud_deps=False
            )
            clouded = ex.contract_exposure(
                graph_c, nodes, population, deps, k, spec, include_cloud_deps=True
            )

            board = ex.leaderboard(normal, clouded, labels, top=top)
            _print_leaderboard(board, len(population), "contracts")
            _check_labels(board)

            table = ex.platform_table(normal, clouded, len(population))
            _print_platform_table(table)

            print("\n  Network-level robustness (PDF section 4):")
            for view, series, graph in [
                ("normal", normal, graph_n),
                ("clouded", clouded, graph_c),
            ]:
                summary = ex.exposure_summary(series, graph)
                _print_summary(f"{view}", summary)
                summaries.append(
                    {"analysis": "contract", "spec": spec, "view": view, **summary}
                )

            board_path = out_dir / f"leaderboard_contract_{spec}.csv"
            table_path = out_dir / f"exposure_by_platform_{spec}.csv"
            board.to_csv(board_path, index=False)
            table.to_csv(table_path, index=False)
            written += [board_path.name, table_path.name]

    # --- Analysis 1: entity exposure ----------------------------------------
    if analysis in ("entity", "both"):
        print("\n\n" + "#" * 80)
        print("# ANALYSIS: ENTITY EXPOSURE (permissive by construction)")
        print("#" * 80)
        print(f"\n  Population: {len(nodes):,} entities.")
        print("  I_k(v) = share of OTHER entities within k dependency tiers of v.")

        normal = ex.entity_exposure(g_normal, nodes, k)
        clouded = ex.entity_exposure(g_clouded, nodes, k)

        board = ex.leaderboard(normal, clouded, labels, top=top)
        _print_leaderboard(board, len(nodes) - 1, "entities")
        _check_labels(board)

        table = ex.platform_table(normal, clouded, len(nodes) - 1)
        _print_platform_table(table)

        print("\n  Network-level robustness (PDF section 4):")
        for view, series, graph in [
            ("normal", normal, g_normal),
            ("clouded", clouded, g_clouded),
        ]:
            summary = ex.exposure_summary(series, graph)
            _print_summary(f"{view}", summary)
            summaries.append(
                {"analysis": "entity", "spec": "permissive", "view": view, **summary}
            )

        board_path = out_dir / "leaderboard_entity.csv"
        table_path = out_dir / "exposure_by_platform_entity.csv"
        board.to_csv(board_path, index=False)
        table.to_csv(table_path, index=False)
        written += [board_path.name, table_path.name]

    summary_path = out_dir / "exposure_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    written.append(summary_path.name)

    # --- Limitations --------------------------------------------------------
    print("\n\n" + "#" * 80)
    print("# LIMITATIONS")
    print("#" * 80)
    print("  - USAspending reports no Tier 2 subcontracts; k > 2 relies on the entity")
    print("    projection (chains assembled across contracts via role-switching).")
    print("  - Exposure is not failure: no substitution, redundancy or multi-homing is")
    print("    modelled, and no outage is posited.")
    print("  - Subaward reporting thresholds leave most contracts isolated in the")
    print("    normal view; this is an observability limit, not a supply-chain fact.")
    print("  - Multi-platform contracts count as fully exposed to each platform.")
    print(
        "  - Node roles differ in kind (resellers, staffing, integrators, platforms)."
    )

    print("\n\n" + "=" * 80)
    print("EXPOSURE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n  Output directory: {out_dir}")
    for name in written:
        print(f"    {name}")
    print(f"\n  Elapsed: {time.time() - start:.1f} seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Tier depth for the headline metric (default: 2)",
    )
    parser.add_argument(
        "--top", type=int, default=15, help="Leaderboard length (default: 15)"
    )
    parser.add_argument(
        "--analysis",
        choices=["contract", "entity", "both"],
        default="both",
        help="Which population to score (default: both)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_exposure_analysis(k=args.k, top=args.top, analysis=args.analysis)

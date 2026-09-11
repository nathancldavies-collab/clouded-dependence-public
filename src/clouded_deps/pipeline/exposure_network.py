#!/usr/bin/env python3
"""
Nth-Order Dependency Exposure: Normal vs Clouded View
======================================================
Implements the exposure metric from `nth_order_dependency_exposure.pdf` under two
edge conventions, per PLAN_EXPOSURE_ANALYSIS.md.

  normal view  : edges from observable contractual relationships only (prime -> subaward)
  clouded view : the above plus platform dependency edges from Stage 1 + Stage 2

Framing: this measures *exposure* only — which contracts stand in a dependency
relation to a node, i.e. which could be affected. No outage is posited and no
failure is predicted (see PDF section 5).

Metric
------
Distance is counted in TIERS, not raw edges. A contract is identified with its
performing entity at tier 0; only entity->entity hops increment the tier counter.

    I_k(v) = |D_k(v)| / |population|

where D_k(v) is the set of nodes OTHER than v's own that reach v within k tiers.
Only tiers t >= 1 contribute: tier 0 establishes identity and is excluded.

Two analyses, each over a single node type:

  entity exposure   : population = entities. Permissive by construction (the entity
                      graph has no contracts to confine propagation to).
  contract exposure : population = all filtered prime awards. Computed under both
                      propagation specs; `conservative` is primary.

Propagation specs (contract exposure only)
------------------------------------------
  conservative : a contract is exposed only via dependencies recorded ON THAT
                 CONTRACT. Platform reachability comes solely from per-contract
                 attribution, so the normal entity graph is used for deeper hops.
  permissive   : a contract additionally inherits every dependency of its
                 performing entity, from any of that entity's contracts. Uses the
                 clouded entity graph. An upper bound.

Implementation
--------------
networkx is the readable reference implementation (graph construction, single-target
BFS). scipy.sparse is the fast path that scores every node at once via boolean
reachability products. `tests/test_exposure_network.py` pins the two together.
"""

from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from clouded_deps.pipeline.platform_attribution import (
    _CONTRACTOR_NAME_PATTERNS,
    CONCRETE_PLATFORMS,
    PLATFORM_ENTITIES,
    _parse_platform_mentions,
)

PLATFORM_PREFIX = "PLATFORM:"

Spec = Literal["conservative", "permissive"]
View = Literal["normal", "clouded"]

# Tier offsets for contract-level dependencies (see plan section 3.1).
TIER_SUBCONTRACTOR = 1
TIER_PLATFORM_ON_PRIME = 1
TIER_PLATFORM_ON_SUBCONTRACT = 2


def platform_node(platform: str) -> str:
    """Return the fused graph node id for a platform name."""
    return f"{PLATFORM_PREFIX}{platform}"


def is_platform_node(node: object) -> bool:
    """True if `node` is a fused platform node."""
    return isinstance(node, str) and node.startswith(PLATFORM_PREFIX)


def display_label(node: object, labels: dict[str, str]) -> str:
    """Human-readable label for a node, falling back to the raw id."""
    if is_platform_node(node):
        return str(node)[len(PLATFORM_PREFIX) :]
    return labels.get(str(node), str(node))


# =============================================================================
# ENTITY RESOLUTION AND PLATFORM FUSION
# =============================================================================


def _uei_name_pairs(
    primes_df: pd.DataFrame, subs_raw: pd.DataFrame
) -> list[tuple[str, str]]:
    """Collect every (uei, name) pair observable in primes and raw subawards."""
    pairs: list[tuple[str, str]] = []

    prime_cols = [
        ("resolved_entity", "Contractor Name"),
        ("Contractor UEI", "Contractor Name"),
        ("Contractor Parent UEI", "Contractor Name"),
    ]
    for uei_col, name_col in prime_cols:
        if uei_col in primes_df.columns and name_col in primes_df.columns:
            sub = primes_df[[uei_col, name_col]].dropna()
            pairs.extend(zip(sub[uei_col].astype(str), sub[name_col].astype(str)))

    sub_cols = [
        ("Subcontractor UEI", "Subcontractor"),
        ("Prime Contractor UEI", "Prime Contractor"),
        ("Prime Contractor Parent UEI", "Prime Contractor"),
    ]
    for uei_col, name_col in sub_cols:
        if uei_col in subs_raw.columns and name_col in subs_raw.columns:
            sub = subs_raw[[uei_col, name_col]].dropna()
            pairs.extend(zip(sub[uei_col].astype(str), sub[name_col].astype(str)))

    return pairs


def build_platform_alias(
    primes_df: pd.DataFrame, subs_raw: pd.DataFrame
) -> dict[str, str]:
    """
    Map every UEI belonging to a platform vendor onto a single fused platform node.

    Fusing the platform with its corporate entity keeps |V| identical across the
    normal and clouded views, so the comparison isolates topology rather than
    confounding it with a change in population.

    Args:
        primes_df: Filtered prime awards (needs UEI + Contractor Name columns).
        subs_raw: Raw subawards (needs UEI + name columns).

    Returns:
        Mapping of UEI -> "PLATFORM:<name>".
    """
    entities_upper = {k.upper(): v for k, v in PLATFORM_ENTITIES.items()}

    def _match(name: str) -> str | None:
        upper = name.strip().upper()
        if upper in entities_upper:
            return entities_upper[upper]
        for entity_name, platform in entities_upper.items():
            if entity_name in upper:
                return platform
        lower = name.lower()
        for platform, patterns in _CONTRACTOR_NAME_PATTERNS.items():
            for pattern in patterns:
                if _word_in(pattern, lower):
                    return platform
        return None

    alias: dict[str, str] = {}
    for uei, name in _uei_name_pairs(primes_df, subs_raw):
        if uei in alias:
            continue
        platform = _match(name)
        if platform is not None:
            alias[uei] = platform_node(platform)
    return alias


def _word_in(pattern: str, haystack_lower: str) -> bool:
    """Whole-word containment check, mirroring Stage 2's name matching."""
    import re

    return re.search(r"\b" + re.escape(pattern) + r"\b", haystack_lower) is not None


def build_entity_labels(
    primes_df: pd.DataFrame, subs_raw: pd.DataFrame, alias: dict[str, str]
) -> dict[str, str]:
    """
    Map every node id to a human-readable name.

    Labels are drawn from primes AND raw subawards so that entities appearing only
    as subcontractors still carry a name (plan NQ2). The longest observed name wins,
    which prefers fully-qualified legal names over abbreviations.
    """
    labels: dict[str, str] = {}
    for uei, name in _uei_name_pairs(primes_df, subs_raw):
        node = alias.get(uei, uei)
        if is_platform_node(node):
            continue
        existing = labels.get(node)
        if existing is None or len(name) > len(existing):
            labels[node] = name
    return labels


def resolve_subaward_endpoints(
    subs_raw: pd.DataFrame, alias: dict[str, str]
) -> pd.DataFrame:
    """
    Resolve each raw subaward to (prime_node, sub_node).

    Uses `Prime Contractor Parent UEI` where available, else `Prime Contractor UEI`,
    matching the `resolved_entity` convention used elsewhere in the pipeline. Built
    from the RAW subaward file rather than the merged dataset, which preserves the
    subawards that cannot be joined to a filtered prime by Award ID (plan section 2.4).
    """
    df = subs_raw.copy()
    prime_uei = df["Prime Contractor Parent UEI"].fillna(df["Prime Contractor UEI"])
    out = pd.DataFrame(
        {
            "prime_node": prime_uei.astype("string"),
            "sub_node": df["Subcontractor UEI"].astype("string"),
        }
    ).dropna()
    out["prime_node"] = out["prime_node"].map(lambda u: alias.get(u, u))
    out["sub_node"] = out["sub_node"].map(lambda u: alias.get(u, u))
    return out[out["prime_node"] != out["sub_node"]].reset_index(drop=True)


# =============================================================================
# CONTRACT POPULATION AND DEPENDENCIES
# =============================================================================


def build_contract_population(
    primes_df: pd.DataFrame, alias: dict[str, str]
) -> pd.DataFrame:
    """
    Build the contract population from the FILTERED PRIMES file.

    Deliberately not taken from the merged dataset: where subawards consume the whole
    award the merge emits no `prime_retained` row, which would silently drop those
    awards from the population (plan NQ3). This analysis has no dollar-conservation
    invariant to maintain, so it reads the population directly.

    Returns:
        DataFrame with columns [contract_id, performing_node], one row per award.
    """
    df = primes_df[["Award ID", "resolved_entity"]].dropna(subset=["Award ID"]).copy()
    df = df.drop_duplicates("Award ID")
    node = df["resolved_entity"].astype("string")
    df["performing_node"] = node.map(lambda u: alias.get(u, u) if pd.notna(u) else u)
    df = df.rename(columns={"Award ID": "contract_id"})
    return df[["contract_id", "performing_node"]].reset_index(drop=True)


def _cloud_platforms_per_record(attributed_df: pd.DataFrame) -> pd.Series:
    """
    Platforms each cloud record depends on: final_platform plus platform_mentions.

    Multi-platform records yield MULTIPLE platforms rather than a dollar split:
    exposure is a reachability question, so a contract depending on both AWS and
    Azure is fully exposed to each. This deliberately differs from
    `expand_platform_mentions()`, which splits dollars for HHI.
    """
    finals = attributed_df["final_platform"]
    mentions = attributed_df.get("platform_mentions")
    out = []
    for i in range(len(attributed_df)):
        plats = set()
        final = finals.iloc[i]
        if final in CONCRETE_PLATFORMS:
            plats.add(final)
        if mentions is not None:
            plats.update(
                m
                for m in _parse_platform_mentions(mentions.iloc[i])
                if m in CONCRETE_PLATFORMS
            )
        out.append(plats)
    return pd.Series(out, index=attributed_df.index)


def build_contract_deps(
    attributed_df: pd.DataFrame, alias: dict[str, str]
) -> pd.DataFrame:
    """
    Build per-contract direct dependencies with their tier offsets.

    Only tiers >= 1 are emitted; the tier-0 performing entity lives in the contract
    population instead, because it establishes identity rather than dependency.

    Returns:
        DataFrame with columns [contract_id, dep_node, tier, kind], where kind is
        "sub" (an observed subcontract) or "cloud" (an attributed platform).
    """
    rows: list[dict[str, Any]] = []

    subs = attributed_df[attributed_df["record_type"] == "subcontract"]
    subs = subs.dropna(subset=["original_prime_id", "contractor"])
    for cid, entity in zip(subs["original_prime_id"], subs["contractor"]):
        node = alias.get(entity, entity)
        rows.append(
            {
                "contract_id": cid,
                "dep_node": node,
                "tier": TIER_SUBCONTRACTOR,
                "kind": "sub",
            }
        )

    cloud = attributed_df[attributed_df["is_cloud"].fillna(False).astype(bool)]
    cloud = cloud.dropna(subset=["original_prime_id"])
    platforms = _cloud_platforms_per_record(cloud)
    for cid, record_type, plats in zip(
        cloud["original_prime_id"], cloud["record_type"], platforms
    ):
        tier = (
            TIER_PLATFORM_ON_SUBCONTRACT
            if record_type == "subcontract"
            else TIER_PLATFORM_ON_PRIME
        )
        for platform in plats:
            rows.append(
                {
                    "contract_id": cid,
                    "dep_node": platform_node(platform),
                    "tier": tier,
                    "kind": "cloud",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["contract_id", "dep_node", "tier", "kind"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def all_nodes(
    population: pd.DataFrame, edges: pd.DataFrame, deps: pd.DataFrame
) -> list[str]:
    """
    The node set, held IDENTICAL across both views so |V| is invariant.

    Includes every platform node whether or not it is currently reachable, so the
    normal view contains the same vertices as the clouded view.
    """
    nodes = set(population["performing_node"].dropna().astype(str))
    nodes |= set(edges["prime_node"].astype(str))
    nodes |= set(edges["sub_node"].astype(str))
    nodes |= set(deps["dep_node"].dropna().astype(str))
    nodes |= {platform_node(p) for p in CONCRETE_PLATFORMS}
    return sorted(nodes)


def build_entity_graph(nodes: list[str], edges: pd.DataFrame) -> nx.DiGraph:
    """
    Normal-view entity graph: X -> Y where X subcontracts to Y.

    Direction is dependent -> dependency, per the PDF. Disruption is evaluated by
    traversing the reverse graph. Edges are deduplicated (reachability is binary);
    the observation count is kept as an edge attribute for the pruning sensitivity.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    counts = edges.groupby(["prime_node", "sub_node"]).size()
    for (src, dst), weight in counts.items():
        graph.add_edge(str(src), str(dst), observations=int(weight), kind="sub")
    return graph


def add_cloud_edges(
    graph: nx.DiGraph, attributed_df: pd.DataFrame, alias: dict[str, str]
) -> nx.DiGraph:
    """
    Clouded-view entity graph: adds X -> PLATFORM:P for every entity X performing an
    attributed contract. Returns a copy; the input graph is left unchanged.
    """
    out = graph.copy()
    cloud = attributed_df[attributed_df["is_cloud"].fillna(False).astype(bool)]
    platforms = _cloud_platforms_per_record(cloud)
    for entity, plats in zip(cloud["contractor"], platforms):
        if pd.isna(entity):
            continue
        src = alias.get(entity, entity)
        for platform in plats:
            dst = platform_node(platform)
            if src != dst:
                out.add_edge(str(src), dst, kind="cloud")
    return out


def select_graph(
    view: View, spec: Spec, g_normal: nx.DiGraph, g_clouded: nx.DiGraph
) -> nx.DiGraph:
    """
    Pick the entity graph implied by (view, spec), keeping the specs symmetric.

    The conservative spec uses the NORMAL entity graph even in the clouded view:
    platform reachability comes solely from per-contract attribution, never from
    another contract of the same entity. The permissive spec uses the clouded graph.
    """
    if view == "normal":
        return g_normal
    return g_clouded if spec == "permissive" else g_normal


# =============================================================================
# SPARSE REACHABILITY (fast path)
# =============================================================================


def _adjacency(graph: nx.DiGraph, nodes: list[str]) -> sp.csr_matrix:
    """Boolean adjacency in `nodes` order: A[i, j] = i depends on j."""
    index = {n: i for i, n in enumerate(nodes)}
    rows, cols = [], []
    for src, dst in graph.edges():
        if src in index and dst in index:
            rows.append(index[src])
            cols.append(index[dst])
    n = len(nodes)
    data = np.ones(len(rows), dtype=bool)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=bool)


def _binarise(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Collapse an accumulated product back to boolean."""
    out = matrix.tocsr()
    out.data = np.ones_like(out.data, dtype=bool)
    out = out.astype(bool)
    out.eliminate_zeros()
    return out


def reachability_matrices(
    graph: nx.DiGraph, nodes: list[str], max_hops: int
) -> list[sp.csr_matrix]:
    """
    Boolean reachability within 0..max_hops entity hops.

    Returns:
        List `R` of length max_hops + 1 where R[e][i, j] is True if i reaches j in
        at most e hops. R[0] is the identity, so a dependency counts as reaching
        itself in zero hops.
    """
    adjacency = _adjacency(graph, nodes)
    identity = sp.identity(len(nodes), dtype=bool, format="csr")
    reach = [identity]
    frontier = identity
    for _ in range(max_hops):
        frontier = _binarise(frontier @ adjacency)
        reach.append(_binarise(reach[-1] + frontier))
    return reach


def _incidence(
    keys: np.ndarray, node_ids: np.ndarray, n_keys: int, n_nodes: int
) -> sp.csr_matrix:
    """Boolean incidence matrix from paired (row, column) index arrays."""
    data = np.ones(len(keys), dtype=bool)
    matrix = sp.csr_matrix(
        (data, (keys, node_ids)), shape=(n_keys, n_nodes), dtype=bool
    )
    return _binarise(matrix)


# =============================================================================
# ENTITY EXPOSURE (Analysis 1)
# =============================================================================


def entity_exposure(graph: nx.DiGraph, nodes: list[str], k: int) -> pd.Series:
    """
    I_k(v) over the entity population, for every node at once.

    Entity exposure is permissive by construction: the entity graph has no contracts
    to confine propagation to, so the conservative/permissive distinction does not
    arise (plan NQ1).

    Returns:
        Series indexed by node, values in [0, 1]. Self is excluded from both the
        numerator and the denominator (|V| - 1), per the PDF's "other nodes".
    """
    reach = reachability_matrices(graph, nodes, k)
    within = reach[k].tolil()
    within.setdiag(False)
    counts = np.asarray(within.tocsr().sum(axis=0)).ravel()
    denominator = max(len(nodes) - 1, 1)
    return pd.Series(counts / denominator, index=nodes, name=f"I{k}")


# =============================================================================
# CONTRACT EXPOSURE (Analysis 2)
# =============================================================================


def _contract_exposure_matrix(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
    k: int,
    spec: Spec,
    include_cloud_deps: bool,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Boolean (contract x node) exposure matrix, plus the identity incidence to exclude.

    A contract C is exposed to v within k tiers if some dependency (d, t) of C
    satisfies t + hops(d, v) <= k, i.e. R[k - t][d, v]. Under the permissive spec the
    performing entity also seeds traversal, at tier 0.
    """
    node_index = {n: i for i, n in enumerate(nodes)}
    contract_index = {c: i for i, c in enumerate(population["contract_id"])}
    n_contracts, n_nodes = len(contract_index), len(nodes)

    reach = reachability_matrices(graph, nodes, k)

    used = deps if include_cloud_deps else deps[deps["kind"] != "cloud"]
    exposed = sp.csr_matrix((n_contracts, n_nodes), dtype=bool)

    for tier, group in used.groupby("tier"):
        hops = k - int(tier)
        if hops < 0:
            continue
        rows = group["contract_id"].map(contract_index)
        cols = group["dep_node"].map(node_index)
        keep = rows.notna() & cols.notna()
        if not keep.any():
            continue
        incidence = _incidence(
            rows[keep].to_numpy(dtype=np.int64),
            cols[keep].to_numpy(dtype=np.int64),
            n_contracts,
            n_nodes,
        )
        exposed = _binarise(exposed + incidence @ reach[hops])

    perf_rows = population["contract_id"].map(contract_index)
    perf_cols = population["performing_node"].map(node_index)
    keep = perf_rows.notna() & perf_cols.notna()
    identity = _incidence(
        perf_rows[keep].to_numpy(dtype=np.int64),
        perf_cols[keep].to_numpy(dtype=np.int64),
        n_contracts,
        n_nodes,
    )

    if spec == "permissive":
        exposed = _binarise(exposed + identity @ reach[k])

    return exposed, identity


def contract_exposure(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
    k: int,
    spec: Spec = "conservative",
    include_cloud_deps: bool = True,
) -> pd.Series:
    """
    I_k(v) over the contract population, for every node at once.

    A node's own contracts are excluded from the numerator: D_k(v) is the set of
    OTHER nodes reaching v, and including them turns the metric into a contract-volume
    ranking (plan section 3.2).

    Returns:
        Series indexed by node, values in [0, 1].
    """
    exposed, identity = _contract_exposure_matrix(
        graph, nodes, population, deps, k, spec, include_cloud_deps
    )
    others = _binarise(exposed > identity)  # drop each node's own contracts
    counts = np.asarray(others.sum(axis=0)).ravel()
    denominator = max(len(population), 1)
    return pd.Series(counts / denominator, index=nodes, name=f"I{k}")


def contract_exposure_counts(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
    k: int,
    spec: Spec = "conservative",
    include_cloud_deps: bool = True,
) -> pd.Series:
    """Absolute contract counts rather than shares (plan section 4, your Q4 note)."""
    shares = contract_exposure(
        graph, nodes, population, deps, k, spec, include_cloud_deps
    )
    return (shares * len(population)).round().astype(int)


# =============================================================================
# NETWORKX REFERENCE IMPLEMENTATION (single target, for validation)
# =============================================================================


def contract_exposure_networkx(
    graph: nx.DiGraph,
    population: pd.DataFrame,
    deps: pd.DataFrame,
    target: str,
    k: int,
    spec: Spec = "conservative",
    include_cloud_deps: bool = True,
) -> int:
    """
    Reference implementation for one target, via reverse BFS.

    Deliberately straightforward and slow. `tests/test_exposure_network.py` asserts
    it agrees with the sparse path, which keeps the fast path trustworthy while
    leaving something step-throughable for debugging.
    """
    distances = nx.single_source_shortest_path_length(
        graph.reverse(copy=False), target, cutoff=k
    )

    own = set(population.loc[population["performing_node"] == target, "contract_id"])

    used = deps if include_cloud_deps else deps[deps["kind"] != "cloud"]
    exposed: set = set()
    for cid, dep, tier in zip(used["contract_id"], used["dep_node"], used["tier"]):
        hops = distances.get(dep)
        if hops is not None and hops + int(tier) <= k:
            exposed.add(cid)

    if spec == "permissive":
        for cid, perf in zip(population["contract_id"], population["performing_node"]):
            hops = distances.get(perf)
            if hops is not None and hops <= k:
                exposed.add(cid)

    return len(exposed - own)


def entity_exposure_networkx(graph: nx.DiGraph, target: str, k: int) -> int:
    """Reference implementation of entity exposure for one target."""
    distances = nx.single_source_shortest_path_length(
        graph.reverse(copy=False), target, cutoff=k
    )
    return sum(1 for node, hops in distances.items() if node != target and hops <= k)


# =============================================================================
# REPORTING
# =============================================================================


def leaderboard(
    normal: pd.Series,
    clouded: pd.Series,
    labels: dict[str, str],
    top: int = 15,
) -> pd.DataFrame:
    """
    Rank nodes by exposure within each view and lay the two rankings side by side.

    This is the primary output: the claim is the change in identity and magnitude of
    the most-exposed node, not a per-provider delta.
    """
    rows = []
    for view, series in [("normal", normal), ("clouded", clouded)]:
        ordered = series.sort_values(ascending=False).head(top)
        for position, (node, value) in enumerate(ordered.items(), start=1):
            rows.append(
                {
                    "view": view,
                    "rank": position,
                    "node": node,
                    "label": display_label(node, labels),
                    "is_platform": is_platform_node(node),
                    "exposure": value,
                }
            )
    return pd.DataFrame(rows)


def platform_table(
    normal: pd.Series, clouded: pd.Series, population_size: int
) -> pd.DataFrame:
    """Per-platform exposure in both views: absolute counts, shares, hidden exposure."""
    rows = []
    for platform in sorted(CONCRETE_PLATFORMS):
        node = platform_node(platform)
        if node not in normal.index:
            continue
        share_n = float(normal.get(node, 0.0))
        share_c = float(clouded.get(node, 0.0))
        rows.append(
            {
                "platform": platform,
                "normal_count": round(share_n * population_size),
                "clouded_count": round(share_c * population_size),
                "normal_pct": share_n * 100,
                "clouded_pct": share_c * 100,
                "hidden_pp": (share_c - share_n) * 100,
                "ratio": (share_c / share_n) if share_n > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("clouded_count", ascending=False)


def exposure_summary(series: pd.Series, graph: nx.DiGraph) -> dict[str, float]:
    """
    Network-level robustness summary (PDF section 4).

    Computed over nodes with at least one in-edge; the zero-exposure count is
    reported separately rather than silently deflating the mean and p95.
    """
    from clouded_deps.pipeline.baseline_merged_hhi import calculate_gini

    connected = [n for n in series.index if graph.in_degree(n) > 0]
    values = series.loc[connected]
    return {
        "n_scored": len(values),
        "n_zero_exposure": int((series == 0).sum()),
        "max": float(values.max()) if len(values) else 0.0,
        "p95": float(values.quantile(0.95)) if len(values) else 0.0,
        "mean": float(values.mean()) if len(values) else 0.0,
        "gini": calculate_gini(values[values > 0]) if (values > 0).any() else 0.0,
        "n_above_10pct": int((values > 0.10).sum()),
    }

"""Tests for the Nth-order dependency exposure metric."""

from itertools import pairwise

import networkx as nx
import pandas as pd
import pytest

from clouded_deps.pipeline import exposure_network as ex

# ---------------------------------------------------------------------------
# Fixture: a hand-built network with a known role-switching chain.
#
#   contracts          performing entity      subcontractor
#   C1                 A                      B
#   C2                 B                      P (= PLATFORM:Azure, as a party)
#   C3                 C                      -            (cloud, attributed Azure)
#   C4                 A                      -            (no deps at all)
#
# Entity edges (dependent -> dependency):  A -> B,  B -> PLATFORM:Azure
# So A reaches Azure in 2 hops via the role-switcher B: this is the multi-tier
# chain that only exists because B is a sub on C1 and a prime on C2.
# ---------------------------------------------------------------------------

AZURE = ex.platform_node("Azure")


@pytest.fixture
def population() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_id": ["C1", "C2", "C3", "C4"],
            "performing_node": ["A", "B", "C", "A"],
        }
    )


@pytest.fixture
def deps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"contract_id": "C1", "dep_node": "B", "tier": 1, "kind": "sub"},
            {"contract_id": "C2", "dep_node": AZURE, "tier": 1, "kind": "sub"},
            {"contract_id": "C3", "dep_node": AZURE, "tier": 1, "kind": "cloud"},
        ]
    )


@pytest.fixture
def nodes() -> list[str]:
    return sorted(["A", "B", "C", AZURE])


@pytest.fixture
def graph(nodes: list[str]) -> nx.DiGraph:
    edges = pd.DataFrame(
        [
            {"prime_node": "A", "sub_node": "B"},
            {"prime_node": "B", "sub_node": AZURE},
        ]
    )
    return ex.build_entity_graph(nodes, edges)


# ---------------------------------------------------------------------------
# Exact values on the fixture
# ---------------------------------------------------------------------------


def test_contract_exposure_exact_values(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """Azure's exposure grows with k as the role-switching chain unfolds."""
    counts = {
        k: ex.contract_exposure_counts(graph, nodes, population, deps, k=k)[AZURE]
        for k in (1, 2, 3)
    }
    # k=1: C2 (Azure is a recorded subcontractor) and C3 (attributed cloud dep).
    assert counts[1] == 2
    # k=2: adds C1, whose subcontractor B depends on Azure one hop further out.
    assert counts[2] == 3
    # k=3: nothing new -- C4 has no dependencies at all.
    assert counts[3] == 3


def test_identity_term_excluded(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """A node's own contracts never count toward its exposure (plan section 3.2)."""
    # A performs C1 and C4. Under the permissive spec A reaches nothing that leads
    # back to itself, so its exposure must be 0 despite performing two contracts.
    for spec in ("conservative", "permissive"):
        counts = ex.contract_exposure_counts(
            graph, nodes, population, deps, k=3, spec=spec
        )
        assert counts["A"] == 0, spec


def test_tier_zero_does_not_contribute(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """Only tiers >= 1 enter the numerator."""
    # C performs C3 and nothing depends on C, so C's exposure is 0 at every k.
    counts = ex.contract_exposure_counts(graph, nodes, population, deps, k=3)
    assert counts["C"] == 0


def test_cloud_deps_can_be_excluded(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """The normal view drops cloud dependencies, losing C3."""
    normal = ex.contract_exposure_counts(
        graph, nodes, population, deps, k=1, include_cloud_deps=False
    )
    clouded = ex.contract_exposure_counts(
        graph, nodes, population, deps, k=1, include_cloud_deps=True
    )
    assert normal[AZURE] == 1  # C2 only
    assert clouded[AZURE] == 2  # C2 + C3


def test_entity_exposure_exact_values(graph: nx.DiGraph, nodes: list[str]) -> None:
    """Entity exposure counts other entities, over a |V| - 1 denominator."""
    # A and B both reach Azure within 2 hops; 4 nodes total.
    assert ex.entity_exposure(graph, nodes, k=1)[AZURE] == pytest.approx(1 / 3)
    assert ex.entity_exposure(graph, nodes, k=2)[AZURE] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_monotonic_in_k(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """I_k is non-decreasing in k."""
    series = [
        ex.contract_exposure(graph, nodes, population, deps, k=k) for k in (1, 2, 3, 4)
    ]
    for earlier, later in pairwise(series):
        assert (later >= earlier - 1e-12).all()


def test_clouded_dominates_normal(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """The clouded view can only ever add exposure, never remove it."""
    for k in (1, 2, 3):
        normal = ex.contract_exposure(
            graph, nodes, population, deps, k=k, include_cloud_deps=False
        )
        clouded = ex.contract_exposure(
            graph, nodes, population, deps, k=k, include_cloud_deps=True
        )
        assert (clouded >= normal - 1e-12).all()


def test_permissive_dominates_conservative(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """The permissive spec is an upper bound on the conservative one."""
    for k in (1, 2, 3):
        conservative = ex.contract_exposure(
            graph, nodes, population, deps, k=k, spec="conservative"
        )
        permissive = ex.contract_exposure(
            graph, nodes, population, deps, k=k, spec="permissive"
        )
        assert (permissive >= conservative - 1e-12).all()


def test_node_set_identical_across_views(graph: nx.DiGraph) -> None:
    """|V| must be invariant, or the comparison confounds topology with population."""
    attributed = pd.DataFrame(
        {
            "contractor": ["C"],
            "record_type": ["prime_no_subs"],
            "original_prime_id": ["C3"],
            "is_cloud": [True],
            "final_platform": ["Azure"],
            "platform_mentions": ["['Azure']"],
        }
    )
    clouded = ex.add_cloud_edges(graph, attributed, alias={})
    assert set(clouded.nodes()) == set(graph.nodes())
    assert clouded.number_of_edges() >= graph.number_of_edges()


def test_cycles_terminate() -> None:
    """Reachability over a cycle must not hang or double-count."""
    edges = pd.DataFrame(
        [
            {"prime_node": "A", "sub_node": "B"},
            {"prime_node": "B", "sub_node": "C"},
            {"prime_node": "C", "sub_node": "A"},
        ]
    )
    cyclic_nodes = sorted(["A", "B", "C"])
    graph = ex.build_entity_graph(cyclic_nodes, edges)
    # Every other node reaches every node once the cycle is traversed.
    assert ex.entity_exposure(graph, cyclic_nodes, k=5)["A"] == pytest.approx(1.0)


def test_repeated_paths_counted_once() -> None:
    """Two distinct paths to the same node yield one unit of exposure."""
    edges = pd.DataFrame(
        [
            {"prime_node": "A", "sub_node": "B"},
            {"prime_node": "A", "sub_node": "C"},
            {"prime_node": "B", "sub_node": "D"},
            {"prime_node": "C", "sub_node": "D"},
        ]
    )
    diamond = sorted(["A", "B", "C", "D"])
    graph = ex.build_entity_graph(diamond, edges)
    # A, B, C all reach D -- A via two distinct paths, counted once.
    assert ex.entity_exposure(graph, diamond, k=3)["D"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# networkx / scipy parity
# ---------------------------------------------------------------------------


def test_parity_on_fixture(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """The sparse fast path must agree with the networkx reference."""
    for spec in ("conservative", "permissive"):
        for include_cloud in (True, False):
            for k in (1, 2, 3):
                fast = ex.contract_exposure_counts(
                    graph,
                    nodes,
                    population,
                    deps,
                    k=k,
                    spec=spec,
                    include_cloud_deps=include_cloud,
                )
                for target in nodes:
                    slow = ex.contract_exposure_networkx(
                        graph,
                        population,
                        deps,
                        target,
                        k=k,
                        spec=spec,
                        include_cloud_deps=include_cloud,
                    )
                    assert fast[target] == slow, (spec, include_cloud, k, target)


def test_entity_parity_on_fixture(graph: nx.DiGraph, nodes: list[str]) -> None:
    """Entity exposure fast path must agree with the networkx reference."""
    for k in (1, 2, 3):
        fast = ex.entity_exposure(graph, nodes, k=k)
        for target in nodes:
            slow = ex.entity_exposure_networkx(graph, target, k=k)
            assert fast[target] * (len(nodes) - 1) == pytest.approx(slow), (k, target)

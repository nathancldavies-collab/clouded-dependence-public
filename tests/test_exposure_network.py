"""Tests for the Nth-order dependency exposure metric."""

from itertools import pairwise

import networkx as nx
import numpy as np
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


# ---------------------------------------------------------------------------
# Barbell (paired) output
# ---------------------------------------------------------------------------


def _paired() -> tuple[pd.Series, pd.Series, dict[str, str]]:
    """Two views whose top-2 sets overlap in only one node."""
    normal = pd.Series({"A": 0.50, "B": 0.40, "C": 0.10, "D": 0.00})
    clouded = pd.Series({"A": 0.50, "B": 0.45, "C": 0.90, "D": 0.80})
    return normal, clouded, {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"}


def test_barbell_takes_union_of_both_top_sets() -> None:
    """A node in only one view's top-N still appears, with both endpoints."""
    normal, clouded, labels = _paired()
    out = ex.barbell_data(normal, clouded, labels, population_size=100, top=2)
    # normal top-2 = {A, B}; clouded top-2 = {C, D}; union has all four.
    assert set(out["node"]) == {"A", "B", "C", "D"}
    assert len(out) == 4


def test_barbell_carries_counterpart_values() -> None:
    """Nodes present only in the clouded top-N keep their normal-view value."""
    normal, clouded, labels = _paired()
    out = ex.barbell_data(normal, clouded, labels, population_size=100, top=2)
    row = out[out["node"] == "D"].iloc[0]
    assert row["normal"] == pytest.approx(0.0)  # not in normal's top 2
    assert row["clouded"] == pytest.approx(0.8)
    assert not row["in_top_normal"]
    assert row["in_top_clouded"]


def test_barbell_counts_and_deltas() -> None:
    """Counts scale by the population and deltas are clouded minus normal."""
    normal, clouded, labels = _paired()
    out = ex.barbell_data(normal, clouded, labels, population_size=100, top=4)
    row = out[out["node"] == "C"].iloc[0]
    assert row["normal_count"] == 10
    assert row["clouded_count"] == 90
    assert row["delta_count"] == 80
    assert row["ratio"] == pytest.approx(9.0)
    assert row["label"] == "Gamma"


def test_barbell_ranks_come_from_the_full_series() -> None:
    """Ranks reflect position across all nodes, not just the selected union."""
    normal, clouded, labels = _paired()
    out = ex.barbell_data(normal, clouded, labels, population_size=100, top=2)
    ranks = out.set_index("node")
    assert ranks.loc["A", "rank_normal"] == 1
    assert ranks.loc["C", "rank_clouded"] == 1
    assert ranks.loc["C", "rank_normal"] == 3


def test_barbell_ratio_is_nan_when_normal_is_zero() -> None:
    """A node invisible in the normal view has no finite ratio."""
    normal, clouded, labels = _paired()
    out = ex.barbell_data(normal, clouded, labels, population_size=100, top=4)
    assert pd.isna(out[out["node"] == "D"].iloc[0]["ratio"])


# ---------------------------------------------------------------------------
# Exposed contract value (weighted readout)
# ---------------------------------------------------------------------------


def test_unit_weights_reproduce_the_count_exactly(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """`weights = ones` must recover I_k -- the two readouts are one measure."""
    ones_c = np.ones(len(population))
    ones_e = np.ones(len(nodes))
    for k in (1, 2, 3):
        counts = ex.contract_exposure(graph, nodes, population, deps, k=k)
        weighted = ex.contract_exposure(
            graph, nodes, population, deps, k=k, weights=ones_c
        )
        assert np.allclose(counts.to_numpy(), weighted.to_numpy())

        ent_counts = ex.entity_exposure(graph, nodes, k=k)
        ent_weighted = ex.entity_exposure(graph, nodes, k=k, weights=ones_e)
        # Entity count divides by |V| - 1, the weighted form by sum(w) = |V|.
        assert np.allclose(
            ent_counts.to_numpy() * (len(nodes) - 1),
            ent_weighted.to_numpy() * len(nodes),
        )


def test_value_readout_weights_by_award_size(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """A large exposed award moves S_k far more than a small one moves it."""
    # C2 and C3 are Azure-exposed at k=1; make C2 dominate by value.
    weights = np.array([1.0, 97.0, 1.0, 1.0])  # C1, C2, C3, C4
    value = ex.contract_exposure(graph, nodes, population, deps, k=1, weights=weights)
    count = ex.contract_exposure(graph, nodes, population, deps, k=1)
    # By count Azure sits atop 2 of 4 contracts; by value, 98 of 100 dollars.
    assert count[AZURE] == pytest.approx(0.5)
    assert value[AZURE] == pytest.approx(0.98)


def test_value_readout_excludes_a_nodes_own_contracts(
    graph: nx.DiGraph,
    nodes: list[str],
    population: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    """Self-exclusion carries over to the weighted form (plan section 3.6)."""
    # A performs C1 and C4; give them all the value. A's own value must not count.
    weights = np.array([50.0, 0.0, 0.0, 50.0])
    value = ex.contract_exposure(
        graph, nodes, population, deps, k=3, spec="permissive", weights=weights
    )
    assert value["A"] == pytest.approx(0.0)


def test_negative_award_values_are_clamped(population: pd.DataFrame) -> None:
    """A negative obligation must never reduce an exposure total."""
    primes = pd.DataFrame(
        {
            "Award ID": ["C1", "C2", "C3", "C4"],
            "Total Dollars Obligated": [100.0, -25.0, 0.0, 50.0],
        }
    )
    weights = ex.build_contract_weights(primes, population)
    assert (weights >= 0).all()
    assert weights.tolist() == [100.0, 0.0, 0.0, 50.0]


def test_entity_weights_align_to_node_order(nodes: list[str]) -> None:
    """Entity weights come back in `nodes` order, with absent nodes at zero."""
    attributed = pd.DataFrame(
        {
            "contractor": ["A", "A", "B", "ZZZ"],
            "dollars": [10.0, 5.0, 7.0, 99.0],
        }
    )
    weights = ex.build_entity_weights(attributed, alias={}, nodes=nodes)
    by_node = dict(zip(nodes, weights))
    assert by_node["A"] == pytest.approx(15.0)
    assert by_node["B"] == pytest.approx(7.0)
    assert by_node["C"] == pytest.approx(0.0)  # present as a node, absent from data

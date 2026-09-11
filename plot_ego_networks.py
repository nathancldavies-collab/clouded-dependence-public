#!/usr/bin/env python3
"""
Ego Networks: Normal-view vs Clouded-view Most-Exposed Node
============================================================
Two ego networks side by side, at entity level:

  LEFT   the normal view's most-exposed node (Carahsoft) and the entities that
         depend on it within k tiers
  RIGHT  the clouded view's most-exposed node (Azure) and the same

Each panel draws the CLOUDED neighbourhood, coloured by whether the contracting
record alone would have revealed it:

  grey       also reachable in the normal view — visible without attribution
  dark blue  reachable only in the clouded view — revealed by platform attribution

Node area is proportional to the dollars that entity holds. Only the ego is
labelled. Rings are tier distance: inner = 1 tier, outer = 2 tiers.

Run from project root:
    uv run plot_ego_networks.py [--k 2]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from clouded_deps.directories import DATA_DIR, OUTPUTS_DIR
from clouded_deps.pipeline import exposure_network as ex
from clouded_deps.pipeline.create_merged_dataset import load_primes
from plot_exposure_barbell import humanise

NORMAL_FILL = "#9a9a95"
CLOUDED_FILL = "#17538f"
EDGE_COLOR = "#d5d4cf"
EGO_FILL = "#0b0b0b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

# Nodes per unit area, held CONSTANT across both panels. This is what gives the
# layout its gravity: a tier's annulus is sized so its area is proportional to the
# number of nodes in it, so the disc a node set occupies grows with the count and
# the two panels are directly comparable by eye. Carahsoft's 106 dependents occupy
# roughly half the diameter of Azure's 429.
NODE_DENSITY = 27.4
CENTRE_HOLE = 0.52  # keeps the ego marker and its label clear of the first tier
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
SEED = 20260911


def _load() -> tuple:
    """Build the entity graphs, labels and dollar weights."""
    primes = load_primes(
        DATA_DIR / "02_processed/01_filtered/prime_services_filtered.csv"
    )
    attributed = pd.read_csv(
        DATA_DIR / "02_processed/03_classified/attributed_dataset.csv", low_memory=False
    )
    subs = pd.read_csv(
        DATA_DIR / "01_raw/subawards.csv", encoding="utf-8-sig", low_memory=False
    )

    alias = ex.build_platform_alias(primes, subs)
    labels = ex.build_entity_labels(primes, subs, alias)
    population = ex.build_contract_population(primes, alias)
    edges = ex.resolve_subaward_endpoints(subs, alias)
    deps = ex.build_contract_deps(attributed, alias)
    nodes = ex.all_nodes(population, edges, deps)

    g_normal = ex.build_entity_graph(nodes, edges)
    g_clouded = ex.add_cloud_edges(g_normal, attributed, alias)
    weights = ex.build_entity_weights(attributed, alias, nodes)
    return g_normal, g_clouded, labels, dict(zip(nodes, weights))


def ego_members(
    g_normal: nx.DiGraph, g_clouded: nx.DiGraph, target: str, k: int
) -> tuple[dict[str, int], set[str]]:
    """
    Who depends on `target` within k tiers, and which of them the normal view sees.

    Traversal is on the REVERSED graph: edges point dependent -> dependency, so the
    dependents of `target` are its ancestors.

    Returns:
        (clouded members mapped to their tier distance, subset also in the normal view)
    """
    clouded = nx.single_source_shortest_path_length(
        g_clouded.reverse(copy=False), target, cutoff=k
    )
    normal = nx.single_source_shortest_path_length(
        g_normal.reverse(copy=False), target, cutoff=k
    )
    members = {n: d for n, d in clouded.items() if n != target and d >= 1}
    seen_normally = {n for n in members if n in normal and n != target}
    return members, seen_normally


def annulus_layout(
    members: dict[str, int], inner_first: set[str] | None = None
) -> tuple[dict, float]:
    """
    Place each member in a band packed at constant density, visibility band first.

    Bands are ordered by VISIBILITY first and tier second: every entity the
    contracting record already reveals occupies the inner disc (ordered by tier
    within it), and the clouded-only ones form the shell outside it. Each band b gets
    the annulus between r[b-1] and r[b], with r chosen so the annulus AREA is
    proportional to the number of nodes in it:

        r[b] = sqrt(r[b-1]^2 + n[b] / (pi * NODE_DENSITY))

    Because the density constant is shared across panels, the radius a node set
    reaches encodes how many nodes it has -- a small neighbourhood draws a small
    disc, and the outer radius is unchanged by how the nodes are banded. Within an
    annulus, nodes are placed by phyllotaxis (golden angle, radius by equal-area
    quantile), which fills it evenly without banding or the visible seam a fixed
    angular step leaves.

    Returns:
        (node -> (x, y), outer radius of the whole disc)
    """
    pos: dict[str, tuple[float, float]] = {}
    priority = inner_first or set()
    bands = [
        sorted(
            (n for n in members if (n in priority) == visible),
            key=lambda n: (members[n], str(n)),
        )
        for visible in (True, False)
    ]
    inner = CENTRE_HOLE
    index_offset = 0
    for band in bands:
        count = len(band)
        if count == 0:
            continue
        outer = np.sqrt(inner**2 + count / (np.pi * NODE_DENSITY))
        for i, node in enumerate(band):
            # Equal-area radial quantile keeps density flat across the annulus.
            frac = (i + 0.5) / count
            radius = np.sqrt(inner**2 + frac * (outer**2 - inner**2))
            angle = (index_offset + i) * GOLDEN_ANGLE
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
        index_offset += count
        inner = outer
    return pos, inner


def _sizes(members: dict[str, int], weights: dict[str, float], scale: float) -> list:
    """Node area proportional to dollars held, with a visible floor."""
    return [12.0 + scale * np.sqrt(max(weights.get(n, 0.0), 0.0)) for n in members]


def draw_ego(
    ax: plt.Axes,
    graph: nx.DiGraph,
    members: dict[str, int],
    seen_normally: set[str],
    target: str,
    labels: dict[str, str],
    weights: dict[str, float],
    size_scale: float,
    axis_limit: float,
) -> None:
    """Draw one ego network panel."""
    pos, _ = annulus_layout(members, seen_normally)
    pos[target] = (0.0, 0.0)

    induced = graph.subgraph([target, *members]).copy()
    nx.draw_networkx_edges(
        induced,
        pos,
        ax=ax,
        edge_color=EDGE_COLOR,
        width=0.45,
        alpha=0.3,
        arrows=False,
    )

    order = list(members)
    ax.scatter(
        [pos[n][0] for n in order],
        [pos[n][1] for n in order],
        s=_sizes(members, weights, size_scale),
        c=[NORMAL_FILL if n in seen_normally else CLOUDED_FILL for n in order],
        edgecolors=SURFACE,
        linewidths=0.6,
        zorder=3,
    )

    ax.scatter(
        [0], [0], s=420, c=EGO_FILL, edgecolors=SURFACE, linewidths=2.5, zorder=5
    )
    raw = ex.display_label(target, labels)
    ax.annotate(
        humanise(raw, ex.is_platform_node(target)),
        xy=(0, 0),
        xytext=(0, -26),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=TEXT_PRIMARY,
        zorder=6,
        bbox={"boxstyle": "round,pad=0.34", "fc": SURFACE, "ec": "none", "alpha": 0.92},
    )

    # Shared limits across panels, so the disc sizes are comparable by eye.
    ax.set_aspect("equal")
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.axis("off")


def plot_ego_networks(left: str, right: str, k: int, output_path: Path) -> Path:
    """Render both ego networks with a shared size scale and save."""
    g_normal, g_clouded, labels, weights = _load()

    panels = []
    for target in (left, right):
        members, seen = ego_members(g_normal, g_clouded, target, k)
        panels.append((target, members, seen))
        print(
            f"  {ex.display_label(target, labels):28s} "
            f"{len(members):4d} dependents  ({len(seen)} in normal view, "
            f"{len(members) - len(seen)} clouded-only)"
        )

    # One size scale across both panels, so areas are comparable between them.
    peak = max(
        (max((weights.get(n, 0.0) for n in m), default=0.0) for _, m, _ in panels),
        default=1.0,
    )
    size_scale = 300.0 / np.sqrt(peak) if peak > 0 else 1.0

    radii = [annulus_layout(m, seen)[1] for _, m, seen in panels]
    axis_limit = max(radii) * 1.10

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    for ax, (target, members, seen), radius in zip(axes, panels, radii):
        draw_ego(
            ax,
            g_clouded,
            members,
            seen,
            target,
            labels,
            weights,
            size_scale,
            axis_limit,
        )
        print(f"    disc radius {radius:.2f} for {len(members)} entities")

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=12,
            markerfacecolor=NORMAL_FILL,
            markeredgecolor=SURFACE,
            label=lbl,
        )
        for lbl in ["Visible in the contracting record"]
    ] + [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=12,
            markerfacecolor=CLOUDED_FILL,
            markeredgecolor=SURFACE,
            label="Revealed only by platform attribution",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=17,
            markerfacecolor="none",
            markeredgecolor=TEXT_SECONDARY,
            label="Node area = dollars held by that entity",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        ncol=3,
        fontsize=14,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.subplots_adjust(top=0.97, bottom=0.1, left=0.02, right=0.98, wspace=0.05)

    # Dashed rule separating the two views.
    fig.add_artist(
        Line2D(
            [0.5, 0.5],
            [0.09, 0.97],
            transform=fig.transFigure,
            color=TEXT_SECONDARY,
            linestyle=(0, (6, 6)),
            linewidth=1.1,
            alpha=0.55,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--k", type=int, default=2, help="Tier depth (default: 2)")
    parser.add_argument(
        "--left",
        default=None,
        help="Left ego node id (default: the normal view's top node)",
    )
    parser.add_argument(
        "--right",
        default=ex.platform_node("Azure"),
        help="Right ego node id (default: PLATFORM:Azure)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUTS_DIR / "exposure" / "ego_networks.png",
        help="Output PNG path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    left = args.left
    if left is None:
        board = pd.read_csv(
            OUTPUTS_DIR / "exposure" / "leaderboard_entity_permissive_count.csv"
        )
        left = board[board["view"] == "normal"].sort_values("rank").iloc[0]["node"]
    saved = plot_ego_networks(left, args.right, args.k, args.out)
    print(f"\n  Saved ego networks: {saved}")

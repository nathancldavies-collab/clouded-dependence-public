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

The figure is sized to print at the same height as the barbell when the two sit
side by side in LaTeX (see clouded_deps.plot_style), so run
plot_exposure_barbell.py first.

Run from project root:
    uv run plot_ego_networks.py [--k 3] [--match barbell.pdf]
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
from clouded_deps.plot_style import (
    BARBELL_WIDTH_FRAC,
    DEFAULT_K,
    EGO_WIDTH_FRAC,
    NAIVE_COLOR,
    figure_size_in,
    partner_size_in,
)
from plot_exposure_barbell import humanise

NORMAL_FILL = NAIVE_COLOR
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

# Marker areas are in points^2, so they do not scale with the axes. They were tuned
# at this many inches per layout unit; at any other scale they are rescaled by the
# square of the ratio, so nodes keep the same size relative to the disc.
REFERENCE_INCHES_PER_UNIT = 1.36
AXIS_PAD = 1.10  # panel half-extent as a multiple of its disc radius
PANEL_GAP_IN = 0.15
LEGEND_IN = 0.8  # two legend rows at LEGEND_FONTSIZE
TOP_PAD_IN = 0.05
EGO_LABEL_FONTSIZE = 14
# Ego labels sit over the disc, so long legal names are shortened.
EGO_SHORT_NAMES = {"Carahsoft Technology Corp.": "Carahsoft"}
LEGEND_FONTSIZE = 12  # the two views must share one row at this width


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
    half_width: float,
    half_height: float,
    area_factor: float,
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
        s=[area_factor * s for s in _sizes(members, weights, size_scale)],
        c=[NORMAL_FILL if n in seen_normally else CLOUDED_FILL for n in order],
        edgecolors=SURFACE,
        linewidths=0.6,
        zorder=3,
    )

    ax.scatter(
        [0],
        [0],
        s=420 * area_factor,
        c=EGO_FILL,
        edgecolors=SURFACE,
        linewidths=2.5,
        zorder=5,
    )
    raw = ex.display_label(target, labels)
    name = humanise(raw, ex.is_platform_node(target))
    ax.annotate(
        EGO_SHORT_NAMES.get(name, name),
        xy=(0, 0),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=EGO_LABEL_FONTSIZE,
        fontweight="bold",
        color=TEXT_PRIMARY,
        zorder=6,
        bbox={"boxstyle": "round,pad=0.34", "fc": SURFACE, "ec": "none", "alpha": 0.92},
    )

    # Panels share one data-to-inch scale, so the disc sizes are comparable by eye.
    ax.set_aspect("equal")
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_height, half_height)
    ax.axis("off")


def plot_ego_networks(
    left: str, right: str, k: int, output_path: Path, figsize: tuple[float, float]
) -> Path:
    """
    Render both ego networks with a shared size scale and save at exactly `figsize`.

    Each panel is as wide as its own disc, at one data-to-inch scale shared by both,
    so the smaller neighbourhood does not waste width on empty margin.
    """
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
    fig_w, fig_h = figsize
    avail_w = fig_w - PANEL_GAP_IN
    avail_h = fig_h - LEGEND_IN - TOP_PAD_IN
    inches_per_unit = min(
        avail_w / (2 * AXIS_PAD * sum(radii)),
        avail_h / (2 * AXIS_PAD * max(radii)),
    )
    area_factor = (inches_per_unit / REFERENCE_INCHES_PER_UNIT) ** 2
    half_height = AXIS_PAD * max(radii)
    panel_h = 2 * half_height * inches_per_unit
    panel_ws = [2 * AXIS_PAD * r * inches_per_unit for r in radii]

    fig = plt.figure(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # Centre the pair horizontally, and the panels in the space above the legend.
    x = (fig_w - sum(panel_ws) - PANEL_GAP_IN) / 2
    y = LEGEND_IN + (avail_h - panel_h) / 2
    axes = []
    for width in panel_ws:
        axes.append(
            fig.add_axes((x / fig_w, y / fig_h, width / fig_w, panel_h / fig_h))
        )
        x += width + PANEL_GAP_IN
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
            AXIS_PAD * radius,
            half_height,
            area_factor,
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
        for lbl in ["Naive view (contracting record)"]
    ] + [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=12,
            markerfacecolor=CLOUDED_FILL,
            markeredgecolor=SURFACE,
            label="Unclouded view (platform attribution)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=17,
            markerfacecolor="none",
            markeredgecolor=TEXT_SECONDARY,
            label="Node area = dollars held",
        ),
    ]
    # The two views share a row; the size key sits on its own row beneath.
    legend_style = {
        "frameon": False,
        "loc": "lower center",
        "fontsize": LEGEND_FONTSIZE,
        "labelcolor": TEXT_SECONDARY,
        "handletextpad": 0.3,
        "columnspacing": 1.0,
    }
    views = fig.legend(
        handles=handles[:2],
        ncol=2,
        bbox_to_anchor=(0.5, 0.4 / fig_h),
        **legend_style,
    )
    fig.add_artist(views)
    fig.legend(handles=handles[2:], bbox_to_anchor=(0.5, 0.0), **legend_style)

    # Dashed rule separating the two views, centred in the gap between panels.
    gap_x = (axes[0].get_position().x1 + axes[1].get_position().x0) / 2
    fig.add_artist(
        Line2D(
            [gap_x, gap_x],
            [y / fig_h, (y + panel_h) / fig_h],
            transform=fig.transFigure,
            color=TEXT_SECONDARY,
            linestyle=(0, (6, 6)),
            linewidth=1.1,
            alpha=0.55,
        )
    )

    # No tight bbox: the saved page must be exactly `figsize` for the heights to
    # match the barbell in LaTeX.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help=f"Tier depth (default: {DEFAULT_K})"
    )
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
        default=None,
        help="Output path (default: outputs/exposure/ego_networks.<file-type>)",
    )
    parser.add_argument(
        "--file-type",
        choices=["png", "pdf"],
        default="png",
        help="Output file format when --out is not given (default: png)",
    )
    parser.add_argument(
        "--match",
        type=Path,
        default=None,
        help=(
            "Barbell figure to match in height when set side by side in LaTeX "
            "(default: outputs/exposure/exposure_barbell_entity_value_k<k>.<file-type>)"
        ),
    )
    # Ignore unknown flags so `just plot-exposure` can pass the barbell's args here too.
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    left = args.left
    if left is None:
        board = pd.read_csv(
            OUTPUTS_DIR / "exposure" / "leaderboard_entity_permissive_count.csv"
        )
        left = board[board["view"] == "normal"].sort_values("rank").iloc[0]["node"]
    out = args.out or OUTPUTS_DIR / "exposure" / f"ego_networks.{args.file_type}"
    match = args.match or (
        OUTPUTS_DIR
        / "exposure"
        / f"exposure_barbell_entity_value_k{args.k}.{args.file_type}"
    )
    if not match.exists():
        raise SystemExit(f"Missing {match}\nRun plot_exposure_barbell.py first.")
    figsize = partner_size_in(figure_size_in(match), BARBELL_WIDTH_FRAC, EGO_WIDTH_FRAC)
    print(f"  Matching {match.name}: {figsize[0]:.2f} x {figsize[1]:.2f} in")
    saved = plot_ego_networks(left, args.right, args.k, out, figsize)
    print(f"\n  Saved ego networks: {saved}")

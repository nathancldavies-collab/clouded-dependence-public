#!/usr/bin/env python3
"""
Node Diagram: One Buyer's Suppliers in the Naive and Unclouded Views
====================================================================
A qualitative example for the persistence analysis: one awarding office over
a few consecutive fiscal years, drawn as two rows of nodes.

  - Top (naive view): a node per contractor the office paid for cloud each
    year. The contractor changes every year, so nothing looks retained.
  - Bottom (unclouded view): the same records keyed by platform. The platform
    behind those contractors stays the same, so it is retained every year.

Thin vertical ties link each contractor to the platform it is keyed by; an
arrow between years marks a supplier retained from one year to the next.

The example is found, not hand-picked. A candidate is an office and window
where one platform is present every year, every record on it is directly
attributed (no imputation), and the contractors behind it share none from one
year to the next. Candidates are ranked by how few other records the office
has in the window (the fewer, the cleaner the picture), then by dollars on
the platform. The top candidate for three years is GSA Region 10, FY2021-2023,
whose Microsoft support contract for the Office of Special Counsel ran through
a different reseller each year.

The page is saved at exactly the size of `plot_persistence.py`'s figure, so
the two sit side by side at the same scale; the gap between the rows
stretches to fill the height.

Run from project root (after `uv run plot_persistence.py`):
    uv run plot_persistence_example.py [--years 3] [--rank 1] [--buyer "..."]
                                       [--start 2021] [--file-type pdf] [--find]

Design notes:
  - The naive row is the grey baseline, as in the other figures. Contractors
    are told apart by shape, so the churn reads without colour.
  - The platform followed takes the unclouded blue; every other supplier is
    grey, since it is context, not the story.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Patch, RegularPolygon
from matplotlib.transforms import Bbox

from clouded_deps.directories import DATA_DIR, OUTPUTS_DIR
from clouded_deps.plot_style import (
    NAIVE_COLOR,
    SPINE,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    UNCLOUDED_COLOR,
    figure_size_in,
)
from run_persistence_analysis import SCOPES, load_scope

COLUMN_WIDTH = 2.4  # data units from a year's last node to the next year's first
NODE_GAP = 1.6  # between suppliers in the same year
NODE_RADIUS = 0.26
VIEWS = ("naive", "unclouded")  # top row, bottom row
ROW_LABELS = {
    "naive": "Naive view\n(by contractor)",
    "unclouded": "Unclouded view\n(by platform)",
}
LABEL_WIDTH = 14  # characters before a supplier name wraps
LABEL_GAP = 4  # points between a node and its label
FONT_SIZE = 11

# Naive-row shapes in order of first appearance: (vertices, orientation, size).
# None draws a circle. Sizes are scaled by eye so the shapes look equally heavy.
SHAPES = [
    (None, 0.0, 1.0),  # circle
    (4, np.pi / 4, 1.15),  # square
    (3, 0.0, 1.35),  # triangle
    (4, 0.0, 1.3),  # diamond
    (6, 0.0, 1.1),  # hexagon
    (5, 0.0, 1.15),  # pentagon
]

# Legal-form suffixes dropped from contractor names on the figure.
SUFFIXES = (
    "LIMITED LIABILITY COMPANY",
    "PROFESSIONAL SERVICES",
    "CORPORATION",
    "L.L.C",
    "L.P",
    "INC",
    "LLC",
    "CORP",
)
# Short words that stay lower case, and acronyms title case would mangle.
LOWER_WORDS = {"And", "Of", "The", "For"}
ACRONYMS = {"MERP"}


def short_name(name: str) -> str:
    """
    A contractor's name in title case, without its legal-form suffix.

    Words of up to three letters (REI, IBM) are taken as acronyms and kept.
    """
    name = name.strip().rstrip(",. ")
    stripped = True
    while stripped:
        stripped = False
        for suffix in SUFFIXES:
            if name.upper().endswith(" " + suffix):
                name = name[: -len(suffix)].rstrip(",. ")
                stripped = True
    words = []
    for i, word in enumerate(name.split()):
        titled = word.title()
        if i and titled in LOWER_WORDS:
            titled = titled.lower()
        elif word in ACRONYMS or (len(word) <= 3 and word.isalpha()):
            titled = word
        words.append(titled)
    return " ".join(words)


def format_dollars(value: float) -> str:
    """Dollars as $1.2M or $325K."""
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value / 1e3:.0f}K"


def load_names(data_path: Path) -> pd.Series:
    """Contractor name by UEI."""
    names = pd.read_csv(data_path, usecols=["contractor", "contractor_name"])
    return names.drop_duplicates("contractor").set_index("contractor")[
        "contractor_name"
    ]


def find_examples(records: pd.DataFrame, years: int) -> pd.DataFrame:
    """
    Offices and windows of `years` fiscal years where one platform is present
    every year, all of its records are attributed, and the contractors behind
    it share none from one year to the next. Best candidates first.
    """
    rows = []
    for buyer, office in records.groupby("buyer"):
        for start in sorted(office["fiscal_year"].unique()):
            span = office[office["fiscal_year"].between(start, start + years - 1)]
            if span["fiscal_year"].nunique() < years:
                continue
            platforms = span[span["key_source"] != "contractor"]
            for platform, group in platforms.groupby("unclouded"):
                if not (group["key_source"] == "attributed").all():
                    continue
                by_year = group.groupby("fiscal_year")["naive"].agg(set)
                if len(by_year) < years or any(
                    a & b for a, b in zip(by_year, by_year.iloc[1:])
                ):
                    continue
                rows.append(
                    {
                        "buyer": buyer,
                        "start": start,
                        "platform": platform,
                        "contractors": group["naive"].nunique(),
                        "other_records": int((span["unclouded"] != platform).sum()),
                        "dollars": group["dollars"].sum(),
                    }
                )
    if not rows:
        raise SystemExit(f"No office fits the pattern over {years} years.")
    return (
        pd.DataFrame(rows)
        .sort_values(["other_records", "dollars"], ascending=[True, False])
        .reset_index(drop=True)
        .rename(lambda i: i + 1)
    )


def node_patch(row: pd.Series) -> Patch:
    """A supplier's node: its shape, filled with its colour and ringed in ink."""
    vertices, orientation, size = SHAPES[row["shape"]]
    style = {
        "facecolor": row["color"],
        "edgecolor": TEXT_PRIMARY,
        "linewidth": 1.0,
        "zorder": 3,
    }
    if vertices is None:
        return Circle((row["x"], row["y"]), NODE_RADIUS * size, **style)
    return RegularPolygon(
        (row["x"], row["y"]),
        vertices,
        radius=NODE_RADIUS * size,
        orientation=orientation,
        **style,
    )


def node_table(
    records: pd.DataFrame, names: pd.Series, platform: str, gap: float
) -> pd.DataFrame:
    """
    One row per (view, year, supplier) with its label, look and position; the
    naive row sits `gap` data units above the unclouded one.

    Within a year the platform followed and its contractors come first, so
    they line up down the column; other suppliers follow in descending dollars.
    Naive-row shapes follow the contractor, in that order of first appearance.
    """
    focal = records["unclouded"] == platform
    # Each year starts after the widest row of the year before it.
    widest = (
        pd.concat(
            [records.groupby("fiscal_year")[view].nunique() for view in VIEWS], axis=1
        )
        .max(axis=1)
        .sort_index()
    )
    starts = (COLUMN_WIDTH + (widest - 1) * NODE_GAP).cumsum().shift(fill_value=0)
    frames = []
    for view in VIEWS:
        nodes = (
            records.assign(focal=focal)
            .groupby(["fiscal_year", view], as_index=False)
            .agg(dollars=("dollars", "sum"), focal=("focal", "any"))
            .rename(columns={view: "key"})
            .sort_values(
                ["fiscal_year", "focal", "dollars"], ascending=[True, False, False]
            )
        )
        rank = nodes.groupby("fiscal_year").cumcount()
        # The focal node leads its year; any others fan out to its right.
        nodes["x"] = nodes["fiscal_year"].map(starts) + rank * NODE_GAP
        nodes["y"] = gap if view == "naive" else 0.0
        nodes["view"] = view
        if view == "naive":
            order = {
                key: i % len(SHAPES) for i, key in enumerate(nodes["key"].unique())
            }
            nodes["shape"] = nodes["key"].map(order)
            nodes["color"] = NAIVE_COLOR
        else:
            nodes["shape"] = 0
            nodes["color"] = np.where(
                nodes["key"] == platform, UNCLOUDED_COLOR, NAIVE_COLOR
            )
        # Keys are UEIs (named by their contractor) or platform names.
        uei = nodes["key"].isin(names.index)
        nodes["label"] = nodes["key"].where(
            ~uei, nodes["key"].map(names).fillna("").map(short_name)
        )
        frames.append(nodes)
    nodes = pd.concat(frames, ignore_index=True)
    # Vertical extent of each shape, for ties and labels to meet its edge.
    extents = [
        node_patch(row)
        .get_patch_transform()
        .transform(node_patch(row).get_path().vertices)[:, 1]
        for _, row in nodes.iterrows()
    ]
    nodes["top"] = [ys.max() for ys in extents]
    nodes["bottom"] = [ys.min() for ys in extents]
    return nodes


def draw_node(ax: plt.Axes, row: pd.Series) -> plt.Annotation:
    """One supplier: its node, with its name and dollars above or below."""
    ax.add_patch(node_patch(row))
    name = textwrap.fill(row["label"], LABEL_WIDTH)
    above = row["view"] == "naive"
    return ax.annotate(
        f"{name}\n{format_dollars(row['dollars'])}",
        xy=(row["x"], row["top"] if above else row["bottom"]),
        xytext=(0, LABEL_GAP if above else -LABEL_GAP),
        textcoords="offset points",
        ha="center",
        va="bottom" if above else "top",
        fontsize=FONT_SIZE - 1,
        color=TEXT_PRIMARY if row["focal"] else TEXT_SECONDARY,
        linespacing=1.15,
    )


def draw_retained(ax: plt.Axes, nodes: pd.DataFrame) -> None:
    """
    An arrow for every supplier kept from one year to the next, per row.

    An arrow that would pass through another node bows into the gap between
    the rows instead.
    """
    for view, row in nodes.groupby("view"):
        # arc3's rad bends to the left of travel: positive is down, left to right.
        bow = 0.22 if view == "naive" else -0.22
        for _, key_nodes in row.sort_values("fiscal_year").groupby("key"):
            pairs = zip(
                key_nodes.iloc[:-1].itertuples(), key_nodes.iloc[1:].itertuples()
            )
            for a, b in pairs:
                if b.fiscal_year != a.fiscal_year + 1:
                    continue
                blocked = row["x"].between(a.x, b.x, inclusive="neither").any()
                ax.add_patch(
                    FancyArrowPatch(
                        (a.x + NODE_RADIUS, a.y),
                        (b.x - NODE_RADIUS, b.y),
                        arrowstyle="-|>",
                        mutation_scale=14,
                        color=a.color,
                        linewidth=2,
                        shrinkA=3,
                        shrinkB=3,
                        connectionstyle=f"arc3,rad={bow if blocked else 0}",
                        zorder=2,
                    )
                )


def draw_ties(ax: plt.Axes, nodes: pd.DataFrame, records: pd.DataFrame) -> None:
    """Thin ties from each contractor to the platform it is keyed by."""
    position = nodes.set_index(["view", "fiscal_year", "key"])
    links = records[["fiscal_year", "naive", "unclouded"]].drop_duplicates()
    for link in links.itertuples():
        top = position.loc[("naive", link.fiscal_year, link.naive)]
        bottom = position.loc[("unclouded", link.fiscal_year, link.unclouded)]
        ax.plot(
            [top["x"], bottom["x"]],
            [top["bottom"], bottom["top"]],
            color=SPINE,
            linewidth=1.2,
            zorder=1,
        )


def fit_to_width(fig: plt.Figure, ax: plt.Axes, width: float) -> float:
    """
    Crop the axes to its contents and size the figure to `width` inches, at
    one scale in x and y. Returns that scale, in inches per data unit.

    Text is sized in points and nodes in data units, so each pass changes the
    data extent a little; a few passes settle it.
    """
    ax.set_position([0, 0, 1, 1])
    for _ in range(5):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # The drawn artists only: the axes' own tight bbox is the whole axes.
        box = Bbox.union(
            [
                artist.get_window_extent(renderer)
                for artist in [*ax.texts, *ax.patches, *ax.lines]
            ]
        )
        (x0, y0), (x1, y1) = ax.transData.inverted().transform(box.get_points())
        pad = 0.03 * (x1 - x0)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        scale = width / (x1 - x0 + 2 * pad)
        fig.set_size_inches(width, scale * (y1 - y0 + 2 * pad), forward=False)
    return scale


def draw_figure(
    records: pd.DataFrame,
    names: pd.Series,
    platform: str,
    gap: float,
    width: float,
) -> tuple[plt.Figure, plt.Axes, float]:
    """The two rows of nodes, `gap` data units apart, fitted to `width` inches."""
    nodes = node_table(records, names, platform, gap)

    fig, ax = plt.subplots(figsize=(width, 4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_xlim(nodes["x"].min() - 1, nodes["x"].max() + 1)
    ax.set_ylim(-1.5, gap + 1.5)

    draw_ties(ax, nodes, records)
    draw_retained(ax, nodes)
    labels = [draw_node(ax, row) for _, row in nodes.iterrows()]

    # Year labels go below the deepest platform label, offset in points so the
    # gap holds whatever scale the fit settles on.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = 72 / fig.dpi
    below = nodes["view"] == "unclouded"
    depth = max(
        label.get_window_extent(renderer).height * points
        for label, is_below in zip(labels, below)
        if is_below
    )
    lowest = nodes.loc[below, "bottom"].min()
    for year, x in nodes.groupby("fiscal_year")["x"].min().items():
        ax.annotate(
            f"FY{year}",
            xy=(x, lowest),
            xytext=(0, -(LABEL_GAP + depth + 10)),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=FONT_SIZE + 1,
            color=TEXT_SECONDARY,
        )
    left = nodes["x"].min() - NODE_RADIUS * max(size for *_, size in SHAPES)
    for view, y in zip(VIEWS, (gap, 0.0)):
        ax.annotate(
            ROW_LABELS[view],
            xy=(left, y),
            xytext=(-14, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=FONT_SIZE + 1,
            color=TEXT_PRIMARY,
            fontweight="bold",
        )

    return fig, ax, fit_to_width(fig, ax, width)


def plot_example(
    records: pd.DataFrame,
    names: pd.Series,
    platform: str,
    size: tuple[float, float],
    output_path: Path,
) -> Path:
    """
    Draw the two rows of nodes for one office and save at exactly `size`
    inches.

    The width sets the scale; the gap between the rows is then stretched until
    the height matches. At a fixed scale the height grows linearly with the
    gap, so two trial drafts give it, and one more pass absorbs the small drift
    in scale.
    """
    width, height = size

    def natural_height(gap: float) -> float:
        fig, _, _ = draw_figure(records, names, platform, gap, width)
        plt.close(fig)
        return fig.get_size_inches()[1]

    gaps = [1.0, 2.0]
    heights = [natural_height(gap) for gap in gaps]
    for _ in range(2):
        slope = (heights[-1] - heights[-2]) / (gaps[-1] - gaps[-2])
        gaps.append(gaps[-1] + (height - heights[-1]) / slope)
        heights.append(natural_height(gaps[-1]))
    gap = gaps[-1]
    if gap < 2 * NODE_RADIUS * max(size for *_, size in SHAPES):
        raise SystemExit(f"Too little room: {width:.2f} x {height:.2f} in is too flat.")

    fig, ax, scale = draw_figure(records, names, platform, gap, width)
    # Absorb what the solve left over by padding the vertical limits equally.
    low, high = ax.get_ylim()
    extra = (height / scale - (high - low)) / 2
    ax.set_ylim(low - extra, high + extra)
    fig.set_size_inches(width, height, forward=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # No tight bbox: the page must be exactly `size` to match its partner.
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "02_processed" / "03_classified" / "attributed_dataset.csv",
        help="Attributed dataset from the pipeline",
    )
    parser.add_argument(
        "--primes",
        type=Path,
        default=DATA_DIR
        / "02_processed"
        / "01_filtered"
        / "prime_services_filtered.csv",
        help="Filtered prime awards, for the awarding office",
    )
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="all",
        help="Records in scope, as in run_persistence_analysis.py (default: all)",
    )
    parser.add_argument(
        "--years", type=int, default=3, help="Number of fiscal years (default: 3)"
    )
    parser.add_argument(
        "--buyer",
        help="Only candidates whose 'department | office' name contains this",
    )
    parser.add_argument("--start", type=int, help="Only candidates starting this FY")
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Which of the remaining candidates to draw (default: 1, the best)",
    )
    parser.add_argument(
        "--find",
        action="store_true",
        help="List the candidates instead of drawing one",
    )
    parser.add_argument(
        "--file-type",
        choices=["png", "pdf"],
        default="png",
        help="Output file format (default: png)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = "" if args.scope == "all" else f"_{args.scope}"
    out_dir = OUTPUTS_DIR / "persistence"
    partner = out_dir / f"persistence_ratio{suffix}.{args.file_type}"
    if not args.find and not partner.exists():
        raise SystemExit(
            f"Missing {partner}\nRun `uv run plot_persistence.py --scope {args.scope}"
            f" --file-type {args.file_type}` first; this figure matches its size."
        )

    records = load_scope(args.data, args.primes, args.scope)
    candidates = find_examples(records, args.years)
    if args.buyer:
        candidates = candidates[
            candidates["buyer"].str.contains(args.buyer, regex=False)
        ]
    if args.start:
        candidates = candidates[candidates["start"] == args.start]
    if args.find:
        with pd.option_context("display.width", 200, "display.max_colwidth", 80):
            print(candidates.head(30).to_string())
        raise SystemExit
    if args.rank > len(candidates):
        raise SystemExit(f"Only {len(candidates)} candidates match; see --find.")

    pick = candidates.iloc[args.rank - 1]
    print(f"  example: {pick['buyer']}, FY{pick['start']}, {pick['platform']}")
    in_window = records["fiscal_year"].between(
        pick["start"], pick["start"] + args.years - 1
    )
    saved = plot_example(
        records[(records["buyer"] == pick["buyer"]) & in_window],
        load_names(args.data),
        pick["platform"],
        figure_size_in(partner),
        out_dir / f"persistence_example{suffix}.{args.file_type}",
    )
    print(f"  saved: {saved}")

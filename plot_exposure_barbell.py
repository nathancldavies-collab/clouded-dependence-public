#!/usr/bin/env python3
"""
Barbell Plot: Normal vs Clouded Dependency Exposure
====================================================
Renders `outputs_results_v2/exposure/barbell_data.csv` as a faceted barbell
(dumbbell) chart: one row per node, two endpoints showing its exposure in the
normal and clouded views, connected to show the movement.

Run `run_exposure_analysis.py` first to produce the data.

Run from project root:
    uv run plot_exposure_barbell.py [--top 5] [--out path.png]

Design notes:
  - A dumbbell shows one measure in two states, so the two endpoints are two
    SHADES OF ONE HUE, not two hues. The pair below is validated for colourblind
    separation (dE 21.8 protan/deutan/tritan) and >= 3:1 contrast on the surface.
  - Facets carry independent x-scales: conservative exposure runs to ~0.5% of
    contracts while permissive runs to ~60%, so a shared scale would flatten the
    conservative facet to a single tick. Each facet is labelled accordingly.
  - Where a node does not move, the markers overlap exactly; a surface-coloured
    ring keeps both readable.
  - Platform nodes are bold, so platform-vs-contractor identity never rests on
    colour alone.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from clouded_deps.directories import OUTPUTS_DIR

# --- Palette -----------------------------------------------------------------
# Two shades of the repo's series blue. Validated light-mode, surface #fcfcfb:
# lightness band PASS, chroma floor PASS, CVD separation PASS, normal-vision
# PASS, contrast PASS.
NORMAL_COLOR = "#4197dd"
CLOUDED_COLOR = "#17538f"
CONNECTOR = "#c2d6ea"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"
SPINE = "#d8d7d1"

FACET_TITLES = {
    ("entity", "permissive"): "Entities — primary",
    ("contract", "conservative"): "Contracts — conservative (supporting)",
    ("contract", "permissive"): "Contracts — permissive (upper bound)",
}
FACET_UNITS = {"contract": "contracts", "entity": "entities"}

READOUT_AXIS = {
    "count": "Share of {unit} exposed (%)",
    "value": r"Exposed contract value (% of \$237.4B total)",
}
READOUTS = ("count", "value")
READOUT_NOTE = {
    "count": "Bold labels are cloud platforms.",
    "value": (
        "Exposed contract value counts the WHOLE award, not the portion attributable"
        " to the dependency. Bold labels are cloud platforms."
    ),
}

# Contractor names arrive SHOUTING from USAspending. Title-casing them is right
# for ordinary words but wrong for acronyms, so those are listed explicitly -- a
# length heuristic mis-fires on short ordinary words (BOOZ, ALLEN).
_UPPER_TOKENS = {
    "LLC",
    "L.L.C.",
    "LP",
    "L.P.",
    "LLP",
    "PLC",
    "AG",
    "NV",
    "SA",
    "SHI",
    "DLT",
    "CDW",
    "IBM",
    "SAIC",
    "CGI",
    "GDIT",
    "NCI",
    "ECS",
    "IT",
    "US",
    "USA",
    "DC",
    "HP",
    "AT&T",
    "NTT",
    "KPMG",
    "EY",
    "PWC",
}
_BRAND_CASING = {
    "Teksystems": "TEKsystems",
    "Servicenow": "ServiceNow",
    "Salesforce.Com": "Salesforce.com",
    "Amazon.Com": "Amazon.com",
}
# Entity is primary (plan section 3.3), so it leads the figure.
FACET_ORDER = [
    ("entity", "permissive"),
    ("contract", "conservative"),
    ("contract", "permissive"),
]


def humanise(label: str, is_platform: bool) -> str:
    """
    Case a node label for display without mangling acronyms.

    Platform names are already correctly cased ("AWS", "ServiceNow") and pass
    through untouched. Contractor names arrive SHOUTING from USAspending, so they
    are title-cased -- except tokens that were short and all-caps in the source,
    which are acronyms and stay as they are.
    """
    if is_platform:
        return label
    out = []
    for token in str(label).split():
        cased = token.title()
        bare = cased.strip(".,()")
        if bare.upper() in _UPPER_TOKENS:
            cased = cased.replace(bare, bare.upper())
        elif bare in _BRAND_CASING:
            cased = cased.replace(bare, _BRAND_CASING[bare])
        out.append(cased)
    return " ".join(out)


def load_barbell(
    path: Path, readout: str = "count", top: int | None = None
) -> pd.DataFrame:
    """Load one readout from the barbell data, optionally trimming each facet."""
    if not path.exists():
        raise SystemExit(
            f"Missing {path}\nRun `just exposure` first to produce the data."
        )
    df = pd.read_csv(path)
    if "readout" in df.columns:
        available = sorted(df["readout"].unique())
        df = df[df["readout"] == readout]
        if df.empty:
            raise SystemExit(
                f"No rows with readout={readout!r} in {path}. Available: {available}"
            )
    elif readout != "count":
        raise SystemExit(
            f"{path} predates the value readout; re-run `just exposure` to regenerate."
        )
    if top is not None:
        df = (
            df.sort_values("clouded", ascending=False)
            .groupby(["analysis", "spec"], sort=False)
            .head(top)
        )
    return df.reset_index(drop=True)


def _facets(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Facet keys present in the data, in a stable reporting order."""
    present = set(map(tuple, df[["analysis", "spec"]].drop_duplicates().to_numpy()))
    ordered = [key for key in FACET_ORDER if key in present]
    return ordered + sorted(present - set(ordered))


def _draw_facet(
    ax: plt.Axes,
    facet: pd.DataFrame,
    analysis: str,
    spec: str,
    show_series_labels: bool,
    readout: str = "count",
) -> None:
    """Draw one facet: rows sorted by clouded exposure, highest at the top."""
    rows = facet.sort_values("clouded", ascending=True).reset_index(drop=True)
    positions = range(len(rows))

    for y, row in zip(positions, rows.itertuples()):
        ax.plot(
            [row.normal * 100, row.clouded * 100],
            [y, y],
            color=CONNECTOR,
            linewidth=2,
            solid_capstyle="round",
            zorder=1,
        )

    # Surface-coloured ring so coincident markers stay legible where delta is 0.
    # The normal marker is drawn larger so that a node which does not move (its
    # two endpoints coincide exactly, e.g. delta == 0) still reads as two marks:
    # a dark core inside a light annulus. Both stay centred on their true value.
    ax.scatter(
        rows["normal"] * 100,
        positions,
        s=105,
        color=NORMAL_COLOR,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=3,
        label="Normal view",
    )
    ax.scatter(
        rows["clouded"] * 100,
        positions,
        s=45,
        color=CLOUDED_COLOR,
        edgecolors=SURFACE,
        linewidths=1.0,
        zorder=4,
        label="Clouded view",
    )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(
        [humanise(row.label, row.is_platform) for row in rows.itertuples()],
        fontsize=9,
    )
    for tick, is_platform in zip(ax.get_yticklabels(), rows["is_platform"]):
        tick.set_color(TEXT_PRIMARY if is_platform else TEXT_SECONDARY)
        tick.set_fontweight("bold" if is_platform else "normal")

    span = max(rows["clouded"].max() * 100, 1e-6)
    ax.set_xlim(-span * 0.04, span * 1.08)
    ax.set_ylim(-0.7, len(rows) - 0.3)

    # Counts in the right margin: the contrast obligation is met with visible
    # numbers, and they carry the absolute scale the percentages hide.
    unit = FACET_UNITS.get(analysis, "nodes")

    def _amount(value: float) -> str:
        # Escape the dollar sign: matplotlib parses a $...$ pair as mathtext, which
        # would swallow the signs and italicise the magnitude suffix.
        if readout == "value":
            if value >= 1e9:
                return rf"\${value / 1e9:,.1f}B"
            return rf"\${value / 1e6:,.0f}M"
        return f"{value:,.0f}"

    for y, row in zip(positions, rows.itertuples()):
        ax.annotate(
            f"{_amount(row.normal_count)} → {_amount(row.clouded_count)}",
            xy=(1.02, y),
            xycoords=("axes fraction", "data"),
            va="center",
            ha="left",
            fontsize=8,
            color=TEXT_MUTED,
        )
    header = (
        "value: normal → clouded" if readout == "value" else f"{unit}: normal → clouded"
    )
    ax.annotate(
        header,
        xy=(1.02, 1.045),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=8,
        color=TEXT_MUTED,
        style="italic",
    )

    if show_series_labels and len(rows) > 0:
        top_row = rows.iloc[-1]
        for value, color, text, align in [
            (top_row["normal"], NORMAL_COLOR, "normal", "right"),
            (top_row["clouded"], CLOUDED_COLOR, "clouded", "left"),
        ]:
            offset = -span * 0.015 if align == "right" else span * 0.015
            ax.annotate(
                text,
                xy=(value * 100 + offset, len(rows) - 1 + 0.42),
                ha=align,
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    ax.set_title(
        FACET_TITLES.get((analysis, spec), f"{analysis} — {spec}"),
        loc="left",
        fontsize=10.5,
        color=TEXT_PRIMARY,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(
        READOUT_AXIS[readout].format(unit=unit),
        fontsize=9,
        color=TEXT_SECONDARY,
        labelpad=6,
    )

    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)


def plot_barbell(
    df: pd.DataFrame, output_path: Path, k: int | None = None, readout: str = "count"
) -> Path:
    """Render every facet as a stacked barbell chart and save it."""
    facets = _facets(df)
    counts = [len(df[(df["analysis"] == a) & (df["spec"] == s)]) for a, s in facets]

    header_in, legend_in, row_in, facet_pad_in = 1.15, 1.05, 0.30, 0.92
    body_in = sum(row_in * n + facet_pad_in for n in counts)
    height = header_in + legend_in + body_in
    hspace = facet_pad_in / (row_in * (sum(counts) / len(counts)))
    fig, axes = plt.subplots(
        nrows=len(facets),
        figsize=(10.5, height),
        dpi=200,
        gridspec_kw={"height_ratios": counts, "hspace": hspace},
    )
    if len(facets) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for index, (ax, (analysis, spec)) in enumerate(zip(axes, facets)):
        facet = df[(df["analysis"] == analysis) & (df["spec"] == spec)]
        _draw_facet(
            ax, facet, analysis, spec, show_series_labels=index == 0, readout=readout
        )

    depth = f" (k = {k})" if k is not None else ""
    fig.subplots_adjust(
        left=0.26,
        right=0.82,
        top=1 - header_in / height,
        bottom=legend_in / height,
    )

    measure = "Exposed contract value" if readout == "value" else "Dependency exposure"
    fig.text(
        0.035,
        1 - 0.34 / height,
        f"{measure}: what the contracting record shows"
        " vs. what platform attribution reveals",
        ha="left",
        va="top",
        fontsize=13.5,
        color=TEXT_PRIMARY,
        fontweight="bold",
    )
    fig.text(
        0.035,
        1 - 0.62 / height,
        "Union of the top-ranked nodes in either view. "
        + READOUT_NOTE[readout]
        + f" Exposure only{depth}: which nodes stand in a dependency"
        " relation, not a prediction of failure.",
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT_SECONDARY,
    )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label,
        )
        for color, label in [
            (NORMAL_COLOR, "Normal view (contracting record only)"),
            (CLOUDED_COLOR, "Clouded view (with platform attribution)"),
        ]
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.035, 0.18 / height),
        ncol=2,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_table(df: pd.DataFrame) -> None:
    """Print the underlying numbers, so the chart is never the only view."""
    for analysis, spec in _facets(df):
        facet = df[(df["analysis"] == analysis) & (df["spec"] == spec)]
        facet = facet.sort_values("clouded", ascending=False)
        print(f"\n  {FACET_TITLES.get((analysis, spec), f'{analysis} - {spec}')}")
        print(
            f"    {'node':<34s} {'normal':>9s} {'clouded':>9s} "
            f"{'normal %':>9s} {'clouded %':>10s}"
        )
        print(f"    {'-' * 74}")
        for row in facet.itertuples():
            marker = "*" if row.is_platform else " "
            print(
                f"    {marker}{row.label[:33]:<33s} {row.normal_count:>12,.0f} "
                f"{row.clouded_count:>12,.0f} {row.normal * 100:>8.3f}% "
                f"{row.clouded * 100:>9.3f}%"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=OUTPUTS_DIR / "exposure" / "barbell_data.csv",
        help="Barbell data CSV from run_exposure_analysis.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUTS_DIR / "exposure",
        help="Directory for the output PNGs",
    )
    parser.add_argument(
        "--readout",
        choices=["count", "value", "both"],
        default="both",
        help="Which readout to render (default: both, as separate figures)",
    )
    parser.add_argument(
        "--top", type=int, default=None, help="Trim each facet to this many rows"
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Tier depth, for the subtitle only"
    )
    parser.add_argument(
        "--table", action="store_true", help="Also print the numbers as a table"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    readouts = READOUTS if args.readout == "both" else (args.readout,)
    for readout in readouts:
        data = load_barbell(args.data, readout=readout, top=args.top)
        if args.table:
            print_table(data)
        out = args.out_dir / f"exposure_barbell_{readout}.png"
        saved = plot_barbell(data, out, k=args.k, readout=readout)
        print(f"  Saved {readout:5s} barbell: {saved}")

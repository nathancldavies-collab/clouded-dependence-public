"""
Shared figure style
===================
Palette, markers and defaults used across the plotting scripts, so the figures
read as one set: the naive view is always the same grey baseline and the
unclouded view always carries the finding in the repo's dark blue.
"""

import re
from pathlib import Path

from PIL import Image

# --- Palette -----------------------------------------------------------------
# The naive view is grey because it is the baseline the reader already has. The
# grey deliberately fails the chroma floor (it should read as grey); CVD
# separation from the blue and contrast against the surface both pass.
NAIVE_COLOR = "#8a8880"
UNCLOUDED_COLOR = "#17538f"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"
SPINE = "#d8d7d1"

# Shape carries the series distinction alongside colour, so the two views stay
# separable in greyscale and for readers who cannot tell the colours apart.
NAIVE_MARKER = "o"
UNCLOUDED_MARKER = "*"

# --- Defaults ----------------------------------------------------------------
# Dependency tier depth reported in the exposure figures.
DEFAULT_K = 3

# --- Subfigure pairing -------------------------------------------------------
# The barbell and ego-network figures sit side by side in LaTeX (subcaption) at
# these fractions of \textwidth.
BARBELL_WIDTH_FRAC = 0.60
EGO_WIDTH_FRAC = 0.39


def figure_size_in(path: Path, dpi: float = 200) -> tuple[float, float]:
    """
    The (width, height) in inches of a saved figure.

    PDFs are read from the page's MediaBox (in points); raster images from their
    pixel size at `dpi`.
    """
    if path.suffix == ".pdf":
        match = re.search(rb"/MediaBox\s*\[\s*([\d.\s-]+)\]", path.read_bytes())
        if match is None:
            raise ValueError(f"No MediaBox in {path}")
        x0, y0, x1, y1 = map(float, match.group(1).split())
        return (x1 - x0) / 72, (y1 - y0) / 72
    with Image.open(path) as image:
        width, height = image.size
    return width / dpi, height / dpi


def partner_size_in(
    partner: tuple[float, float], partner_frac: float, own_frac: float
) -> tuple[float, float]:
    """
    Figure size that prints at the partner's height and the same scale.

    Each subfigure is scaled to its share of \\textwidth. Sizing this figure's
    width in the ratio of the two shares, at the partner's height, gives both the
    same scale factor -- so they print at equal height with matching font sizes.
    """
    width, height = partner
    return width * own_frac / partner_frac, height

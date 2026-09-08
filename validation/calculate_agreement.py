#!/usr/bin/env python3
"""
Krippendorff's Alpha for the Validation Samples
===============================================
Computes nominal Krippendorff's alpha for either validation part:

  Part A: classification codes (non-cloud | cloud-dependent | cloud-infrastructure)
  Part B: platform codes (AWS | Azure | … | N/A)

Rating files are read through small loaders into one intermediate long format
(``Ratings``: one row per unit x rater), so supporting a new file layout later
only means adding a loader to ``LOADERS`` — nothing downstream changes.

Ratings can be filtered by the rater's own confidence (``--confidence``) and,
for part A, collapsed from the three-way classification to binary cloud /
non-cloud (``--binary``).

Run from project root:
    uv run validation/calculate_agreement.py --part a
    uv run validation/calculate_agreement.py --part a --binary --confidence medium high
    uv run validation/calculate_agreement.py --part b --input some/other/file.csv
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd

from clouded_deps.directories import OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RATINGS_DIR = OUTPUTS_DIR / "validation"

# Filename globs used when no explicit --input is given.
PART_GLOBS = {
    "a": "part_a_classification_rating_*.xlsx",
    "b": "part_b_attribution_rating_*.xlsx",
}
PART_LABELS = {"a": "classification", "b": "platform"}

MIN_RATERS = 2

# Rater self-reported confidence, ordered from least to most sure.
CONFIDENCE_LEVELS = ["low", "medium", "high"]

# Part A only: collapse the three-way classification to cloud / non-cloud.
BINARY_MAP = {
    "non-cloud": "non-cloud",
    "cloud-dependent": "cloud",
    "cloud-infrastructure": "cloud",
}

# ---------------------------------------------------------------------------
# Intermediate format
# ---------------------------------------------------------------------------
UNIT_COL = "unit_id"
RATER_COL = "rater"
LABEL_COL = "label"
CONFIDENCE_COL = "confidence"
LONG_COLS = [UNIT_COL, RATER_COL, LABEL_COL, CONFIDENCE_COL]
REQUIRED_COLS = [UNIT_COL, RATER_COL, LABEL_COL]


@dataclass(frozen=True)
class Ratings:
    """Long-format ratings: exactly one row per (unit, rater) with a label.

    ``long`` always has the columns ``unit_id``, ``rater``, ``label`` and
    ``confidence`` (the rater's own low/medium/high call, possibly missing);
    labels are stripped strings and rows with a missing label are dropped by
    the loaders.
    """

    long: pd.DataFrame

    @classmethod
    def from_frame(cls, df: pd.DataFrame, source: str) -> "Ratings":
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{source}: missing column(s) {', '.join(missing)}")
        out = df.reindex(columns=LONG_COLS).copy()
        for col in LONG_COLS:
            out[col] = out[col].astype("string").str.strip()
        out = out[out[LABEL_COL].notna() & (out[LABEL_COL] != "")]
        out[CONFIDENCE_COL] = out[CONFIDENCE_COL].replace("", pd.NA).str.lower()
        return cls(out.reset_index(drop=True))

    @property
    def raters(self) -> list[str]:
        return sorted(self.long[RATER_COL].dropna().unique().tolist())

    @property
    def units(self) -> list[str]:
        return sorted(self.long[UNIT_COL].dropna().unique().tolist())

    @property
    def labels(self) -> list[str]:
        return sorted(self.long[LABEL_COL].dropna().unique().tolist())

    def wide(self) -> pd.DataFrame:
        """Units x raters matrix of labels (NaN where a rater did not code)."""
        return self.long.pivot(index=UNIT_COL, columns=RATER_COL, values=LABEL_COL)

    def overlapping_units(self) -> list[str]:
        """Units coded by at least two raters."""
        counts = self.long.groupby(UNIT_COL)[RATER_COL].nunique()
        return sorted(counts[counts >= MIN_RATERS].index.tolist())

    def subset(self, raters: list[str]) -> "Ratings":
        return Ratings(
            self.long[self.long[RATER_COL].isin(raters)].reset_index(drop=True)
        )

    def filter_confidence(self, levels: list[str]) -> "Ratings":
        """Keep only ratings the rater marked with one of ``levels``.

        Ratings with no confidence recorded are dropped, since they cannot be
        shown to meet the filter.
        """
        keep = self.long[CONFIDENCE_COL].isin(levels)
        return Ratings(self.long[keep].reset_index(drop=True))

    def map_labels(self, mapping: dict[str, str]) -> "Ratings":
        """Recode labels (e.g. collapse to binary); unmapped labels are kept."""
        out = self.long.copy()
        out[LABEL_COL] = (
            out[LABEL_COL].map(lambda v: mapping.get(v, v)).astype("string")
        )
        return Ratings(out)


def combine(parts: list[Ratings]) -> Ratings:
    return Ratings(pd.concat([p.long for p in parts], ignore_index=True))


# ---------------------------------------------------------------------------
# Loaders — file path -> Ratings
# ---------------------------------------------------------------------------
def load_rating_sheet(path: Path) -> Ratings:
    """Load a rating spreadsheet produced by build_validation_sample.py."""
    read = pd.read_csv if path.suffix.lower() == ".csv" else pd.read_excel
    raw = read(path)
    df = raw.rename(
        columns={"original_prime_id": UNIT_COL, "classification": LABEL_COL}
    )
    return Ratings.from_frame(df, source=path.name)


def load_long(path: Path) -> Ratings:
    """Load a file that is already in the intermediate long format."""
    read = pd.read_csv if path.suffix.lower() == ".csv" else pd.read_excel
    return Ratings.from_frame(read(path), source=path.name)


Loader = Callable[[Path], Ratings]

LOADERS: dict[str, Loader] = {
    "rating_sheet": load_rating_sheet,
    "long": load_long,
}
DEFAULT_LOADER = "rating_sheet"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(ratings: Ratings) -> list[str]:
    """Return a list of blocking problems (empty means the data is usable)."""
    problems = []

    if ratings.long.empty:
        return ["no rated rows left (blank `classification` cells, or filtered out)"]

    if len(ratings.raters) < MIN_RATERS:
        problems.append(
            f"need ratings from at least {MIN_RATERS} raters, "
            f"found {len(ratings.raters)}: {', '.join(ratings.raters) or 'none'}"
        )

    dupes = ratings.long.duplicated(subset=[UNIT_COL, RATER_COL], keep=False)
    if dupes.any():
        pairs = ratings.long.loc[dupes, [UNIT_COL, RATER_COL]].drop_duplicates()
        problems.append(
            f"{len(pairs)} (unit, rater) pair(s) rated more than once, "
            f"e.g. {pairs.iloc[0][UNIT_COL]} / {pairs.iloc[0][RATER_COL]}"
        )

    if len(ratings.raters) >= MIN_RATERS and not ratings.overlapping_units():
        problems.append("no unit was rated by two or more raters — alpha is undefined")

    return problems


# ---------------------------------------------------------------------------
# Alpha
# ---------------------------------------------------------------------------
def reliability_matrix(ratings: Ratings) -> np.ndarray:
    """Raters x units matrix of integer label codes, np.nan where uncoded."""
    codes = {label: i for i, label in enumerate(ratings.labels)}
    wide = ratings.wide().apply(lambda col: col.map(codes)).astype("Float64")
    # Units a rater did not code stay missing, which krippendorff ignores.
    return wide.T.to_numpy(dtype=float, na_value=np.nan)


def alpha(ratings: Ratings) -> float:
    return float(
        krippendorff.alpha(
            reliability_data=reliability_matrix(ratings),
            level_of_measurement="nominal",
        )
    )


def percent_agreement(ratings: Ratings) -> float | None:
    """Share of two-rater units where both raters chose the same label."""
    wide = ratings.wide()
    both = wide.dropna()
    if both.empty or both.shape[1] != MIN_RATERS:
        return None
    return float((both.iloc[:, 0] == both.iloc[:, 1]).mean())


def pairwise_alphas(ratings: Ratings) -> dict[tuple[str, str], float | None]:
    raters = ratings.raters
    out: dict[tuple[str, str], float | None] = {}
    for i, first in enumerate(raters):
        for second in raters[i + 1 :]:
            pair = ratings.subset([first, second])
            out[(first, second)] = alpha(pair) if pair.overlapping_units() else None
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(ratings: Ratings, part: str, scheme: str):
    overlap = ratings.overlapping_units()
    print(f"\n=== Krippendorff's alpha — part {part.upper()} ({scheme}) ===")
    print(f"  Raters:            {', '.join(ratings.raters)}")
    print(f"  Units rated:       {len(ratings.units)}")
    print(f"  Units with >=2 raters: {len(overlap)}")
    print(f"  Distinct labels:   {len(ratings.labels)}")

    print(f"\n  alpha (nominal):   {alpha(ratings):.3f}")
    agreement = percent_agreement(ratings)
    if agreement is not None:
        print(f"  raw agreement:     {agreement:.1%}")

    pairs = pairwise_alphas(ratings)
    if len(pairs) > 1:
        print("\n  Pairwise alpha:")
        for (first, second), value in pairs.items():
            shown = "n/a (no shared units)" if value is None else f"{value:.3f}"
            print(f"    {first} vs {second}: {shown}")

    print("\n  Label counts per rater:")
    counts = ratings.long.groupby([LABEL_COL, RATER_COL]).size().unstack(fill_value=0)
    for line in counts.to_string().splitlines():
        print(f"    {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [Path(p) for p in args.input]
    return sorted(args.ratings_dir.glob(PART_GLOBS[args.part]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part", choices=sorted(PART_GLOBS), required=True, help="Validation part."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="Rating files to read (default: all rater files for the part).",
    )
    parser.add_argument(
        "--ratings-dir",
        type=Path,
        default=RATINGS_DIR,
        help=f"Directory searched when --input is omitted (default: {RATINGS_DIR}).",
    )
    parser.add_argument(
        "--confidence",
        nargs="+",
        choices=CONFIDENCE_LEVELS,
        help=(
            "Keep only ratings the rater marked with these confidence levels "
            "(default: all; ratings with no confidence recorded are dropped "
            "when this filter is used)."
        ),
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Part A only: collapse classifications to cloud / non-cloud.",
    )
    parser.add_argument(
        "--loader",
        choices=sorted(LOADERS),
        default=DEFAULT_LOADER,
        help=f"Input format (default: {DEFAULT_LOADER}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = resolve_inputs(args)
    if not paths:
        print(
            f"[error] no rating files found for part {args.part} in {args.ratings_dir}",
            file=sys.stderr,
        )
        return 1

    loader = LOADERS[args.loader]
    loaded = []
    for path in paths:
        ratings = loader(path)
        print(
            f"Loaded {path}: {len(ratings.long)} rated rows ({', '.join(ratings.raters)})"
        )
        loaded.append(ratings)

    ratings = combine(loaded)

    if args.confidence:
        before = len(ratings.long)
        ratings = ratings.filter_confidence(args.confidence)
        print(
            f"Confidence filter [{', '.join(args.confidence)}]: "
            f"kept {len(ratings.long)} of {before} ratings"
        )

    scheme = PART_LABELS[args.part]
    if args.binary:
        if args.part != "a":
            print(
                "[error] --binary only applies to part A (classification codes)",
                file=sys.stderr,
            )
            return 1
        unknown = sorted(set(ratings.labels) - set(BINARY_MAP))
        if unknown:
            print(
                f"[warn] labels left uncollapsed: {', '.join(unknown)}",
                file=sys.stderr,
            )
        ratings = ratings.map_labels(BINARY_MAP)
        scheme = "classification, binary cloud/non-cloud"

    problems = validate(ratings)
    if problems:
        print("\n[error] cannot compute alpha:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    report(ratings, args.part, scheme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
The naive and unclouded views of cloud spending, shared by the analyses.

The two views differ ONLY in attribution: same records, same dollars. The naive
view keys each record by the contractor on the record (resolved UEI); the
unclouded view reassigns every platform-attributed record to its platform.
"""

from pathlib import Path

import pandas as pd

# FY2025 is a partial year in the data; the paper's range is FY2017-2024.
FIRST_FY, LAST_FY = 2017, 2024

ATTRIBUTED_METHODS = {"phase1_direct", "phase2_description", "phase3_pattern"}


def load_records(path: Path) -> pd.DataFrame:
    """Load the cloud records of the attributed dataset with a key for each view."""
    if not path.exists():
        raise SystemExit(f"Missing {path}\nRun `just pipeline` first.")
    df = pd.read_csv(
        path,
        usecols=[
            "contractor",
            "original_prime_id",
            "dollars",
            "fiscal_year",
            "is_cloud",
            "final_platform",
            "attribution_method",
        ],
    )
    # Non-positive dollars are de-obligations; the baseline and platform HHI
    # both drop them, so this does too.
    df = df[
        df["is_cloud"]
        & (df["dollars"] > 0)
        & df["fiscal_year"].between(FIRST_FY, LAST_FY)
    ]
    df = df.assign(fiscal_year=df["fiscal_year"].astype(int))

    # Keying the unattributed rows by contractor UEI (not `final_platform`, which
    # falls back to contractor NAME) keeps them identical across the two views.
    attributed = df["attribution_method"].isin(ATTRIBUTED_METHODS)
    return df.assign(
        naive=df["contractor"],
        unclouded=df["final_platform"].where(attributed, df["contractor"]),
    )

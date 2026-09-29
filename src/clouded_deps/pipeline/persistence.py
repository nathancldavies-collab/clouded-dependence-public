"""
Supplier persistence: are buyers' suppliers more persistent by platform than by
contractor, beyond what merging contractors into platforms explains?

For buyer g (an awarding office) and fiscal year t, a supplier s is *at risk*
if it holds cloud dollars from g in t and g buys cloud again in t+1. It is
*retained* if it also holds cloud dollars from g in t+1. Pooling over buyers
and years:

    cA = retained / at risk, suppliers keyed by contractor  (naive view)
    cB = retained / at risk, suppliers keyed by platform    (unclouded view)
    PR = cB / cA                                            (persistence ratio)

Each record's unclouded key has a source (`key_source`):

    attributed   its own platform attribution
    imputed      unattributed, keyed by its contractor's modal platform
    contractor   unattributed, contractor never attributed: keeps its UEI

The unclouded view is a coarsening of the naive one, so PR >= 1 almost
mechanically. The null is therefore a record-level label shuffle: platforms
are permuted across records within each year, which keeps every platform's
yearly record count but breaks any tie between buyer and platform.
Contractor-keyed records are the same in both views and stay fixed.

Everything below runs on integer codes, so a permutation or a bootstrap draw
costs a few array passes rather than a pandas groupby.
"""

import numpy as np
import pandas as pd

from clouded_deps.pipeline.views import ATTRIBUTED_METHODS

VIEWS = ("naive", "unclouded")
KEY_SOURCES = ("attributed", "imputed", "contractor")


def attach_buyer(records: pd.DataFrame, primes: pd.DataFrame) -> pd.DataFrame:
    """
    Add `buyer` (department | awarding office) from the prime award.

    Subcontract records share their prime's `original_prime_id`, so they take
    the prime's buyer. The department is part of the key because some office
    names recur across departments.
    """
    offices = primes.set_index("Award ID")
    department = offices["Awarding Department/Agency"].fillna("")
    buyer = department + " | " + offices["Awarding Office"]
    return records.assign(buyer=records["original_prime_id"].map(buyer))


def impute_platforms(records: pd.DataFrame) -> pd.DataFrame:
    """
    Key the unattributed records of every contractor with at least one
    attributed record by its modal attributed platform, and add `key_source`.

    Without this, a contractor attributed in one year and not the next would
    change key in the unclouded view only, which reads as spurious platform
    churn. Ties between platforms go to the first name alphabetically.
    Unattributed records of never-attributed contractors keep their UEI.
    """
    attributed = records["attribution_method"].isin(ATTRIBUTED_METHODS)
    modal = (
        records[attributed]
        .groupby(["contractor", "final_platform"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["n", "final_platform"], ascending=[False, True])
        .drop_duplicates("contractor")
        .set_index("contractor")["final_platform"]
    )
    imputed = ~attributed & records["contractor"].isin(modal.index)
    source = np.select([attributed, imputed], KEY_SOURCES[:2], KEY_SOURCES[2])
    return records.assign(
        unclouded=records["unclouded"].where(
            ~imputed, records["contractor"].map(modal)
        ),
        key_source=source,
    )


def retention_counts(
    buyer: np.ndarray, year: np.ndarray, key: np.ndarray, n_buyers: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    At-risk and retained supplier pairs per buyer.

    `buyer`, `year` and `key` are non-negative integer codes, one per record;
    `year` counts from the first fiscal year. Returns two arrays of length
    `n_buyers`.
    """
    n_years = int(year.max()) + 2
    n_keys = int(key.max()) + 1
    # One cell per (buyer, year, supplier); +n_keys steps a cell forward a year.
    cells = np.unique((buyer * n_years + year) * n_keys + key)
    buyer_year = cells // n_keys
    active = np.unique(buyer_year)

    at_risk = np.isin(buyer_year + 1, active)
    retained = at_risk & np.isin(cells + n_keys, cells)
    cell_buyer = buyer_year // n_years
    return (
        np.bincount(cell_buyer[at_risk], minlength=n_buyers),
        np.bincount(cell_buyer[retained], minlength=n_buyers),
    )


def shuffle_within_year(
    key: np.ndarray, year: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute `key` across records of the same year."""
    by_year = np.argsort(year, kind="stable")
    shuffled = np.lexsort((rng.random(len(key)), year))
    out = np.empty_like(key)
    out[by_year] = key[shuffled]
    return out


def shuffle_platforms(
    codes: dict[str, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """
    One draw of the null for the unclouded key.

    Platforms (attributed and imputed alike) are permuted across records within
    each year. Contractor-keyed records have no platform and stay fixed.
    """
    key = codes["unclouded"]
    out = key.copy()
    platform = codes["source"] != "contractor"
    out[platform] = shuffle_within_year(key[platform], codes["year"][platform], rng)
    return out


def encode(records: pd.DataFrame) -> dict[str, np.ndarray]:
    """Integer codes for buyer, year and each view's supplier key."""
    codes = {
        "buyer": pd.factorize(records["buyer"])[0],
        "year": (records["fiscal_year"] - records["fiscal_year"].min()).to_numpy(),
        "source": records["key_source"].to_numpy(),
    }
    for view in VIEWS:
        codes[view] = pd.factorize(records[view])[0]
    return codes


def persistence(
    records: pd.DataFrame, n_boot: int, n_perm: int, seed: int
) -> dict[str, np.ndarray | float | int]:
    """
    cA, cB and PR, with their bootstrap and jackknife replicates and the
    permutation null for PR.

    The bootstrap resamples buyers with replacement, each carrying all its
    pairs; cA and cB share the resamples, so PR's draws are paired. The
    jackknife leaves out one buyer at a time, in closed form.
    """
    rng = np.random.default_rng(seed)
    codes = encode(records)
    n_buyers = int(codes["buyer"].max()) + 1
    counts = {
        view: retention_counts(codes["buyer"], codes["year"], codes[view], n_buyers)
        for view in VIEWS
    }
    # Both views share buyer-years, so their at-risk buyers are the same.
    at_risk_buyers = counts["naive"][0] > 0
    n = {view: c[0][at_risk_buyers] for view, c in counts.items()}
    r = {view: c[1][at_risk_buyers] for view, c in counts.items()}
    n_clusters = int(at_risk_buyers.sum())

    result: dict[str, np.ndarray | float | int] = {
        "n_buyers": n_clusters,
        "n_records": len(records),
    }
    for view, label in zip(VIEWS, ("cA", "cB"), strict=True):
        result[f"n_pairs_{label}"] = int(n[view].sum())
        result[label] = r[view].sum() / n[view].sum()
        # Leave one buyer out: drop its pairs from both sums.
        result[f"{label}_jack"] = (r[view].sum() - r[view]) / (n[view].sum() - n[view])

    draws = {label: np.empty(n_boot) for label in ("cA", "cB")}
    for b in range(n_boot):
        weights = np.bincount(
            rng.integers(0, n_clusters, n_clusters), minlength=n_clusters
        )
        for view, label in zip(VIEWS, ("cA", "cB"), strict=True):
            draws[label][b] = (weights @ r[view]) / (weights @ n[view])
    for label, values in draws.items():
        result[f"{label}_boot"] = values

    result["pr"] = result["cB"] / result["cA"]
    result["pr_boot"] = draws["cB"] / draws["cA"]
    result["pr_jack"] = result["cB_jack"] / result["cA_jack"]

    # The shuffle leaves the naive view untouched, so cA is fixed under the null.
    null = np.empty(n_perm)
    for p in range(n_perm):
        shuffled = shuffle_platforms(codes, rng)
        at_risk, retained = retention_counts(
            codes["buyer"], codes["year"], shuffled, n_buyers
        )
        null[p] = retained.sum() / at_risk.sum() / result["cA"]
    result["pr_null"] = null
    return result

"""Tests for the supplier persistence analysis."""

import numpy as np
import pandas as pd
import pytest

from clouded_deps.pipeline import persistence as ps

# ---------------------------------------------------------------------------
# Fixture: three offices over FY2017-19. X and Y both resell Azure; Z resells AWS.
#
#   office   2017        2018        2019
#   O1       X (Azure)   Y (Azure)   Y (Azure)    contractor churns, platform stays
#   O2       Z (AWS) x2  Z (AWS)     Z (AWS)      both persist; 2017 row duplicated
#   O3       X (Azure)   -           X (Azure)    gap year: never at risk
#
# At-risk pairs (supplier at t, office active at t+1):
#   naive      O1: (X,17) lost, (Y,18) kept   O2: (Z,17) kept, (Z,18) kept
#   unclouded  O1: (Azure,17), (Azure,18) kept  O2: (AWS,17), (AWS,18) kept
# So cA = 3/4, cB = 4/4, PR = 4/3.
# ---------------------------------------------------------------------------

ROWS = [
    ("O1", 2017, "X", "Azure"),
    ("O1", 2018, "Y", "Azure"),
    ("O1", 2019, "Y", "Azure"),
    ("O2", 2017, "Z", "AWS"),
    ("O2", 2017, "Z", "AWS"),
    ("O2", 2018, "Z", "AWS"),
    ("O2", 2019, "Z", "AWS"),
    ("O3", 2017, "X", "Azure"),
    ("O3", 2019, "X", "Azure"),
]


@pytest.fixture
def records() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=["buyer", "fiscal_year", "naive", "unclouded"])
    return df.assign(dollars=1.0, key_source="attributed")


def _counts(records: pd.DataFrame, view: str) -> dict[str, tuple[int, int]]:
    codes = ps.encode(records)
    buyers = pd.factorize(records["buyer"])[1]
    at_risk, retained = ps.retention_counts(
        codes["buyer"], codes["year"], codes[view], len(buyers)
    )
    return {
        b: (int(n), int(r)) for b, n, r in zip(buyers, at_risk, retained, strict=True)
    }


def test_retention_counts_per_buyer(records: pd.DataFrame) -> None:
    assert _counts(records, "naive") == {"O1": (2, 1), "O2": (2, 2), "O3": (0, 0)}
    assert _counts(records, "unclouded") == {"O1": (2, 2), "O2": (2, 2), "O3": (0, 0)}


def test_persistence_ratio(records: pd.DataFrame) -> None:
    result = ps.persistence(records, n_boot=50, n_perm=20, seed=0)
    assert result["cA"] == pytest.approx(3 / 4)
    assert result["cB"] == pytest.approx(1.0)
    assert result["pr"] == pytest.approx(4 / 3)
    # O3 has no at-risk pairs, so it is not a bootstrap cluster.
    assert result["n_buyers"] == 2
    assert result["n_pairs_cA"] == result["n_pairs_cB"] == 4


# Refitting on one office leaves its own jackknife as 0/0, which is expected.
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_jackknife_matches_refit(records: pd.DataFrame) -> None:
    """The closed-form leave-one-buyer-out equals recomputing without the buyer."""
    result = ps.persistence(records, n_boot=10, n_perm=1, seed=0)
    for i, office in enumerate(["O1", "O2"]):
        refit = ps.persistence(
            records[records["buyer"] != office], n_boot=10, n_perm=1, seed=0
        )
        assert result["cA_jack"][i] == pytest.approx(refit["cA"])
        assert result["cB_jack"][i] == pytest.approx(refit["cB"])


def test_reseller_switch_lowers_platform_retention(records: pd.DataFrame) -> None:
    """Same contractor, new platform: retained in the naive view, lost in the other."""
    switch = pd.DataFrame(
        [("O4", 2017, "Z", "AWS"), ("O4", 2018, "Z", "Azure")],
        columns=["buyer", "fiscal_year", "naive", "unclouded"],
    ).assign(dollars=1.0, key_source="attributed")
    counts = _counts(pd.concat([records, switch]), "unclouded")
    assert counts["O4"] == (1, 0)
    assert _counts(pd.concat([records, switch]), "naive")["O4"] == (1, 1)


def test_shuffle_keeps_yearly_counts(records: pd.DataFrame) -> None:
    codes = ps.encode(records)
    rng = np.random.default_rng(0)
    for _ in range(20):
        shuffled = ps.shuffle_within_year(codes["unclouded"], codes["year"], rng)
        for year in np.unique(codes["year"]):
            in_year = codes["year"] == year
            assert sorted(shuffled[in_year]) == sorted(codes["unclouded"][in_year])


def test_impute_platforms_uses_modal_platform() -> None:
    """Unattributed rows take their contractor's modal platform, if it has one."""
    rows = pd.DataFrame(
        [
            # contractor, final_platform, attribution_method
            ("X", "Azure", "phase2_description"),
            ("X", "Azure", "phase1_direct"),
            ("X", "AWS", "phase2_description"),
            ("X", "X Inc", "cloud_unattributed_contractor"),
            ("Y", "Azure", "phase3_pattern"),
            ("Y", "AWS", "phase3_pattern"),
            ("Y", "Y Inc", "cloud_unattributed_contractor"),
            ("Z", "Z Inc", "cloud_unattributed_contractor"),
        ],
        columns=["contractor", "final_platform", "attribution_method"],
    )
    # views.load_records keys unattributed rows by contractor in both views.
    rows = rows.assign(
        unclouded=rows["final_platform"].where(
            rows["attribution_method"] != "cloud_unattributed_contractor",
            rows["contractor"],
        )
    )
    imputed = ps.impute_platforms(rows)
    # Y's AWS/Azure tie goes to AWS alphabetically; Z was never attributed.
    assert imputed["unclouded"].tolist() == [
        "Azure",
        "Azure",
        "AWS",
        "Azure",
        "Azure",
        "AWS",
        "AWS",
        "Z",
    ]
    assert imputed["key_source"].tolist() == [
        "attributed",
        "attributed",
        "attributed",
        "imputed",
        "attributed",
        "attributed",
        "imputed",
        "contractor",
    ]


def test_null_shuffles_platforms_by_record() -> None:
    """
    Platform labels, attributed and imputed alike, move between records of the
    same year; contractor-keyed records stay put.
    """
    rows = pd.DataFrame(
        [
            # buyer, year, contractor, platform, source
            ("O1", 2017, "A", "Azure", "attributed"),
            ("O2", 2017, "B", "AWS", "attributed"),
            ("O3", 2018, "C", "GCP", "attributed"),
            ("O1", 2018, "D", "Oracle", "attributed"),
            ("O1", 2017, "A", "Azure", "imputed"),
            ("O2", 2018, "A", "Azure", "imputed"),
            ("O3", 2017, "B", "AWS", "imputed"),
            ("O1", 2018, "B", "AWS", "imputed"),
            ("O2", 2017, "C", "GCP", "imputed"),
            ("O3", 2018, "Z", "Z", "contractor"),
        ],
        columns=["buyer", "fiscal_year", "naive", "unclouded", "key_source"],
    )
    codes = ps.encode(rows)
    key, source = codes["unclouded"], codes["source"]
    platform = source != "contractor"
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(50):
        shuffled = ps.shuffle_platforms(codes, rng)
        for year in (0, 1):
            in_year = platform & (codes["year"] == year)
            assert sorted(shuffled[in_year]) == sorted(key[in_year])
        assert (shuffled[~platform] == key[~platform]).all()
        seen.add(tuple(shuffled[platform]))
    assert len(seen) > 1


def test_attach_buyer_uses_department_and_prime() -> None:
    primes = pd.DataFrame(
        {
            "Award ID": ["A1", "A2"],
            "Awarding Department/Agency": ["Dept of X", None],
            "Awarding Office": ["OFFICE 1", "OFFICE 1"],
        }
    )
    # A subcontract row carries its prime's id, so it takes the prime's buyer.
    records = pd.DataFrame({"original_prime_id": ["A1", "A1", "A2", "A3"]})
    buyers = ps.attach_buyer(records, primes)["buyer"].tolist()
    assert buyers[:3] == ["Dept of X | OFFICE 1", "Dept of X | OFFICE 1", " | OFFICE 1"]
    assert pd.isna(buyers[3])

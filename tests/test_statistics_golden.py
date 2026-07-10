"""Characterization ("golden file") tests for morphometrics_stats.statistics().

statistics() is a public API used by external / legacy user scripts (e.g.
old_scripts/mitochondria_statistics.py). Its CONTRACT is the exact bytes of the two
CSVs it writes. These tests pin that contract so a refactor of the internals cannot
change the numbers.

Golden files in tests/golden/ were snapshotted from the pre-refactor implementation
(see scratchpad/gen_golden.py). The fixture below MUST reproduce them.

Refactor policy:
  * rawstats.csv                -> must stay byte-identical, forever.
  * pairwise CSV *data rows*    -> must stay byte-identical, forever.
  * pairwise CSV *header*       -> may change ONLY by the intended column renames,
                                   tracked in EXPECTED_PAIRWISE_HEADER.
"""
import os

import pytest

from surface_morphometrics import morphometrics_stats as ms

GOLD = os.path.join(os.path.dirname(__file__), "golden")

# Fixture: 3 classes x 6 tomograms. ER is far from both (significant); PM is
# interleaved with OMM (non-significant -> exercises the " " star token).
DATASETS = [
    [12.1, 11.8, 12.4, 12.0, 11.9, 12.2],
    [25.0, 24.8, 25.3, 24.9, 25.1, 25.2],
    [12.15, 11.75, 12.45, 11.95, 11.85, 12.25],
]
CONDITIONS = ["WT", "WT", "Mut"]
MORPHS = ["OMM", "ER", "PM"]

# Post-Phase-1 header: the three KS columns are renamed to make explicit that this KS
# compares per-tomogram SUMMARY statistics (not pooled triangle distributions), and the
# stray space before "Sample B Condition" is fixed. Data cells are unchanged.
EXPECTED_PAIRWISE_HEADER = (
    "Base Experiment,Stat Type,Utest_Stars,KS_summary_Stars,"
    "Sample A Condition,Sample A Morph,Sample A Mean,Sample A 95% Error,"
    "Sample B Condition,Sample B Morph,Sample B Mean,Sample B 95% Error,"
    "U,P_U,T,P_T,KS_summary_stat,P_KS_summary,n_A,n_B"
)

# The exact renames Phase 1 makes to the header (old token -> new token). The test
# below asserts the header changed ONLY by these.
HEADER_RENAMES = {
    "KStest_Stars": "KS_summary_Stars",
    ",KS,": ",KS_summary_stat,",
    ",P_KS,": ",P_KS_summary,",
    ", Sample B Condition": ",Sample B Condition",   # stray-space fix
}


def _run(tmp_path):
    """Run statistics() into tmp_path; return (pairwise_path, rawstats_path)."""
    pairwise = str(tmp_path / "gold_violin.csv")
    ms.statistics(DATASETS, "Golden Fixture", CONDITIONS, MORPHS,
                  test_type="Peaks", filename=pairwise, ylabel="Distance (nm)")
    return pairwise, str(tmp_path / "gold_rawstats.csv")


def test_rawstats_byte_identical(tmp_path):
    pairwise, rawstats = _run(tmp_path)
    with open(rawstats, "rb") as f, open(f"{GOLD}/statistics_rawstats.csv", "rb") as g:
        assert f.read() == g.read()


def test_pairwise_data_rows_byte_identical(tmp_path):
    pairwise, _rawstats = _run(tmp_path)
    got = open(pairwise).read().splitlines(keepends=True)
    gold = open(f"{GOLD}/statistics_pairwise.csv").read().splitlines(keepends=True)
    # Every data row (everything after the header) must be byte-identical.
    assert got[1:] == gold[1:]


def test_pairwise_header_matches_expected(tmp_path):
    pairwise, _rawstats = _run(tmp_path)
    assert open(pairwise).readline().rstrip("\n") == EXPECTED_PAIRWISE_HEADER


def test_pairwise_header_differs_from_golden_only_by_intended_renames():
    """The refactor may change the header ONLY by the documented renames."""
    old = open(f"{GOLD}/statistics_pairwise.csv").readline().rstrip("\n")
    transformed = old
    for src, dst in HEADER_RENAMES.items():
        transformed = transformed.replace(src, dst)
    assert transformed == EXPECTED_PAIRWISE_HEADER


# --- Characterization of edge cases (Phase 0) ----------------------------------
# Documents CURRENT behavior on two edge cases so the refactor is provably no worse.
#
# NOTE on the `except e:` in statistics(): it is a latent bug (evaluating an unbound
# `e` would raise NameError), but modern scipy returns NaN with a warning instead of
# raising on degenerate input, so the except clause is never entered and no crash
# occurs today. Phase 1 removes the dead try/except entirely.

def test_current_degenerate_pair_does_not_crash(tmp_path):
    """All-identical inputs currently succeed: MWU/KS p=1.0 -> ' ', ttest -> nan."""
    pairwise = str(tmp_path / "deg_violin.csv")
    ms.statistics([[5, 5, 5], [5, 5, 5]], "deg", ["A", "B"], ["x", "y"],
                  filename=pairwise, test_type="median")
    row = open(pairwise).read().splitlines()[1]
    fields = row.split(",")
    assert fields[2] == " " and fields[3] == " "     # Utest_Stars, KStest_Stars
    assert fields[14] == "nan"                        # T is nan for identical inputs


def test_more_than_12_datasets_no_longer_crashes(tmp_path):
    """Phase 1: colors wrap with modulo, so >12 datasets works (was IndexError)."""
    data = [[float(i), float(i) + 1, float(i) + 2] for i in range(13)]
    pairwise = str(tmp_path / "many_violin.csv")
    ms.statistics(data, "many", [str(i) for i in range(13)], [""] * 13, filename=pairwise)
    assert os.path.exists(pairwise)
    # 13 datasets -> C(13,2) = 78 pairwise rows (+ header)
    assert len(open(pairwise).read().splitlines()) == 1 + 78

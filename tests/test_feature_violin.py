"""Tests for the `morphometrics violin` feature comparison across classes."""
import numpy as np
import pandas as pd
import pytest
import yaml

from surface_morphometrics import feature_violin as fv
from surface_morphometrics import morphometrics_stats as ms
from surface_morphometrics.morphometrics_stats import significance_stars, violin


def test_summary_statistic_dispatch():
    values = np.array([1.0, 2.0, 3.0])
    areas = np.ones(3)
    assert abs(fv.summary_statistic(values, areas, "mean") - 2.0) < 1e-9
    assert abs(fv.summary_statistic(values, areas, "median") - 2.0) < 1e-9
    # peak falls back to the data range when none is given
    peak = fv.summary_statistic(values, areas, "peak", bins=2)
    assert 1.0 <= peak <= 3.0
    with pytest.raises(ValueError):
        fv.summary_statistic(values, areas, "mode")


def test_summary_statistic_is_area_weighted():
    values = np.array([0.0, 10.0])
    areas = np.array([9.0, 1.0])          # heavily weights the 0.0
    assert abs(fv.summary_statistic(values, areas, "mean") - 1.0) < 1e-9
    assert fv.summary_statistic(values, areas, "median") == 0.0


def _write_dataset(tmp_path, feature="IMM_dist"):
    seg, work = tmp_path / "seg", tmp_path / "work"
    seg.mkdir(); work.mkdir()
    rng = np.random.default_rng(0)
    for tomo in ("T1", "T2"):
        (seg / f"{tomo}.mrc").touch()
        for label, mu in (("OMM", 12.0), ("ER", 25.0)):
            pd.DataFrame({feature: rng.normal(mu, 1.0, 200),
                          "area": np.ones(200)}
                         ).to_csv(work / f"{tomo}_{label}.AVV_rh9.csv", index=False)
        # IMM has no `feature` column -> must be skipped
        pd.DataFrame({"area": np.ones(200)}).to_csv(work / f"{tomo}_IMM.AVV_rh9.csv", index=False)
    config = {"seg_dir": str(seg) + "/", "work_dir": str(work) + "/",
              "segmentation_values": {"OMM": 1, "IMM": 2, "ER": 3},
              "curvature_measurements": {"radius_hit": 9}}
    return config


def test_collect_feature_skips_classes_without_the_column(tmp_path):
    config = _write_dataset(tmp_path)
    labels, records = fv.collect_feature(config, "IMM_dist")
    assert labels == ["OMM", "IMM", "ER"]          # config order preserved
    found = {label for label, _unit, _strat, _v, _a in records}
    assert found == {"OMM", "ER"}                   # IMM lacks IMM_dist
    assert len(records) == 4                        # 2 classes x 2 tomograms
    # not split -> unit == stratum == tomogram
    for _label, unit, stratum, _v, _a in records:
        assert unit == stratum


def test_collect_feature_drops_nonfinite_and_zero_area(tmp_path):
    config = _write_dataset(tmp_path)
    work = config["work_dir"]
    pd.DataFrame({"IMM_dist": [1.0, np.nan, 3.0, 4.0],
                  "area": [1.0, 1.0, 0.0, 2.0]}).to_csv(work + "T1_OMM.AVV_rh9.csv", index=False)
    _labels, records = fv.collect_feature(config, "IMM_dist")
    values = next(v for label, unit, _s, v, _a in records if label == "OMM" and unit == "T1")
    assert list(values) == [1.0, 4.0]               # NaN and zero-area rows dropped


def test_collect_feature_split_components(tmp_path):
    seg, work = tmp_path / "seg", tmp_path / "work"
    seg.mkdir(); work.mkdir()
    (seg / "T1.mrc").touch()
    # one OMM surface with two organelles (component_number 1 and 2), plus id 0 (no patch)
    pd.DataFrame({"IMM_dist": [10.0, 11.0, 20.0, 21.0, 99.0],
                  "area": [1.0, 1.0, 1.0, 1.0, 1.0],
                  "component_number": [1, 1, 2, 2, 0]}
                 ).to_csv(work / "T1_OMM.AVV_rh9.csv", index=False)
    config = {"seg_dir": str(seg) + "/", "work_dir": str(work) + "/",
              "segmentation_values": {"OMM": 1}, "curvature_measurements": {"radius_hit": 9}}
    _labels, records = fv.collect_feature(config, "IMM_dist", split_components=True)
    assert len(records) == 2                                   # two organelles, id 0 dropped
    units = sorted(r[1] for r in records)
    assert units == ["T1#c1", "T1#c2"]
    assert all(r[2] == "T1" for r in records)                 # stratum is the tomogram
    vals = {r[1]: sorted(r[3]) for r in records}
    assert vals["T1#c1"] == [10.0, 11.0] and vals["T1#c2"] == [20.0, 21.0]


def test_violin_writes_svg_and_png(tmp_path):
    out = tmp_path / "v.svg"
    violin([[1.0, 2.0, 3.0], [4.0, 5.0]], ["OMM", "ER"], "title", "y", filename=str(out))
    assert out.exists()
    assert (tmp_path / "v.png").exists()


def test_violin_with_significance_annotations(tmp_path):
    out = tmp_path / "v.svg"
    violin([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ["OMM", "ER"], "title", "y",
           filename=str(out), annotations=[(0, 1, "**")])
    assert out.exists()


def test_significance_stars_thresholds():
    assert significance_stars(0.0005) == "****"
    assert significance_stars(0.002) == "***"
    assert significance_stars(0.007) == "**"
    assert significance_stars(0.03) == "*"
    assert significance_stars(0.2) == "ns"
    assert significance_stars(np.nan) == "n/a"
    assert significance_stars(None) == "n/a"


def test_pairwise_tests_reports_all_three_tests():
    a = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0]
    b = [9.0, 9.1, 8.9, 9.05, 8.95, 9.0]      # clearly separated
    rows = ms.pairwise_tests([a, b], ["OMM", "ER"])
    assert len(rows) == 1
    row = rows[0]
    assert row["class_a"] == "OMM" and row["class_b"] == "ER"
    assert row["n_a"] == 6 and row["n_b"] == 6
    for key in ("mwu_U", "mwu_p", "mwu_stars", "ttest_t", "ttest_p", "ttest_stars",
                "ks_stat", "ks_p", "ks_stars", "ci95_a", "ci95_b", "mean_a", "mean_b"):
        assert key in row, key
    assert row["mwu_stars"] not in ("ns", "n/a")
    assert row["ttest_p"] < 0.001
    assert row["ks_stat"] == 1.0                       # fully separated
    # ci95 = sem*1.96
    assert abs(row["ci95_a"] - ms.st.sem(a) * 1.96) < 1e-12


def test_pairwise_tests_include_ks_toggle_and_ns_tokens():
    no_ks = ms.pairwise_tests([[1, 2, 3], [4, 5, 6]], ["A", "B"], include_ks=False)[0]
    assert "ks_stat" not in no_ks and "mwu_U" in no_ks
    # custom ns/na tokens (statistics() uses " ")
    blank = ms.pairwise_tests([[5, 5, 5], [5, 5, 5]], ["A", "B"], ns=" ", na=" ")[0]
    assert blank["mwu_stars"] == " " and blank["ttest_stars"] == " "


def test_pairwise_tests_three_classes_and_degenerate_input():
    rows = ms.pairwise_tests([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ["A", "B", "C"])
    assert [(r["class_a"], r["class_b"]) for r in rows] == [("A", "B"), ("A", "C"), ("B", "C")]
    # All-identical groups must not raise; the t-test is undefined -> "n/a".
    degenerate = ms.pairwise_tests([[5, 5, 5], [5, 5, 5]], ["A", "B"])[0]
    assert degenerate["ttest_stars"] == "n/a"
    assert degenerate["mwu_stars"] == "ns"

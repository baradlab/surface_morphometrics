"""Paired comparisons of membrane classes within tomograms (compare: classes)."""
import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from surface_morphometrics import morphometrics_stats as ms
from surface_morphometrics import spatial_stats as ss


def _paired_tomos(n, shift, tomo_sd=0.02, seed=0, size=300):
    """n tomograms; each has a random offset shared by classes A and B (the pairing)."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(n):
        base = 0.05 + rng.normal(0, tomo_sd)
        a.append(rng.normal(base, 0.01, size))
        b.append(rng.normal(base + shift, 0.01, size))
    return a, b


# --- paired_permutation_test ---

def test_paired_permutation_exact_floor_and_detection():
    a, b = _paired_tomos(5, shift=0.01)
    res = ss.paired_permutation_test(a, b, reps=5000)
    assert res["exact"] and res["reps"] == 32 and res["n_pairs"] == 5
    assert res["min_possible_p"] == pytest.approx(2 / 32)
    assert res["p_value"] == pytest.approx(2 / 32)          # every tomogram agrees
    assert res["observed"] > 0.2


def test_paired_permutation_monte_carlo_when_too_many_patterns():
    a, b = _paired_tomos(14, shift=0.01, seed=1)
    res = ss.paired_permutation_test(a, b, reps=500, seed=2)
    assert not res["exact"] and res["reps"] == 500
    assert res["p_value"] == pytest.approx(1 / 501)


def test_paired_permutation_is_calibrated_under_the_null():
    hits = 0
    for seed in range(200):
        a, b = _paired_tomos(6, shift=0.0, seed=seed, size=80)
        hits += ss.paired_permutation_test(a, b, reps=64)["p_value"] <= 0.05
    assert hits / 200 <= 0.08          # nominal 0.05 (floor 2/64 ~ 0.031 is attainable)


def test_centering_keeps_the_test_calibrated_with_large_tomogram_offsets():
    hits = 0
    for seed in range(200):
        a, b = _paired_tomos(6, shift=0.0, tomo_sd=0.05, seed=1000 + seed, size=80)
        hits += ss.paired_permutation_test(a, b, reps=64)["p_value"] <= 0.05
    assert hits / 200 <= 0.08


def test_pairing_beats_unpaired_when_tomograms_vary():
    # A small class shift buried under large tomogram-to-tomogram variation: the paired
    # test sees it, the unpaired (condition-style) permutation does not.
    a, b = _paired_tomos(7, shift=0.004, tomo_sd=0.03, seed=3)
    paired = ss.paired_permutation_test(a, b, reps=5000)["p_value"]
    unpaired = ss.permutation_test(a + b, ["A"] * 7 + ["B"] * 7, reps=2000)["p_value"]
    assert paired < 0.05 < unpaired


def test_uncentered_pooled_ks_loses_the_pairing():
    # Why center="median" is the default: with big tomogram offsets each tomogram's
    # class difference sits at a different place on the axis, the pooled KS reports the
    # largest one regardless of the swaps, and the pairing is wasted.
    a, b = _paired_tomos(5, shift=0.01, tomo_sd=0.05, seed=4)
    centered = ss.paired_permutation_test(a, b)
    raw = ss.paired_permutation_test(a, b, center=None)
    assert centered["p_value"] == pytest.approx(2 / 32)
    assert raw["p_value"] > 0.1
    assert raw["observed"] == pytest.approx(centered["observed_uncentered"])


def test_paired_permutation_wasserstein_and_validation():
    a, b = _paired_tomos(4, shift=0.01)
    assert ss.paired_permutation_test(a, b, statistic="wasserstein")["observed"] > 0
    with pytest.raises(ValueError):
        ss.paired_permutation_test(a, b[:3])
    with pytest.raises(ValueError):
        ss.paired_permutation_test(a, b, statistic="median")
    with pytest.raises(ValueError):
        ss.paired_permutation_test(a, b, center="mode")


# --- paired_tests ---

def test_paired_tests_use_common_tomograms_and_correct():
    by = {"OMM": {"T1": 1.0, "T2": 2.0, "T3": 3.0, "T4": 4.0, "T5": 5.0, "T6": 6.0},
          "IMM": {"T1": 1.5, "T2": 2.6, "T3": 3.4, "T4": 4.7, "T5": 5.5, "T6": 6.4},
          "ER": {"T1": 1.0, "T2": 2.0, "T3": 3.0}}
    rows = ms.paired_tests(by, [("OMM", "IMM"), ("OMM", "ER")], correction="holm")
    r1, r2 = rows
    assert r1["n_pairs"] == 6 and r1["mean_diff"] == pytest.approx(-0.5166667, abs=1e-6)
    assert r1["ttest_rel_p"] < 0.01
    assert r2["n_pairs"] == 3 and r2["wilcoxon_p"] == 1.0   # identical values
    assert r1["wilcoxon_p_adj"] >= r1["wilcoxon_p"] and r1["n_comparisons"] == 2


def test_paired_tests_too_few_pairs_is_nan():
    row = ms.paired_tests({"A": {"T1": 1.0}, "B": {"T1": 2.0, "T2": 3.0}}, [("A", "B")])[0]
    assert row["n_pairs"] == 1 and np.isnan(row["wilcoxon_p"])
    assert row["wilcoxon_stars"] == "n/a"


# --- compare_batch ---

def _write_study(tmp_path, analyses, n_tomos=6):
    seg, work = tmp_path / "seg", tmp_path / "work"
    seg.mkdir(); work.mkdir()
    rng = np.random.default_rng(0)
    for cond in ("V", "T"):
        for i in range(n_tomos):
            tomo = f"{cond}{i}"
            (seg / f"{tomo}.mrc").touch()
            base = 0.05 + rng.normal(0, 0.02)
            for label, shift in (("OMM", 0.0), ("IMM", 0.01), ("ER", 0.0)):
                df = pd.DataFrame({"curvedness_VV": np.abs(rng.normal(base + shift, 0.01, 200)),
                                   "area": np.ones(200)})
                if label == "ER":
                    df = df.drop(columns="curvedness_VV")      # ER lacks the feature
                df.to_csv(work / f"{tomo}_{label}.AVV_rh9.csv", index=False)
    config = {"seg_dir": str(seg) + "/", "work_dir": str(work) + "/",
              "segmentation_values": {"OMM": 1, "IMM": 2, "ER": 3},
              "groups": {"condition": {"Vehicle": ["V*"], "Treated": ["T*"]}},
              "comparison": {"group": "condition"},
              "analyses": analyses}
    cfgp = tmp_path / "study.yml"
    yaml.safe_dump(config, open(cfgp, "w"))
    return str(cfgp)


def _run(args):
    from surface_morphometrics.batch_compare import compare_batch_cli
    return CliRunner().invoke(compare_batch_cli, args)


CLASS_ANALYSIS = {"name": "omm_vs_imm", "compare": "classes",
                  "classes": ["OMM", "IMM", "ER"], "condition": "Vehicle",
                  "feature": "curvedness_VV", "statistic": "median", "range": [0, 0.1]}


def test_batch_class_analysis_end_to_end(tmp_path):
    cfgp = _write_study(tmp_path, [CLASS_ANALYSIS])
    r = _run([cfgp, "--reps", "5000"])
    assert r.exit_code == 0, r.output
    assert "no surfaces with 'curvedness_VV' for ER -- dropped" in r.output
    out = tmp_path / "compare_batch"
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary["class"]) == ["OMM", "IMM"]
    assert set(summary["condition"]) == {"Vehicle"} and set(summary["n_tomograms"]) == {6}
    tests = pd.read_csv(out / "tests.csv")
    s = tests[tests["level"] == "per_tomogram_paired_summary"].iloc[0]
    assert s["n_pairs"] == 6 and s["mean_diff"] < 0 and s["wilcoxon_p"] < 0.05
    p = tests[tests["level"] == "pooled_distribution_paired_permutation"].iloc[0]
    assert p["exact"] and p["n_pairs"] == 6
    assert p["p_permutation"] == pytest.approx(2 / 64)
    assert p["diff"] < 0 and p["diff_ci_high"] < 0          # IMM higher, CI excludes 0
    assert (out / "omm_vs_imm_violin.svg").exists() and (out / "omm_vs_imm_hist.svg").exists()


def test_batch_class_analysis_mixed_with_condition_analysis(tmp_path):
    analyses = [dict(CLASS_ANALYSIS, condition=None, classes=["OMM", "IMM"]),
                {"name": "imm_conditions", "class": "IMM", "feature": "curvedness_VV"}]
    cfgp = _write_study(tmp_path, analyses)
    r = _run([cfgp, "--reps", "200", "--no-plots", "--correction-family", "batch"])
    assert r.exit_code == 0, r.output
    tests = pd.read_csv(tmp_path / "compare_batch" / "tests.csv")
    assert set(tests["level"]) == {"per_tomogram_paired_summary",
                                   "pooled_distribution_paired_permutation",
                                   "per_tomogram_summary",
                                   "pooled_distribution_permutation"}
    paired = tests[tests["level"] == "per_tomogram_paired_summary"].iloc[0]
    assert paired["n_pairs"] == 12 and paired["condition"] == "all"
    assert "wilcoxon_p_adj" in tests.columns and "mwu_p_adj" in tests.columns


def test_batch_class_analysis_reference_and_validation(tmp_path):
    a = dict(CLASS_ANALYSIS, classes=["OMM", "IMM"], reference="IMM")
    cfgp = _write_study(tmp_path, [a])
    r = _run([cfgp, "--reps", "100", "--no-plots"])
    assert r.exit_code == 0, r.output
    t = pd.read_csv(tmp_path / "compare_batch" / "tests.csv")
    assert set(zip(t["class_a"], t["class_b"])) == {("IMM", "OMM")}
    for k, (bad, msg) in enumerate([({"classes": ["OMM"]}, "at least two"),
                                    ({"reference": "PM"}, "reference"),
                                    ({"pairs": [["OMM", "PM"]]}, "pairs"),
                                    ({"compare": "tomograms"}, "compare must be"),
                                    ({"center": "mode"}, "center")]):
        sub = tmp_path / f"bad{k}"
        sub.mkdir()
        r = _run([_write_study(sub, [dict(a, **bad)]), "--no-plots"])
        assert r.exit_code != 0 and msg in r.output, (bad, r.output)


def test_batch_class_analysis_unknown_condition(tmp_path):
    cfgp = _write_study(tmp_path, [dict(CLASS_ANALYSIS, condition="Nope")])
    r = _run([cfgp, "--no-plots"])
    assert r.exit_code != 0 and "Nope" in r.output

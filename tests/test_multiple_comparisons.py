"""Multiple-comparison correction: adjust_pvalues / apply_correction and the commands."""
import glob

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from surface_morphometrics import morphometrics_stats as ms


# --- adjust_pvalues ---

@pytest.mark.parametrize("method", ["holm", "fdr_bh", "bonferroni"])
def test_adjust_matches_statsmodels(method):
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    rng = np.random.default_rng(0)
    for m in (1, 2, 5, 17):
        p = np.concatenate([rng.uniform(0, 0.05, m // 2), rng.uniform(0, 1, m - m // 2)])
        rng.shuffle(p)
        expected = multipletests(p, method=method)[1]
        np.testing.assert_allclose(ms.adjust_pvalues(p, method), expected, rtol=1e-12)


def test_adjust_hand_worked_holm_and_bh():
    p = [0.01, 0.04, 0.03, 0.005]
    # Holm: sorted 0.005,0.01,0.03,0.04 -> x4,x3,x2,x1 = 0.02,0.03,0.06,0.04 -> monotone
    np.testing.assert_allclose(ms.adjust_pvalues(p, "holm"), [0.03, 0.06, 0.06, 0.02])
    # BH: p*m/rank = 0.02,0.02,0.04,0.04 -> step-up minimum from the top
    np.testing.assert_allclose(ms.adjust_pvalues(p, "fdr_bh"), [0.02, 0.04, 0.04, 0.02])


def test_adjust_nan_passthrough_and_family_size():
    adj = ms.adjust_pvalues([0.01, np.nan, 0.02], "bonferroni")
    assert np.isnan(adj[1])
    np.testing.assert_allclose(adj[[0, 2]], [0.02, 0.04])     # m = 2, not 3
    assert np.all(np.isnan(ms.adjust_pvalues([np.nan, np.nan], "holm")))


def test_adjust_none_caps_and_rejects_unknown():
    np.testing.assert_array_equal(ms.adjust_pvalues([0.2, 0.7], "none"), [0.2, 0.7])
    assert ms.adjust_pvalues([0.6, 0.7], "bonferroni").max() == 1.0
    with pytest.raises(ValueError, match="correction"):
        ms.adjust_pvalues([0.1], "sidak")


# --- apply_correction / pairwise_tests ---

def test_apply_correction_recomputes_stars_from_adjusted_p():
    rows = [{"mwu_p": 0.03, "mwu_stars": "*"}, {"mwu_p": 0.04, "mwu_stars": "*"},
            {"mwu_p": 0.5, "mwu_stars": "ns"}]
    ms.apply_correction(rows, "holm")
    assert [r["mwu_p_adj"] for r in rows] == pytest.approx([0.09, 0.09, 0.5])
    assert all(r["mwu_stars"] == "ns" for r in rows)       # nothing survives
    assert rows[0]["correction"] == "holm" and rows[0]["n_comparisons"] == 3
    assert rows[0]["mwu_p"] == 0.03                         # raw p kept


def test_apply_correction_custom_column_without_stars():
    rows = [{"p_permutation": 0.01}, {"p_permutation": 0.02}]
    ms.apply_correction(rows, "bonferroni", pcols=("p_permutation",))
    assert [r["p_permutation_adj"] for r in rows] == pytest.approx([0.02, 0.04])


def test_pairwise_tests_default_is_legacy_uncorrected():
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    legacy = ms.pairwise_tests(data, ["A", "B", "C"])
    assert "mwu_p_adj" not in legacy[0] and "correction" not in legacy[0]
    corrected = ms.pairwise_tests(data, ["A", "B", "C"], correction="holm")
    for leg, cor in zip(legacy, corrected):
        assert cor["mwu_p"] == leg["mwu_p"]
        assert cor["mwu_p_adj"] >= cor["mwu_p"]
        assert cor["ks_p_adj"] >= cor["ks_p"]
    assert corrected[0]["n_comparisons"] == 3


# --- commands ---

def _write_run(tmp_path):
    seg, work = tmp_path / "seg", tmp_path / "work"
    seg.mkdir(); work.mkdir()
    rng = np.random.default_rng(0)
    for tomo in ("T1", "T2", "T3", "T4"):
        (seg / f"{tomo}.mrc").touch()
        for label, mu in (("OMM", 12.0), ("ER", 12.5), ("IMM", 30.0)):
            pd.DataFrame({"IMM_dist": rng.normal(mu, 1.0, 200), "area": np.ones(200)}
                         ).to_csv(work / f"{tomo}_{label}.AVV_rh9.csv", index=False)
    config = {"seg_dir": str(seg) + "/", "work_dir": str(work) + "/",
              "segmentation_values": {"OMM": 1, "IMM": 2, "ER": 3},
              "groups": {"condition": {"A": ["T1", "T2"], "B": "rest"}}}
    cfgp = tmp_path / "config.yml"
    yaml.safe_dump(config, open(cfgp, "w"))
    return str(cfgp), work


@pytest.mark.parametrize("method", ["holm", "none"])
def test_violin_tests_csv_has_adjusted_columns(tmp_path, method):
    from surface_morphometrics.feature_violin import violin_cli
    cfgp, work = _write_run(tmp_path)
    r = CliRunner().invoke(violin_cli, [cfgp, "-n", "IMM_dist", "--test", "mwu",
                                        "--correction", method])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(glob.glob(str(work / "*_tests.csv"))[0])
    assert set(df["correction"]) == {method} and set(df["n_comparisons"]) == {3}
    if method == "none":
        np.testing.assert_allclose(df["mwu_p_adj"], df["mwu_p"])
    else:
        assert (df["mwu_p_adj"] >= df["mwu_p"]).all()
    assert f"adjusted by {method} over 3 pair(s)" in r.output


def test_compare_adjusts_across_classes(tmp_path):
    from surface_morphometrics.feature_compare import compare_cli
    cfgp, work = _write_run(tmp_path)
    r = CliRunner().invoke(compare_cli, [cfgp, "-n", "IMM_dist", "-g", "condition",
                                         "--reps", "200", "--correction", "bonferroni"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(glob.glob(str(work / "*_compare.csv"))[0])
    assert len(df) == 3 and set(df["n_comparisons"]) == {3}
    np.testing.assert_allclose(df["p_value_adj"], np.minimum(df["p_value"] * 3, 1.0))
    assert "adjusted by bonferroni across 3 classes" in r.output


def _batch_config(tmp_path, **comparison):
    rng = np.random.default_rng(1)
    for src, mu in (("a", 0.04), ("b", 0.06), ("c", 0.05)):
        work = tmp_path / src
        work.mkdir()
        for tomo in ("T1", "T2", "T3"):
            for label in ("OMM", "IMM"):
                pd.DataFrame({"curvedness_VV": np.abs(rng.normal(mu, 0.01, 200)),
                              "area": np.ones(200)}
                             ).to_csv(work / f"{tomo}_{label}.AVV_rh9.csv", index=False)
    comp = {"group": "condition", "conditions": ["A", "B", "C"]}
    comp.update(comparison)
    config = {"classes": ["OMM", "IMM"],
              "datasets": {s: {"work_dir": s, "groups": {"condition": {s.upper(): "*"}}}
                           for s in ("a", "b", "c")},
              "comparison": comp,
              "analyses": [{"name": "imm", "class": "IMM", "feature": "curvedness_VV"},
                           {"name": "omm", "class": "OMM", "feature": "curvedness_VV"}]}
    cfgp = tmp_path / "batch.yml"
    yaml.safe_dump(config, open(cfgp, "w"))
    return str(cfgp)


def _run_batch(args):
    from surface_morphometrics.batch_compare import compare_batch_cli
    return CliRunner().invoke(compare_batch_cli, args)


def test_batch_corrects_within_each_analysis_by_default(tmp_path):
    cfgp = _batch_config(tmp_path)
    r = _run_batch([cfgp, "--reps", "100", "--no-plots"])
    assert r.exit_code == 0, r.output
    t = pd.read_csv(tmp_path / "compare_batch" / "tests.csv")
    assert set(t["correction"]) == {"holm"}
    assert set(t["n_comparisons"]) == {3}                 # 3 pairs per analysis
    perm = t[t["level"] == "pooled_distribution_permutation"]
    for _, grp in perm.groupby("analysis"):
        np.testing.assert_allclose(grp["p_permutation_adj"],
                                   ms.adjust_pvalues(grp["p_permutation"], "holm"))
    assert "holm-adj" in r.output


def test_batch_family_spans_the_run(tmp_path):
    cfgp = _batch_config(tmp_path, correction="fdr_bh", correction_family="batch")
    r = _run_batch([cfgp, "--reps", "100", "--no-plots"])
    assert r.exit_code == 0, r.output
    t = pd.read_csv(tmp_path / "compare_batch" / "tests.csv")
    assert set(t["n_comparisons"]) == {6}                 # 2 analyses x 3 pairs
    perm = t[t["level"] == "pooled_distribution_permutation"]
    np.testing.assert_allclose(perm["p_permutation_adj"],
                               ms.adjust_pvalues(perm["p_permutation"], "fdr_bh"))
    assert "across the batch" in r.output


def test_batch_cli_overrides_and_rejects_bad_config(tmp_path):
    cfgp = _batch_config(tmp_path, correction="holm")
    r = _run_batch([cfgp, "--reps", "50", "--no-plots", "--correction", "none"])
    assert r.exit_code == 0, r.output
    t = pd.read_csv(tmp_path / "compare_batch" / "tests.csv")
    s = t[t["level"] == "per_tomogram_summary"]
    np.testing.assert_allclose(s["mwu_p_adj"], s["mwu_p"])
    (tmp_path / "x").mkdir()
    bad = _batch_config(tmp_path / "x", correction="sidak")
    r = _run_batch([bad, "--no-plots"])
    assert r.exit_code != 0 and "correction" in r.output

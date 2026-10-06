"""Tests for `morphometrics compare_batch` -- config-driven N-way analyses."""
import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from surface_morphometrics.batch_compare import (compare_batch_cli, parse_conditions,
                                                 validate_analyses)


def _write(path, tomos, mu, seed):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for tomo in tomos:
        for label in ("OMM", "IMM"):
            n = 300
            pd.DataFrame({"curvedness_VV": np.abs(rng.normal(mu, 0.01, n)),
                          "OMM_dist": rng.uniform(0, 40, n),
                          "component_number": rng.integers(1, 3, n),
                          "area": np.ones(n)}
                         ).to_csv(path / f"{tomo}_{label}.AVV_rh9.csv", index=False)


def _config(tmp_path, analyses=None, **comparison):
    _write(tmp_path / "a", ["T1", "T2", "T3"], 0.04, 0)
    _write(tmp_path / "b", ["T1", "T2", "T3"], 0.06, 1)
    _write(tmp_path / "c", ["T1", "T2"], 0.05, 2)
    comp = {"group": "condition",
            "conditions": [{"name": "A", "short": "a", "color": "#D55E00"}, "B", "C"]}
    comp.update(comparison)
    config = {
        "classes": ["OMM", "IMM"],
        "datasets": {s: {"work_dir": s, "groups": {"condition": {s.upper(): "*"}}}
                     for s in ("a", "b", "c")},
        "comparison": comp,
        "analyses": analyses or [
            {"name": "imm_curv", "class": "IMM", "feature": "curvedness_VV",
             "statistic": "peak", "range": [0, 0.1]},
            {"name": "crista_spread", "class": "IMM", "feature": "curvedness_VV",
             "statistic": "std", "filters": ["IMM:OMM_dist>20"]},
            {"name": "omm_far", "class": "OMM", "feature": "curvedness_VV",
             "statistic": "median", "filters": ["OMM:OMM_dist>20"],
             "ci_statistic": "median"},
        ]}
    cfgp = tmp_path / "batch.yml"
    yaml.safe_dump(config, open(cfgp, "w"))
    return str(cfgp)


def _run(args):
    return CliRunner().invoke(compare_batch_cli, args)


def test_batch_writes_both_levels(tmp_path):
    cfgp = _config(tmp_path)
    r = _run([cfgp, "--reps", "200", "--no-plots"])
    assert r.exit_code == 0, r.output
    summary = pd.read_csv(tmp_path / "compare_batch" / "summary.csv")
    tests = pd.read_csv(tmp_path / "compare_batch" / "tests.csv")
    assert set(summary["analysis"]) == {"imm_curv", "crista_spread", "omm_far"}
    assert len(summary) == 9                                   # 3 analyses x 3 conditions
    perm = tests[tests["level"] == "pooled_distribution_permutation"]
    # no reference -> all 3 pairs, for the two analyses that run the pooled test
    assert set(perm["analysis"]) == {"imm_curv", "omm_far"}
    assert len(perm) == 6
    row = perm[(perm["analysis"] == "imm_curv") & (perm["class_a"] == "A")
               & (perm["class_b"] == "B")].iloc[0]
    assert row["n_tomograms_a"] == 3 and row["n_tomograms_b"] == 3
    assert row["ks"] > 0.5 and row["mean_b"] > row["mean_a"]
    assert set(perm[perm["analysis"] == "omm_far"]["ci_statistic"]) == {"median"}
    summ = tests[tests["level"] == "per_tomogram_summary"]
    assert set(summ["analysis"]) == {"imm_curv", "crista_spread", "omm_far"}


def test_batch_reference_and_conditions_and_only(tmp_path):
    cfgp = _config(tmp_path, reference="A")
    r = _run([cfgp, "--reps", "100", "--no-plots", "--only", "imm_curv",
              "--conditions", "A", "--conditions", "B"])
    assert r.exit_code == 0, r.output
    tests = pd.read_csv(tmp_path / "compare_batch" / "tests.csv")
    perm = tests[tests["level"] == "pooled_distribution_permutation"]
    assert list(zip(perm["class_a"], perm["class_b"])) == [("A", "B")]
    assert set(tests["analysis"]) == {"imm_curv"}


def test_batch_reference_tests_against_reference_only(tmp_path):
    cfgp = _config(tmp_path, reference="B")
    r = _run([cfgp, "--reps", "100", "--no-plots", "--only", "imm_curv"])
    assert r.exit_code == 0, r.output
    perm = pd.read_csv(tmp_path / "compare_batch" / "tests.csv").query(
        "level == 'pooled_distribution_permutation'")
    assert sorted(zip(perm["class_a"], perm["class_b"])) == [("B", "A"), ("B", "C")]


def test_batch_split_components_nests(tmp_path):
    cfgp = _config(tmp_path)
    r = _run([cfgp, "--reps", "100", "--no-plots", "--only", "imm_curv",
              "--split-components"])
    assert r.exit_code == 0, r.output
    perm = pd.read_csv(tmp_path / "compare_batch" / "tests.csv").query(
        "level == 'pooled_distribution_permutation'")
    row = perm.iloc[0]
    assert row["n_units_a"] > row["n_tomograms_a"]


def test_batch_seed_is_reproducible(tmp_path):
    cfgp = _config(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    assert _run([cfgp, "--reps", "100", "--no-plots", "--output", str(out1)]).exit_code == 0
    assert _run([cfgp, "--reps", "100", "--no-plots", "--output", str(out2)]).exit_code == 0
    pd.testing.assert_frame_equal(pd.read_csv(out1 / "tests.csv"),
                                  pd.read_csv(out2 / "tests.csv"))


def test_batch_plots(tmp_path):
    cfgp = _config(tmp_path)
    r = _run([cfgp, "--reps", "50", "--only", "imm_curv"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "compare_batch" / "imm_curv_violin.svg").exists()
    assert (tmp_path / "compare_batch" / "imm_curv_hist.svg").exists()


def test_batch_single_dataset_config(tmp_path):
    seg, work = tmp_path / "seg", tmp_path / "work"
    seg.mkdir()
    _write(work, ["TF1", "TF2", "TE1", "TE2"], 0.05, 0)
    for t in ("TF1", "TF2", "TE1", "TE2"):
        (seg / f"{t}.mrc").touch()
    config = {"seg_dir": str(seg) + "/", "work_dir": str(work) + "/",
              "segmentation_values": {"OMM": 1, "IMM": 2},
              "groups": {"condition": {"Tg": ["TF*"], "Vehicle": ["TE*"]}},
              "analyses": [{"name": "c", "class": "IMM", "feature": "curvedness_VV"}]}
    cfgp = tmp_path / "single.yml"
    yaml.safe_dump(config, open(cfgp, "w"))
    r = _run([str(cfgp), "--reps", "50", "--no-plots"])
    assert r.exit_code == 0, r.output
    summary = pd.read_csv(tmp_path / "compare_batch" / "summary.csv")
    assert list(summary["condition"]) == ["Tg", "Vehicle"]


def test_batch_rejects_unknown_condition(tmp_path):
    cfgp = _config(tmp_path)
    config = yaml.safe_load(open(cfgp))
    config["comparison"]["conditions"].append("Nope")
    yaml.safe_dump(config, open(cfgp, "w"))
    r = _run([cfgp, "--no-plots"])
    assert r.exit_code != 0 and "Nope" in r.output


def test_validate_analyses():
    with pytest.raises(ValueError, match="missing"):
        validate_analyses([{"name": "x", "class": "IMM"}])
    with pytest.raises(ValueError, match="duplicate"):
        validate_analyses([{"name": "x", "class": "I", "feature": "f"}] * 2)
    with pytest.raises(ValueError, match="statistic"):
        validate_analyses([{"name": "x", "class": "I", "feature": "f", "statistic": "max"}])
    with pytest.raises(ValueError, match="bad filter"):
        validate_analyses([{"name": "x", "class": "I", "feature": "f", "filters": ["a~1"]}])


def test_parse_conditions_defaults():
    out = parse_conditions({}, ["Tg", "Vehicle"])
    assert [c["name"] for c in out] == ["Tg", "Vehicle"]
    assert out[0]["short"] == "Tg" and out[0]["color"].startswith("#")


def test_batch_explicit_pairs(tmp_path):
    cfgp = _config(tmp_path, pairs=[["A", "B"], ["C", "B"]], reference="A")
    r = _run([cfgp, "--reps", "50", "--no-plots", "--only", "imm_curv"])
    assert r.exit_code == 0, r.output
    tests = pd.read_csv(tmp_path / "compare_batch" / "tests.csv")
    perm = tests.query("level == 'pooled_distribution_permutation'")
    assert list(zip(perm["class_a"], perm["class_b"])) == [("A", "B"), ("C", "B")]
    summ = tests.query("level == 'per_tomogram_summary'")
    assert {frozenset(p) for p in zip(summ["class_a"], summ["class_b"])} == \
        {frozenset(("A", "B")), frozenset(("B", "C"))}


def test_batch_rejects_bad_pair(tmp_path):
    cfgp = _config(tmp_path, pairs=[["A", "Z"]])
    r = _run([cfgp, "--no-plots"])
    assert r.exit_code != 0 and "pairs" in r.output

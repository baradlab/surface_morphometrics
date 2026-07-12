"""Tests for `morphometrics compare` -- the treatment comparison command (phase 5)."""
import glob

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from surface_morphometrics.feature_compare import compare_cli


def _write_grouped_run(tmp_path, sep=True):
    """A run with Tg (TF*) and Vehicle (TE*) tomograms; sep -> the conditions differ."""
    seg, work = tmp_path / "seg", tmp_path / "work"
    seg.mkdir(); work.mkdir()
    rng = np.random.default_rng(0)
    spec = [("TF1", 0.06), ("TF2", 0.062), ("TF3", 0.058),
            ("TE1", 0.04), ("TE2", 0.042), ("TE3", 0.039)]
    for tomo, mu in spec:
        if not sep:
            mu = 0.05
        (seg / f"{tomo}.mrc").touch()
        for label in ("OMM", "IMM"):
            pd.DataFrame({"curvedness_VV": np.abs(rng.normal(mu, 0.01, 400)),
                          "OMM_dist": rng.uniform(0, 40, 400),
                          "component_number": rng.integers(1, 3, 400),
                          "area": np.ones(400)}
                         ).to_csv(work / f"{tomo}_{label}.AVV_rh9.csv", index=False)
    config = {"seg_dir": str(seg) + "/", "work_dir": str(work) + "/",
              "segmentation_values": {"OMM": 1, "IMM": 2},
              "curvature_measurements": {"radius_hit": 9},
              "groups": {"condition": {"Tg": ["TF*"], "Vehicle": ["TE*"]}}}
    cfgp = tmp_path / "config.yml"
    yaml.safe_dump(config, open(cfgp, "w"))
    return str(cfgp), str(work) + "/"


def _run(args):
    return CliRunner().invoke(compare_cli, args)


def test_compare_detects_difference(tmp_path):
    cfgp, work = _write_grouped_run(tmp_path, sep=True)
    r = _run([cfgp, "-n", "curvedness_VV", "-g", "condition", "--reps", "2000"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(glob.glob(work + "*_compare.csv")[0])
    assert set(df["class"]) == {"OMM", "IMM"}
    row = df[df["class"] == "IMM"].iloc[0]
    assert row["condition_a"] == "Tg" and row["condition_b"] == "Vehicle"
    assert row["observed"] > 0.5                       # clearly separated distributions
    assert row["n_blocks_a"] == 3 and row["n_blocks_b"] == 3
    # means recover the two levels with the right ordering
    assert row["mean_a"] > row["mean_b"]
    assert row["p_value"] == pytest.approx(row["min_possible_p"], abs=0.02)  # at the floor


def test_compare_split_components_permutes_at_tomogram_level(tmp_path):
    cfgp, work = _write_grouped_run(tmp_path, sep=True)
    r = _run([cfgp, "-n", "curvedness_VV", "-g", "condition", "--split-components",
              "--reps", "1000"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(glob.glob(work + "*_compare.csv")[0])
    row = df[df["class"] == "IMM"].iloc[0]
    # organelles are the units, but blocks (the p-value floor) are still the 3+3 tomograms
    assert row["n_units_a"] > row["n_blocks_a"]
    assert row["n_blocks_a"] == 3 and row["n_blocks_b"] == 3


def test_compare_with_filter(tmp_path):
    cfgp, work = _write_grouped_run(tmp_path, sep=True)
    r = _run([cfgp, "-n", "curvedness_VV", "-g", "condition",
              "--filter", "OMM_dist>=20", "--reps", "500"])
    assert r.exit_code == 0, r.output
    assert "triangle filter: OMM_dist>=20" in r.output


def test_compare_missing_group_errors(tmp_path):
    cfgp, _work = _write_grouped_run(tmp_path)
    r = _run([cfgp, "-n", "curvedness_VV", "-g", "morphology"])
    assert r.exit_code != 0
    assert "no group 'morphology'" in r.output


def test_compare_requires_two_buckets(tmp_path):
    cfgp, _work = _write_grouped_run(tmp_path)
    config = yaml.safe_load(open(cfgp))
    config["groups"]["condition"]["Extra"] = ["ZZ*"]     # now three buckets
    yaml.safe_dump(config, open(cfgp, "w"))
    r = _run([cfgp, "-n", "curvedness_VV", "-g", "condition"])
    assert r.exit_code != 0
    assert "exactly" in r.output and "two conditions" in r.output


def test_compare_missing_feature_errors(tmp_path):
    cfgp, _work = _write_grouped_run(tmp_path)
    r = _run([cfgp, "-n", "no_such_feature", "-g", "condition"])
    assert r.exit_code != 0
    assert "no surfaces" in r.output

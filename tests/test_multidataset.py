"""Tests for multi-dataset federation, the pickle backend, and cross-source `compare`."""
import glob
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from surface_morphometrics.dataset import Dataset, PickleDataset, read_experiment_pickle
from surface_morphometrics.feature_compare import compare_cli
from surface_morphometrics.morphometrics_stats import Experiment, Tomogram
from surface_morphometrics.multidataset import MultiDataset, load_for_analysis


def _surface(rng, mu, n=300):
    return pd.DataFrame({"curvedness_VV": np.abs(rng.normal(mu, 0.01, n)),
                         "OMM_dist": rng.uniform(0, 40, n),
                         "component_number": rng.integers(1, 3, n),
                         "area": np.ones(n)})


def _write_work_dir(path, tomos, mu, seed=0, labels=("OMM", "IMM")):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for tomo in tomos:
        for label in labels:
            _surface(rng, mu).to_csv(path / f"{tomo}_{label}.AVV_rh9.csv", index=False)
    return str(path) + "/"


def _write_pickle(path, tomos, mu, seed=1, labels=("OMM", "IMM")):
    rng = np.random.default_rng(seed)
    exp = Experiment("legacy")
    for tomo in tomos:
        t = Tomogram(tomo, [], [])
        for label in labels:
            t[label] = _surface(rng, mu)
        exp[tomo] = t
    with open(path, "wb") as handle:
        pickle.dump(exp, handle)
    return str(path)


# --- Dataset.from_work_dir ---

def test_from_work_dir_discovers_from_csv_names(tmp_path):
    work = _write_work_dir(tmp_path / "w", ["T1", "T2_labels"], 0.05)
    ds = Dataset.from_work_dir(work, ["OMM", "IMM"])
    assert ds.tomograms() == ["T1", "T2_labels"]
    ds2 = Dataset.from_work_dir(work, ["OMM", "IMM"], exclude_tomograms=["T2*"])
    assert ds2.tomograms() == ["T1"] and ds2.excluded_tomograms == ["T2_labels"]


# --- PickleDataset ---

def test_pickle_dataset_matches_csv_backend(tmp_path):
    rng = np.random.default_rng(3)
    frames = {(t, l): _surface(rng, 0.05) for t in ("A", "B") for l in ("OMM", "IMM")}
    work = tmp_path / "w"; work.mkdir()
    exp = Experiment("x")
    for t in ("A", "B"):
        tomo = Tomogram(t, [], [])
        for l in ("OMM", "IMM"):
            frames[(t, l)].to_csv(work / f"{t}_{l}.AVV_rh9.csv", index=False)
            tomo[l] = pd.read_csv(work / f"{t}_{l}.AVV_rh9.csv")
        exp[t] = tomo
    pkl = tmp_path / "x.pkl"
    pickle.dump(exp, open(pkl, "wb"))

    csv_ds = Dataset.from_work_dir(str(work), ["OMM", "IMM"])
    pk_ds = PickleDataset.from_pickle(str(pkl))
    assert pk_ds.labels == ["IMM", "OMM"]          # discovered, sorted
    pk_ds = PickleDataset.from_pickle(str(pkl), labels=["OMM", "IMM"])
    assert pk_ds.exists("A", "OMM") and not pk_ds.exists("A", "ER")
    clauses = [{"cls": None, "prop": "OMM_dist", "op": ">=", "value": 20.0,
                "spec": "OMM_dist>=20"}]
    a, _ = csv_ds.collect_feature("curvedness_VV", filters=clauses)
    b, _ = pk_ds.collect_feature("curvedness_VV", filters=clauses)
    assert [r[:3] for r in a] == [r[:3] for r in b]
    for ra, rb in zip(a, b):
        np.testing.assert_allclose(ra[3], rb[3])


def test_legacy_pickle_recorded_under_main_loads(tmp_path, monkeypatch):
    """Script-made pickles name their classes `__main__.Experiment`; remap them."""
    import __main__
    for cls in (Experiment, Tomogram):
        monkeypatch.setattr(cls, "__module__", "__main__")
        monkeypatch.setattr(__main__, cls.__name__, cls, raising=False)
    path = _write_pickle(tmp_path / "legacy.pkl", ["T1"], 0.05)
    monkeypatch.undo()
    assert b"__main__" in open(path, "rb").read()
    with pytest.raises(AttributeError):
        pickle.load(open(path, "rb"))           # the failure the research scripts hit
    exp = read_experiment_pickle(path)
    assert isinstance(exp, Experiment) and exp["T1"].has_key("OMM")


# --- MultiDataset ---

def _two_sources(tmp_path):
    w1 = _write_work_dir(tmp_path / "ctrl", ["T1", "T2", "T3"], 0.04, seed=0)
    w2 = _write_work_dir(tmp_path / "mut", ["T1", "T2", "T3"], 0.06, seed=1)
    m1 = Dataset.from_work_dir(w1, ["OMM", "IMM"], groups={"condition": {"Control": "*"}})
    m2 = Dataset.from_work_dir(w2, ["OMM", "IMM"], groups={"condition": {"Mutant": "*"}})
    return MultiDataset({"ctrl": m1, "mut": m2}), m1, m2


def test_multidataset_namespaces_units_and_strata(tmp_path):
    md, m1, m2 = _two_sources(tmp_path)
    assert len(md) == 6 and md.labels == ["OMM", "IMM"]
    recs, diag = md.collect_feature("curvedness_VV")
    strata = {r[2] for r in recs}
    # the same tomogram name in both sources stays two distinct strata
    assert {"ctrl/T1", "mut/T1"} <= strata and len(strata) == 6
    assert diag["used_tomograms"][0] == "ctrl/T1"
    assert [r[0] for r in recs] == sorted([r[0] for r in recs], key=["OMM", "IMM"].index)


def test_multidataset_collect_parity_with_members(tmp_path):
    md, m1, m2 = _two_sources(tmp_path)
    recs, _ = md.collect_feature("curvedness_VV", split_components=True)
    by_hand = []
    for src, ds in (("ctrl", m1), ("mut", m2)):
        r, _ = ds.collect_feature("curvedness_VV", split_components=True)
        by_hand += [(lab, f"{src}/{u}") for lab, u, *_ in r]
    assert sorted((r[0], r[1]) for r in recs) == sorted(by_hand)
    assert any("#c" in r[1] for r in recs)


def test_multidataset_metadata_merges_and_adds_dataset(tmp_path):
    md, _, _ = _two_sources(tmp_path)
    assert md.metadata("mut/T2") == {"dataset": "mut", "condition": "Mutant"}
    assert md.group_values("condition") == ["Control", "Mutant"]
    assert md.group_values("dataset") == ["ctrl", "mut"]


def test_from_config_mixed_backends_and_relative_paths(tmp_path):
    _write_work_dir(tmp_path / "ctrl", ["C1", "C2"], 0.04)
    _write_pickle(tmp_path / "legacy.pkl", ["P1", "P2", "P3"], 0.06)
    config = {"classes": ["OMM", "IMM"],
              "datasets": {
                  "ctrl": {"work_dir": "ctrl", "groups": {"condition": {"Control": "*"}}},
                  "old": {"work_dir": "gone/", "pickle": "legacy.pkl",
                          "exclude_tomograms": ["P3"],
                          "groups": {"condition": {"Pos": ["P1"], "Neg": "rest"}}}}}
    md = MultiDataset.from_config(config, base_dir=str(tmp_path))
    assert isinstance(md["old"], PickleDataset) and not isinstance(md["ctrl"], PickleDataset)
    assert md.tomograms() == ["ctrl/C1", "ctrl/C2", "old/P1", "old/P2"]
    assert md.metadata("old/P1")["condition"] == "Pos"
    assert md.metadata("old/P2")["condition"] == "Neg"


def test_from_config_missing_data_errors_unless_optional(tmp_path):
    _write_work_dir(tmp_path / "ctrl", ["C1"], 0.04)
    config = {"classes": ["OMM"], "datasets": {"ctrl": {"work_dir": "ctrl"},
                                               "lost": {"work_dir": "nope"}}}
    with pytest.raises(ValueError, match="lost"):
        MultiDataset.from_config(config, base_dir=str(tmp_path))
    config["datasets"]["lost"]["optional"] = True
    md = MultiDataset.from_config(config, base_dir=str(tmp_path))
    assert list(md.members) == ["ctrl"]


def test_from_config_validates_member_groups(tmp_path):
    _write_work_dir(tmp_path / "ctrl", ["C1", "C2"], 0.04)
    config = {"classes": ["OMM"], "datasets": {"ctrl": {
        "work_dir": "ctrl", "groups": {"condition": {"A": ["C1"]}}}}}
    with pytest.raises(ValueError, match="C2"):
        MultiDataset.from_config(config, base_dir=str(tmp_path))


def test_load_for_analysis_single_vs_multi(tmp_path):
    work = _write_work_dir(tmp_path / "w", ["T1"], 0.05)
    seg = tmp_path / "seg"; seg.mkdir(); (seg / "T1.mrc").touch()
    single = {"seg_dir": str(seg) + "/", "work_dir": work,
              "segmentation_values": {"OMM": 1}}
    assert isinstance(load_for_analysis(single), Dataset)
    multi = {"classes": ["OMM"], "datasets": {"a": {"work_dir": work}}}
    assert isinstance(load_for_analysis(multi, str(tmp_path / "c.yml")), MultiDataset)


# --- cross-source compare ---

def _multi_config(tmp_path, sep=True):
    _write_work_dir(tmp_path / "ctrl", ["T1", "T2", "T3"], 0.04, seed=0)
    _write_work_dir(tmp_path / "mut", ["T1", "T2", "T3"], 0.06 if sep else 0.04, seed=1)
    config = {"classes": ["OMM", "IMM"],
              "datasets": {"ctrl": {"work_dir": "ctrl",
                                    "groups": {"condition": {"Control": "*"}}},
                           "mut": {"work_dir": "mut",
                                   "groups": {"condition": {"Mutant": "*"}}}}}
    cfgp = tmp_path / "multi.yml"
    yaml.safe_dump(config, open(cfgp, "w"))
    return str(cfgp)


def test_compare_across_datasets(tmp_path):
    cfgp = _multi_config(tmp_path)
    r = CliRunner().invoke(compare_cli, [cfgp, "-n", "curvedness_VV", "-g", "condition",
                                         "--reps", "500"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(glob.glob(str(tmp_path / "*_compare.csv"))[0])
    row = df[df["class"] == "IMM"].iloc[0]
    assert row["n_blocks_a"] == 3 and row["n_blocks_b"] == 3    # T1 in each is distinct
    assert row["observed"] > 0.5 and row["mean_b"] > row["mean_a"]


def test_compare_implicit_dataset_group(tmp_path):
    cfgp = _multi_config(tmp_path)
    r = CliRunner().invoke(compare_cli, [cfgp, "-n", "curvedness_VV", "-g", "dataset",
                                         "--reps", "200"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(glob.glob(str(tmp_path / "*_dataset_compare.csv"))[0])
    assert set(df["condition_a"]) == {"ctrl"}


def test_compare_conditions_picks_two_of_many(tmp_path):
    cfgp = _multi_config(tmp_path)
    config = yaml.safe_load(open(cfgp))
    _write_work_dir(tmp_path / "third", ["T1", "T2"], 0.05, seed=2)
    config["datasets"]["third"] = {"work_dir": "third",
                                   "groups": {"condition": {"Third": "*"}}}
    yaml.safe_dump(config, open(cfgp, "w"))
    r = CliRunner().invoke(compare_cli, [cfgp, "-n", "curvedness_VV", "-g", "condition"])
    assert r.exit_code != 0 and "--conditions" in r.output
    r = CliRunner().invoke(compare_cli, [cfgp, "-n", "curvedness_VV", "-g", "condition",
                                         "--conditions", "Control", "Third", "--reps", "200"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(glob.glob(str(tmp_path / "*_condition_compare.csv"))[0])
    assert set(df["condition_b"]) == {"Third"} and set(df["n_blocks_b"]) == {2}

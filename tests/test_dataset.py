"""Tests for the lazy, metadata-aware Dataset model (phase 3) and parquet cache (phase 4)."""
import os
from glob import glob

import numpy as np
import pandas as pd
import pytest

from surface_morphometrics.dataset import Dataset, assign_groups, SurfaceView


def _make_run(tmp_path, tomograms=("TF1", "TE1", "UF3"), labels=("OMM", "IMM"),
              radius_hit=9):
    """Write a minimal run: seg MRCs + per-surface CSVs, return a config dict."""
    seg, work = tmp_path / "seg", tmp_path / "work"
    seg.mkdir(); work.mkdir()
    for tomo in tomograms:
        (seg / f"{tomo}.mrc").touch()
        for label in labels:
            pd.DataFrame({"IMM_dist": np.arange(5.0), "area": np.ones(5)}
                         ).to_csv(work / f"{tomo}_{label}.AVV_rh{radius_hit}.csv", index=False)
    config = {"seg_dir": str(seg) + "/", "work_dir": str(work) + "/",
              "segmentation_values": {label: i + 1 for i, label in enumerate(labels)},
              "curvature_measurements": {"radius_hit": radius_hit}}
    return config


# --- group assignment / validation --------------------------------------------------

def test_assign_groups_by_glob():
    groups = {"condition": {"Tg": ["TF*", "UF*"], "Vehicle": ["TE*", "UE*"]},
              "morphology": {"fragmented": ["?F*"], "elongated": ["?E*"]}}
    meta = assign_groups(["TF1", "TE1", "UF3"], groups)
    assert meta["TF1"] == {"condition": "Tg", "morphology": "fragmented"}
    assert meta["TE1"] == {"condition": "Vehicle", "morphology": "elongated"}
    assert meta["UF3"] == {"condition": "Tg", "morphology": "fragmented"}


def test_assign_groups_unassigned_raises():
    groups = {"condition": {"Tg": ["TF*"]}}          # nothing matches TE1
    with pytest.raises(ValueError, match="no bucket in group 'condition'"):
        assign_groups(["TF1", "TE1"], groups)


def test_assign_groups_ambiguous_raises():
    groups = {"condition": {"Tg": ["T*"], "Vehicle": ["TF*"]}}   # TF1 matches both
    with pytest.raises(ValueError, match="matches 2 buckets"):
        assign_groups(["TF1"], groups)


def test_assign_groups_empty_is_noop():
    assert assign_groups(["TF1", "TE1"], {}) == {"TF1": {}, "TE1": {}}


def test_assign_groups_default_bucket_catches_the_rest():
    # "these named tomograms are Positive, everything else is Negative"
    groups = {"drp1": {"Positive": ["TF1", "TF2"], "Negative": []}}
    meta = assign_groups(["TF1", "TF2", "TE1", "UF3"], groups)
    assert meta["TF1"] == {"drp1": "Positive"}
    assert meta["TE1"] == {"drp1": "Negative"}
    assert meta["UF3"] == {"drp1": "Negative"}


def test_assign_groups_rejects_two_default_buckets():
    groups = {"drp1": {"Positive": [], "Negative": []}}
    with pytest.raises(ValueError, match="more than one default"):
        assign_groups(["TF1"], groups)


# --- construction / selection from config -------------------------------------------

def test_from_config_discovers_tomograms(tmp_path):
    config = _make_run(tmp_path)
    ds = Dataset.from_config(config)
    assert sorted(ds.tomograms()) == ["TE1", "TF1", "UF3"]
    assert ds.labels == ["OMM", "IMM"]
    assert len(ds) == 3


def test_from_config_applies_selection(tmp_path):
    config = _make_run(tmp_path)
    config["statistics"] = {"exclude_tomograms": ["UF3"], "include_tomograms": ["T*"]}
    ds = Dataset.from_config(config)
    assert sorted(ds.tomograms()) == ["TE1", "TF1"]      # UF3 excluded, UF* wouldn't match T*
    assert "UF3" not in ds


def test_from_config_reads_groups(tmp_path):
    config = _make_run(tmp_path)
    config["groups"] = {"condition": {"Tg": ["?F*"], "Vehicle": ["?E*"]}}
    ds = Dataset.from_config(config)
    assert ds.metadata("TF1") == {"condition": "Tg"}
    assert ds.metadata("TE1") == {"condition": "Vehicle"}


def test_bad_groups_raise_at_construction(tmp_path):
    config = _make_run(tmp_path)
    config["groups"] = {"condition": {"Tg": ["TF*"]}}    # TE1, UF3 unassigned
    with pytest.raises(ValueError, match="exactly one bucket"):
        Dataset.from_config(config)


# --- lazy loading & access ----------------------------------------------------------

def test_lazy_load_caches_and_defers(tmp_path):
    config = _make_run(tmp_path)
    ds = Dataset.from_config(config)
    assert ds._cache == {}                                # nothing read yet
    df = ds["TF1"]["OMM"]
    assert list(df["IMM_dist"]) == [0, 1, 2, 3, 4]
    assert ("TF1", "OMM") in ds._cache
    assert ds.load("TF1", "OMM") is df                    # same cached object


def test_getitem_access_pattern_matches_experiment(tmp_path):
    config = _make_run(tmp_path)
    ds = Dataset.from_config(config)
    # dataset[tomo][label] -> dataframe, like Experiment[tomo][label]
    assert isinstance(ds["TF1"]["IMM"], pd.DataFrame)
    assert "IMM" in ds["TF1"] and ds["TF1"].classes() == ["OMM", "IMM"]
    with pytest.raises(KeyError):
        ds["NOPE"]


def test_load_missing_surface_raises(tmp_path):
    config = _make_run(tmp_path, tomograms=("TF1",), labels=("OMM",))
    ds = Dataset.from_config(config)
    with pytest.raises(KeyError, match="no surface file"):
        ds.load("TF1", "IMM")
    assert not ds.surface("TF1", "IMM").exists()


def test_surface_view_path_and_metadata(tmp_path):
    config = _make_run(tmp_path)
    config["groups"] = {"condition": {"Tg": ["?F*"], "Vehicle": ["?E*"]}}
    ds = Dataset.from_config(config)
    view = ds.surface("TF1", "OMM")
    assert isinstance(view, SurfaceView)
    assert view.path.endswith("TF1_OMM.AVV_rh9.csv")
    assert view.exists()
    assert view.metadata == {"condition": "Tg"}


# --- metadata-aware selection -------------------------------------------------------

def test_surfaces_selects_by_class_and_metadata(tmp_path):
    config = _make_run(tmp_path)
    config["groups"] = {"condition": {"Tg": ["?F*"], "Vehicle": ["?E*"]}}
    ds = Dataset.from_config(config)
    # all IMM surfaces of Tg tomograms (TF1, UF3)
    imm_tg = sorted(v.tomo for v in ds.surfaces(label="IMM", condition="Tg"))
    assert imm_tg == ["TF1", "UF3"]
    # every surface (both classes) of Vehicle tomograms
    vehicle = sorted((v.tomo, v.label) for v in ds.surfaces(condition="Vehicle"))
    assert vehicle == [("TE1", "IMM"), ("TE1", "OMM")]


def test_surfaces_exists_only(tmp_path):
    config = _make_run(tmp_path, tomograms=("TF1",), labels=("OMM",))
    ds = Dataset.from_config(config)
    # IMM has no file; exists_only (default) skips it, exists_only=False yields it
    assert [v.label for v in ds.surfaces()] == ["OMM"]
    labels = sorted(v.label for v in ds.surfaces(exists_only=False))
    assert labels == ["OMM"]                              # only OMM configured here


# --- parquet parse-cache (phase 4) --------------------------------------------------

pyarrow = pytest.importorskip("pyarrow")


def test_resolve_cache_dir():
    assert Dataset._resolve_cache_dir(False, "/w/") is None
    assert Dataset._resolve_cache_dir(None, "/w/") is None
    assert Dataset._resolve_cache_dir(True, "/w/") == "/w/.morphometrics_cache/"
    assert Dataset._resolve_cache_dir("/tmp/c", "/w/") == "/tmp/c/"


def test_from_config_enables_cache_from_statistics(tmp_path):
    config = _make_run(tmp_path)
    config["statistics"] = {"cache": True}
    ds = Dataset.from_config(config)
    assert ds.cache_dir == config["work_dir"] + ".morphometrics_cache/"


def test_cache_writes_parquet_and_reads_it_back(tmp_path, monkeypatch):
    config = _make_run(tmp_path)
    cache_dir = str(tmp_path / "cache") + "/"
    ds = Dataset.from_config(config, cache=cache_dir)
    df = ds.load("TF1", "OMM")
    parquets = glob(cache_dir + "TF1_OMM-*.parquet")
    assert len(parquets) == 1                             # parse cached to parquet

    # A fresh Dataset (empty in-memory cache) must read the parquet, not re-parse the CSV:
    # make read_csv blow up so a cache miss would be obvious.
    ds2 = Dataset.from_config(config, cache=cache_dir)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("read_csv called on a cache hit"))
    df2 = ds2.load("TF1", "OMM")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))


def test_cache_invalidates_when_source_changes(tmp_path):
    config = _make_run(tmp_path)
    cache_dir = str(tmp_path / "cache") + "/"
    ds = Dataset.from_config(config, cache=cache_dir)
    ds.load("TF1", "OMM")
    first = glob(cache_dir + "TF1_OMM-*.parquet")[0]

    # Rewrite the source with different content; the key (mtime/size) changes.
    path = ds.surface_path("TF1", "OMM")
    pd.DataFrame({"IMM_dist": np.arange(9.0), "area": np.ones(9)}).to_csv(path, index=False)
    ds2 = Dataset.from_config(config, cache=cache_dir)
    df = ds2.load("TF1", "OMM")
    assert len(df) == 9                                   # re-parsed the new CSV
    remaining = glob(cache_dir + "TF1_OMM-*.parquet")
    assert len(remaining) == 1 and remaining[0] != first  # stale parquet pruned


def test_cache_off_by_default(tmp_path):
    config = _make_run(tmp_path)
    ds = Dataset.from_config(config)
    assert ds.cache_dir is None
    ds.load("TF1", "OMM")
    assert not glob(str(tmp_path / "**" / "*.parquet"), recursive=True)


def test_cache_disables_gracefully_when_write_fails(tmp_path, monkeypatch, capsys):
    config = _make_run(tmp_path)
    ds = Dataset.from_config(config, cache=str(tmp_path / "cache") + "/")
    monkeypatch.setattr(pd.DataFrame, "to_parquet",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError("no engine")))
    df = ds.load("TF1", "OMM")                            # must still succeed from CSV
    assert len(df) == 5 and ds._parquet_disabled
    assert "parquet caching disabled" in capsys.readouterr().out


def test_collect_feature_engine(tmp_path):
    config = _make_run(tmp_path, tomograms=("TF1", "TE1"))
    config["groups"] = {"condition": {"Tg": ["?F*"], "Vehicle": ["?E*"]}}
    ds = Dataset.from_config(config)
    records, diag = ds.collect_feature("IMM_dist")
    # 2 classes x 2 tomograms, each with 5 finite triangles
    assert len(records) == 4
    assert diag["dropped_units"] == 0 and sorted(diag["used_tomograms"]) == ["TE1", "TF1"]
    for _label, unit, stratum, values, areas in records:
        assert unit == stratum and len(values) == 5 and len(areas) == 5
    # size threshold drops everything (each unit has 5 triangles)
    _records, diag2 = ds.collect_feature("IMM_dist", min_triangles=10)
    assert _records == [] and diag2["dropped_units"] == 4


def test_to_experiment_and_pickle_roundtrip(tmp_path):
    config = _make_run(tmp_path, tomograms=("TF1", "TE1"))
    ds = Dataset.from_config(config)
    exp = ds.to_experiment(name="expt")
    assert exp.name == "expt"
    assert set(exp.tomograms) == {"TF1", "TE1"}
    pd.testing.assert_frame_equal(exp["TF1"]["OMM"], ds.load("TF1", "OMM"))

    import pickle
    pkl = tmp_path / "expt.pkl"
    ds.to_pickle(str(pkl))
    with open(pkl, "rb") as handle:
        loaded = pickle.load(handle)
    assert set(loaded.tomograms) == {"TF1", "TE1"}

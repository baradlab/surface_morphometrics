"""Tests for the CSV/Euclidean semivariogram, pooling, and `morphometrics variogram`."""
import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from scipy.ndimage import gaussian_filter

from surface_morphometrics import spatial_stats as ss
from surface_morphometrics.feature_variogram import POOLED, variogram_cli


def _field(n=80, smooth=4.0, nugget_sd=0.0, seed=0):
    """A flat n x n grid (spacing 1) carrying a smooth random field plus white noise."""
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(size=(n, n)), smooth, mode="wrap")
    f = f / f.std()
    f = f + rng.normal(0, nugget_sd, f.shape)
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    xyz = np.column_stack([g.astype(float), np.zeros(len(g))])
    return xyz, f.ravel()


def test_euclidean_variogram_rises_then_plateaus():
    xyz, v = _field(smooth=4.0)
    h, gamma, counts = ss.euclidean_semivariogram(xyz, v, max_h=30, n_bins=15,
                                                  n_seeds=300)
    assert len(h) == 15 and np.all(counts >= 30)
    assert gamma[0] < 0.3 * gamma[-1]                # strongly correlated at short lag
    fit = ss.fit_correlation_length(h, gamma, counts=counts, model="auto")
    assert fit["ok"] and 2 < fit["ell"] < 15


def test_smoother_field_has_longer_correlation_length():
    ells = []
    for smooth in (2.0, 5.0):
        xyz, v = _field(smooth=smooth, seed=1)
        h, g, c = ss.euclidean_semivariogram(xyz, v, max_h=40, n_bins=20, n_seeds=300)
        ells.append(ss.fit_correlation_length(h, g, counts=c)["ell"])
    assert ells[1] > 1.5 * ells[0]


def test_white_noise_is_mostly_nugget():
    xyz, v = _field(smooth=3.0, seed=2)
    rng = np.random.default_rng(3)
    v = rng.normal(size=v.shape)                     # no spatial structure at all
    h, g, c = ss.euclidean_semivariogram(xyz, v, max_h=20, n_bins=10, n_seeds=200)
    s = ss.summarize_variogram_fit(ss.fit_correlation_length(h, g, counts=c), variance=1.0)
    assert not s["ok"] or s["structured_fraction"] < 0.3
    assert np.allclose(g, 1.0, atol=0.15)            # flat at the variance


def test_nugget_recovers_added_noise():
    xyz, v = _field(smooth=4.0, nugget_sd=0.5, seed=4)
    h, g, c = ss.euclidean_semivariogram(xyz, v, max_h=30, n_bins=15, n_seeds=300,
                                         robust=False)
    fit = ss.fit_correlation_length(h, g, counts=c, model="auto")
    assert fit["ok"] and fit["model"] == "gaussian"   # a smoothed field
    assert 0.18 < fit["nugget"] < 0.35               # true nugget 0.25
    # the exponential model can only mimic a smooth rise by giving up the nugget
    expo = ss.fit_correlation_length(h, g, counts=c, model="exponential")
    assert expo["nugget"] < 0.1 and expo["wsse"] > fit["wsse"]
    s = ss.summarize_variogram_fit(fit, variance=np.var(v), total_area=len(v))
    assert 0.5 < s["structured_fraction"] < 0.95 and s["neff"] > 0
    assert s["model"] == "gaussian"


def test_fit_rejects_unknown_model():
    with pytest.raises(ValueError):
        ss.fit_correlation_length(np.arange(1, 10.0), np.ones(9), model="spherical")


def test_mask_restricts_seeds_and_targets():
    xyz, v = _field(smooth=3.0)
    mask = xyz[:, 0] < 40
    v2 = v.copy()
    v2[~mask] = 1e6                                  # would dominate if it leaked in
    h, g, c = ss.euclidean_semivariogram(xyz, v2, max_h=20, mask=mask)
    assert np.all(g < 10)


def test_variogram_sums_pool_like_more_seeds():
    xyz, v = _field(smooth=3.0)
    _, _, _, s1 = ss.euclidean_semivariogram(xyz, v, max_h=20, n_bins=10, seed=0,
                                             return_sums=True)
    _, _, _, s2 = ss.euclidean_semivariogram(xyz, v, max_h=20, n_bins=10, seed=1,
                                             return_sums=True)
    cnt1 = s1.cnt.copy()
    s1 += s2
    assert np.all(s1.cnt == cnt1 + s2.cnt)
    with pytest.raises(ValueError):
        s1 += ss.VariogramSums(max_h=20, n_bins=5)


def test_geodesic_variogram_on_a_lattice():
    gt = pytest.importorskip("graph_tool")
    from graph_tool.generation import lattice
    n = 40
    g = lattice([n, n])
    dist = g.new_edge_property("double")
    dist.a = 1.0
    g.ep["distance"] = dist
    xyz, v = _field(n=n, smooth=3.0)
    # lattice vertex k is (k // n, k % n) -- the same ordering as _field
    h, gamma, c = ss.geodesic_semivariogram(g, v, max_h=15, n_bins=15, n_seeds=100,
                                            min_pairs=10)
    he, ge, _ = ss.euclidean_semivariogram(xyz, v, max_h=15, n_bins=15, n_seeds=100,
                                           min_pairs=10)
    assert gamma[0] < gamma[-1]
    # Manhattan geodesics are >= Euclidean, so the geodesic curve rises more slowly
    assert np.mean(gamma[:5]) <= np.mean(ge[:5]) * 1.2


# --- CLI ---

def _write_run(tmp_path, with_noise=False):
    seg, work = tmp_path / "seg", tmp_path / "work"
    seg.mkdir(); work.mkdir()
    for i, tomo in enumerate(("T1", "T2")):
        (seg / f"{tomo}.mrc").touch()
        xyz, v = _field(n=50, smooth=3.0, seed=i)
        pd.DataFrame({"thickness": 4 + 0.3 * v, "area": np.ones(len(v)),
                      "OMM_dist": xyz[:, 0], "xyz_x": xyz[:, 0], "xyz_y": xyz[:, 1],
                      "xyz_z": xyz[:, 2]}).to_csv(work / f"{tomo}_IMM.AVV_rh9.csv",
                                                  index=False)
    if with_noise:
        pd.DataFrame({"surface": ["T1_IMM", "T2_IMM"],
                      "thickness_noise_var": [0.01, 0.02],
                      "thickness_noise_var_robust": [0.01, 0.02]}
                     ).to_csv(work / "thickness_noise.csv", index=False)
    config = {"seg_dir": str(seg) + "/", "work_dir": str(work) + "/",
              "segmentation_values": {"IMM": 1, "OMM": 2}}
    cfgp = tmp_path / "config.yml"
    yaml.safe_dump(config, open(cfgp, "w"))
    return str(cfgp), work


def test_variogram_cli_writes_per_surface_and_pooled(tmp_path):
    cfgp, work = _write_run(tmp_path, with_noise=True)
    r = CliRunner().invoke(variogram_cli, [cfgp, "-n", "thickness", "--max-h", "25",
                                           "--bins", "12"])
    assert r.exit_code == 0, r.output
    assert "neighborhood-averaged" in r.output
    df = pd.read_csv(work / "thickness_variogram.csv")
    assert list(df["surface"]) == ["T1", "T2", POOLED]
    pooled = df[df["surface"] == POOLED].iloc[0]
    assert pooled["ok"] and pooled["ell"] > 0 and pooled["area"] == 5000
    t1 = df[df["surface"] == "T1"].iloc[0]
    assert t1["split_half_noise_var"] == 0.01
    assert t1["split_half_reliability"] == pytest.approx(1 - 0.01 / t1["variance"])
    assert (work / "thickness_variogram_IMM.svg").exists()
    assert "split-half noise SD" in r.output


def test_variogram_cli_filter_and_missing_feature(tmp_path):
    cfgp, work = _write_run(tmp_path)
    r = CliRunner().invoke(variogram_cli, [cfgp, "-n", "thickness", "--filter",
                                           "OMM_dist<25", "--no-plots", "--max-h", "15"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(work / "thickness_variogram.csv")
    assert set(df["n_triangles"][df["surface"] != POOLED]) == {25 * 50}
    r = CliRunner().invoke(variogram_cli, [cfgp, "-n", "nope", "--no-plots"])
    assert r.exit_code != 0 and "nope" in r.output
    r = CliRunner().invoke(variogram_cli, [cfgp, "-n", "thickness", "-c", "ER"])
    assert r.exit_code != 0 and "unknown class" in r.output


def test_variogram_cli_on_multidataset(tmp_path):
    for i, src in enumerate(("a", "b")):
        work = tmp_path / src
        work.mkdir()
        xyz, v = _field(n=40, smooth=3.0, seed=i)
        pd.DataFrame({"thickness": 4 + 0.3 * v, "area": np.ones(len(v)),
                      "xyz_x": xyz[:, 0], "xyz_y": xyz[:, 1], "xyz_z": xyz[:, 2]}
                     ).to_csv(work / "T1_IMM.AVV_rh9.csv", index=False)
    config = {"classes": ["IMM"], "datasets": {"a": {"work_dir": "a"},
                                               "b": {"work_dir": "b"}}}
    cfgp = tmp_path / "study.yml"
    yaml.safe_dump(config, open(cfgp, "w"))
    r = CliRunner().invoke(variogram_cli, [str(cfgp), "-n", "thickness", "--no-plots",
                                           "--max-h", "15"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(tmp_path / "thickness_variogram.csv")
    assert list(df["surface"]) == ["a/T1", "b/T1", POOLED]

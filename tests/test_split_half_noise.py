"""Tests for the split-half thickness noise estimate (measure_thickness --noise-estimate)."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from surface_morphometrics import _thickness_worker as tw
from surface_morphometrics import measure_thickness as mt


def test_split_half_neighbors_are_disjoint_and_complete():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 20, (300, 3))
    d, idx = cKDTree(xyz).query(xyz, k=40, distance_upper_bound=5.0)
    halves = rng.integers(0, 2, len(xyz))
    rows = np.arange(0, 300, 7)
    da, ia, db, ib = mt.split_half_neighbors(d, idx, rows, halves)
    for r, row in enumerate(rows):
        a = set(ia[r][np.isfinite(da[r])])
        b = set(ib[r][np.isfinite(db[r])])
        full = set(idx[row][np.isfinite(d[row])])
        assert a.isdisjoint(b) and a | b == full
        assert all(halves[i] == 0 for i in a) and all(halves[i] == 1 for i in b)


def test_split_half_noise_recovers_known_noise():
    rng = np.random.default_rng(1)
    truth = rng.normal(4.0, 0.5, 20000)         # real spatial signal, SD 0.5
    sigma = 0.2                                  # noise SD of the FULL estimate
    full = truth + rng.normal(0, sigma, truth.size)
    # each half averages half the data -> variance 2 sigma^2
    a = truth + rng.normal(0, np.sqrt(2) * sigma, truth.size)
    b = truth + rng.normal(0, np.sqrt(2) * sigma, truth.size)
    a[:50] = np.nan                              # failed fits are skipped
    res = mt.split_half_noise(a, b, full)
    assert res["n_both"] == truth.size - 50
    assert abs(np.sqrt(res["noise_var"]) - sigma) < 0.01
    assert abs(np.sqrt(res["noise_var_robust"]) - sigma) < 0.01
    expected_rel = 0.5 ** 2 / (0.5 ** 2 + sigma ** 2)
    assert abs(res["reliability"] - expected_rel) < 0.02


def test_split_half_noise_too_few_pairs_is_nan():
    res = mt.split_half_noise([1.0, np.nan], [1.1, 2.0], [1.0, 1.1, 1.2])
    assert np.isnan(res["noise_var"]) and res["n_both"] == 1
    assert res["total_var"] > 0


def _noisy_bilayer_surface(noise, n_side=24, seed=0):
    """A flat grid of triangles whose profiles are a 3.5 nm bilayer plus white noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-10, 10, 81)
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side)), -1).reshape(-1, 2)
    xyz = np.column_stack([g * 1.0, np.zeros(len(g))])
    clean = (0.01 + tw._monogaussian(x, 0.02, -1.75, 1.0)
             + tw._monogaussian(x, 0.02, 1.75, 1.0))
    profiles = clean + rng.normal(0, noise, (len(g), len(x)))
    d, idx = cKDTree(xyz).query(xyz, k=60, distance_upper_bound=4.0)
    return -profiles, d, idx, x


def _fit_in_process(thickness_arr, d, idx, x):
    tw.init_worker(thickness_arr, d, idx, x, raw_average=True)
    try:
        res = tw.fit_triangle_chunk(list(range(len(d))))
    finally:
        tw.init_worker(None, None, None, x)
    return np.array([r[0] for r in res], dtype=float)


def test_split_half_noise_tracks_profile_noise_end_to_end():
    estimates = []
    for noise in (0.002, 0.008):
        arr, d, idx, x = _noisy_bilayer_surface(noise)
        full = _fit_in_process(arr, d, idx, x)
        rng = np.random.default_rng(0)
        rows = np.arange(len(d))
        da, ia, db, ib = mt.split_half_neighbors(d, idx, rows, rng.integers(0, 2, len(d)))
        res = mt.split_half_noise(_fit_in_process(arr, da, ia, x),
                                  _fit_in_process(arr, db, ib, x), full)
        assert res["n_both"] > 0.8 * len(d)
        # the field has no real signal, so nearly all its variance is noise
        assert res["reliability"] < 0.6
        estimates.append(res["noise_var"])
    assert estimates[1] > 4 * estimates[0]        # 4x profile noise -> much larger


def test_write_noise_table_upserts(tmp_path):
    path = str(tmp_path / mt.NOISE_CSV)
    mt.write_noise_table([{"surface": "T1_IMM", "thickness_noise_var": 1.0},
                          {"surface": "T1_OMM", "thickness_noise_var": 2.0}], path)
    mt.write_noise_table([{"surface": "T1_IMM", "thickness_noise_var": 3.0}], path)
    df = pd.read_csv(path).set_index("surface")
    assert df.loc["T1_IMM", "thickness_noise_var"] == 3.0
    assert df.loc["T1_OMM", "thickness_noise_var"] == 2.0 and len(df) == 2

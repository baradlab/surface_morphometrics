"""Tests for spatial_stats: N_eff estimators, weighted effect sizes, the correlation
length fit, and the tomogram-level permutation test. Pure numpy/scipy (no graph-tool)."""
import numpy as np
import pytest
from scipy import stats as scipy_stats

from surface_morphometrics import spatial_stats as ss


# --- N_eff estimators ----------------------------------------------------------

def test_kish_neff():
    assert ss.kish_neff(np.ones(10)) == pytest.approx(10.0)      # equal weights -> n
    # one dominant weight -> ~1 independent observation
    assert ss.kish_neff([1000.0, 1.0, 1.0]) < 1.01
    assert ss.kish_neff([0.0, 0.0]) == 0.0


def test_neff_from_neighbors_and_correlation_length():
    assert ss.neff_from_neighbors(61501, 276.5) == pytest.approx(61501 / 276.5)
    # A / (2 pi ell^2)
    assert ss.neff_from_correlation_length(118195, 8.2) == pytest.approx(
        118195 / (2 * np.pi * 8.2 ** 2))


# --- weighted effect sizes -----------------------------------------------------

def test_weighted_ks_matches_scipy_when_unweighted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0.5, 1, 200)
    d_ours = ss.weighted_ks_statistic(a, b)
    d_scipy = scipy_stats.ks_2samp(a, b).statistic
    assert d_ours == pytest.approx(d_scipy, abs=1e-9)


def test_weighted_ks_bounds():
    assert ss.weighted_ks_statistic([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert ss.weighted_ks_statistic([0, 0, 0], [1, 1, 1]) == pytest.approx(1.0)


def test_weights_change_ks():
    # up-weight the tail of a so its ECDF shifts toward b
    a = np.array([0.0, 0.0, 10.0])
    unweighted = ss.weighted_ks_statistic(a, [10.0, 10.0, 10.0])
    weighted = ss.weighted_ks_statistic(a, [10.0, 10.0, 10.0], weights_a=[1, 1, 100])
    assert weighted < unweighted


def test_weighted_wasserstein_shift():
    # a rigid shift of c gives Wasserstein distance c
    a = np.array([0.0, 1.0, 2.0, 3.0])
    assert ss.weighted_wasserstein(a, a + 2.5) == pytest.approx(2.5)


# --- correlation-length fit (the estimator core, validated on a synthetic gamma) ---

def test_fit_correlation_length_recovers_known_ell():
    ell_true = 7.0
    h = np.linspace(0.5, 60, 40)
    gamma = ss._variogram_model(h, 0.1, 1.0, ell_true, 0.0)   # no drift, no noise
    fit = ss.fit_correlation_length(h, gamma)
    assert fit["ok"]
    assert fit["ell"] == pytest.approx(ell_true, rel=0.05)


def test_fit_correlation_length_with_drift_and_noise():
    ell_true = 6.0
    h = np.linspace(0.5, 60, 50)
    rng = np.random.default_rng(1)
    gamma = (ss._variogram_model(h, 0.05, 1.0, ell_true, 0.0004)   # includes c2 h^2 drift
             + rng.normal(0, 0.01, len(h)))
    fit = ss.fit_correlation_length(h, gamma)
    assert fit["ok"]
    # the drift term must be recovered, so ell reflects only the short-range structure
    assert fit["ell"] == pytest.approx(ell_true, rel=0.2)
    assert fit["drift"] > 0


# --- permutation test ----------------------------------------------------------

def _units(rng, mus, n=300):
    return [rng.normal(mu, 1.0, n) for mu in mus]


def test_permutation_detects_clear_difference():
    rng = np.random.default_rng(0)
    A = _units(rng, [0, 0, 0, 0, 0, 0])
    B = _units(rng, [4, 4, 4, 4, 4, 4])          # far apart
    res = ss.permutation_test(A + B, ["a"] * 6 + ["b"] * 6, reps=500)
    # Clearly significant. (It cannot hit the theoretical floor at 500 reps: KS is
    # symmetric, so the label-swapped permutation always ties observed D -> p ~ 2/501.)
    assert res["p_value"] < 0.01
    assert res["p_value"] >= res["min_possible_p"] - 1e-9   # can't beat the design floor
    assert res["n_units_a"] == 6 and res["n_units_b"] == 6


def test_permutation_null_gives_large_p():
    rng = np.random.default_rng(2)
    units = _units(rng, [0, 0, 0, 0, 0, 0])       # all identical distribution
    res = ss.permutation_test(units, ["a", "a", "a", "b", "b", "b"], reps=500)
    assert res["p_value"] > 0.1


def test_permutation_reproducible_and_floor():
    rng = np.random.default_rng(3)
    units = _units(rng, [0, 1, 2, 0, 1, 2])
    r1 = ss.permutation_test(units, ["a"] * 3 + ["b"] * 3, reps=300, seed=42)
    r2 = ss.permutation_test(units, ["a"] * 3 + ["b"] * 3, reps=300, seed=42)
    assert r1["p_value"] == r2["p_value"]
    assert r1["min_possible_p"] == pytest.approx(2.0 / 20)   # 2 / C(6,3)


def test_permutation_wasserstein_and_validation():
    rng = np.random.default_rng(4)
    A = _units(rng, [0, 0, 0])
    B = _units(rng, [3, 3, 3])
    res = ss.permutation_test(A + B, ["a"] * 3 + ["b"] * 3, statistic="wasserstein", reps=200)
    assert res["statistic"] == "wasserstein"
    assert res["observed"] > 1.0
    with pytest.raises(ValueError):
        ss.permutation_test(A, ["a", "a", "a"], reps=10)   # only one condition


# --- intraclass correlation + stratified (nested) permutation ------------------

def test_intraclass_correlation_high_and_low():
    # Strong tomogram clustering: each group's units share an offset -> ICC high.
    rng = np.random.default_rng(0)
    offsets = [0.0, 5.0, 10.0]
    vals, groups = [], []
    for gi, off in enumerate(offsets):
        for _ in range(5):
            vals.append(off + rng.normal(0, 0.1)); groups.append(gi)
    assert ss.intraclass_correlation(vals, groups) > 0.9
    # No clustering: group label unrelated to value -> ICC ~ 0.
    v = rng.normal(0, 1, 30)
    g = np.tile([0, 1, 2], 10)
    assert ss.intraclass_correlation(v, g) < 0.2


def test_nested_permutation_floor_is_set_by_tomograms_not_organelles():
    # 2 tomograms/condition, 4 organelles each -> 8 organelles/condition, but the
    # exchangeable blocks are the 4 tomograms, so the floor is 2/C(4,2) = 1/3.
    rng = np.random.default_rng(1)
    units, conds, strata = [], [], []
    for tomo in range(4):
        cond = "a" if tomo < 2 else "b"
        for _ in range(4):
            units.append(rng.normal(0 if cond == "a" else 5, 1, 100))
            conds.append(cond); strata.append(tomo)
    nested = ss.permutation_test(units, conds, strata=strata, reps=500)
    assert nested["n_blocks_a"] == 2 and nested["n_units_a"] == 8
    assert nested["min_possible_p"] == pytest.approx(2 / 6)      # 2 / C(4,2)
    # Flat treats all 16 organelles as blocks -> a far smaller floor (more "power").
    flat = ss.permutation_test(units, conds, reps=500)
    assert flat["min_possible_p"] < nested["min_possible_p"]


def test_nested_permutation_requires_constant_condition_in_stratum():
    units = [np.arange(5) for _ in range(4)]
    conds = ["a", "b", "a", "b"]
    bad_strata = [0, 0, 1, 1]           # stratum 0 has both a and b -> invalid
    with pytest.raises(ValueError):
        ss.permutation_test(units, conds, strata=bad_strata, reps=10)


# --- cluster bootstrap ---------------------------------------------------------

def test_cluster_bootstrap_brackets_estimate_and_reproducible():
    rng = np.random.default_rng(0)
    units = [rng.normal(5.0, 1.0, 200) for _ in range(6)]
    r1 = ss.cluster_bootstrap(units, statistic="mean", reps=500, seed=1)
    r2 = ss.cluster_bootstrap(units, statistic="mean", reps=500, seed=1)
    assert r1["ci_low"] < r1["estimate"] < r1["ci_high"]
    assert r1["estimate"] == pytest.approx(5.0, abs=0.3)
    assert (r1["ci_low"], r1["ci_high"]) == (r2["ci_low"], r2["ci_high"])   # seeded
    assert r1["n_units"] == 6


def test_cluster_bootstrap_wider_than_iid_triangle_bootstrap():
    # Correlated units: each tomogram has its own offset, so whole-tomogram resampling
    # sees more variability than resampling individual (within-tomogram-similar) triangles.
    rng = np.random.default_rng(2)
    offsets = rng.normal(0, 1.0, 8)
    units = [off + rng.normal(0, 0.05, 300) for off in offsets]   # tight within, spread between
    weights = [np.ones(len(u)) for u in units]
    cluster = ss.cluster_bootstrap(units, weights, statistic="mean", reps=600, seed=3)
    cluster_w = cluster["ci_high"] - cluster["ci_low"]
    # naive i.i.d. triangle bootstrap of the pooled values
    pooled = np.concatenate(units)
    rng2 = np.random.default_rng(3)
    iid = np.array([pooled[rng2.integers(0, len(pooled), len(pooled))].mean() for _ in range(600)])
    iid_w = np.percentile(iid, 95) - np.percentile(iid, 5)
    assert cluster_w > 3 * iid_w      # cluster CI is much wider (honest); iid under-covers


def test_cluster_bootstrap_callable_and_median():
    rng = np.random.default_rng(4)
    units = [rng.normal(2.0, 1.0, 100) for _ in range(5)]
    med = ss.cluster_bootstrap(units, statistic="median", reps=300, seed=0)
    assert med["ci_low"] < med["estimate"] < med["ci_high"]
    # callable: 95th percentile as the summary
    q = ss.cluster_bootstrap(units, statistic=lambda v, w: float(np.percentile(v, 95)),
                             reps=200, seed=0)
    assert q["estimate"] > med["estimate"]
    with pytest.raises(ValueError):
        ss.cluster_bootstrap([np.arange(5)], statistic="mean")   # need >= 2 units


def test_cluster_t_interval_mechanics():
    # Per-unit means are exactly known -> check the t-interval arithmetic.
    units = [np.full(10, m) for m in (1.0, 2.0, 3.0, 4.0, 5.0)]   # unit means 1..5
    res = ss.cluster_t_interval(units, statistic="mean", ci=0.90)
    from scipy.stats import t as t_dist
    m = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    se = m.std(ddof=1) / np.sqrt(5)
    half = t_dist.ppf(0.95, 4) * se
    assert res["estimate"] == pytest.approx(3.0)
    assert res["ci_low"] == pytest.approx(3.0 - half)
    assert res["ci_high"] == pytest.approx(3.0 + half)
    assert res["unit_values"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(ValueError):
        ss.cluster_t_interval([np.arange(3)])                    # need >= 2 units

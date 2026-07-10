#! /usr/bin/env python
"""Spatially-aware comparison of membrane-surface feature distributions.

Triangle values on a membrane are strongly spatially autocorrelated, so the number of
triangles massively overstates the number of independent observations. A naive pooled
two-sample test with n = n_triangles is wildly anticonservative -- it rejects ~90% of
the time even when two conditions are identical (see PLAN_stats_refactor.md, Phase 4).
Two defensible routes are provided.

1. Permutation test (the assumption-light default):
  * permutation_test(...)  -- keeps a distributional effect size (area-weighted KS D or
                             Wasserstein) but gets the p-value by permuting the CONDITION
                             label across the independent unit (the tomogram). Exact under
                             exchangeability of tomograms; no correlation model, no
                             stationarity, no N_eff. Validated calibrated with full power.

2. Effective sample size N_eff (for a corrected pooled KS a la ks_statistics):
  * kish_neff(areas)                     -- (Σw)²/Σw², the NO-correlation upper bound
  * neff_from_neighbors(n_tri, avg_nbr)  -- n_triangles / pycurv's avg_num_neighbors
  * geodesic_semivariogram(...)  -- γ(h) over geodesic distance on the triangle graph
  * fit_correlation_length(...)  -- fit γ(h) = c0 + c1(1 - e^{-h/ℓ}) + c2 h², so the
                                    smooth non-stationary drift (c2 h²) is modelled
                                    rather than mistaken for correlation. Returns ℓ.
  * neff_from_correlation_length(A, ℓ) = A / (2π ℓ²)   [integral range A_c = 2π ℓ²,
                                    the same for exponential and Gaussian covariance]
  N_eff is sound GIVEN a correct ℓ, but Type-I error is asymmetrically sensitive to it:
  underestimating ℓ (overestimating N_eff) is anticonservative (ℓ 20% low -> Type-I
  doubles). Prefer a conservative (upper) ℓ, or just use permutation_test.

Distances are GEODESIC (along the surface), because two membrane sheets can be nm apart
in 3D yet far apart on the surface.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import numpy as np


# ---------------------------------------------------------------------------
# Cheap, fit-free estimators
# ---------------------------------------------------------------------------

def kish_neff(areas):
    """Kish effective sample size (Σw)²/Σw² for area weights w.

    This is the effective n IF the observations were independent — i.e. the upper
    bound. Spatial correlation makes the true N_eff much smaller.
    """
    w = np.asarray(areas, dtype=float)
    denom = np.sum(w ** 2)
    return float(w.sum() ** 2 / denom) if denom > 0 else 0.0


def neff_from_neighbors(n_triangles, avg_num_neighbors):
    """N_eff ≈ n_triangles / avg_num_neighbors (pycurv's mean geodesic-disk occupancy).

    A free estimate: pycurv writes avg_num_neighbors to its *_runtimes.csv. Treats one
    radius_hit geodesic disk as one independent patch.
    """
    return float(n_triangles / avg_num_neighbors)


def neff_from_correlation_length(total_area, ell):
    """N_eff = total_area / integral_range, with integral range A_c = 2π ℓ².

    A_c = ∫ρ(h) dA = 2π ∫₀^∞ ρ(h) h dh = 2π ℓ² for both exponential (ρ=e^{-h/ℓ}) and
    Gaussian (ρ=e^{-h²/2ℓ²}) correlation, so this holds regardless of which the
    variogram fit assumed.
    """
    return float(total_area / (2.0 * np.pi * ell ** 2))


# ---------------------------------------------------------------------------
# Geodesic semivariogram
# ---------------------------------------------------------------------------

def geodesic_semivariogram(graph, values, weights_ep="distance", areas=None,
                           max_h=60.0, n_bins=30, n_seeds=200, robust=True,
                           min_pairs=30, seed=0):
    """Empirical semivariogram γ(h) over geodesic distance on a triangle graph.

    Parameters
    ----------
    graph : graph_tool.Graph
        Triangle graph (vertices = triangles) with an edge-length property.
    values : (n,) array
        Per-triangle feature values.
    weights_ep : str
        Name of the edge property holding geodesic edge lengths (default "distance").
    areas : (n,) array or None
        Per-triangle areas for area-weighted increments; None = unweighted.
    max_h : float
        Maximum geodesic lag to include.
    n_bins, n_seeds : int
        Number of lag bins and random seed triangles.
    robust : bool
        Cressie–Hawkins robust estimator (default) vs the classical Matheron mean.
    min_pairs : int
        Drop lag bins with fewer than this many pairs.

    Returns
    -------
    (h_centers, gamma, counts) for bins that met `min_pairs`.
    """
    from graph_tool.topology import shortest_distance

    values = np.asarray(values, dtype=float)
    n = graph.num_vertices()
    if areas is None:
        areas = np.ones(n)
    areas = np.asarray(areas, dtype=float)
    rng = np.random.default_rng(seed)
    wt = graph.ep[weights_ep]

    edges = np.linspace(0.0, max_h, n_bins + 1)
    # Accumulate per bin. For the robust estimator we accumulate mean(|Δ|^0.5); for the
    # classical one, mean(Δ²)/2. Both are area-weighted.
    wsum = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)

    seeds = rng.choice(n, size=min(n_seeds, n), replace=False)
    for s in seeds:
        d = shortest_distance(graph, source=graph.vertex(int(s)), weights=wt,
                              max_dist=max_h).get_array()
        m = np.isfinite(d) & (d > 0) & (d <= max_h)
        if not m.any():
            continue
        idx = np.clip(np.digitize(d[m], edges) - 1, 0, n_bins - 1)
        delta = values[int(s)] - values[m]
        w = areas[int(s)] * areas[m]
        contrib = np.sqrt(np.abs(delta)) if robust else 0.5 * delta ** 2
        np.add.at(acc, idx, w * contrib)
        np.add.at(wsum, idx, w)
        np.add.at(cnt, idx, 1.0)

    ok = cnt >= min_pairs
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)
    if robust:
        # Cressie–Hawkins: γ = 0.5 * mean(|Δ|^0.5)^4 / (0.457 + 0.494/N + 0.045/N²)
        N = np.maximum(cnt, 1)
        gamma = 0.5 * mean ** 4 / (0.457 + 0.494 / N + 0.045 / N ** 2)
    else:
        gamma = mean
    return centers[ok], gamma[ok], cnt[ok]


# ---------------------------------------------------------------------------
# Correlation-length fit
# ---------------------------------------------------------------------------

def _variogram_model(h, c0, c1, ell, c2):
    """Nested model: nugget + short-range exponential residual + smooth drift."""
    return c0 + c1 * (1.0 - np.exp(-h / ell)) + c2 * h ** 2


def fit_correlation_length(h, gamma, counts=None, ell0=None):
    """Fit γ(h) = c0 + c1(1 - e^{-h/ℓ}) + c2 h² and return the correlation length ℓ.

    The c2 h² term absorbs a smooth non-stationary drift, so ℓ reflects only the
    short-range (stationary) structure — the whole point of the nested model.

    Returns a dict: ell, ell_stderr, sill (c1), nugget (c0), drift (c2), params, and
    `ok` (False if the fit failed or ℓ is not identifiable).
    """
    from scipy.optimize import curve_fit

    h = np.asarray(h, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    good = np.isfinite(h) & np.isfinite(gamma)
    h, gamma = h[good], gamma[good]
    result = {"ell": np.nan, "ell_stderr": np.nan, "sill": np.nan,
              "nugget": np.nan, "drift": np.nan, "params": None, "ok": False}
    if len(h) < 5:
        return result

    span = gamma.max() - gamma.min()
    if ell0 is None:
        ell0 = 0.25 * h.max()
    # Drift c2 >= 0: a semivariogram is non-decreasing, so a negative quadratic term is
    # unphysical and lets the optimiser find degenerate fits (ell running to the boundary).
    p0 = [max(gamma.min(), 0.0), max(span, 1e-9), ell0, 0.0]
    bounds = ([0.0, 0.0, h.min() if h.min() > 0 else 1e-3, 0.0],
              [gamma.max() + span, 3 * span + 1e-9, h.max(), np.inf])
    sigma = None
    if counts is not None:
        counts = np.asarray(counts, dtype=float)[good]
        sigma = 1.0 / np.sqrt(np.maximum(counts, 1.0))   # more pairs -> more weight
    try:
        popt, pcov = curve_fit(_variogram_model, h, gamma, p0=p0, bounds=bounds,
                               sigma=sigma, maxfev=20000)
    except (RuntimeError, ValueError):
        return result
    c0, c1, ell, c2 = popt
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 4
    result.update(ell=float(ell), ell_stderr=float(perr[2]), sill=float(c1),
                  nugget=float(c0), drift=float(c2), params=tuple(map(float, popt)),
                  ok=bool(np.isfinite(ell) and ell < 0.99 * h.max() and c1 > 0))
    return result


def estimate_neff(graph, values, total_area, weights_ep="distance", areas=None,
                  max_h=60.0, n_seeds=200, n_bins=30, robust=True, seed=0):
    """End-to-end per-feature N_eff: variogram -> fit ℓ -> A / (2π ℓ²).

    Returns a dict with ell, neff, and the underlying fit/variogram, or ok=False if ℓ
    was not identifiable (caller should fall back to neff_from_neighbors).
    """
    h, gamma, counts = geodesic_semivariogram(
        graph, values, weights_ep=weights_ep, areas=areas, max_h=max_h,
        n_bins=n_bins, n_seeds=n_seeds, robust=robust, seed=seed)
    fit = fit_correlation_length(h, gamma, counts=counts)
    out = {"ok": fit["ok"], "ell": fit["ell"], "neff": np.nan,
           "fit": fit, "h": h, "gamma": gamma, "counts": counts}
    if fit["ok"]:
        out["neff"] = neff_from_correlation_length(total_area, fit["ell"])
    return out


# ---------------------------------------------------------------------------
# Weighted two-sample effect sizes
# ---------------------------------------------------------------------------

def _as_weighted(values, weights):
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones(len(values))
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(values):
            raise ValueError("values and weights must have the same length")
    return values, weights


def weighted_ks_statistic(values_a, values_b, weights_a=None, weights_b=None):
    """Area-weighted two-sample Kolmogorov-Smirnov D (sup distance between weighted ECDFs).

    With equal weights this reduces to the usual two-sample KS D.
    """
    a, wa = _as_weighted(values_a, weights_a)
    b, wb = _as_weighted(values_b, weights_b)
    oa, ob = np.argsort(a), np.argsort(b)
    a, wa = a[oa], wa[oa]
    b, wb = b[ob], wb[ob]
    grid = np.concatenate([a, b])
    cwa = np.concatenate([[0.0], np.cumsum(wa)]) / wa.sum()
    cwb = np.concatenate([[0.0], np.cumsum(wb)]) / wb.sum()
    cdf_a = cwa[np.searchsorted(a, grid, side="right")]
    cdf_b = cwb[np.searchsorted(b, grid, side="right")]
    return float(np.max(np.abs(cdf_a - cdf_b)))


def weighted_wasserstein(values_a, values_b, weights_a=None, weights_b=None):
    """Area-weighted 1-Wasserstein (earth-mover) distance between two samples."""
    from scipy.stats import wasserstein_distance
    a, wa = _as_weighted(values_a, weights_a)
    b, wb = _as_weighted(values_b, weights_b)
    return float(wasserstein_distance(a, b, wa, wb))


_STATISTICS = {"ks": weighted_ks_statistic, "wasserstein": weighted_wasserstein}


# ---------------------------------------------------------------------------
# Tomogram-level permutation test
# ---------------------------------------------------------------------------

def permutation_test(unit_values, unit_conditions, statistic="ks",
                     unit_weights=None, reps=2000, seed=0):
    """Two-condition comparison with the tomogram as the independent unit.

    Each "unit" is one tomogram/organelle: a 1D array of that surface's per-triangle
    feature values (optionally with per-triangle area weights). The observed statistic
    pools all units within a condition and compares the two pooled (weighted)
    distributions. The null is built by permuting which units belong to which condition,
    so the p-value treats the tomogram -- not the triangle -- as the unit of replication.
    Exact under exchangeability of tomograms; needs no correlation model or N_eff.

    Parameters
    ----------
    unit_values : list of 1D array-likes, one per tomogram.
    unit_conditions : list of condition labels (exactly two distinct), one per unit.
    statistic : "ks" (area-weighted KS D) or "wasserstein".
    unit_weights : optional list of per-triangle weight arrays matching unit_values.
    reps : number of random label permutations.
    seed : RNG seed.

    Returns
    -------
    dict with: statistic, observed, p_value, condition_a/b, n_units_a/b, reps, and
    min_possible_p (= 2 / C(nA+nB, nA), the two-sided floor set by the design).
    """
    from math import comb

    if statistic not in _STATISTICS:
        raise ValueError(f"statistic must be one of {tuple(_STATISTICS)}")
    stat_fn = _STATISTICS[statistic]

    conditions = list(unit_conditions)
    uniq = sorted(set(conditions))
    if len(uniq) != 2:
        raise ValueError(f"need exactly two conditions, got {uniq}")
    ca, cb = uniq
    idx_a = [i for i, c in enumerate(conditions) if c == ca]
    idx_b = [i for i, c in enumerate(conditions) if c == cb]

    vals = [np.asarray(v, dtype=float) for v in unit_values]
    if unit_weights is None:
        wts = [np.ones(len(v)) for v in vals]
    else:
        wts = [np.asarray(w, dtype=float) for w in unit_weights]

    def pooled_stat(ia, ib):
        va = np.concatenate([vals[i] for i in ia])
        wa = np.concatenate([wts[i] for i in ia])
        vb = np.concatenate([vals[i] for i in ib])
        wb = np.concatenate([wts[i] for i in ib])
        return stat_fn(va, vb, wa, wb)

    observed = pooled_stat(idx_a, idx_b)

    all_idx = idx_a + idx_b
    n_a = len(idx_a)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(reps):
        perm = rng.permutation(all_idx)
        if pooled_stat(perm[:n_a], perm[n_a:]) >= observed - 1e-12:
            ge += 1
    return {
        "statistic": statistic,
        "observed": observed,
        "p_value": (1 + ge) / (1 + reps),
        "condition_a": ca, "condition_b": cb,
        "n_units_a": n_a, "n_units_b": len(idx_b),
        "reps": reps,
        "min_possible_p": 2.0 / comb(len(all_idx), n_a),
    }

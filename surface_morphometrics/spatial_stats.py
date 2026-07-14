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
                             label across an exchangeable unit. Exact under exchangeability;
                             no correlation model, no stationarity, no N_eff. Validated
                             calibrated with full power. The unit is the whole tomogram by
                             default; pass `strata` (tomogram id per unit) to make the unit
                             a connected component/organelle while keeping the p-value's
                             replication level at the tomogram (nested).
  * intraclass_correlation(...)  -- between-tomogram vs within-tomogram variance of a
                             per-organelle summary; tells you whether organelles can be
                             treated as independent samples (low ICC) or cluster by
                             tomogram/cell (high ICC -> use the nested permutation).

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

For confidence intervals on a per-condition summary statistic (tomogram = unit):
  * cluster_t_interval(...)  -- RECOMMENDED for a mean: one summary per tomogram, then a
                             t-interval on those. Best-calibrated at small n.
  * cluster_bootstrap(...)   -- resamples WHOLE tomograms with replacement (for non-mean
                             statistics); under-covers somewhat at small n. Both beat the
                             legacy i.i.d. triangle bootstrap, which assumes independence
                             (coverage ~0.20 at nominal 0.90 -> CIs far too narrow).

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


def _shared_grid(unit_values, max_points):
    """Evaluation grid for the pooled ECDFs: every distinct value, or quantiles of them.

    A KS/Wasserstein distance between step functions is determined by the ECDFs at the
    data points, so the exact statistic needs the full set of distinct values. For
    membrane data that is millions of points per condition, so above `max_points` we fall
    back to a quantile grid: the statistic becomes a (very close) approximation, but the
    permutation test stays exactly valid because observed and permuted replicates are
    scored with the SAME grid.
    """
    pooled = np.concatenate(unit_values)
    uniq = np.unique(pooled)
    if len(uniq) <= max_points:
        return uniq
    grid = np.quantile(pooled, np.linspace(0.0, 1.0, max_points))
    return np.unique(grid)


def _ecdf_on_grid(values, weights, grid):
    """A unit's weight-normalized ECDF evaluated on `grid` (weights sum to 1)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.concatenate([[0.0], np.cumsum(w)])
    return cw[np.searchsorted(v, grid, side="right")] / cw[-1]


# ---------------------------------------------------------------------------
# Tomogram-level permutation test
# ---------------------------------------------------------------------------

def intraclass_correlation(values, groups):
    """ICC(1): the fraction of variance in a per-unit summary that is BETWEEN groups.

    Used to judge whether connected components (organelles) can be treated as
    independent samples. `values` is one summary statistic per unit (organelle),
    `groups` is that unit's grouping label (its tomogram). ICC near 0 means organelles
    within a tomogram are no more alike than across tomograms -> they are ~independent
    and a "flat" per-organelle analysis is justified. ICC near 1 means strong
    tomogram/cell-level clustering -> treating organelles as independent replicates is
    anticonservative; use tomogram-stratified (nested) inference.

    One-way random-effects ICC(1,1) from an ANOVA decomposition; returns 0.0 for a
    negative estimate (no detectable clustering) and NaN if it cannot be computed.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    k, N = len(uniq), len(values)
    if k < 2 or N <= k:
        return float("nan")
    grand = values.mean()
    ns = np.array([np.sum(groups == g) for g in uniq])
    ssb = float(np.sum([n * (values[groups == g].mean() - grand) ** 2
                        for g, n in zip(uniq, ns)]))
    ssw = float(np.sum([((values[groups == g] - values[groups == g].mean()) ** 2).sum()
                        for g in uniq]))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(ns ** 2) / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    if denom <= 0:
        return 0.0
    return float(max((msb - msw) / denom, 0.0))


def permutation_test(unit_values, unit_conditions, statistic="ks",
                     unit_weights=None, strata=None, reps=2000, seed=0,
                     grid_points=4096):
    """Two-condition comparison whose p-value treats a chosen level as the unit.

    Each "unit" is one surface (a whole tomogram, or one connected component/organelle
    when splitting): a 1D array of per-triangle feature values with optional area
    weights. The observed effect size pools all units within a condition and compares
    the two pooled (weighted) distributions with `statistic`. The null permutes the
    condition label across EXCHANGEABLE BLOCKS:

      * strata=None  -> each unit is its own block (FLAT). Valid when the unit itself is
        the level at which condition is exchangeable -- e.g. organelles when there is no
        tomogram/cell-level clustering (check `intraclass_correlation` first).
      * strata given -> units sharing a stratum move together and the condition is
        permuted at the STRATUM level (NESTED). Use strata = tomogram id when condition
        is assigned per tomogram: the effect size still uses all organelles, but the
        p-value's floor is set by the number of tomograms, not organelles. Condition
        must be constant within a stratum.

    Each unit's weighted ECDF is precomputed once on a shared grid (`grid_points`), so a
    replicate is a cheap weighted mixture of those ECDFs rather than a re-sort of every
    pooled triangle. That is what makes the test usable on real surfaces: at ~6M triangles
    per condition the naive version takes ~40 minutes per comparison, this takes seconds.
    See :func:`_shared_grid` for why the resulting statistic stays a valid test.

    Returns a dict: statistic, observed, p_value, condition_a/b, n_units_a/b,
    n_blocks_a/b, reps, min_possible_p (= 2 / C(n_blocks, n_blocks_a)).
    """
    from math import comb

    if statistic not in _STATISTICS:
        raise ValueError(f"statistic must be one of {tuple(_STATISTICS)}")

    conditions = list(unit_conditions)
    uniq = sorted(set(conditions))
    if len(uniq) != 2:
        raise ValueError(f"need exactly two conditions, got {uniq}")
    ca, cb = uniq

    vals = [np.asarray(v, dtype=float) for v in unit_values]
    if unit_weights is None:
        wts = [np.ones(len(v)) for v in vals]
    else:
        wts = [np.asarray(w, dtype=float) for w in unit_weights]

    # Exchangeable blocks: one unit each (flat), or one per stratum (nested).
    if strata is None:
        blocks = [[i] for i in range(len(vals))]
    else:
        strata = list(strata)
        by_stratum = {}
        for i, s in enumerate(strata):
            by_stratum.setdefault(s, []).append(i)
        blocks = list(by_stratum.values())
    block_cond = []
    for b in blocks:
        cs = {conditions[i] for i in b}
        if len(cs) != 1:
            raise ValueError("condition must be constant within a stratum/block")
        block_cond.append(cs.pop())

    blocks_a = [b for b, c in zip(blocks, block_cond) if c == ca]
    blocks_b = [b for b, c in zip(blocks, block_cond) if c == cb]
    if not blocks_a or not blocks_b:
        raise ValueError("each condition needs at least one block")
    all_blocks = blocks_a + blocks_b
    n_blocks_a = len(blocks_a)

    # Precompute each unit's weighted ECDF on a shared grid, plus its total weight. A
    # condition's pooled ECDF is then the weight-weighted mixture of its units' ECDFs, so
    # each permutation is O(n_units x grid) instead of O(n_triangles log n_triangles).
    grid = _shared_grid(vals, grid_points)
    ecdfs = np.array([_ecdf_on_grid(v, w, grid) for v, w in zip(vals, wts)])
    totals = np.array([w.sum() for w in wts])

    def pooled_ecdf(unit_idx):
        idx = np.asarray(unit_idx, dtype=int)
        return (totals[idx, None] * ecdfs[idx]).sum(axis=0) / totals[idx].sum()

    def pooled_stat(block_idx_a, block_idx_b):
        fa = pooled_ecdf([i for b in block_idx_a for i in b])
        fb = pooled_ecdf([i for b in block_idx_b for i in b])
        diff = np.abs(fa - fb)
        if statistic == "ks":
            return float(diff.max())
        return float(np.trapezoid(diff, grid))   # 1-Wasserstein = integral |F_a - F_b|

    observed = pooled_stat(blocks_a, blocks_b)

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(reps):
        order = rng.permutation(len(all_blocks))
        pa = [all_blocks[j] for j in order[:n_blocks_a]]
        pb = [all_blocks[j] for j in order[n_blocks_a:]]
        if pooled_stat(pa, pb) >= observed - 1e-12:
            ge += 1
    return {
        "statistic": statistic,
        "observed": observed,
        "p_value": (1 + ge) / (1 + reps),
        "condition_a": ca, "condition_b": cb,
        "n_units_a": sum(1 for c in conditions if c == ca),
        "n_units_b": sum(1 for c in conditions if c == cb),
        "n_blocks_a": n_blocks_a, "n_blocks_b": len(blocks_b),
        "reps": reps,
        "min_possible_p": 2.0 / comb(len(all_blocks), n_blocks_a),
    }


# ---------------------------------------------------------------------------
# Cluster (whole-tomogram) bootstrap
# ---------------------------------------------------------------------------

def _weighted_mean(values, weights):
    return float(np.sum(weights * values) / np.sum(weights))


def _weighted_median(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


_SUMMARIES = {"mean": _weighted_mean, "median": _weighted_median}


def cluster_bootstrap(unit_values, unit_weights=None, statistic="mean",
                      reps=1000, ci=0.90, seed=0):
    """Whole-tomogram bootstrap confidence interval for a pooled summary statistic.

    Resamples the independent units (tomograms) WITH REPLACEMENT and recomputes the
    area-weighted summary of the pooled triangles. This respects within-tomogram
    spatial correlation -- unlike an i.i.d. triangle resample (the legacy
    morphometrics_stats.bootstrap), which assumes independence and produces confidence
    intervals that are far too narrow.

    Parameters
    ----------
    unit_values : list of 1D array-likes, one per tomogram.
    unit_weights : optional list of per-triangle area weights matching unit_values.
    statistic : "mean", "median", or a callable (values, weights) -> float
        (e.g. pass morphometrics_stats.weighted_histogram_peak with functools.partial).
    reps : bootstrap resamples.
    ci : central interval mass (0.90 -> 5th/95th percentiles).
    seed : RNG seed.

    Returns
    -------
    dict: estimate, ci_low, ci_high, ci, reps, n_units.
    """
    if callable(statistic):
        stat_fn = statistic
    elif statistic in _SUMMARIES:
        stat_fn = _SUMMARIES[statistic]
    else:
        raise ValueError(f"statistic must be callable or one of {tuple(_SUMMARIES)}")

    vals = [np.asarray(v, dtype=float) for v in unit_values]
    if unit_weights is None:
        wts = [np.ones(len(v)) for v in vals]
    else:
        wts = [np.asarray(w, dtype=float) for w in unit_weights]
    n = len(vals)
    if n < 2:
        raise ValueError("cluster bootstrap needs at least 2 units")

    def pooled(idx):
        return stat_fn(np.concatenate([vals[i] for i in idx]),
                       np.concatenate([wts[i] for i in idx]))

    estimate = pooled(range(n))
    rng = np.random.default_rng(seed)
    boot = np.array([pooled(rng.integers(0, n, n)) for _ in range(reps)])
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(boot, lo)),
        "ci_high": float(np.percentile(boot, hi)),
        "ci": ci, "reps": reps, "n_units": n,
    }


def cluster_t_interval(unit_values, unit_weights=None, statistic="mean", ci=0.90):
    """Cluster-level t confidence interval -- the recommended CI for a per-condition mean.

    Each tomogram contributes ONE number (its area-weighted `statistic`); the interval
    is mean +/- t_{n-1} * SE over those per-tomogram values. This treats the tomogram as
    the unit of replication and, at the small tomogram counts typical here (n~6), is
    much better calibrated than the percentile `cluster_bootstrap` (in coverage sims the
    t-interval reaches ~0.86 at nominal 0.90, the percentile bootstrap ~0.77, and the
    legacy i.i.d. triangle bootstrap ~0.20). Tomograms are weighted EQUALLY (the standard
    biological-replicate analysis), so one large surface cannot dominate.

    `statistic` is "mean", "median", or a callable (values, weights) -> float. Use the
    percentile `cluster_bootstrap` only for statistics where a per-unit summary + t is
    not appropriate, and treat its small-n coverage as approximate.

    Returns dict: estimate, ci_low, ci_high, ci, n_units, unit_values (the per-tomogram
    summaries).
    """
    from scipy.stats import t as t_dist

    if callable(statistic):
        stat_fn = statistic
    elif statistic in _SUMMARIES:
        stat_fn = _SUMMARIES[statistic]
    else:
        raise ValueError(f"statistic must be callable or one of {tuple(_SUMMARIES)}")

    vals = [np.asarray(v, dtype=float) for v in unit_values]
    if unit_weights is None:
        wts = [np.ones(len(v)) for v in vals]
    else:
        wts = [np.asarray(w, dtype=float) for w in unit_weights]
    n = len(vals)
    if n < 2:
        raise ValueError("cluster t-interval needs at least 2 units")

    per_unit = np.array([stat_fn(v, w) for v, w in zip(vals, wts)])
    est = float(per_unit.mean())
    se = float(per_unit.std(ddof=1) / np.sqrt(n))
    half = t_dist.ppf((1 + ci) / 2, n - 1) * se
    return {
        "estimate": est,
        "ci_low": est - half, "ci_high": est + half,
        "ci": ci, "n_units": n,
        "unit_values": [float(x) for x in per_unit],
    }

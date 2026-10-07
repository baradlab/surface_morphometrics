"""Tests for the pure fitting/alignment helpers in _thickness_worker (numpy/scipy)."""
import numpy as np
import scipy.optimize as opt

from surface_morphometrics import _thickness_worker as tw


def _fit_bilayer(dat, x):
    """Mirror the dual-Gaussian offset fit used in fit_triangle_chunk_offsets."""
    mid = len(dat) // 2
    lm = np.argmin(dat[:mid])
    rm = np.argmin(dat[mid:]) + mid
    a, b = x[lm + 2:rm - 2], dat[lm + 2:rm - 2]
    center_seed, half_seed, _ = tw._seed_bilayer_center(a, b)
    af, bf = tw._symmetric_fit_window(a, b, center_seed, half_seed)
    p0 = [0.02, 0.02, 1.5, center_seed, half_seed, 0.0]
    bounds = ([0.005, 0.005, 0.8, center_seed - 3.0, tw.MIN_THICKNESS / 2.0, -1],
              [0.04, 0.04, 2.2, center_seed + 3.0, tw.MAX_THICKNESS / 2.0, 1])
    p, _ = opt.curve_fit(tw._dual_gaussian_shared_width, af, bf, p0, bounds=bounds)
    return p[3], p[4]  # center, half_sep


def test_dual_gaussian_centered_is_two_separated_peaks():
    x = np.linspace(-8, 8, 200)
    y = tw._dual_gaussian_centered(x, 1.0, 1.0, 1.0, 1.0, center=0.5, half_sep=2.0, o=0.1)
    # equals two Gaussians at center -/+ half_sep, plus offset
    expected = (tw._monogaussian(x, 1, 0.5 - 2.0, 1)
                + tw._monogaussian(x, 1, 0.5 + 2.0, 1) + 0.1)
    assert np.allclose(y, expected)


def test_seed_bilayer_center_finds_two_leaflets():
    x = np.linspace(-10, 10, 201)
    # leaflets at -0.6 and +3.0 -> center +1.2, half 1.8
    b = tw._monogaussian(x, 1.0, -0.6, 1.0) + tw._monogaussian(x, 0.8, 3.0, 1.0)
    center, half, n_peaks = tw._seed_bilayer_center(x, b)
    assert n_peaks == 2
    assert abs(center - 1.2) < 0.2 and abs(half - 1.8) < 0.2


def test_seed_bilayer_center_reports_single_unresolved_peak():
    x = np.linspace(-10, 10, 201)
    b = tw._monogaussian(x, 1.0, 1.0, 2.5)   # one broad, unresolved peak
    center, half, n_peaks = tw._seed_bilayer_center(x, b)
    assert n_peaks == 1                       # only one leaflet resolved
    assert abs(center - 1.0) < 0.2            # center seeded at the single peak


def test_seed_rejects_small_shoulders_as_single_peak():
    # A single central membrane peak flanked by two smaller protein-density
    # shoulders. The shoulders are local maxima but have tiny prominence, so they
    # must NOT be taken as leaflets -- the region is one (unresolved) membrane.
    x = np.linspace(-10, 10, 161)
    b = (tw._monogaussian(x, 0.020, 0.0, 1.5)      # central membrane
         + tw._monogaussian(x, 0.008, -4.0, 1.0)   # protein shoulder
         + tw._monogaussian(x, 0.008, 4.0, 1.0))   # protein shoulder
    center, half, n_resolved = tw._seed_bilayer_center(x, b)
    assert n_resolved == 1            # shoulders rejected, only the membrane counts
    assert abs(center) < 0.3          # centered on the membrane, not a shoulder/midpoint


def test_seed_resolves_near_symmetric_bilayer_despite_prominence_asymmetry():
    # Regression: a well-resolved bilayer whose two leaflets are near-equal height
    # (a hair apart) with a shallow inter-leaflet saddle. find_peaks assigns the
    # marginally taller leaflet a large prominence (down to the solvent base) and
    # the other only a small one (down to the saddle), so their prominence *ratio*
    # is ~0.2 even though both are real leaflets. The seeder must still resolve two
    # (it compares height above the shared base, not prominence) -- otherwise
    # essentially every clean bilayer wrongly falls back to a single Gaussian.
    x = np.linspace(-10, 10, 161)
    b = (0.011
         + tw._monogaussian(x, 0.0062, -1.75, 1.0)
         + tw._monogaussian(x, 0.0063, 1.75, 1.0))   # +1.75 leaflet a hair taller
    mid = len(b) // 2
    lm = np.argmin(b[:mid]); rm = np.argmin(b[mid:]) + mid
    a, bb = x[lm + 2:rm - 2], b[lm + 2:rm - 2]
    center, half, n_resolved = tw._seed_bilayer_center(a, bb)
    assert n_resolved == 2
    assert abs(center) < 0.3 and abs(2 * half - 3.5) < 0.6


def test_seed_resolves_merged_bilayer_with_near_flat_saddle():
    # Regression (the OMM case): a thin/merged bilayer whose two leaflets are only
    # ~2.3 nm apart with an almost-flat saddle between them. The second leaflet's
    # find_peaks prominence is near zero, so any prominence floor drops it; the
    # seeder must instead keep it on height-above-base + separation and resolve two.
    x = np.linspace(-10, 10, 161)
    b = (0.011
         + tw._monogaussian(x, 0.011, -1.25, 0.95)
         + tw._monogaussian(x, 0.0112, 1.25, 0.95))   # near-equal, barely separated
    mid = len(b) // 2
    lm = np.argmin(b[:mid]); rm = np.argmin(b[mid:]) + mid
    a, bb = x[lm + 2:rm - 2], b[lm + 2:rm - 2]
    # the inter-leaflet saddle is only just below the peaks (a near-flat dip)
    center, half, n_resolved = tw._seed_bilayer_center(a, bb)
    assert n_resolved == 2
    assert abs(center) < 0.4 and 2 * half >= tw.MIN_THICKNESS


def test_seed_keeps_asymmetric_bilayer_as_two_leaflets():
    # Genuinely asymmetric bilayer (one leaflet taller) must still resolve as two.
    x = np.linspace(-10, 10, 161)
    b = tw._monogaussian(x, 0.018, -0.6, 1.0) + tw._monogaussian(x, 0.015, 3.0, 1.0)
    center, half, n_resolved = tw._seed_bilayer_center(x, b)
    assert n_resolved == 2
    assert abs(center - 1.2) < 0.3


def test_partial_mixing_with_shallow_dip_still_fits_dual():
    # Overlapping leaflets that still leave a shallow saddle (partial mixing) must
    # resolve as two and the dual fit must recover the center exactly.
    x = np.linspace(-10, 10, 201)
    dat = 0.011 + tw._monogaussian(x, 0.017, -1.75, 1.0) + tw._monogaussian(x, 0.017, 1.75, 1.0)
    mid = len(dat) // 2
    lm = np.argmin(dat[:mid]); rm = np.argmin(dat[mid:]) + mid
    _, _, n_resolved = tw._seed_bilayer_center(x[lm + 2:rm - 2], dat[lm + 2:rm - 2])
    assert n_resolved == 2
    center, half = _fit_bilayer(dat, x)
    assert abs(center) < 0.1 and abs(2 * half - 3.5) < 0.4


def test_single_resolved_peak_falls_back_to_single_gaussian():
    # A poorly-resolved region (one broad peak): the dual fit must NOT be forced;
    # n_resolved < 2 routes it to the single-Gaussian step, which still centers on
    # the peak (no spurious ~2 nm bilayer reported).
    x = np.linspace(-10, 10, 161)
    dat = 0.011 + tw._monogaussian(x, 0.02, 1.0, 2.5)
    mid = len(dat) // 2
    lm = np.argmin(dat[:mid]); rm = np.argmin(dat[mid:]) + mid
    a, b = x[lm + 2:rm - 2], dat[lm + 2:rm - 2]
    _, _, n_resolved = tw._seed_bilayer_center(a, b)
    assert n_resolved < 2                      # dual Gaussian is skipped
    # the single-Gaussian fallback recovers the peak center
    p_mono, _ = opt.curve_fit(tw._monogaussian_with_offset, a, b,
                              [0.03, 1.0, 2.5, 0],
                              bounds=([0.005, -6, 1.0, -1], [0.06, 6, 5.0, 1]))
    assert abs(p_mono[1] - 1.0) < 0.3


def test_fit_recovers_center_of_asymmetric_bilayer_with_confounder():
    # The reported failure: taller leaflet + nearby density made the old fit put
    # both Gaussians in one leaflet. The centered+windowed fit must recover the
    # true bilayer center (+1.2 nm), not a single leaflet position.
    x = np.linspace(-10, 10, 161)
    dat = (0.011
           + tw._monogaussian(x, 0.018, 1.2 - 1.8, 1.0)   # taller left leaflet
           + tw._monogaussian(x, 0.015, 1.2 + 1.8, 1.0)   # shorter right leaflet
           + tw._monogaussian(x, 0.014, 7.0, 1.0))        # confounding nearby density
    center, half = _fit_bilayer(dat, x)
    assert abs(center - 1.2) < 0.3                          # bilayer center, not a leaflet
    assert tw.MIN_THICKNESS <= 2 * half <= tw.MAX_THICKNESS  # peaks stay separated
    assert half >= tw.MIN_THICKNESS / 2.0


def test_dual_gaussian_shared_width_ties_both_widths():
    # The shared-width model must equal the independent-width model with w1 == w2.
    # Tying the widths is what stops the fit from absorbing asymmetric outer density
    # into unequal leaflet widths (which, with a shared center, biases the center).
    x = np.linspace(-8, 8, 200)
    shared = tw._dual_gaussian_shared_width(x, 1.0, 0.8, 1.3, center=0.5, half_sep=2.0, o=0.1)
    expected = tw._dual_gaussian_centered(x, 1.0, 1.3, 0.8, 1.3, center=0.5, half_sep=2.0, o=0.1)
    assert np.allclose(shared, expected)


def test_symmetric_fit_window_is_balanced_about_seed():
    # When the profile extends further on one side of the seed than the other, the
    # window must be trimmed to the shorter side so the fit sees an equal x-range each
    # way -- an unbalanced range gives the least-squares fit more leverage on one side
    # and pulls the center off the midpoint.
    a = np.linspace(-4.0, 7.0, 111)            # extends much further to the right
    b = np.ones_like(a)
    af, bf = tw._symmetric_fit_window(a, b, center_seed=0.0, half_seed=1.75)
    assert abs(af.min() + af.max()) < 1e-6     # symmetric about 0
    assert af.min() >= -4.0 and af.max() <= 7.0


def test_fit_keeps_well_centered_bilayer_centered():
    x = np.linspace(-10, 10, 161)
    dat = 0.011 + tw._monogaussian(x, 0.017, -2.0, 1.1) + tw._monogaussian(x, 0.017, 2.0, 1.1)
    center, half = _fit_bilayer(dat, x)
    assert abs(center) < 0.2 and abs(2 * half - 4.0) < 0.5  # no spurious move


def test_global_seed_still_gates_dual_per_triangle():
    # Even when a resolved global fit is supplied (its center/half seed every
    # triangle), fit_triangle_chunk_offsets must still gate the dual fit on each
    # triangle's own resolution. A genuinely single-peak (poorly-resolved) triangle
    # must fall back to the single Gaussian (method 2), not be forced into a dual that
    # invents a ~2 nm bilayer and -- for a skewed peak -- mis-centers the vertex.
    x = np.linspace(-10, 10, 81)
    bilayer = (0.01 + tw._monogaussian(x, 0.02, -1.75, 1.0)
               + tw._monogaussian(x, 0.02, 1.75, 1.0))          # two leaflets
    single_broad = 0.01 + tw._monogaussian(x, 0.02, 0.0, 2.2)   # one broad peak
    skewed = (0.01 + tw._monogaussian(x, 0.02, 0.6, 1.3)
              + tw._monogaussian(x, 0.008, 3.2, 1.8))           # skewed single peak
    profiles = np.vstack([bilayer, single_broad, skewed])
    n = len(profiles)
    # Worker multiplies by -1; give each triangle exactly one neighbour (itself).
    tw.init_worker(-profiles, np.zeros((n, 1)), np.arange(n).reshape(n, 1), x,
                   use_xcorr=False, global_fit_params=(-1.75, 1.4, 1.75, 1.4))
    try:
        res = tw.fit_triangle_chunk_offsets(list(range(n)))
    finally:
        tw.init_worker(None, None, None, x)        # reset module globals
    methods = [r[3] for r in res]
    centers = [r[0] for r in res]
    assert methods == [1, 2, 2]                    # bilayer dual; single peaks -> single
    assert abs(centers[0]) < 0.3                   # bilayer centered on the midpoint
    assert abs(centers[1]) < 0.4                   # broad single peak centered on its peak
    assert abs(centers[2]) < 2.0                   # skewed peak not flung out to a fake leaflet
    # the dual fit uses a single shared width, so both reported sigmas match
    assert res[0][1] == res[0][2]


def test_thickness_quality_gate_reports_nan_for_unresolved_triangles():
    # measure_thickness's per-triangle fit (fit_triangle_chunk) must only report a
    # thickness when two leaflets are resolved and the fit is physical. Single-peak,
    # skewed, and sub-minimum-separation profiles have no valid bilayer thickness and
    # must return NaN rather than a forced dual's spurious value.
    x = np.linspace(-10, 10, 81)
    bilayer = (0.01 + tw._monogaussian(x, 0.02, -1.75, 1.0)
               + tw._monogaussian(x, 0.02, 1.75, 1.0))          # ~3.5 nm bilayer
    single_broad = 0.01 + tw._monogaussian(x, 0.02, 0.0, 2.2)   # one broad peak
    skewed = (0.01 + tw._monogaussian(x, 0.02, 0.6, 1.3)
              + tw._monogaussian(x, 0.008, 3.2, 1.8))           # skewed single peak
    too_thin = (0.01 + tw._monogaussian(x, 0.02, -0.6, 1.0)
                + tw._monogaussian(x, 0.02, 0.6, 1.0))          # ~1.2 nm < MIN_THICKNESS
    profiles = np.vstack([bilayer, single_broad, skewed, too_thin])
    n = len(profiles)
    tw.init_worker(-profiles, np.zeros((n, 1)), np.arange(n).reshape(n, 1), x,
                   use_xcorr=False, raw_average=True)
    try:
        res = tw.fit_triangle_chunk(list(range(n)))
    finally:
        tw.init_worker(None, None, None, x)
    thicknesses = [r[0] for r in res]
    assert tw.MIN_THICKNESS <= thicknesses[0] <= tw.MAX_THICKNESS  # real bilayer measured
    assert abs(thicknesses[0] - 3.5) < 0.6
    assert all(np.isnan(t) for t in thicknesses[1:])              # the rest gated to NaN


def test_recovery_tier_needs_global_prior_to_recover_merged_bilayer():
    # A real ~2.8 nm bilayer so merged that the seeder resolves only one leaflet
    # (n_resolved < 2). With a global bilayer prior the recovery tier must accept it
    # as a centered dual (method 1); without a prior the strict gate routes it to the
    # single Gaussian (method 2). This is the "1 in 3 -> most" middle ground: merged
    # bilayers on a resolved surface still go through dual, anomalies do not.
    x = np.linspace(-10, 10, 81)
    merged = (0.01 + tw._monogaussian(x, 0.02, -1.4, 1.3)
              + tw._monogaussian(x, 0.02, 1.4, 1.3))
    mid = len(merged) // 2
    lm = np.argmin(merged[:mid]); rm = np.argmin(merged[mid:]) + mid
    a, b = x[lm + 2:rm - 2], merged[lm + 2:rm - 2]
    assert tw._seed_bilayer_center(a, b)[2] < 2          # locally merged: one leaflet

    prof = -merged[None, :]
    tw.init_worker(prof, np.zeros((1, 1)), np.array([[0]]), x, use_xcorr=False,
                   global_fit_params=(-1.4, 1.3, 1.4, 1.3))
    try:
        r_prior = tw.fit_triangle_chunk_offsets([0])[0]
    finally:
        tw.init_worker(None, None, None, x)
    tw.init_worker(prof, np.zeros((1, 1)), np.array([[0]]), x,
                   use_xcorr=False, global_fit_params=None)
    try:
        r_noprior = tw.fit_triangle_chunk_offsets([0])[0]
    finally:
        tw.init_worker(None, None, None, x)

    assert r_prior[3] == 1                                # prior -> recovered as dual
    assert abs(r_prior[0]) < 0.3                          # centered on the midpoint
    assert tw.MIN_THICKNESS <= 2 * r_prior[1] <= tw.MAX_THICKNESS
    assert r_noprior[3] == 2                              # no prior -> single Gaussian


def test_thickness_recovery_and_resolution_score():
    # measure_thickness path with a global bilayer prior: a clearly-resolved bilayer
    # is measured at high confidence (score ~1), a merged bilayer is recovered via the
    # prior but flagged low-confidence (score < 0.5), and a genuine single peak is
    # still NaN. The resolution score lets downstream separate the two.
    x = np.linspace(-10, 10, 81)
    resolved = (0.01 + tw._monogaussian(x, 0.02, -1.75, 1.0)
                + tw._monogaussian(x, 0.02, 1.75, 1.0))
    merged = (0.01 + tw._monogaussian(x, 0.02, -1.4, 1.3)
              + tw._monogaussian(x, 0.02, 1.4, 1.3))
    single = 0.01 + tw._monogaussian(x, 0.02, 0.0, 2.4)
    profiles = np.vstack([resolved, merged, single])
    n = len(profiles)
    tw.init_worker(-profiles, np.zeros((n, 1)), np.arange(n).reshape(n, 1), x,
                   use_xcorr=False, global_fit_params=(-1.75, 1.4, 1.75, 1.4))
    try:
        res = tw.fit_triangle_chunk(list(range(n)))
    finally:
        tw.init_worker(None, None, None, x)
    (t_res, _, s_res, r_res), (t_mer, _, s_mer, r_mer), (t_sin, _, s_sin, r_sin) = res
    # resolved: measured, high-confidence score
    assert tw.MIN_THICKNESS <= t_res <= tw.MAX_THICKNESS and s_res >= 0.5
    # merged: recovered (measured) but flagged low-confidence
    assert tw.MIN_THICKNESS <= t_mer <= tw.MAX_THICKNESS and s_mer < 0.5
    # genuine single peak: no thickness
    assert np.isnan(t_sin)
    # only the merged triangle's thickness came from the recovery tier
    assert (r_res, r_mer, r_sin) == (0, 1, 0)


def test_thickness_no_prior_stays_strict():
    # Without a global prior the thickness path stays strict: the merged bilayer that
    # the recovery tier would rescue is NaN instead.
    x = np.linspace(-10, 10, 81)
    merged = (0.01 + tw._monogaussian(x, 0.02, -1.4, 1.3)
              + tw._monogaussian(x, 0.02, 1.4, 1.3))
    tw.init_worker(-merged[None, :], np.zeros((1, 1)), np.array([[0]]), x,
                   use_xcorr=False, global_fit_params=None)
    try:
        thk, _, score, recovered = tw.fit_triangle_chunk([0])[0]
    finally:
        tw.init_worker(None, None, None, x)
    assert np.isnan(thk)            # no prior -> no recovery
    assert score < 0.5             # but the score still reports it was unresolved
    assert recovered == 0


def test_monogaussian_peak_and_symmetry():
    x = np.linspace(-10, 10, 401)
    y = tw._monogaussian(x, h=2.0, c=1.0, w=1.5)
    assert np.isclose(y.max(), 2.0, atol=1e-3)        # peak height
    assert np.isclose(x[np.argmax(y)], 1.0, atol=0.1)  # peak position
    # symmetric about the center
    assert np.isclose(tw._monogaussian(0.0, 2, 1, 1.5), tw._monogaussian(2.0, 2, 1, 1.5))


def test_dual_gaussian_is_sum_plus_offset():
    x = np.linspace(-5, 5, 50)
    a = tw._monogaussian(x, 1, -1, 1)
    b = tw._monogaussian(x, 1, 1, 1)
    assert np.allclose(tw._dual_gaussian(x, 1, -1, 1, 1, 1, 1, 0.3), a + b + 0.3)
    assert np.allclose(tw._monogaussian_with_offset(x, 1, -1, 1, 0.3), a + 0.3)


def test_compute_r_squared():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert tw._compute_r_squared(y, y) == 1.0          # perfect fit
    # predicting the mean everywhere -> R^2 == 0
    assert np.isclose(tw._compute_r_squared(y, np.full_like(y, y.mean())), 0.0)
    # constant truth -> defined as 0
    assert tw._compute_r_squared(np.ones(4), np.ones(4)) == 0.0


def test_weighted_centroid_recovers_peak_position():
    x = np.linspace(-10, 10, 201)
    y = tw._monogaussian(x, 1.0, 2.0, 1.0)             # peak at +2 (within center_range)
    assert np.isclose(tw._weighted_centroid(x, y, center_range=5.0), 2.0, atol=0.2)
    # no samples in range -> 0.0
    assert tw._weighted_centroid(np.array([20.0, 30.0]), np.array([1.0, 1.0])) == 0.0


def test_xcorr_offset_recovers_known_shift():
    x = np.linspace(-10, 10, 201)
    ref = tw._monogaussian(x, 1.0, 0.0, 1.5)
    shifted = tw._monogaussian(x, 1.0, 1.0, 1.5)       # shifted +1 nm
    off = tw._xcorr_offset(shifted, ref, x)
    assert np.isclose(off, 1.0, atol=0.15)
    # flat profile -> zero offset (no information)
    assert tw._xcorr_offset(np.ones_like(x), ref, x) == 0.0


def test_xcorr_offsets_batch_matches_known_shifts():
    x = np.linspace(-10, 10, 201)
    ref = tw._monogaussian(x, 1.0, 0.0, 1.5)
    shifts = [-2.0, 0.0, 1.5]
    profiles = np.vstack([tw._monogaussian(x, 1.0, s, 1.5) for s in shifts])
    out = tw._xcorr_offsets_batch(profiles, ref, x)
    assert np.allclose(out, shifts, atol=0.2)

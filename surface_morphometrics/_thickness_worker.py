"""
Worker functions for parallel thickness measurement.

This module is intentionally minimal to avoid importing heavy dependencies
(pycurv, graph_tool) in multiprocessing worker processes.
"""

import numpy as np
import scipy.optimize as opt

# Bilayer fit-quality thresholds, shared by the per-triangle and global fits.
R2_THRESHOLD = 0.6
MIN_THICKNESS = 2.0   # nm (minimum accepted peak-to-peak leaflet separation)
MAX_THICKNESS = 7.0   # nm (maximum accepted peak-to-peak leaflet separation)
# A second peak only counts as a real leaflet if it rises at least this fraction
# of the strongest peak's height above the shared solvent baseline. Comparing
# height-above-base (not find_peaks prominence) keeps a merged bilayer's second
# leaflet -- which can have a near-zero saddle and so a tiny prominence -- while
# still rejecting small flanking shoulders (e.g. neighboring protein density).
MIN_LEAFLET_HEIGHT_RATIO = 0.5

# "Prior-recovery" tier: when the surface's global average resolves a clean bilayer,
# a triangle whose locally-averaged profile merges into a single peak can still hold a
# real (just blurred) bilayer. Seeded from the global fit, its dual fit is trusted
# only if it is non-degenerate and center-stable -- distinguishing a recovered merged
# bilayer from a genuine single / skewed peak:
MIN_LEAFLET_AMP_RATIO = 0.3       # both leaflets must have comparable amplitude
MAX_CENTER_DISAGREEMENT = 0.7     # nm; dual center must agree with a single-Gaussian fit
PINNED_THICKNESS_EPS = 0.15       # nm above MIN_THICKNESS; reject floor-pinned (fully merged) fits


def usable_profile_rows(profiles):
    """Which density profiles are fully inside the tomogram, as a boolean mask.

    ``sample_density`` writes NaN for samples that fall outside the tomogram, so a
    row containing any NaN is a linescan that ran off the edge of the volume. Such a
    profile cannot be fitted, and averaging it into a neighborhood would propagate
    the damage to triangles that are themselves fine -- so every site that averages
    profiles over neighbors filters with this first.
    """
    profiles = np.atleast_2d(profiles)
    return np.all(np.isfinite(profiles), axis=1)


def _monogaussian(x, h, c, w):
    """Single gaussian function."""
    return h * np.exp(-(x - c)**2 / (2 * w**2))


def _dual_gaussian(x, h1, c1, w1, h2, c2, w2, o):
    """Dual gaussian function for curve fitting."""
    return _monogaussian(x, h1, c1, w1) + _monogaussian(x, h2, c2, w2) + o


def _monogaussian_with_offset(x, h, c, w, o):
    """Single Gaussian with offset for curve_fit."""
    return _monogaussian(x, h, c, w) + o


def _dual_gaussian_centered(x, h1, w1, h2, w2, center, half_sep, o):
    """Dual Gaussian parameterized by the bilayer center and half-separation.

    The two leaflet peaks sit at ``center - half_sep`` and ``center + half_sep``,
    so bounding ``half_sep`` to a physical range forces one Gaussian into each
    leaflet -- the fit can never collapse both peaks into the same leaflet (the
    failure mode of seeding both Gaussians at the global-max peak). ``center`` is
    the membrane center (the refinement offset); ``2 * half_sep`` is the thickness.
    """
    return (_monogaussian(x, h1, center - half_sep, w1)
            + _monogaussian(x, h2, center + half_sep, w2) + o)


def _dual_gaussian_shared_width(x, h1, h2, w, center, half_sep, o):
    """Dual Gaussian like :func:`_dual_gaussian_centered` but with one shared width.

    A bilayer's two leaflets have essentially the same width, so tying them to a
    single ``w`` is more physical -- and, critically, removes a centering bias:
    with independent widths the fit absorbs *asymmetric* outer density (a heavier
    flank on one side, CTF ringing) by making one leaflet wider than the other, and
    because both peaks share a single ``center`` that width-tilt is converted into a
    systematic shift of the fitted center toward one side. Sharing the width leaves
    the center free to sit at the true bilayer midpoint.
    """
    return _dual_gaussian_centered(x, h1, w, h2, w, center, half_sep, o)


def _symmetric_fit_window(a, b, center_seed, half_seed, buffer=1.5, min_half_width=2.5):
    """Window the profile symmetrically about ``center_seed`` for the dual fit.

    The window is scaled to the seeded bilayer (``half_seed + buffer``) and kept
    symmetric about the seed: if the profile extends further on one side than the
    other, the window is trimmed to the shorter side so the fit sees an equal x-range
    each way. An asymmetric range would otherwise give the least-squares fit more
    leverage on one side and pull the center off the midpoint. Falls back to the full
    region if the symmetric window would be too small to fit.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    half_width = max(half_seed + buffer, min_half_width)
    half_width = min(half_width, center_seed - float(a.min()), float(a.max()) - center_seed)
    mask = (a >= center_seed - half_width) & (a <= center_seed + half_width)
    if int(mask.sum()) >= 5:
        return a[mask], b[mask]
    return a, b


def _seed_bilayer_center(a, b, default_center=0.0):
    """Estimate (center, half_sep, n_resolved) of a bilayer profile region.

    Enumerates every local maximum (no prominence floor), anchors on the tallest
    peak -- the dominant membrane density -- and pairs it with a partner leaflet a
    physical bilayer thickness (MIN..MAX nm) away. The partner is validated by its
    height above the shared solvent baseline (``b.min()``), not by ``find_peaks``
    prominence. This matters for two reasons:

    * A merged/thin bilayer (e.g. an OMM at this resolution) has two real leaflets
      separated by a near-flat saddle, so the shorter leaflet's *prominence* is tiny
      and any prominence floor -- or comparing the two leaflets' prominences -- drops
      it. Height above the common base is independent of saddle depth, so both
      leaflets of a merged bilayer are kept.
    * Comparing height-above-base still rejects small flanking shoulders (neighbouring
      protein density), which sit well below ``MIN_LEAFLET_HEIGHT_RATIO`` of the
      membrane peak, and the MIN..MAX window rejects distant confounders.

    Being permissive here is safe: the caller validates every dual fit by R^2 and
    fitted thickness, so a spurious partner is discarded downstream, whereas a missed
    leaflet would wrongly skip the dual fit entirely. ``n_resolved`` is the number of
    real leaflets (2 -> fit a dual Gaussian; 1 -> a single central peak, fall back to
    a single Gaussian; 0 -> nothing resolved). ``half_sep`` is clamped to the physical
    thickness range.
    """
    from scipy.signal import find_peaks

    a = np.asarray(a)
    b = np.asarray(b)
    # All strict local maxima. No prominence floor: a merged bilayer's second leaflet
    # has near-zero prominence but is still a real leaflet; it is filtered below by
    # height-above-base and separation instead.
    peaks, _ = find_peaks(b)
    if len(peaks) == 0:
        return default_center, MIN_THICKNESS / 2.0, 0

    positions = a[peaks]
    heights = b[peaks]
    base = float(b.min())
    # Anchor on the tallest peak (dominant membrane density), then look for the
    # partner leaflet: a peak a physical bilayer thickness (MIN..MAX) away whose
    # height above the shared solvent base is a real fraction of the anchor's.
    primary = int(np.argmax(heights))
    primary_rise = float(heights[primary] - base)
    partner = None
    best_partner_rise = 0.0
    for j in range(len(peaks)):
        if j == primary:
            continue
        sep = abs(positions[j] - positions[primary])
        if not (MIN_THICKNESS <= sep <= MAX_THICKNESS):
            continue
        partner_rise = float(heights[j] - base)
        if primary_rise <= 0:
            continue
        if (partner_rise >= MIN_LEAFLET_HEIGHT_RATIO * primary_rise
                and partner_rise > best_partner_rise):
            partner = j
            best_partner_rise = partner_rise
    if partner is not None:
        p_left, p_right = sorted([positions[primary], positions[partner]])
        center = 0.5 * (p_left + p_right)
        half = float(np.clip(0.5 * (p_right - p_left),
                             MIN_THICKNESS / 2.0, MAX_THICKNESS / 2.0))
        return float(center), half, 2
    # One real peak (a single, possibly unresolved, membrane with weak shoulders).
    return float(positions[primary]), MIN_THICKNESS / 2.0, 1


def _leaflet_resolution(a, b):
    """Continuous bilayer-resolution score in [0, 1] for a profile region.

    This is the continuous form of the resolution gate: the height (above the shared
    solvent base) of the best second leaflet within a physical bilayer distance of the
    dominant one, as a fraction of the dominant leaflet's height. A clearly resolved
    bilayer scores near 1; a profile with only a weak shoulder scores low; a fully
    merged single peak scores 0. The gate accepts a triangle as strictly resolved at
    ``MIN_LEAFLET_HEIGHT_RATIO`` (0.5); below that a thickness is only obtained by
    prior recovery and reads systematically thin, so the score doubles as a
    per-triangle reliability flag for the reported thickness.
    """
    from scipy.signal import find_peaks

    a = np.asarray(a)
    b = np.asarray(b)
    peaks, _ = find_peaks(b)
    if len(peaks) < 2:
        return 0.0
    positions = a[peaks]
    heights = b[peaks]
    base = float(b.min())
    primary = int(np.argmax(heights))
    primary_rise = float(heights[primary] - base)
    if primary_rise <= 0:
        return 0.0
    best = 0.0
    for j in range(len(peaks)):
        if j == primary:
            continue
        if MIN_THICKNESS <= abs(positions[j] - positions[primary]) <= MAX_THICKNESS:
            best = max(best, float(heights[j] - base) / primary_rise)
    return float(min(best, 1.0))


def _dual_recovery_ok(a, b, popt, r2, center_seed):
    """Accept a global-prior-seeded dual fit on a single-peak (merged) triangle?

    Used only when the surface's global average resolves a bilayer but this triangle's
    local profile shows one peak. The dual fit is trusted only if it is a real,
    non-degenerate, centered bilayer rather than a forced split of a single peak:

    * a good fit with a physical thickness clear of the floor -- a floor-pinned fit
      means the leaflets merged completely, so there is no real separation to report;
    * both leaflets comparable in amplitude (neither Gaussian collapsed to nothing);
    * the dual center agrees with an independent single-Gaussian center -- a genuine
      skewed single peak splits asymmetrically and the two disagree.

    ``popt`` is the ``_dual_gaussian_shared_width`` result (h1, h2, w, center, half, o).
    """
    thickness = 2.0 * popt[4]
    if not (r2 > R2_THRESHOLD
            and MIN_THICKNESS + PINNED_THICKNESS_EPS < thickness <= MAX_THICKNESS):
        return False
    h1, h2 = popt[0], popt[1]
    if min(h1, h2) < MIN_LEAFLET_AMP_RATIO * max(h1, h2):
        return False
    try:
        ps, _ = opt.curve_fit(_monogaussian_with_offset, a, b,
                              [0.03, center_seed, 2.5, 0],
                              bounds=([0.005, center_seed - 6, 1.0, -1],
                                      [0.06, center_seed + 6, 5.0, 1]))
    except Exception:
        return False
    return abs(popt[3] - ps[1]) < MAX_CENTER_DISAGREEMENT


def _compute_r_squared(y_true, y_pred):
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def _weighted_centroid(x, y, center_range=5.0):
    """
    Compute weighted centroid of the profile within center_range of x=0.

    More robust fallback when Gaussian fitting fails.
    """
    central_mask = (x > -center_range) & (x < center_range)
    if not np.any(central_mask):
        return 0.0

    x_central = x[central_mask]
    y_central = y[central_mask]

    # Shift weights to be positive
    weights = y_central - y_central.min()
    weight_sum = weights.sum()

    if weight_sum == 0:
        return 0.0

    return np.average(x_central, weights=weights)


def _xcorr_offset(profile, reference, x_positions):
    """
    Find the offset that best aligns profile with reference using cross-correlation.

    Uses normalized cross-correlation with sub-pixel interpolation for precision.

    Parameters
    ----------
    profile : np.ndarray
        The local density profile to align
    reference : np.ndarray
        The reference profile (e.g., global average)
    x_positions : np.ndarray
        The x positions corresponding to profile samples

    Returns
    -------
    float
        The optimal shift in the same units as x_positions
    """
    # Normalize both signals (zero mean, unit variance)
    prof_norm = profile - np.mean(profile)
    ref_norm = reference - np.mean(reference)

    prof_std = np.std(prof_norm)
    ref_std = np.std(ref_norm)

    if prof_std == 0 or ref_std == 0:
        return 0.0

    prof_norm = prof_norm / prof_std
    ref_norm = ref_norm / ref_std

    # Cross-correlation
    correlation = np.correlate(prof_norm, ref_norm, mode='full')

    # Find peak
    peak_idx = np.argmax(correlation)
    center_idx = len(reference) - 1

    # Sub-pixel interpolation using parabolic fit around peak
    if 1 <= peak_idx < len(correlation) - 1:
        y0 = correlation[peak_idx - 1]
        y1 = correlation[peak_idx]
        y2 = correlation[peak_idx + 1]
        # Parabolic interpolation
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            sub_pixel = (y0 - y2) / denom
        else:
            sub_pixel = 0.0
    else:
        sub_pixel = 0.0

    # Convert to position offset
    sample_spacing = x_positions[1] - x_positions[0] if len(x_positions) > 1 else 1.0
    shift_samples = (peak_idx + sub_pixel) - center_idx
    offset = shift_samples * sample_spacing

    return offset


def _xcorr_offsets_batch(profiles, reference, x_positions):
    """
    Vectorized cross-correlation of N profiles against a single reference.

    Uses FFT-based batch correlation instead of N separate np.correlate calls,
    which is ~10x faster for typical neighborhood sizes (~50 profiles).

    Returns shifts in the same units as x_positions (one value per profile).
    """
    N, L = profiles.shape
    # Normalize each profile (zero mean, unit variance)
    means = profiles.mean(axis=1, keepdims=True)
    prof_norm = profiles - means
    stds = prof_norm.std(axis=1, keepdims=True)
    nonzero = stds.flatten() > 0
    prof_norm[nonzero] /= stds[nonzero]

    ref_norm = reference - reference.mean()
    ref_std = reference.std()
    if ref_std == 0:
        return np.zeros(N)
    ref_norm = ref_norm / ref_std

    # Batch cross-correlation via FFT: correlate(a, b) = IFFT(FFT(a) * conj(FFT(b)))
    # Use next power-of-2 for FFT efficiency
    n_full = 2 * L - 1
    n_fft = 1 << (n_full - 1).bit_length()
    P = np.fft.rfft(prof_norm, n=n_fft, axis=1)   # (N, n_fft//2+1)
    R = np.fft.rfft(ref_norm, n=n_fft)             # (n_fft//2+1,)
    corr = np.fft.irfft(P * np.conj(R)[np.newaxis, :], n=n_fft, axis=1)
    # Full-length slice (same as mode='full' output)
    corr = np.concatenate([corr[:, -(L - 1):], corr[:, :L]], axis=1)  # (N, 2L-1)

    # Find peak and do parabolic sub-pixel refinement (vectorized)
    peak_idx = np.argmax(corr, axis=1)  # (N,)
    center_idx = L - 1
    sub_pixel = np.zeros(N)
    valid_pk = (peak_idx >= 1) & (peak_idx < n_full - 1)
    if np.any(valid_pk):
        rows = np.where(valid_pk)[0]
        pi = peak_idx[rows]
        y0 = corr[rows, pi - 1]
        y1 = corr[rows, pi]
        y2 = corr[rows, pi + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        good = np.abs(denom) > 1e-10
        sub_pixel[rows[good]] = (y0[good] - y2[good]) / denom[good]

    sample_spacing = x_positions[1] - x_positions[0] if len(x_positions) > 1 else 1.0
    shift_samples = (peak_idx + sub_pixel) - center_idx
    return shift_samples * sample_spacing


# Global variables for local thickness computation workers
_lt_value_array = None
_lt_distances = None
_lt_neighbors = None
_lt_x_positions = None


def init_local_thickness_worker(value_array, distances, neighbors, x_positions):
    """Initialize worker process with shared data for local thickness computation."""
    global _lt_value_array, _lt_distances, _lt_neighbors, _lt_x_positions
    _lt_value_array = value_array
    _lt_distances = distances
    _lt_neighbors = neighbors
    _lt_x_positions = x_positions


def compute_thickness_chunk(indices):
    """
    Compute local thickness for a chunk of triangle indices.

    Returns list of thickness values (np.nan for failed fits).
    """
    results = []
    for i in indices:
        valid_mask = _lt_distances[i] != np.inf
        if not np.any(valid_mask):
            results.append(np.nan)
            continue

        l = _lt_distances[i][valid_mask]
        neighbors = _lt_neighbors[i][valid_mask]

        # Drop neighbors whose linescan left the tomogram; if none are usable the
        # triangle simply has no measurement.
        inside = usable_profile_rows(_lt_value_array[neighbors])
        if not np.any(inside):
            results.append(np.nan)
            continue
        l, neighbors = l[inside], neighbors[inside]
        weights = 1.0 / (1.0 + l)

        dat = np.average(_lt_value_array[neighbors], weights=weights, axis=0) * -1
        dat = dat - dat.min()
        dat_sum = dat.sum()
        if dat_sum == 0:
            results.append(np.nan)
            continue
        dat = dat / (80/81 * dat_sum)

        try:
            mid = len(dat) // 2
            left_min = np.argmin(dat[:mid])
            right_min = np.argmin(dat[mid:]) + mid

            a = _lt_x_positions[left_min+2:right_min-2]
            b = dat[left_min+2:right_min-2]

            if len(a) < 7:
                results.append(np.nan)
                continue

            # Same quality gate as fit_triangle_chunk: require two resolved leaflets,
            # then a centered + shared-width fit accepted only on R^2 and a physical
            # thickness; otherwise NaN (no valid bilayer thickness).
            center_seed, half_seed, n_resolved = _seed_bilayer_center(a, b)
            if n_resolved < 2:
                results.append(np.nan)
                continue
            af, bf = _symmetric_fit_window(a, b, center_seed, half_seed)
            p0 = [0.02, 0.02, 1.5, center_seed, half_seed, 0.0]
            bounds = ([0.005, 0.005, 0.8, center_seed - 3.0, MIN_THICKNESS / 2.0, -1],
                      [0.04,  0.04,  2.2, center_seed + 3.0, MAX_THICKNESS / 2.0,  1])
            popt, _ = opt.curve_fit(_dual_gaussian_shared_width, af, bf, p0, bounds=bounds)
            r2 = _compute_r_squared(bf, _dual_gaussian_shared_width(af, *popt))
            thickness = 2.0 * popt[4]
            if r2 > R2_THRESHOLD and MIN_THICKNESS <= thickness <= MAX_THICKNESS:
                results.append(thickness)
            else:
                results.append(np.nan)
        except Exception:
            results.append(np.nan)

    return results


# Global variables for offset fitting workers (set by initializer)
_worker_thickness_arr = None
_worker_distances = None
_worker_neighbor_indices = None
_worker_x = None
_worker_monolayer = False
_worker_reference = None  # Reference profile for cross-correlation
_worker_use_xcorr = False  # Whether to use cross-correlation method
_worker_global_fit_params = None  # (c1, w1, c2, w2) from global dual Gaussian fit
_worker_rolling_xcorr = False  # Whether fit_triangle_chunk aligns neighbors before averaging
_worker_raw_average = True    # If True: average raw profiles then normalize (original behavior)


def init_worker(thickness_arr, distances, neighbor_indices, x, monolayer=False,
                reference=None, use_xcorr=False, global_fit_params=None,
                rolling_xcorr=False, raw_average=True):
    """Initialize worker process with shared data."""
    global _worker_thickness_arr, _worker_distances, _worker_neighbor_indices
    global _worker_x, _worker_monolayer, _worker_reference, _worker_use_xcorr
    global _worker_global_fit_params, _worker_rolling_xcorr, _worker_raw_average
    _worker_thickness_arr = thickness_arr
    _worker_distances = distances
    _worker_neighbor_indices = neighbor_indices
    _worker_x = x
    _worker_monolayer = monolayer
    _worker_reference = reference
    _worker_use_xcorr = use_xcorr
    _worker_global_fit_params = global_fit_params
    _worker_rolling_xcorr = rolling_xcorr
    _worker_raw_average = raw_average


def fit_triangle_chunk(indices):
    """
    Fit dual gaussian to a chunk of triangles' density profiles.

    Processes multiple triangles per call to reduce multiprocessing overhead.

    Returns
    -------
    list of tuples
        [(thickness, offset, resolution), ...] for each triangle in chunk.
        ``thickness`` is NaN where no valid bilayer was measured; ``resolution`` is
        the per-triangle bilayer-resolution score in [0, 1] (NaN if there was no
        profile to score) -- a reliability flag for the reported thickness.
    """
    sample_spacing = _worker_x[1] - _worker_x[0] if len(_worker_x) > 1 else 1.0
    max_shift_samples = max(1, int(round(2.0 / sample_spacing)))

    results = []
    for i in indices:
        # Get valid neighbors (those within distance bound)
        valid_mask = _worker_distances[i] != np.inf
        l = _worker_distances[i][valid_mask]
        neighbors = _worker_neighbor_indices[i][valid_mask]

        if len(neighbors) == 0:
            results.append((np.nan, 0, np.nan))
            continue

        # Drop neighbors whose linescan left the tomogram (NaN samples). Without
        # this, one out-of-bounds neighbor makes the whole weighted average NaN and
        # a triangle that is itself well inside the volume loses its measurement.
        inside = usable_profile_rows(_worker_thickness_arr[neighbors])
        if not np.any(inside):
            results.append((np.nan, 0, np.nan))
            continue
        l, neighbors = l[inside], neighbors[inside]

        weights = 1.0 / (1.0 + l)

        if _worker_raw_average:
            # Original measure_thickness.py behavior: weighted-average raw profiles,
            # then invert and normalize the composite.
            dat = np.average(_worker_thickness_arr[neighbors], weights=weights, axis=0) * -1
            dat = dat - dat.min()
            dat_sum = dat.sum()
            if dat_sum == 0:
                results.append((np.nan, 0, np.nan))
                continue
            dat = dat / (80 / 81 * dat_sum)
        else:
            # Invert and normalize each neighbor's profile individually before averaging.
            indiv = _worker_thickness_arr[neighbors] * -1
            mins = indiv.min(axis=1, keepdims=True)
            indiv = indiv - mins
            row_sums = indiv.sum(axis=1, keepdims=True)
            valid_rows = row_sums.flatten() > 0
            if not np.any(valid_rows):
                results.append((np.nan, 0, np.nan))
                continue
            indiv[valid_rows] /= (80 / 81 * row_sums[valid_rows])

            valid_profs = indiv[valid_rows]
            valid_weights = weights[valid_rows]

            dat0 = np.average(valid_profs, weights=valid_weights, axis=0)
            if _worker_rolling_xcorr:
                # xcorr-align each neighbor to the initial average before re-averaging,
                # sharpening bilayer peaks and improving dual Gaussian success rate.
                shifts_nm = _xcorr_offsets_batch(valid_profs, dat0, _worker_x)
                shift_samples = np.clip(
                    np.round(shifts_nm / sample_spacing).astype(int),
                    -max_shift_samples, max_shift_samples
                )
                n_cols = valid_profs.shape[1]
                col_idx = (np.arange(n_cols)[np.newaxis, :] + shift_samples[:, np.newaxis]) % n_cols
                aligned_profs = valid_profs[np.arange(len(shift_samples))[:, np.newaxis], col_idx]
                dat = np.average(aligned_profs, weights=valid_weights, axis=0)
            else:
                dat = dat0

        # Find minima to constrain fitting region
        mid = len(dat) // 2
        left_min = np.argmin(dat[:mid])
        right_min = np.argmin(dat[mid:]) + mid

        a = _worker_x[left_min+2:right_min-2]
        b = dat[left_min+2:right_min-2]

        if len(a) < 7:  # Need enough points for the fit
            results.append((np.nan, 0, np.nan))
            continue

        # Per-triangle reliability flag reported alongside every thickness: how clearly
        # the two leaflets are resolved (1 = cleanly separated, 0 = a single merged
        # peak). Strict fits score >= MIN_LEAFLET_HEIGHT_RATIO; prior-recovered fits
        # below it (and read systematically thin), so downstream can weight or filter.
        resolution = _leaflet_resolution(a, b)

        # Quality gate, two tiers (mirrors the refinement path):
        #   * resolved (n_resolved >= 2): fit and accept on R^2 + physical thickness;
        #   * prior recovery (n_resolved < 2 but the global average resolved a bilayer):
        #     a locally-merged profile may still hold a real, blurred bilayer -- fit it
        #     seeded from the global prior and accept only a non-degenerate,
        #     center-stable result (_dual_recovery_ok). Without a prior, a single peak
        #     has no measurable thickness -> NaN.
        center_seed, half_seed, n_resolved = _seed_bilayer_center(a, b)
        have_prior = _worker_global_fit_params is not None
        if have_prior:
            c1_g, w1_g, c2_g, w2_g = _worker_global_fit_params
            seed_c = 0.5 * (c1_g + c2_g)
            seed_h = float(np.clip(0.5 * (c2_g - c1_g),
                                   MIN_THICKNESS / 2.0, MAX_THICKNESS / 2.0))
            seed_w = w1_g
        else:
            seed_c, seed_h, seed_w = center_seed, half_seed, 1.5

        recovery = n_resolved < 2 and have_prior
        if n_resolved < 2 and not recovery:
            results.append((np.nan, 0, resolution))
            continue

        try:
            af, bf = _symmetric_fit_window(a, b, seed_c, seed_h)
            p0 = [0.02, 0.02, seed_w, seed_c, seed_h, 0.0]
            bounds = ([0.005, 0.005, 0.8, seed_c - 3.0, MIN_THICKNESS / 2.0, -1],
                      [0.04,  0.04,  2.2, seed_c + 3.0, MAX_THICKNESS / 2.0,  1])
            p3, _ = opt.curve_fit(_dual_gaussian_shared_width, af, bf, p0, bounds=bounds)
            r2 = _compute_r_squared(bf, _dual_gaussian_shared_width(af, *p3))
            thickness = 2.0 * p3[4]
            offset = p3[3]
            if recovery:
                accept = _dual_recovery_ok(a, b, p3, r2, seed_c)
            else:
                accept = r2 > R2_THRESHOLD and MIN_THICKNESS <= thickness <= MAX_THICKNESS
            if accept:
                results.append((thickness, offset, resolution))
            else:
                results.append((np.nan, 0, resolution))
        except Exception:
            results.append((np.nan, 0, resolution))

    return results


def fit_triangle_chunk_offsets(indices):
    """
    Fit gaussian(s) to density profiles and return center offsets and sigmas.

    Two modes available (controlled by _worker_use_xcorr):

    Cross-correlation mode (_worker_use_xcorr=True):
        Uses normalized cross-correlation with reference profile to find
        optimal shift. Very robust to noise and shape variations.

    Gaussian fitting mode (_worker_use_xcorr=False):
        Uses adaptive model selection:
        1. Try dual Gaussian (bilayer model)
        2. If R² > 0.6 and thickness in 2-7 nm range, use it
        3. Otherwise try single Gaussian
        4. If that also fails, fall back to weighted centroid

    When _worker_monolayer is True in Gaussian mode, skips dual Gaussian
    and goes straight to single Gaussian fitting.

    Returns
    -------
    list of tuples
        [(offset, sigma1, sigma2, fit_method), ...] for each triangle in chunk.
        fit_method: 0=failed, 1=dual_gaussian, 2=single_gaussian, 3=centroid, 4=xcorr
    """
    # Fit-quality thresholds are module-level constants (R2_THRESHOLD,
    # MIN_THICKNESS, MAX_THICKNESS).

    # Sample spacing — constant across all triangles, computed once per chunk
    sample_spacing = _worker_x[1] - _worker_x[0] if len(_worker_x) > 1 else 1.0
    max_shift_samples = max(1, int(round(2.0 / sample_spacing)))

    results = []
    for i in indices:
        # Get valid neighbors (those within distance bound)
        valid_mask = _worker_distances[i] != np.inf
        l = _worker_distances[i][valid_mask]
        neighbors = _worker_neighbor_indices[i][valid_mask]

        if len(neighbors) == 0:
            results.append((0, np.nan, np.nan, 0))
            continue

        weights = 1.0 / (1.0 + l)

        # Invert and normalize each neighbor's profile individually
        indiv = _worker_thickness_arr[neighbors] * -1
        mins = indiv.min(axis=1, keepdims=True)
        indiv = indiv - mins
        row_sums = indiv.sum(axis=1, keepdims=True)
        valid_rows = row_sums.flatten() > 0
        if not np.any(valid_rows):
            results.append((0, np.nan, np.nan, 0))
            continue
        indiv[valid_rows] /= (80 / 81 * row_sums[valid_rows])

        valid_profs = indiv[valid_rows]
        valid_weights = weights[valid_rows]

        if _worker_use_xcorr:
            # xcorr mode: simple weighted average, then match against global reference
            dat = np.average(valid_profs, weights=valid_weights, axis=0)
        else:
            # Gaussian mode: xcorr-align each neighbor profile to the initial average
            # before re-averaging.  This removes per-triangle positional variation so the
            # re-averaged profile has sharper bilayer peaks — improving dual Gaussian
            # success rate in low-SNR regions.
            dat0 = np.average(valid_profs, weights=valid_weights, axis=0)
            # Vectorized batch xcorr + roll (no Python loop over neighbors)
            shifts_nm = _xcorr_offsets_batch(valid_profs, dat0, _worker_x)
            shift_samples = np.clip(
                np.round(shifts_nm / sample_spacing).astype(int),
                -max_shift_samples, max_shift_samples
            )
            n_cols = valid_profs.shape[1]
            col_idx = (np.arange(n_cols)[np.newaxis, :] + shift_samples[:, np.newaxis]) % n_cols
            aligned_profs = valid_profs[np.arange(len(shift_samples))[:, np.newaxis], col_idx]
            dat = np.average(aligned_profs, weights=valid_weights, axis=0)

        # Cross-correlation mode - use template matching with reference profile
        if _worker_use_xcorr and _worker_reference is not None:
            offset = _xcorr_offset(dat, _worker_reference, _worker_x)
            results.append((offset, np.nan, np.nan, 4))
            continue

        # Find initial peak and set up fitting bounds
        ipk = _worker_x[np.argmax(dat)]

        # Find minima to constrain fitting region
        mid = len(dat) // 2
        left_min = np.argmin(dat[:mid])
        right_min = np.argmin(dat[mid:]) + mid

        a = _worker_x[left_min+2:right_min-2]
        b = dat[left_min+2:right_min-2]

        if len(a) < 5:  # Need enough points for fitting
            # Fall back to centroid
            offset = _weighted_centroid(_worker_x, dat)
            results.append((offset, np.nan, np.nan, 3))
            continue

        # Track best result
        best_offset = None
        best_sigma1 = np.nan
        best_sigma2 = np.nan
        best_method = 0

        # Step 1: Try dual Gaussian (unless monolayer mode). Parameterized by the
        # bilayer center + half-separation so the two leaflet peaks are forced to
        # opposite sides and cannot collapse into the same leaflet.
        #
        # The global average fit (when available) gives more reliable seeds for the
        # center/half/width. The dual fit runs in two tiers:
        #   * resolved (n_resolved >= 2): a clear two-leaflet profile -> fit and accept
        #     on R^2 + physical thickness (high confidence);
        #   * prior recovery (n_resolved < 2 but the global average resolved a bilayer):
        #     a locally-merged profile may still hold a real, blurred bilayer. Seed it
        #     from the global fit but accept only a non-degenerate, center-stable result
        #     (_dual_recovery_ok), so a genuine single / skewed peak is still rejected
        #     and falls through to the single Gaussian rather than inventing a bilayer.
        tri_center, tri_half, n_resolved = _seed_bilayer_center(a, b)
        have_prior = _worker_global_fit_params is not None
        if have_prior:
            c1_g, w1_g, c2_g, w2_g = _worker_global_fit_params
            center_seed = 0.5 * (c1_g + c2_g)
            half_seed = float(np.clip(0.5 * (c2_g - c1_g),
                                      MIN_THICKNESS / 2.0, MAX_THICKNESS / 2.0))
            w1_seed = w2_seed = w1_g
        else:
            center_seed, half_seed = tri_center, tri_half
            w1_seed = w2_seed = 1.5

        recovery = n_resolved < 2 and have_prior      # tier 2
        if not _worker_monolayer and (n_resolved >= 2 or recovery):
            try:
                # Restrict the fit to a window symmetric about the seeded bilayer so
                # distant density (adjacent membranes, CTF ringing) cannot pull the
                # fit out and an asymmetric range cannot bias the center.
                af, bf = _symmetric_fit_window(a, b, center_seed, half_seed)

                # params: h1, h2, shared_w, center, half_sep, offset. A single shared
                # width keeps asymmetric flanks from tilting the fit and shifting the
                # center (see _dual_gaussian_shared_width).
                w_seed = 0.5 * (w1_seed + w2_seed)
                p0 = [0.02, 0.02, w_seed, center_seed, half_seed, 0.0]
                bounds = ([0.005, 0.005, 0.8, center_seed - 3.0, MIN_THICKNESS / 2.0, -1],
                          [0.04,  0.04,  2.2, center_seed + 3.0, MAX_THICKNESS / 2.0,  1])
                p3, _ = opt.curve_fit(_dual_gaussian_shared_width, af, bf, p0, bounds=bounds)

                # Validate fit (stricter, center-stable check for the recovery tier).
                r2 = _compute_r_squared(bf, _dual_gaussian_shared_width(af, *p3))
                thickness = 2.0 * p3[4]
                if recovery:
                    accept = _dual_recovery_ok(a, b, p3, r2, center_seed)
                else:
                    accept = r2 > R2_THRESHOLD and MIN_THICKNESS <= thickness <= MAX_THICKNESS

                if accept:
                    best_offset = p3[3]          # bilayer center
                    best_sigma1 = p3[2]
                    best_sigma2 = p3[2]
                    best_method = 1
            except Exception:
                pass

        # Step 2: Try single Gaussian if dual didn't work. For an unresolved triangle
        # the global membrane center (when available) is a more robust seed than this
        # triangle's noisy argmax, which can latch onto a spurious bump; bounds stay
        # wide so a genuinely shifted membrane can still be found.
        if best_method == 0:
            try:
                mono_seed = center_seed if _worker_global_fit_params is not None else ipk
                p0_mono = [0.03, mono_seed, 2.5, 0]
                bounds_mono = ([0.005, mono_seed-6, 1.0, -1],
                               [0.06, mono_seed+6, 5.0, 1])
                p_mono, _ = opt.curve_fit(_monogaussian_with_offset, a, b, p0_mono, bounds=bounds_mono)

                # Validate fit
                y_pred = _monogaussian_with_offset(a, *p_mono)
                r2 = _compute_r_squared(b, y_pred)

                if r2 > R2_THRESHOLD:
                    best_offset = p_mono[1]
                    best_sigma1 = p_mono[2]
                    best_sigma2 = np.nan
                    best_method = 2
            except Exception:
                pass

        # Step 3: Fall back to weighted centroid
        if best_method == 0:
            best_offset = _weighted_centroid(_worker_x, dat)
            best_sigma1 = np.nan
            best_sigma2 = np.nan
            best_method = 3

        results.append((best_offset, best_sigma1, best_sigma2, best_method))

    return results

"""Density samples outside the tomogram must be NaN, not extrapolated.

`interpn(..., fill_value=None)` extrapolates linearly from the edge, which is
unbounded: a linescan leaving the volume produced invented values that the
thickness fitter could not distinguish from real data. These pin the honest
behavior and the downstream handling.
"""
import numpy as np
import pytest

from surface_morphometrics._thickness_worker import usable_profile_rows


def _volume(n=12):
    """A tiny tomogram with a bilayer-ish dark band, plus its coordinate axes."""
    axes = tuple(np.arange(float(n)) for _ in range(3))
    data = np.zeros((n, n, n))
    data[:, :, n // 2 - 1] = -1.0
    data[:, :, n // 2 + 1] = -1.0
    return axes, data


def test_samples_outside_the_tomogram_are_nan_not_extrapolated():
    from scipy import interpolate as interp
    axes, data = _volume()
    inside = [[5.0, 5.0, 5.0]]
    outside = [[5.0, 5.0, 50.0], [-30.0, 5.0, 5.0]]

    values = interp.interpn(axes, data, np.array(inside + outside),
                            method="linear", bounds_error=False, fill_value=np.nan)
    assert np.isfinite(values[0])
    assert np.all(np.isnan(values[1:]))

    # The old setting invented finite values for exactly those points.
    extrapolated = interp.interpn(axes, data, np.array(outside),
                                  method="linear", bounds_error=False, fill_value=None)
    assert np.all(np.isfinite(extrapolated))


def test_interpolate_marks_out_of_bounds_samples():
    from surface_morphometrics.sample_density import interpolate
    axes, data = _volume()
    # Two triangles: one mid-volume, one far outside along +z.
    xyz = np.array([[6.0, 500.0], [6.0, 500.0], [6.0, 500.0]])
    n_v = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    values = interpolate(data, axes, xyz, n_v,
                         sample_spacing=1.0, angstroms=False, scan_range=3)
    assert np.all(np.isfinite(values[0])), "in-bounds triangle should be measured"
    assert np.all(np.isnan(values[1])), "out-of-bounds triangle should be all NaN"


def test_usable_profile_rows_rejects_any_nan():
    profiles = np.array([
        [1.0, 2.0, 3.0],            # fully inside
        [1.0, np.nan, 3.0],         # partly outside
        [np.nan, np.nan, np.nan],   # entirely outside
    ])
    assert list(usable_profile_rows(profiles)) == [True, False, False]


def test_usable_profile_rows_handles_a_single_profile():
    assert list(usable_profile_rows(np.array([1.0, 2.0, 3.0]))) == [True]
    assert list(usable_profile_rows(np.array([1.0, np.nan]))) == [False]


def test_one_bad_neighbor_does_not_poison_the_average():
    # The motivating failure: a triangle well inside the volume averages its profile
    # over neighbors within average_radius. Without filtering, a single neighbor whose
    # scan left the tomogram makes the whole weighted average NaN.
    profiles = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 2.2, 1.0],
        [np.nan, np.nan, np.nan],   # neighbor near the tomogram edge
    ])
    weights = np.array([1.0, 1.0, 1.0])

    naive = np.average(profiles, weights=weights, axis=0)
    assert np.all(np.isnan(naive)), "unfiltered average is destroyed by one bad neighbor"

    inside = usable_profile_rows(profiles)
    filtered = np.average(profiles[inside], weights=weights[inside], axis=0)
    assert np.all(np.isfinite(filtered))
    assert filtered == pytest.approx([1.0, 2.1, 1.0])


def test_all_neighbors_outside_yields_no_measurement():
    profiles = np.full((4, 5), np.nan)
    assert not np.any(usable_profile_rows(profiles))


def _histogram(tmp_path, data, areas, labels, title="t"):
    import matplotlib
    matplotlib.use("Agg")
    from surface_morphometrics.morphometrics_stats import histogram
    return histogram(data=data, areas=areas, labels=labels, title=title, xlabel="Thickness (nm)",
                     filename=str(tmp_path / "hist.svg"))


def test_all_nan_series_skips_the_histogram_instead_of_raising(tmp_path):
    # Previously: ValueError("autodetected range of [nan, nan] is not finite"), thrown at
    # the very end of measure_thickness after all the per-surface work was done.
    nan_values = [float("nan")] * 20
    assert _histogram(tmp_path, [nan_values], [[1.0] * 20], ["IMM"]) is False
    assert not (tmp_path / "hist.svg").exists()


def test_unmeasured_series_is_omitted_but_others_still_plot(tmp_path):
    rng = np.random.default_rng(0)
    good = list(rng.normal(4.0, 0.3, 200))
    nan_values = [float("nan")] * 20
    assert _histogram(tmp_path, [good, nan_values],
                      [[1.0] * 200, [1.0] * 20], ["OMM", "IMM"]) is True
    assert (tmp_path / "hist.svg").exists()


def test_dropping_nans_does_not_bias_the_area_weighted_density():
    # The histogram is area-weighted with density=True, so excluding unmeasured
    # triangles must drop them from both the counts and the normalization.
    rng = np.random.default_rng(0)
    good = rng.normal(4.0, 0.3, 200)
    mixed = np.concatenate([good, np.full(20, np.nan)])
    weights = np.ones(220)

    clean, _ = np.histogram(good, bins=20, range=(3, 5), weights=np.ones(200), density=True)
    finite = np.isfinite(mixed)
    filtered, _ = np.histogram(mixed[finite], bins=20, range=(3, 5),
                               weights=weights[finite], density=True)
    assert np.allclose(clean, filtered)

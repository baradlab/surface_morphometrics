"""Cubic-spline density sampling (density_sampling.interpolation: cubic).

Linear interpolation blurs most at points halfway between voxels; at ~1 nm/px that
extra blur is enough to merge a bilayer's two leaflets. These pin that the cubic path
is sharper than linear on a near-Nyquist signal, keeps the out-of-bounds -> NaN rule,
and that computing spline coefficients on a cropped block matches the whole volume.
"""
import numpy as np
import pytest

pytest.importorskip("mrcfile")
from surface_morphometrics import sample_density as sd  # noqa: E402


def _wave(n=40, period=4.0):
    """A volume varying along z only: a cosine with a 4-voxel (≈ 2× Nyquist) period."""
    z = np.arange(n, dtype=float)
    data = np.broadcast_to(np.cos(2 * np.pi * z / period), (n, n, n)).astype(np.float32)
    axes = tuple(np.arange(float(n)) for _ in range(3))
    return axes, np.ascontiguousarray(data)


def _profile(axes, data, interpolation, z0=18.0):
    # One "triangle" with its normal along z, sampled every 0.25 voxel over ±4 voxels.
    xyz = np.array([[20.0], [20.0], [z0]])
    n_v = np.array([[0.0], [0.0], [1.0]])
    return sd.interpolate(data, axes, xyz, n_v, sample_spacing=0.25, scan_range=4,
                          interpolation=interpolation)[0]


def test_cubic_is_sharper_than_linear_between_voxels():
    axes, data = _wave()
    s = np.linspace(-4, 4, 33)
    truth = np.cos(2 * np.pi * (18.0 + s) / 4.0)
    err_lin = np.abs(_profile(axes, data, "linear") - truth).max()
    err_cub = np.abs(_profile(axes, data, "cubic") - truth).max()
    assert err_cub < 0.5 * err_lin
    # on the voxel grid both reproduce the data exactly
    on_grid = slice(0, None, 4)
    assert np.allclose(_profile(axes, data, "cubic")[on_grid], truth[on_grid], atol=1e-4)


def test_cubic_outside_the_volume_is_nan():
    axes, data = _wave(n=20)
    pts = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 25.0], [-3.0, 5.0, 5.0], [5.0, 5.0, 19.0]])
    vals = sd._spline_sample(data, pts)
    assert np.isfinite(vals[0]) and np.isfinite(vals[3])   # inside, incl. the last voxel
    assert np.isnan(vals[1]) and np.isnan(vals[2])         # outside -> missing


def test_cropped_block_matches_full_volume_spline():
    from scipy import ndimage
    rng = np.random.default_rng(0)
    data = rng.normal(size=(48, 48, 48)).astype(np.float32)
    pts = rng.uniform(18, 30, size=(200, 3))               # touch only a central block
    full = ndimage.map_coordinates(ndimage.spline_filter(data, order=3, mode="mirror"),
                                   pts.T, order=3, prefilter=False, mode="mirror")
    assert np.allclose(sd._spline_sample(data, pts), full, atol=1e-3)


def test_unknown_interpolation_is_rejected():
    with pytest.raises(ValueError, match="interpolation"):
        sd._interpolation_order("quintic")
    assert sd._interpolation_order("Cubic") == 3 and sd._interpolation_order("linear") == 1

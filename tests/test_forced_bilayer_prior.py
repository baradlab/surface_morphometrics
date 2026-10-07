"""The opt-in forced whole-surface bilayer prior (thickness_measurements.force_bilayer_prior).

At ~1 nm/px an OMM's two leaflets merge, so even the whole-surface average profile is a
single flat-topped peak. Without a prior from that average the recovery tier never runs
and merged triangles are NaN. With ``force=True`` the average is fitted as a bilayer
anyway; that must work for a merged bilayer, be refused for a genuine single peak, and
leave already-resolved surfaces (and the default) unchanged.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
from surface_morphometrics import _thickness_worker as tw  # noqa: E402

mt = pytest.importorskip("surface_morphometrics.measure_thickness")

X = np.linspace(-10, 10, 81)


def _bilayer(half, width):
    return (0.01 + tw._monogaussian(X, 0.02, -half, width)
            + tw._monogaussian(X, 0.02, half, width))


def _surface(profile, n=20, seed=0):
    # The pipeline stores raw tomogram density (membrane dark), i.e. the inverted profile.
    rng = np.random.default_rng(seed)
    rows = -profile[None, :] + rng.normal(0, 1e-5, (n, len(X)))
    return pd.DataFrame(rows, columns=[f"{v}" for v in X])


def test_merged_average_needs_force():
    merged = _surface(_bilayer(1.4, 1.3))
    params, forced = mt._global_bilayer_prior(merged, X)
    assert params is None and forced is False            # default: strict, no prior
    params, forced = mt._global_bilayer_prior(merged, X, force=True)
    assert forced is True and params is not None
    c1, w, c2, _ = params
    assert abs(0.5 * (c1 + c2)) < 0.3                     # centered
    assert tw.MIN_THICKNESS < c2 - c1 <= tw.MAX_THICKNESS


def test_resolved_average_is_not_forced():
    resolved = _surface(_bilayer(1.75, 1.0))
    p_default, f_default = mt._global_bilayer_prior(resolved, X)
    p_forced, f_forced = mt._global_bilayer_prior(resolved, X, force=True)
    assert f_default is False and f_forced is False
    assert np.allclose(p_default, p_forced)


def test_force_refuses_a_genuine_single_peak():
    single = _surface(0.01 + tw._monogaussian(X, 0.02, 0.0, 1.0))
    params, forced = mt._global_bilayer_prior(single, X, force=True)
    assert params is None and forced is False


def test_forced_prior_recovers_merged_triangles_and_flags_them():
    merged = _bilayer(1.4, 1.3)
    resolved = _bilayer(1.75, 1.0)
    profiles = np.vstack([merged, resolved])
    params, forced = mt._global_bilayer_prior(_surface(merged), X, force=True)
    assert forced
    n = len(profiles)
    tw.init_worker(-profiles, np.zeros((n, 1)), np.arange(n).reshape(n, 1), X,
                   use_xcorr=False, global_fit_params=params)
    try:
        res = tw.fit_triangle_chunk(list(range(n)))
    finally:
        tw.init_worker(None, None, None, X)
    (t_mer, _, _, r_mer), (t_res, _, _, r_res) = res
    assert np.isfinite(t_mer) and r_mer == 1     # recovered via the forced prior
    assert np.isfinite(t_res) and r_res == 0     # resolved on its own: not flagged

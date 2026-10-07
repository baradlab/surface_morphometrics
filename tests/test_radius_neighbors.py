"""radius_neighbors keeps the full averaging radius on finely meshed surfaces.

A plain cKDTree k=500 query keeps only the nearest 500 triangles, so on a dense mesh
the effective averaging radius silently shrinks below the configured one.
"""
import numpy as np
from scipy import spatial

from surface_morphometrics._thickness_worker import radius_neighbors


def _plane(spacing, half=40.0):
    g = np.arange(-half, half + 1e-9, spacing)
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])


def test_dense_mesh_neighborhood_spans_the_full_radius():
    xyz = _plane(0.5)                         # ~1800 points within 12 nm: overflows 500
    center = np.array([[0.0, 0.0, 0.0]])
    d_old, _ = spatial.cKDTree(xyz).query(center, k=500, distance_upper_bound=12)
    d_new, i_new = radius_neighbors(xyz, 12, query_xyz=center)
    assert d_old[0][np.isfinite(d_old[0])].max() < 8      # the old cap: ~7 nm, not 12
    valid = np.isfinite(d_new[0])
    assert d_new[0][valid].max() > 11                     # now reaches the full radius
    assert valid.sum() >= 250                             # still a large sample
    assert np.all(d_new[0][valid] <= 12)
    # indices refer to the original array and distances match them
    assert np.allclose(np.linalg.norm(xyz[i_new[0][valid]], axis=1), d_new[0][valid])


def test_sparse_mesh_is_identical_to_plain_query():
    xyz = _plane(2.0)                         # ~110 points within 12 nm: no overflow
    q = xyz[::50]
    d_old, i_old = spatial.cKDTree(xyz).query(q, k=500, distance_upper_bound=12)
    d_new, i_new = radius_neighbors(xyz, 12, query_xyz=q)
    assert np.array_equal(i_old, i_new) and np.array_equal(d_old, d_new)


def test_padding_matches_ckdtree_convention_and_is_reproducible():
    xyz = _plane(0.5)
    d1, i1 = radius_neighbors(xyz, 12, query_xyz=xyz[:5])
    d2, i2 = radius_neighbors(xyz, 12, query_xyz=xyz[:5])
    assert d1.shape == (5, 500) and np.array_equal(i1, i2)
    assert np.all(i1[~np.isfinite(d1)] == len(xyz))       # missing slots -> len(xyz)

"""Tests for xyz2ply.xyz_to_ply (the Screened Poisson meshing step; needs pymeshlab)."""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("pymeshlab", reason="pymeshlab is conda-only and not installed in CI")

from surface_morphometrics import xyz2ply  # noqa: E402  (after the skip guard)


RADIUS = 10.0


def _sphere_cloud(path, n=4000, radius=RADIUS, seed=0):
    """Write an evenly-covered spherical shell of points (nm) as an .xyz cloud."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    np.savetxt(path, directions * radius, fmt="%.4f", delimiter=" ")


def _read_ply(path):
    """Read a PLY mesh -> (vertices (N,3), triangle vertex indices (M,3))."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkPLYReader()
    reader.SetFileName(path)
    reader.Update()
    poly = reader.GetOutput()
    verts = vtk_to_numpy(poly.GetPoints().GetData())
    conn = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return verts, conn


def _mean_triangle_area(path):
    verts, conn = _read_ply(path)
    tri = verts[conn]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(np.mean(0.5 * np.linalg.norm(cross, axis=1)))


def test_xyz_to_ply_reconstructs_a_sphere():
    with tempfile.TemporaryDirectory() as tmp:
        xyz = os.path.join(tmp, "sphere.xyz")
        ply = os.path.join(tmp, "sphere.ply")
        _sphere_cloud(xyz)

        assert xyz2ply.xyz_to_ply(xyz, ply, pointweight=0.7, k_neighbors=100,
                                 deldist=1.5, depth=7, target_area=1.0) == 0
        assert os.path.exists(ply) and os.path.getsize(ply) > 0

        verts, conn = _read_ply(ply)
        assert len(conn) > 0
        # The reconstruction should sit on the input shell, not somewhere else.
        radii = np.linalg.norm(verts, axis=1)
        assert abs(float(np.mean(radii)) - RADIUS) < 1.0


def test_isotropic_remesh_targets_the_requested_triangle_area():
    # Guards the target_area -> target_edge_length conversion for equilateral
    # triangles (edge = sqrt(4 * area / sqrt(3))) in xyz_to_ply.
    with tempfile.TemporaryDirectory() as tmp:
        xyz = os.path.join(tmp, "sphere.xyz")
        _sphere_cloud(xyz)
        areas = {}
        for target in (1.0, 4.0):
            ply = os.path.join(tmp, f"sphere_{target}.ply")
            assert xyz2ply.xyz_to_ply(xyz, ply, pointweight=0.7, k_neighbors=100,
                                      deldist=1.5, depth=7, isotropic_remesh=True,
                                      target_area=target) == 0
            areas[target] = _mean_triangle_area(ply)
        # Remeshing is approximate, so allow a wide band, but each result should
        # track its own target and a 4x larger target must give larger triangles.
        for target, got in areas.items():
            assert 0.4 * target < got < 2.5 * target
        assert areas[4.0] > areas[1.0]


def test_no_reconstruction_returns_1():
    # Too few points for Screened Poisson to produce any faces: the function
    # reports failure (return 1) rather than writing a bogus mesh or raising.
    with tempfile.TemporaryDirectory() as tmp:
        xyz = os.path.join(tmp, "degenerate.xyz")
        ply = os.path.join(tmp, "degenerate.ply")
        np.savetxt(xyz, np.zeros((4, 3)), fmt="%.4f", delimiter=" ")
        assert xyz2ply.xyz_to_ply(xyz, ply, k_neighbors=3, depth=4) == 1
        assert not os.path.exists(ply)


def test_cli_forwards_every_option_to_xyz_to_ply(monkeypatch):
    # A click option that is declared but never forwarded is silently inert (this
    # is exactly how --ultrafine rotted). Assert every option reaches xyz_to_ply.
    from click.testing import CliRunner

    captured = {}
    monkeypatch.setattr(xyz2ply, "xyz_to_ply",
                        lambda *args, **kwargs: captured.update(kwargs) or 0)

    declared = {p.name for p in xyz2ply.xyz_to_ply_from_CLI.params
                if p.name not in ("xyzfile", "plyfile")}
    result = CliRunner().invoke(xyz2ply.xyz_to_ply_from_CLI, ["in.xyz", "out.ply"])
    assert result.exit_code == 0, result.output
    assert declared == set(captured), f"not forwarded: {declared - set(captured)}"

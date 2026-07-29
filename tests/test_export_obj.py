"""Tests for the OBJ/MTL/colormap writer (numpy + matplotlib; no vtk needed)."""
import os
import tempfile

import numpy as np

from surface_morphometrics import export_obj as eo


def _mesh():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2]])
    return pts, faces


def test_write_obj_mtl_creates_files_and_uvs():
    pts, faces = _mesh()
    vals = np.array([3.0, 4.0, 5.0])
    out = os.path.join(tempfile.mkdtemp(), "s_thickness")
    lo, hi = eo.write_obj_mtl(out, pts, faces, vals, "thickness", "viridis",
                              vmin=3.0, vmax=5.0, nan_color="lightgrey")
    for ext in (".obj", ".mtl", ".png"):
        assert os.path.exists(out + ext)
    obj = open(out + ".obj").read()
    assert "mtllib s_thickness.mtl" in obj and "usemtl quant_thickness" in obj
    assert obj.count("\nv ") == len(pts)
    assert obj.count("\nf ") == len(faces)
    assert "map_Kd s_thickness.png" in open(out + ".mtl").read()


def test_write_obj_mtl_nan_swatch():
    import matplotlib.image as mpimg
    from matplotlib.colors import to_rgb
    pts, faces = _mesh()
    vals = np.array([3.0, np.nan, 5.0])
    out = os.path.join(tempfile.mkdtemp(), "s_thickness")
    eo.write_obj_mtl(out, pts, faces, vals, "thickness", "viridis",
                     vmin=3.0, vmax=5.0, nan_color="lightgrey")
    img = mpimg.imread(out + ".png")
    assert img.shape[1] == eo._RAMP_W + eo._NAN_W          # ramp + swatch
    assert np.allclose(img[0, eo._RAMP_W + 4, :3], to_rgb("lightgrey"), atol=0.02)
    # the NaN face's u samples the swatch
    u = [float(l.split()[1]) for l in open(out + ".obj") if l.startswith("vt ")]
    assert abs(u[1] - (eo._RAMP_W + eo._NAN_W / 2.0) / (eo._RAMP_W + eo._NAN_W)) < 1e-6


def test_coordinate_scale_units():
    # nm surface: default Angstrom output, or nm passthrough
    assert eo.coordinate_scale(False, scale_to_angstroms=True) == 10.0
    assert eo.coordinate_scale(False, scale_to_angstroms=False) == 1.0
    # Angstrom surface: passthrough, or back to nm
    assert eo.coordinate_scale(True, scale_to_angstroms=True) == 1.0
    assert eo.coordinate_scale(True, scale_to_angstroms=False) == 0.1


def test_coordinate_scale_voxels():
    # 5 A/px: an nm-scale surface doubles, an Angstrom-scale one is divided by 5
    assert eo.coordinate_scale(False, scale_to_voxels=5.0) == 2.0
    assert eo.coordinate_scale(True, scale_to_voxels=5.0) == 0.2
    # voxel space wins regardless of the (defaulted) Angstrom flag
    assert eo.coordinate_scale(False, scale_to_angstroms=True, scale_to_voxels=10.0) == 1.0


def test_coordinate_scale_rejects_nonpositive_voxel_size():
    import click
    import pytest
    with pytest.raises(click.UsageError):
        eo.coordinate_scale(False, scale_to_voxels=0.0)
    with pytest.raises(click.UsageError):
        eo.coordinate_scale(False, scale_to_voxels=-5.0)


def test_scale_to_voxels_conflicts_with_scale_to_angstroms():
    from click.testing import CliRunner
    with tempfile.TemporaryDirectory() as d:
        cfg = os.path.join(d, "config.yml")
        with open(cfg, "w") as f:
            f.write(f"work_dir: {d}\n")
        result = CliRunner().invoke(
            eo.export_obj_cli,
            [cfg, "--feature", "thickness", "--scale_to_voxels", "5.0",
             "--scale_to_angstroms", "true"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output


def test_write_obj_mtl_nan_disabled():
    import matplotlib.image as mpimg
    pts, faces = _mesh()
    vals = np.array([3.0, np.nan, 5.0])
    out = os.path.join(tempfile.mkdtemp(), "s_thickness")
    eo.write_obj_mtl(out, pts, faces, vals, "thickness", "viridis",
                     vmin=3.0, vmax=5.0, nan_color=None)
    img = mpimg.imread(out + ".png")
    assert img.shape[1] == eo._RAMP_W                       # no swatch
    u = [float(l.split()[1]) for l in open(out + ".obj") if l.startswith("vt ")]
    assert abs(u[1] - 0.5 / eo._RAMP_W) < 1e-6              # NaN -> low end

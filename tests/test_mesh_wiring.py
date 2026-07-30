"""Tests that every surface_generation setting actually reaches the meshing step.

`ultrafine` was documented in config_template.yml and present in DEFAULTS for
several releases while being silently ignored, because nothing carried it from the
config to the meshing subprocess. These tests close that gap: no pymeshlab needed
(the subprocess is stubbed), so they run in CI.
"""
import os
import tempfile

from surface_morphometrics import segmentation_to_meshes as s2m
from surface_morphometrics.config_utils import DEFAULTS


# Every surface_generation key, and which consumer is responsible for it. A new
# setting must be added here *and* wired up, or test_no_setting_is_inert fails.
_XYZ2PLY_FLAGS = {
    "point_weight": "--pointweight",
    "simplify": "--simplify",
    "simplify_max_triangles": "--num_faces",
    "neighbor_count": "--k_neighbors",
    "extrapolation_distance": "--deldist",
    "smoothing_iterations": "--smooth_iter",
    "octree_depth": "--depth",
    "isotropic_remesh": "--isotropic_remesh",
    "target_area": "--target_area",
}
# Consumed earlier in the pipeline, by mrc2xyz.mrc_to_xyz rather than by xyz2ply.
_MRC2XYZ_KEYS = {"angstroms"}


def test_no_setting_is_inert():
    """Every documented surface_generation key has a consumer."""
    documented = set(DEFAULTS["surface_generation"])
    consumed = set(_XYZ2PLY_FLAGS) | _MRC2XYZ_KEYS
    assert documented == consumed, (
        f"unconsumed (documented but inert): {documented - consumed}; "
        f"unknown to the config: {consumed - documented}"
    )


def test_run_xyz_to_ply_passes_every_meshing_setting(monkeypatch):
    """Each xyz2ply setting reaches the subprocess command line with its value."""
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        # cmd is [python, -m, module, xyz_file, ply_file, ...]; run_xyz_to_ply
        # reports success by the ply existing and being non-empty.
        with open(cmd[4], "w") as handle:
            handle.write("ply\n")

        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr(s2m.subprocess, "run", fake_run)

    # Non-default values throughout, so a flag wired to the wrong key is caught.
    surface_config = dict(DEFAULTS["surface_generation"])
    surface_config.update({
        "point_weight": 0.55, "simplify": True, "simplify_max_triangles": 12345,
        "neighbor_count": 77, "extrapolation_distance": 2.25,
        "smoothing_iterations": 3, "octree_depth": 11,
        "isotropic_remesh": False, "target_area": 2.5,
    })

    with tempfile.TemporaryDirectory() as tmp:
        ply = os.path.join(tmp, "out.ply")
        assert s2m.run_xyz_to_ply(os.path.join(tmp, "in.xyz"), ply, surface_config) == 0

    cmd = captured["cmd"]
    assert cmd[1:3] == ["-m", "surface_morphometrics.xyz2ply"]
    for key, flag in _XYZ2PLY_FLAGS.items():
        assert flag in cmd, f"{key} is not passed to the meshing subprocess ({flag} missing)"
        assert cmd[cmd.index(flag) + 1] == str(surface_config[key]), (
            f"{flag} does not carry the configured {key}"
        )


def test_run_xyz_to_ply_reports_failure_when_no_ply_is_written(monkeypatch):
    # A meshing subprocess that produces nothing must be reported as a failure,
    # so make_meshes skips the surface instead of continuing with a missing ply.
    monkeypatch.setattr(s2m.subprocess, "run", lambda *args, **kwargs: None)
    with tempfile.TemporaryDirectory() as tmp:
        ply = os.path.join(tmp, "missing.ply")
        assert s2m.run_xyz_to_ply(os.path.join(tmp, "in.xyz"), ply,
                                  DEFAULTS["surface_generation"]) == 1

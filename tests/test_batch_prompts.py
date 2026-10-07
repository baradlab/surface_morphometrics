"""The batch "are you sure?" prompt lives on refine_mesh, not distances_orientations."""
import pytest
import yaml
from click.testing import CliRunner

# Both commands import pycurv/graph-tool at module load (conda-only; absent in pip CI).
pytest.importorskip("pycurv")
from surface_morphometrics.measure_distances_orientations import distances_orientations_cli  # noqa: E402


def _config(tmp_path):
    seg, work, tomo = tmp_path / "seg", tmp_path / "work", tmp_path / "tomo"
    for d in (seg, work, tomo):
        d.mkdir()
    cfg = {"seg_dir": str(seg) + "/", "work_dir": str(work) + "/",
           "tomo_dir": str(tomo) + "/", "segmentation_values": {"OMM": 1}}
    path = tmp_path / "config.yml"
    yaml.safe_dump(cfg, open(path, "w"))
    return str(path)


def test_distances_orientations_runs_all_files_without_prompting(tmp_path):
    r = CliRunner().invoke(distances_orientations_cli, [_config(tmp_path)], input="")
    assert r.exit_code == 0, r.output
    assert "Continue?" not in r.output


def test_distances_orientations_force_is_a_hidden_deprecated_noop(tmp_path):
    cfg = _config(tmp_path)
    r = CliRunner().invoke(distances_orientations_cli, [cfg, "-f"])
    assert r.exit_code == 0, r.output
    assert "no longer needed" in r.output
    help_text = CliRunner().invoke(distances_orientations_cli, ["--help"]).output
    assert "--force" not in help_text


def test_refine_mesh_batch_run_asks_first(tmp_path):
    from surface_morphometrics.refine_mesh import refine_mesh_cli
    r = CliRunner().invoke(refine_mesh_cli, [_config(tmp_path)], input="n\n")
    assert "Continue?" in r.output
    assert r.exit_code == 1

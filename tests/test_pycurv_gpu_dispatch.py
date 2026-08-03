"""Unit tests for pycurv-gpu dispatch (no CUDA / no real mesh required)."""
import sys
import textwrap
from types import ModuleType
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

# Stub heavy optional deps only when missing so collection works outside the
# full morphometrics env, without shadowing a real graph-tool install.
for _name in ("graph_tool", "pathos", "pathos.pools", "pycurv", "pyto"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = MagicMock()

from surface_morphometrics import curvature  # noqa: E402
from surface_morphometrics import run_pycurv as run_pycurv_mod  # noqa: E402


def test_require_gpu_backend_fails_without_accelerator():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_api = ModuleType("core.api")
    fake_api.run_pipeline = MagicMock()
    fake_core = ModuleType("core")
    with patch.dict(sys.modules, {
             "torch": fake_torch,
             "core": fake_core,
             "core.api": fake_api,
         }):
        try:
            curvature._require_gpu_backend(None)
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "CUDA is not available" in str(exc)
            assert "will not fall back to CPU torch" in str(exc)


def test_require_gpu_backend_rejects_explicit_cpu():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_api = ModuleType("core.api")
    fake_api.run_pipeline = MagicMock()
    fake_core = ModuleType("core")
    with patch.dict(sys.modules, {
             "torch": fake_torch,
             "core": fake_core,
             "core.api": fake_api,
         }):
        try:
            curvature._require_gpu_backend("cpu")
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "gpu_device='cpu' is not supported" in str(exc)


def test_run_pycurv_dispatches_to_gpu():
    with patch.object(curvature, "run_pycurv_gpu") as mock_gpu:
        curvature.run_pycurv(
            "mesh.surface.vtp", "/tmp/work/",
            use_gpu=True, gpu_device="cuda", gpu_batch_size=512,
        )
    mock_gpu.assert_called_once()
    kwargs = mock_gpu.call_args.kwargs
    assert kwargs["device"] == "cuda"
    assert kwargs["batch_size"] == 512


def test_run_pycurv_gpu_calls_pipeline_with_write_gt():
    mock_pipeline = MagicMock(return_value={})
    fake_api = ModuleType("core.api")
    fake_api.run_pipeline = mock_pipeline
    fake_core = ModuleType("core")
    with patch.object(curvature, "_require_gpu_backend", return_value="cuda"), \
         patch.dict(sys.modules, {"core": fake_core, "core.api": fake_api}):
        curvature.run_pycurv_gpu(
            "mesh.surface.vtp", "/tmp/work/",
            scale=1.0, radius_hit=9, min_component=30, exclude_borders=1,
            device="cuda", batch_size=512,
        )

    mock_pipeline.assert_called_once()
    kwargs = mock_pipeline.call_args.kwargs
    assert kwargs["write_gt"] is True
    assert kwargs["write_vtp"] is True
    assert kwargs["radius_hit"] == 9
    assert kwargs["pixel_size"] == 1.0
    assert kwargs["device"] == "cuda"
    assert kwargs["batch_size"] == 512
    assert kwargs["min_component"] == 30
    assert kwargs["exclude_borders"] == 1


def test_run_pycurv_cpu_path_calls_new_workflow():
    with patch.object(curvature, "new_workflow") as mock_nw, \
         patch.object(curvature, "extract_curvatures_after_new_workflow") as mock_ex, \
         patch.object(curvature, "run_pycurv_gpu") as mock_gpu:
        curvature.run_pycurv(
            "mesh.surface.vtp", "/tmp/work/",
            scale=1.0, radius_hit=9, min_component=30, exclude_borders=1,
            use_gpu=False,
        )
    mock_gpu.assert_not_called()
    mock_nw.assert_called_once()
    mock_ex.assert_called_once()


def test_cli_passes_gpu_flag(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "mesh.surface.vtp").write_text("placeholder")
    cfg = tmp_path / "config.yml"
    cfg.write_text(textwrap.dedent(f"""\
        work_dir: "{work}/"
        cores: 1
        curvature_measurements:
          radius_hit: 9
          min_component: 30
          exclude_borders: 1
          use_gpu: false
    """))

    with patch.object(curvature, "_require_gpu_backend", return_value="cuda") as mock_req, \
         patch.object(curvature, "run_pycurv") as mock_run:
        runner = CliRunner()
        result = runner.invoke(
            run_pycurv_mod.run_pycurv_cli,
            [str(cfg), "mesh.surface.vtp", "--gpu", "-f"],
        )
    assert result.exit_code == 0, result.output
    mock_req.assert_called_once()
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["use_gpu"] is True


def test_cli_reads_use_gpu_from_config(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "mesh.surface.vtp").write_text("placeholder")
    cfg = tmp_path / "config.yml"
    cfg.write_text(textwrap.dedent(f"""\
        work_dir: "{work}/"
        cores: 1
        curvature_measurements:
          radius_hit: 9
          min_component: 30
          exclude_borders: 1
          use_gpu: true
          gpu_device: cuda
          gpu_batch_size: 256
    """))

    with patch.object(curvature, "_require_gpu_backend", return_value="cuda"), \
         patch.object(curvature, "run_pycurv") as mock_run:
        runner = CliRunner()
        result = runner.invoke(
            run_pycurv_mod.run_pycurv_cli,
            [str(cfg), "mesh.surface.vtp", "-f"],
        )
    assert result.exit_code == 0, result.output
    kwargs = mock_run.call_args.kwargs
    assert kwargs["use_gpu"] is True
    assert kwargs["gpu_device"] == "cuda"
    assert kwargs["gpu_batch_size"] == 256

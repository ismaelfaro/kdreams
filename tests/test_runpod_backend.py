"""Tests for kdream.backends.runpod."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kdream.core.recipe import (
    BackendSpecs,
    EntrypointSpec,
    InputSpec,
    LocalBackendSpec,
    ModelDescriptor,
    OutputSpec,
    Recipe,
    RecipeMetadata,
    RecipeSource,
)
from kdream.exceptions import BackendError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recipe(
    name: str = "test-model",
    models: list | None = None,
    inputs: dict | None = None,
    outputs: list | None = None,
    entrypoint_script: str = "run.py",
    entrypoint_type: str = "python",
    skip_package_install: bool = False,
) -> Recipe:
    if models is None:
        models = [
            ModelDescriptor(
                name="test-weights",
                source="huggingface",
                id="org/test-model",
                destination="models/test-model",
            )
        ]
    if inputs is None:
        inputs = {
            "prompt": InputSpec(type="string", required=True),
            "steps": InputSpec(type="integer", required=False, default=20),
        }
    if outputs is None:
        outputs = [OutputSpec(name="image", type="file", path="outputs/{timestamp}.png")]

    return Recipe(
        metadata=RecipeMetadata(name=name, description="Test recipe"),
        source=RecipeSource(
            repo="https://github.com/example/test-model",
            ref="main",
            skip_package_install=skip_package_install,
        ),
        models=models,
        entrypoint=EntrypointSpec(script=entrypoint_script, type=entrypoint_type),
        inputs=inputs,
        outputs=outputs,
        backends=BackendSpecs(local=LocalBackendSpec(requires_gpu=True, min_vram_gb=8)),
    )


# ---------------------------------------------------------------------------
# TestDockerfileBuilder
# ---------------------------------------------------------------------------

class TestDockerfileBuilder:
    def _make_builder(self):
        from kdream.backends.runpod import DockerfileBuilder
        return DockerfileBuilder()

    def test_build_returns_string(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert isinstance(result, str)
        assert len(result) > 100

    def test_from_base_image(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "FROM runpod/pytorch" in result

    def test_git_clone_contains_repo(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "git clone" in result
        assert "https://github.com/example/test-model" in result

    def test_git_clone_contains_name(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe(name="my-model"))
        assert "/workspace/my-model" in result

    def test_uv_pip_install(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "uv pip install" in result

    def test_cmd_runs_handler(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert 'CMD ["python", "/workspace/handler.py"]' in result

    def test_hf_home_env_var(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "HF_HOME=/runpod-volume/.cache/huggingface" in result

    def test_models_dir_env_var(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "MODELS_DIR=/runpod-volume/models" in result

    def test_copy_handler(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "COPY handler.py /workspace/handler.py" in result

    def test_skip_package_install_skips_uv(self):
        builder = self._make_builder()
        recipe = _make_recipe(skip_package_install=True)
        result = builder.build(recipe, skip_package_install=True)
        # requirements.txt install should be absent
        assert "requirements.txt" not in result

    def test_install_extras_included(self):
        builder = self._make_builder()
        recipe = _make_recipe()
        recipe.source.install_extras = ["cuda"]
        result = builder.build(recipe)
        assert "[cuda]" in result

    def test_ref_used_in_clone(self):
        builder = self._make_builder()
        recipe = _make_recipe()
        recipe.source.ref = "v2.0"
        result = builder.build(recipe)
        assert "--branch v2.0" in result

    def test_no_models_does_not_crash(self):
        builder = self._make_builder()
        recipe = _make_recipe(models=[])
        result = builder.build(recipe)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestHandlerBuilder
# ---------------------------------------------------------------------------

class TestHandlerBuilder:
    def _make_builder(self):
        from kdream.backends.runpod import HandlerBuilder
        return HandlerBuilder()

    def test_build_returns_string(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert isinstance(result, str)

    def test_imports_runpod(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "import runpod" in result

    def test_serverless_start(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "runpod.serverless.start" in result

    def test_handler_function_defined(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "def handler(job):" in result

    def test_inputs_extracted_from_job(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert 'job.get("input"' in result or "job.get('input'" in result

    def test_ensure_models_called(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "_ensure_models()" in result

    def test_def_ensure_models(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "def _ensure_models():" in result

    def test_hf_snapshot_download_for_hf_model(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert "snapshot_download" in result

    def test_url_download_uses_urlretrieve(self):
        builder = self._make_builder()
        recipe = _make_recipe(models=[
            ModelDescriptor(
                name="url-model",
                source="url",
                id="https://example.com/model.bin",
                destination="models/url-model",
            )
        ])
        result = builder.build(recipe)
        assert "urlretrieve" in result

    def test_civitai_uses_token(self):
        builder = self._make_builder()
        recipe = _make_recipe(models=[
            ModelDescriptor(
                name="civitai-model",
                source="civitai",
                id="12345",
                destination="models/civitai",
            )
        ])
        result = builder.build(recipe)
        assert "CIVITAI_TOKEN" in result

    def test_subprocess_run_uses_script(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe(entrypoint_script="inference.py"))
        assert "inference.py" in result

    def test_error_returned_on_nonzero_exit(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        assert '"error"' in result or "'error'" in result
        assert "returncode" in result

    def test_no_models_has_pass(self):
        builder = self._make_builder()
        recipe = _make_recipe(models=[])
        result = builder.build(recipe)
        assert "def _ensure_models():" in result
        assert "pass" in result

    def test_defaults_applied_in_handler(self):
        builder = self._make_builder()
        result = builder.build(_make_recipe())
        # steps has default=20
        assert "20" in result


# ---------------------------------------------------------------------------
# TestRunPodBackendInstall
# ---------------------------------------------------------------------------

class TestRunPodBackendInstall:
    def _make_backend(self, **kwargs):
        from kdream.backends.runpod import RunPodBackend
        return RunPodBackend(**kwargs)

    def test_install_creates_dockerfile(self, tmp_path):
        backend = self._make_backend()
        backend.install(_make_recipe(), cache_dir=tmp_path)
        assert (tmp_path / "test-model" / "runpod" / "Dockerfile").exists()

    def test_install_creates_handler(self, tmp_path):
        backend = self._make_backend()
        backend.install(_make_recipe(), cache_dir=tmp_path)
        assert (tmp_path / "test-model" / "runpod" / "handler.py").exists()

    def test_install_creates_config(self, tmp_path):
        backend = self._make_backend()
        backend.install(_make_recipe(), cache_dir=tmp_path)
        assert (tmp_path / "test-model" / "runpod" / "config.json").exists()

    def test_install_returns_package_info(self, tmp_path):
        from kdream.core.runner import PackageInfo
        backend = self._make_backend()
        pkg = backend.install(_make_recipe(), cache_dir=tmp_path)
        assert isinstance(pkg, PackageInfo)
        assert pkg.ready is True

    def test_install_package_info_path(self, tmp_path):
        backend = self._make_backend()
        pkg = backend.install(_make_recipe(), cache_dir=tmp_path)
        assert pkg.path == tmp_path / "test-model" / "runpod"

    def test_install_idempotent_without_force(self, tmp_path):
        backend = self._make_backend()
        pkg1 = backend.install(_make_recipe(), cache_dir=tmp_path)
        # Modify Dockerfile to check it's NOT regenerated
        dockerfile = tmp_path / "test-model" / "runpod" / "Dockerfile"
        dockerfile.write_text("MODIFIED", encoding="utf-8")
        pkg2 = backend.install(_make_recipe(), cache_dir=tmp_path)
        assert dockerfile.read_text() == "MODIFIED"

    def test_install_force_reinstall_regenerates(self, tmp_path):
        backend = self._make_backend()
        backend.install(_make_recipe(), cache_dir=tmp_path)
        dockerfile = tmp_path / "test-model" / "runpod" / "Dockerfile"
        dockerfile.write_text("MODIFIED", encoding="utf-8")
        backend.install(_make_recipe(), cache_dir=tmp_path, force_reinstall=True)
        assert dockerfile.read_text() != "MODIFIED"

    def test_is_installed_false_before_install(self, tmp_path):
        backend = self._make_backend()
        assert backend.is_installed("test-model", tmp_path) is False

    def test_is_installed_true_after_install(self, tmp_path):
        backend = self._make_backend()
        backend.install(_make_recipe(), cache_dir=tmp_path)
        assert backend.is_installed("test-model", tmp_path) is True

    def test_config_json_has_gpu_type(self, tmp_path):
        backend = self._make_backend(gpu_type="NVIDIA A100 80GB PCIe")
        backend.install(_make_recipe(), cache_dir=tmp_path)
        config = json.loads(
            (tmp_path / "test-model" / "runpod" / "config.json").read_text()
        )
        assert config["gpu_type"] == "NVIDIA A100 80GB PCIe"

    def test_config_json_stores_endpoint_id(self, tmp_path):
        backend = self._make_backend(endpoint_id="ep-test123")
        backend.install(_make_recipe(), cache_dir=tmp_path)
        config = json.loads(
            (tmp_path / "test-model" / "runpod" / "config.json").read_text()
        )
        assert config["endpoint_id"] == "ep-test123"


# ---------------------------------------------------------------------------
# TestRunPodBackendRun
# ---------------------------------------------------------------------------

def _make_package(tmp_path: Path, endpoint_id: str = "") -> MagicMock:
    """Return a mock PackageInfo pointing to a temp runpod dir."""
    runpod_dir = tmp_path / "test-model" / "runpod"
    runpod_dir.mkdir(parents=True, exist_ok=True)
    config = {"recipe_name": "test-model", "gpu_type": "...",
               "docker_image": "...", "endpoint_id": endpoint_id}
    (runpod_dir / "config.json").write_text(json.dumps(config))
    pkg = MagicMock()
    pkg.path = runpod_dir
    pkg.recipe_name = "test-model"
    return pkg


class TestRunPodBackendRun:
    def _make_backend(self, **kwargs):
        from kdream.backends.runpod import RunPodBackend
        return RunPodBackend(**kwargs)

    def test_run_raises_without_endpoint_id(self, tmp_path):
        backend = self._make_backend(api_key="key123")
        pkg = _make_package(tmp_path, endpoint_id="")
        with pytest.raises(BackendError, match="endpoint ID"):
            backend.run(pkg, {"prompt": "test"})

    def test_run_raises_without_api_key(self, tmp_path):
        backend = self._make_backend()
        pkg = _make_package(tmp_path, endpoint_id="ep-test123")
        backend.endpoint_id = "ep-test123"
        with pytest.raises(BackendError, match="API key"):
            backend.run(pkg, {"prompt": "test"})

    def test_run_calls_endpoint_run_sync(self, tmp_path):
        backend = self._make_backend(api_key="key123", endpoint_id="ep-abc")
        pkg = _make_package(tmp_path, endpoint_id="ep-abc")

        mock_runpod = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.run_sync.return_value = {"output": "done", "outputs": {}}
        mock_runpod.Endpoint.return_value = mock_endpoint

        with patch.dict("sys.modules", {"runpod": mock_runpod}):
            result = backend.run(pkg, {"prompt": "test"})

        mock_runpod.Endpoint.assert_called_once_with("ep-abc")
        mock_endpoint.run_sync.assert_called_once_with({"prompt": "test"}, timeout=300)
        assert result["output"] == "done"

    def test_run_sets_api_key(self, tmp_path):
        backend = self._make_backend(api_key="my-api-key", endpoint_id="ep-abc")
        pkg = _make_package(tmp_path, endpoint_id="ep-abc")

        mock_runpod = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.run_sync.return_value = {"output": "", "outputs": {}}
        mock_runpod.Endpoint.return_value = mock_endpoint

        with patch.dict("sys.modules", {"runpod": mock_runpod}):
            backend.run(pkg, {})

        assert mock_runpod.api_key == "my-api-key"

    def test_run_raises_on_error_in_response(self, tmp_path):
        backend = self._make_backend(api_key="key123", endpoint_id="ep-abc")
        pkg = _make_package(tmp_path, endpoint_id="ep-abc")

        mock_runpod = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.run_sync.return_value = {"error": "CUDA OOM"}
        mock_runpod.Endpoint.return_value = mock_endpoint

        with patch.dict("sys.modules", {"runpod": mock_runpod}):
            with pytest.raises(BackendError, match="CUDA OOM"):
                backend.run(pkg, {"prompt": "test"})

    def test_run_handles_timeout(self, tmp_path):
        backend = self._make_backend(api_key="key123", endpoint_id="ep-abc")
        pkg = _make_package(tmp_path, endpoint_id="ep-abc")

        mock_runpod = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.run_sync.side_effect = TimeoutError("timeout")
        mock_runpod.Endpoint.return_value = mock_endpoint

        with patch.dict("sys.modules", {"runpod": mock_runpod}):
            with pytest.raises(BackendError, match="timed out"):
                backend.run(pkg, {"prompt": "test"})

    def test_run_reads_endpoint_id_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep-from-env")
        monkeypatch.setenv("RUNPOD_API_KEY", "key-from-env")

        from kdream.backends.runpod import RunPodBackend
        backend = RunPodBackend()
        assert backend.endpoint_id == "ep-from-env"
        assert backend.api_key == "key-from-env"

    def test_run_raises_if_runpod_not_installed(self, tmp_path):
        backend = self._make_backend(api_key="key123", endpoint_id="ep-abc")
        pkg = _make_package(tmp_path, endpoint_id="ep-abc")

        with patch.dict("sys.modules", {"runpod": None}):
            with pytest.raises((BackendError, ImportError)):
                backend.run(pkg, {"prompt": "test"})


# ---------------------------------------------------------------------------
# TestGenerateArtifacts
# ---------------------------------------------------------------------------

class TestGenerateArtifacts:
    def _make_backend(self, **kwargs):
        from kdream.backends.runpod import RunPodBackend
        return RunPodBackend(**kwargs)

    def test_generate_artifacts_creates_dockerfile(self, tmp_path):
        backend = self._make_backend()
        dest = backend.generate_artifacts(_make_recipe(), output_dir=tmp_path)
        assert (tmp_path / "Dockerfile").exists()

    def test_generate_artifacts_creates_handler(self, tmp_path):
        backend = self._make_backend()
        backend.generate_artifacts(_make_recipe(), output_dir=tmp_path)
        assert (tmp_path / "handler.py").exists()

    def test_generate_artifacts_creates_config(self, tmp_path):
        backend = self._make_backend()
        backend.generate_artifacts(_make_recipe(), output_dir=tmp_path)
        assert (tmp_path / "config.json").exists()

    def test_generate_artifacts_returns_path(self, tmp_path):
        backend = self._make_backend()
        result = backend.generate_artifacts(_make_recipe(), output_dir=tmp_path)
        assert result == tmp_path

    def test_generate_artifacts_defaults_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        backend = self._make_backend()
        result = backend.generate_artifacts(_make_recipe())
        assert (result / "Dockerfile").exists()


# ---------------------------------------------------------------------------
# TestRunPodCLI
# ---------------------------------------------------------------------------

class TestRunPodCLI:
    def test_runpod_help(self):
        from click.testing import CliRunner
        from kdream.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["runpod", "--help"])
        assert result.exit_code == 0
        assert "runpod" in result.output.lower()

    def test_runpod_generate_help(self):
        from click.testing import CliRunner
        from kdream.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["runpod", "generate", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output

    def test_runpod_generate_creates_artifacts(self, tmp_path):
        from click.testing import CliRunner
        from kdream.cli import cli
        from kdream.core.recipe import parse_yaml_recipe

        recipe_yaml = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: test-model
  version: 1.0.0
  description: Test
  tags: [test]
  license: MIT
  author: test
source:
  repo: https://github.com/example/test
  ref: main
models: []
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
    required: true
outputs: []
backends:
  local:
    requires_gpu: false
    min_vram_gb: 0
    tested_on: []
"""
        recipe_file = tmp_path / "test-model.yaml"
        recipe_file.write_text(recipe_yaml)

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["runpod", "generate", str(recipe_file),
                                         "--output-dir", str(tmp_path / "out")])
            assert result.exit_code == 0
            assert (tmp_path / "out" / "Dockerfile").exists()

    def test_runpod_pods_help(self):
        from click.testing import CliRunner
        from kdream.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["runpod", "pods", "--help"])
        assert result.exit_code == 0

    def test_runpod_terminate_help(self):
        from click.testing import CliRunner
        from kdream.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["runpod", "terminate", "--help"])
        assert result.exit_code == 0

    def test_runpod_generate_invalid_recipe_shows_error(self, tmp_path):
        from click.testing import CliRunner
        from kdream.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["runpod", "generate", "nonexistent-recipe-xyz"])
        assert result.exit_code != 0 or "Error" in result.output or "error" in result.output.lower()

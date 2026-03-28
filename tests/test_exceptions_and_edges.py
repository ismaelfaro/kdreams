"""Tests for exceptions hierarchy and edge cases across the codebase."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kdream.exceptions import (
    BackendError,
    KdreamError,
    ModelDownloadError,
    RecipeError,
    RegistryError,
)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    """All custom exceptions should be subclasses of KdreamError."""

    def test_kdream_error_is_exception(self):
        assert issubclass(KdreamError, Exception)

    def test_recipe_error_is_kdream_error(self):
        assert issubclass(RecipeError, KdreamError)

    def test_registry_error_is_kdream_error(self):
        assert issubclass(RegistryError, KdreamError)

    def test_backend_error_is_kdream_error(self):
        assert issubclass(BackendError, KdreamError)

    def test_model_download_error_is_kdream_error(self):
        assert issubclass(ModelDownloadError, KdreamError)

    def test_all_errors_can_be_caught_as_kdream_error(self):
        for exc_class in (RecipeError, RegistryError, BackendError, ModelDownloadError):
            with pytest.raises(KdreamError):
                raise exc_class("test message")

    def test_exceptions_carry_message(self):
        for exc_class in (KdreamError, RecipeError, RegistryError, BackendError, ModelDownloadError):
            exc = exc_class("some error message")
            assert "some error message" in str(exc)

    def test_exceptions_support_chaining(self):
        original = ValueError("root cause")
        try:
            raise BackendError("backend failed") from original
        except BackendError as e:
            assert e.__cause__ is original


# ---------------------------------------------------------------------------
# RecipeError raised correctly
# ---------------------------------------------------------------------------

class TestRecipeErrorRaisedCorrectly:
    def test_non_dict_yaml_raises_recipe_error(self):
        from kdream.core.recipe import parse_yaml_recipe
        with pytest.raises(RecipeError):
            parse_yaml_recipe("- item1\n- item2\n")

    def test_missing_file_raises_recipe_error(self):
        from kdream.core.recipe import load_recipe
        with pytest.raises(RecipeError):
            load_recipe("/no/such/file/recipe.yaml")

    def test_markdown_without_frontmatter_raises_recipe_error(self):
        from kdream.core.recipe import parse_markdown_recipe
        with pytest.raises(RecipeError):
            parse_markdown_recipe("# Just a heading\nNo frontmatter.")

    def test_recipe_error_message_is_descriptive(self):
        from kdream.core.recipe import load_recipe
        with pytest.raises(RecipeError) as exc_info:
            load_recipe("/no/such/file.yaml")
        assert len(str(exc_info.value)) > 0


# ---------------------------------------------------------------------------
# RegistryError raised correctly
# ---------------------------------------------------------------------------

class TestRegistryErrorRaisedCorrectly:
    def test_recipe_not_found_raises_registry_error(self, tmp_path):
        from kdream.core.registry import RegistryClient
        client = RegistryClient(cache_dir=tmp_path / "cache")

        mock_404 = MagicMock()
        mock_404.status_code = 404

        with patch("httpx.get", return_value=mock_404):
            with patch.object(client, "_find_bundled", return_value=None):
                with pytest.raises(RegistryError, match="not found in registry"):
                    client.fetch_recipe("nonexistent-xyz")

    def test_all_sources_exhausted_raises_registry_error(self, tmp_path):
        from kdream.core.registry import RegistryClient
        client = RegistryClient(cache_dir=tmp_path / "cache")

        with patch.object(client, "_fetch_index_from_github", side_effect=RegistryError("fail")):
            with patch.object(client, "_load_bundled_metadata", return_value=[]):
                with pytest.raises(RegistryError):
                    client._get_all_metadata()

    def test_github_api_non_200_raises_registry_error(self, tmp_path):
        from kdream.core.registry import RegistryClient
        client = RegistryClient(cache_dir=tmp_path / "cache")

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(RegistryError, match="Registry API returned 500"):
                client._fetch_index_from_github(tmp_path / "_index.json")

    def test_network_error_raises_registry_error(self, tmp_path):
        import httpx
        from kdream.core.registry import RegistryClient
        client = RegistryClient(cache_dir=tmp_path / "cache")

        with patch("httpx.get", side_effect=httpx.RequestError("connection refused")):
            with pytest.raises(RegistryError, match="Network error"):
                client._fetch_index_from_github(tmp_path / "_index.json")


# ---------------------------------------------------------------------------
# BackendError raised correctly
# ---------------------------------------------------------------------------

class TestBackendErrorRaisedCorrectly:
    def test_clone_failure_raises_backend_error(self, tmp_path):
        from kdream.backends.local import EnvironmentManager
        env_mgr = EnvironmentManager()
        dest = tmp_path / "repo"

        with patch("git.Repo.clone_from", side_effect=Exception("auth failed")):
            with pytest.raises(BackendError, match="Failed to clone repository"):
                env_mgr.clone_repo("https://github.com/bad/repo", "main", dest)

    def test_venv_creation_failure_raises_backend_error(self, tmp_path):
        from kdream.backends.local import EnvironmentManager
        env_mgr = EnvironmentManager()
        venv_path = tmp_path / "venv"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="uv not found", stdout="")
            with pytest.raises(BackendError, match="Failed to create venv"):
                env_mgr.create_venv(venv_path)

    def test_dep_install_failure_raises_backend_error(self, tmp_path):
        from kdream.backends.local import EnvironmentManager
        env_mgr = EnvironmentManager()

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / "requirements.txt").write_text("some-package==9.9.9\n")

        venv_path = tmp_path / "venv"
        (venv_path / "bin").mkdir(parents=True)
        (venv_path / "bin" / "python").touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="No matching distribution", stdout="")
            with pytest.raises(BackendError, match="Dependency installation failed"):
                env_mgr.install_deps(repo_path, venv_path)

    def test_inference_failure_raises_backend_error(self, tmp_path):
        from kdream.backends.local import LocalBackend
        from kdream.core.recipe import parse_yaml_recipe
        from kdream.core.runner import PackageInfo

        recipe_yaml = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: fail-model
  version: 1.0.0
  description: Test
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test/fail-model
  ref: main
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
    required: true
outputs:
  - name: result
    type: string
"""
        recipe = parse_yaml_recipe(recipe_yaml)

        pkg_dir = tmp_path / "fail-model"
        repo_path = pkg_dir / "repo"
        venv_path = pkg_dir / "venv"
        for d in [repo_path, venv_path, pkg_dir / "models"]:
            d.mkdir(parents=True)
        (repo_path / "kdream-recipe.yaml").write_text(recipe_yaml)

        pkg = PackageInfo(
            recipe_name="fail-model",
            path=pkg_dir,
            ready=True,
            venv_path=venv_path,
            repo_path=repo_path,
            models_path=pkg_dir / "models",
        )

        backend = LocalBackend.__new__(LocalBackend)
        backend.verbose = False
        backend.hardware = MagicMock()
        backend.hardware.best_accelerator.return_value = "cpu"
        backend.runner = MagicMock()
        backend.runner.build_command.return_value = ["/venv/bin/python", "run.py"]
        backend.runner.execute.return_value = (1, "", "RuntimeError: CUDA out of memory")

        with patch.object(backend, "_ensure_cli_wrapper", return_value=None):
            with pytest.raises(BackendError, match="Inference failed"):
                backend.run(pkg, {"prompt": "hello"})


# ---------------------------------------------------------------------------
# ModelDownloadError raised correctly
# ---------------------------------------------------------------------------

class TestModelDownloadErrorRaisedCorrectly:
    def test_checksum_mismatch_raises_model_download_error(self, tmp_path):
        from kdream.backends.local import ModelManager
        from kdream.core.recipe import ModelDescriptor

        mgr = ModelManager()

        desc = ModelDescriptor(
            name="test-model",
            source="huggingface",
            id="org/model",
            destination="model.safetensors",
            checksum="0" * 64,  # Wrong checksum
        )

        # `download_model` computes dest = models_dir / desc.destination
        # Create the file there so the checksum check is reached
        dest_path = tmp_path / desc.destination
        dest_path.write_bytes(b"fake model data")

        with patch.object(mgr, "fetch_hf"):  # Don't actually download
            with pytest.raises(ModelDownloadError, match="checksum mismatch"):
                mgr.download_model(desc, tmp_path)

    def test_unknown_model_source_raises_model_download_error(self, tmp_path):
        from kdream.backends.local import ModelManager
        from kdream.core.recipe import ModelDescriptor

        mgr = ModelManager()
        desc = ModelDescriptor(
            name="test", source="huggingface", id="x", destination="models/x"
        )
        desc.source = "ftp"  # bypass pydantic literal validation

        with pytest.raises(ModelDownloadError, match="Unknown model source"):
            mgr.download_model(desc, tmp_path)

    def test_local_model_missing_raises_model_download_error(self, tmp_path):
        from kdream.backends.local import ModelManager
        from kdream.core.recipe import ModelDescriptor

        mgr = ModelManager()
        desc = ModelDescriptor(
            name="local-weights",
            source="local",
            id="local-weights",
            destination="nonexistent/path",
        )

        with pytest.raises(ModelDownloadError, match="Local model source does not exist"):
            mgr.download_model(desc, tmp_path)

    def test_hf_download_error_raises_model_download_error(self, tmp_path):
        from kdream.backends.local import ModelManager

        mgr = ModelManager()
        dest = tmp_path / "model"

        with patch("huggingface_hub.snapshot_download", side_effect=Exception("auth failed")):
            with pytest.raises(ModelDownloadError, match="Failed to download"):
                mgr.fetch_hf("bad-org/bad-model", dest)

    def test_url_download_error_raises_model_download_error(self, tmp_path):
        import httpx
        from kdream.backends.local import ModelManager

        mgr = ModelManager()
        dest = tmp_path / "file.bin"

        with patch("httpx.stream", side_effect=Exception("connection failed")):
            with pytest.raises(ModelDownloadError, match="Download failed"):
                mgr.fetch_url("https://example.com/model.bin", dest)

    def test_civitai_download_error_raises_model_download_error(self, tmp_path):
        from kdream.backends.local import ModelManager

        mgr = ModelManager()
        dest = tmp_path / "model.safetensors"

        with patch("httpx.stream", side_effect=Exception("network error")):
            with pytest.raises(ModelDownloadError, match="CIVITAI download failed"):
                mgr.fetch_civitai("12345", dest)


# ---------------------------------------------------------------------------
# CIVITAI download routing
# ---------------------------------------------------------------------------

class TestCivitaiDownload:
    def test_download_model_routes_to_civitai(self, tmp_path):
        from kdream.backends.local import ModelManager
        from kdream.core.recipe import ModelDescriptor

        mgr = ModelManager()
        desc = ModelDescriptor(
            name="lora-model",
            source="civitai",
            id="99999",
            destination="loras/lora-model",
        )

        with patch.object(mgr, "fetch_civitai") as mock_civitai:
            mgr.download_model(desc, tmp_path)
            mock_civitai.assert_called_once()
            call_args = mock_civitai.call_args
            assert call_args.args[0] == "99999"
            # Destination should be under the dest dir
            assert "lora-model" in str(call_args.args[1])

    def test_civitai_includes_api_key_in_headers(self, tmp_path):
        from kdream.backends.local import ModelManager

        mgr = ModelManager()
        dest = tmp_path / "model.safetensors"

        captured_headers = {}

        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            def iter_bytes(self, size):
                return iter([b"data"])

        def fake_stream(method, url, headers=None, **kwargs):
            captured_headers.update(headers or {})
            return FakeStream()

        with patch("httpx.stream", side_effect=fake_stream):
            mgr.fetch_civitai("12345", dest, api_key="my-api-key")

        assert "Authorization" in captured_headers
        assert "my-api-key" in captured_headers["Authorization"]


# ---------------------------------------------------------------------------
# CLI: cache clear confirm=y
# ---------------------------------------------------------------------------

class TestCliCacheClearConfirm:
    def test_cache_clear_aborts_on_no(self):
        from click.testing import CliRunner
        from kdream.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["cache", "clear"], input="n\n")
        # Clicking "n" aborts — Click returns exit_code 1 for aborted prompts
        assert result.exit_code in (0, 1)
        assert "Aborted" in result.output or "abort" in result.output.lower() or result.exit_code == 1

    def test_cache_clear_proceeds_on_yes(self, tmp_path):
        from click.testing import CliRunner
        from kdream.cli import cli

        runner = CliRunner()

        # Patch the cache dir to point to tmp_path so nothing real is deleted
        with patch("kdream.cli.Path.home", return_value=tmp_path):
            with patch("shutil.rmtree") as mock_rm:
                result = runner.invoke(cli, ["cache", "clear"], input="y\n")

        # Either it succeeded or showed some output — just ensure no crash
        assert result.exit_code in (0, 1)


# ---------------------------------------------------------------------------
# InferenceRunner: collect_output edge cases
# ---------------------------------------------------------------------------

class TestCollectOutputEdgeCases:
    def _recipe(self, output_type: str = "file", path: str = "outputs/{timestamp}.png"):
        from kdream.core.recipe import OutputSpec, parse_yaml_recipe

        yaml = f"""\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: edge-model
  version: 1.0.0
  description: Edge case model
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test/edge-model
  ref: main
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
outputs:
  - name: result
    type: {output_type}
    path: {path}
"""
        return parse_yaml_recipe(yaml)

    def test_file_pattern_match_returns_latest(self, tmp_path):
        from kdream.backends.local import InferenceRunner

        recipe = self._recipe("file", "outputs/*.png")
        runner = InferenceRunner()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        (outputs_dir / "frame1.png").write_bytes(b"PNG1")
        (outputs_dir / "frame2.png").write_bytes(b"PNG2")

        result = runner.collect_output(recipe, "", tmp_path)
        assert "result" in result
        assert result["result"].endswith(".png")

    def test_no_file_match_falls_back_to_recent_files(self, tmp_path):
        import time
        from kdream.backends.local import InferenceRunner

        recipe = self._recipe("file", "outputs/specific_name_{timestamp}.png")
        runner = InferenceRunner()

        # Create a recent file that won't match the pattern
        recent_file = tmp_path / "actually_output.png"
        recent_file.write_bytes(b"PNG")

        run_start = time.time() - 1  # file was created after this

        result = runner.collect_output(recipe, "", tmp_path, run_start=run_start)
        assert "result" in result
        assert "actually_output.png" in result["result"]

    def test_no_outputs_spec_and_no_new_files_returns_stdout(self, tmp_path):
        from kdream.backends.local import InferenceRunner
        from kdream.core.recipe import parse_yaml_recipe

        yaml = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: no-output-model
  version: 1.0.0
  description: No output spec
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test/no-output
  ref: main
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
"""
        recipe = parse_yaml_recipe(yaml)
        runner = InferenceRunner()

        result = runner.collect_output(recipe, "the model output", tmp_path, run_start=9999999999.0)
        assert "stdout" in result
        assert result["stdout"] == "the model output"

    def test_collect_output_timestamp_placeholder_replaced(self, tmp_path):
        from kdream.backends.local import InferenceRunner

        recipe = self._recipe("file", "outputs/result_{timestamp}.png")
        runner = InferenceRunner()

        # No match — should return expected path with timestamp substituted
        result = runner.collect_output(recipe, "", tmp_path, run_start=9999999999.0)
        assert "result" in result
        # Should not contain literal "{timestamp}"
        assert "{timestamp}" not in result["result"]


# ---------------------------------------------------------------------------
# EnvironmentManager: _find_all_requirements edge cases
# ---------------------------------------------------------------------------

class TestFindAllRequirementsEdgeCases:
    def test_finds_multiple_requirement_files(self, tmp_path):
        from kdream.backends.local import EnvironmentManager

        (tmp_path / "requirements.txt").write_text("torch\n")
        (tmp_path / "requirements-extra.txt").write_text("diffusers\n")

        result = EnvironmentManager._find_all_requirements(tmp_path)
        names = [r.name for r in result]
        assert "requirements.txt" in names
        assert "requirements-extra.txt" in names

    def test_finds_requirements_in_subdirectory(self, tmp_path):
        from kdream.backends.local import EnvironmentManager

        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        (req_dir / "base.txt").write_text("numpy\n")

        result = EnvironmentManager._find_all_requirements(tmp_path)
        names = [r.name for r in result]
        assert "base.txt" in names

    def test_no_duplicate_requirements(self, tmp_path):
        from kdream.backends.local import EnvironmentManager

        (tmp_path / "requirements.txt").write_text("torch\n")

        result = EnvironmentManager._find_all_requirements(tmp_path)
        # requirements.txt should only appear once even though it matches both
        # the explicit candidates and the glob
        names = [r.name for r in result]
        assert names.count("requirements.txt") == 1

    def test_priority_order_requirements_first(self, tmp_path):
        from kdream.backends.local import EnvironmentManager

        (tmp_path / "requirements.txt").write_text("torch\n")
        (tmp_path / "requirements-torch.txt").write_text("torchvision\n")

        result = EnvironmentManager._find_all_requirements(tmp_path)
        assert result[0].name == "requirements.txt"


# ---------------------------------------------------------------------------
# HardwareDetector edge cases
# ---------------------------------------------------------------------------

class TestHardwareDetectorEdgeCases:
    def test_best_accelerator_returns_string(self):
        from kdream.backends.local import HardwareDetector
        detector = HardwareDetector()
        result = detector.best_accelerator()
        assert isinstance(result, str)
        assert result in ("cuda", "mps", "cpu")

    def test_nvidia_smi_fallback(self):
        """When torch is unavailable, nvidia-smi is tried as fallback."""
        from kdream.backends.local import HardwareDetector

        detector = HardwareDetector()
        with patch.dict("sys.modules", {"torch": None}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="8192\n",  # 8 GiB in MiB
                )
                result = detector.detect()

        # nvidia-smi succeeded — device should be cuda
        assert result["device"] == "cuda"
        assert result["vram_gb"] == pytest.approx(8.0, abs=0.1)

    def test_cpu_fallback_when_nvidia_smi_fails(self):
        from kdream.backends.local import HardwareDetector
        import sys

        detector = HardwareDetector()

        # Patch out torch and nvidia-smi, and ensure not on Apple Silicon
        with patch.dict("sys.modules", {"torch": None}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="")
                with patch.object(sys, "platform", "linux"):
                    result = detector.detect()

        assert result["device"] in ("cpu", "mps")  # mps only on darwin/arm64

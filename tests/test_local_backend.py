"""Tests for kdream.backends.local — LocalBackend."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kdream.backends.local import (
    EnvironmentManager,
    HardwareDetector,
    InferenceRunner,
    LocalBackend,
    ModelManager,
)
from kdream.exceptions import ModelDownloadError


class TestHardwareDetector:
    def test_detect_returns_dict(self):
        detector = HardwareDetector()
        result = detector.detect()
        assert isinstance(result, dict)
        assert "device" in result
        assert result["device"] in ("cuda", "mps", "cpu")
        assert "vram_gb" in result

    def test_cpu_fallback_when_no_gpu(self):
        with patch.dict("sys.modules", {"torch": None}):
            detector = HardwareDetector()
            result = detector.detect()
            assert result["device"] in ("cuda", "mps", "cpu")


class TestEnvironmentManager:
    def test_clone_skips_existing_git_repo(self, tmp_path):
        env_mgr = EnvironmentManager()
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        with patch("git.Repo.clone_from") as mock_clone:
            env_mgr.clone_repo("https://github.com/test/test", "main", repo_path)
            mock_clone.assert_not_called()

    def test_create_venv_calls_uv(self, tmp_path):
        env_mgr = EnvironmentManager()
        venv_path = tmp_path / "venv"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            env_mgr.create_venv(venv_path)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "uv" in cmd
            assert "venv" in cmd

    def test_create_venv_skips_if_exists(self, tmp_path):
        env_mgr = EnvironmentManager()
        venv_path = tmp_path / "venv"
        (venv_path / "bin").mkdir(parents=True)
        (venv_path / "bin" / "python").touch()

        with patch("subprocess.run") as mock_run:
            env_mgr.create_venv(venv_path)
            mock_run.assert_not_called()

    def test_find_all_requirements_txt(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("torch\n")
        result = EnvironmentManager._find_all_requirements(tmp_path)
        assert len(result) == 1
        assert result[0].name == "requirements.txt"

    def test_find_all_requirements_returns_none_when_absent(self, tmp_path):
        result = EnvironmentManager._find_all_requirements(tmp_path)
        assert result == []


class TestModelManager:
    def test_verify_correct_checksum(self, tmp_path):
        import hashlib
        content = b"test content"
        f = tmp_path / "model.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()

        mgr = ModelManager()
        assert mgr.verify(f, expected) is True

    def test_verify_wrong_checksum(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"content")
        mgr = ModelManager()
        assert mgr.verify(f, "0" * 64) is False

    @patch("huggingface_hub.snapshot_download")
    def test_fetch_hf_calls_snapshot_download(self, mock_dl, tmp_path):
        mock_dl.return_value = str(tmp_path)
        mgr = ModelManager()
        dest = tmp_path / "model"
        dest.mkdir()
        (dest / "file.bin").touch()  # mark as non-empty so skip check passes

        # Fresh dest — should call snapshot_download
        fresh = tmp_path / "fresh"
        mgr.fetch_hf("test-org/test-model", fresh)
        mock_dl.assert_called_once()

    def test_download_model_routes_hf(self, tmp_path):
        from kdream.core.recipe import ModelDescriptor
        mgr = ModelManager()
        desc = ModelDescriptor(
            name="test", source="huggingface", id="org/name", destination="models/test"
        )
        with patch.object(mgr, "fetch_hf") as mock_hf:
            mgr.download_model(desc, tmp_path)
            mock_hf.assert_called_once()

    def test_download_model_routes_url(self, tmp_path):
        from kdream.core.recipe import ModelDescriptor
        mgr = ModelManager()
        desc = ModelDescriptor(
            name="test", source="url", id="https://example.com/model.ckpt",
            destination="models/test"
        )
        with patch.object(mgr, "fetch_url") as mock_url:
            mgr.download_model(desc, tmp_path)
            mock_url.assert_called_once()

    def test_download_model_unknown_source_raises(self, tmp_path):
        from kdream.core.recipe import ModelDescriptor
        mgr = ModelManager()
        desc = ModelDescriptor(
            name="test", source="huggingface", id="x", destination="models/x"
        )
        desc.source = "ftp"  # bypass pydantic literal validation
        with pytest.raises(ModelDownloadError):
            mgr.download_model(desc, tmp_path)


class TestInferenceRunner:
    def test_build_command_basic(self, tmp_path, sample_yaml_recipe):
        from kdream.core.recipe import parse_yaml_recipe
        recipe = parse_yaml_recipe(sample_yaml_recipe)

        venv_path = tmp_path / "venv"
        (venv_path / "bin").mkdir(parents=True)
        python = venv_path / "bin" / "python"
        python.touch()

        runner = InferenceRunner()
        cmd = runner.build_command(
            recipe,
            {"prompt": "hello", "steps": 10},
            venv_path,
            tmp_path,
        )
        assert str(python) in cmd
        assert "--prompt" in cmd
        assert "hello" in cmd
        assert "--steps" in cmd
        assert "10" in cmd

    def test_collect_output_string(self, tmp_path, sample_yaml_recipe):
        from kdream.core.recipe import OutputSpec, parse_yaml_recipe
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        recipe.outputs[0].type = "string"

        runner = InferenceRunner()
        result = runner.collect_output(recipe, "generated text", tmp_path)
        assert "image" in result
        assert result["image"] == "generated text"

    def test_collect_output_stdout_fallback(self, tmp_path, sample_yaml_recipe):
        from kdream.core.recipe import parse_yaml_recipe
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        recipe.outputs = []  # no outputs spec

        runner = InferenceRunner()
        result = runner.collect_output(recipe, "some output", tmp_path)
        assert "stdout" in result


class TestLocalBackend:
    def test_is_installed_false_empty_dir(self, tmp_path):
        backend = LocalBackend(cache_dir=tmp_path)
        assert backend.is_installed("test-model", tmp_path) is False

    def test_is_installed_true_when_dirs_exist(self, tmp_path):
        backend = LocalBackend(cache_dir=tmp_path)
        pkg = tmp_path / "test-model"
        (pkg / "repo").mkdir(parents=True)
        (pkg / "venv").mkdir(parents=True)
        assert backend.is_installed("test-model", tmp_path) is True

    def test_validate_inputs_ok(self, sample_yaml_recipe):
        from kdream.core.recipe import parse_yaml_recipe
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        backend = LocalBackend.__new__(LocalBackend)
        errors = backend.validate_inputs(recipe, {"prompt": "hi", "steps": 20})
        assert errors == []

    def test_validate_inputs_missing_required(self, sample_yaml_recipe):
        from kdream.core.recipe import parse_yaml_recipe
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        backend = LocalBackend.__new__(LocalBackend)
        errors = backend.validate_inputs(recipe, {})
        assert any("prompt" in e for e in errors)

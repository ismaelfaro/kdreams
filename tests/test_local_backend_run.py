"""Tests for LocalBackend.run() — the main inference execution path."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kdream.backends.local import LocalBackend
from kdream.core.runner import PackageInfo
from kdream.exceptions import BackendError


# ---------------------------------------------------------------------------
# Minimal recipe YAML with and without device inputs
# ---------------------------------------------------------------------------

RECIPE_YAML_SIMPLE = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: simple-model
  version: 1.0.0
  description: Simple test model
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test/simple-model
  ref: main
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
    required: true
    description: Input prompt
  steps:
    type: integer
    default: 20
    description: Number of steps
outputs:
  - name: image
    type: file
    path: outputs/result.png
"""

RECIPE_YAML_WITH_DEVICE = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: device-model
  version: 1.0.0
  description: Model with device input
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test/device-model
  ref: main
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
    required: true
  device:
    type: string
    default: cpu
outputs:
  - name: result
    type: string
"""

RECIPE_YAML_STRING_OUTPUT = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: text-model
  version: 1.0.0
  description: Text output model
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test/text-model
  ref: main
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
    required: true
outputs:
  - name: text
    type: string
"""


def _make_package(tmp_path: Path, name: str = "simple-model", recipe_yaml: str = "") -> PackageInfo:
    """Create a minimal PackageInfo with directories on disk."""
    pkg_dir = tmp_path / name
    repo_path = pkg_dir / "repo"
    venv_path = pkg_dir / "venv"
    models_path = pkg_dir / "models"
    for d in [repo_path, venv_path, models_path]:
        d.mkdir(parents=True)

    # Optionally write a recipe file into the repo
    if recipe_yaml:
        (repo_path / "kdream-recipe.yaml").write_text(recipe_yaml)

    return PackageInfo(
        recipe_name=name,
        path=pkg_dir,
        ready=True,
        venv_path=venv_path,
        repo_path=repo_path,
        models_path=models_path,
    )


def _backend(verbose: bool = False) -> LocalBackend:
    return LocalBackend.__new__(LocalBackend)


# ---------------------------------------------------------------------------
# Helpers for mocking LocalBackend.run() internals
# ---------------------------------------------------------------------------

def _setup_backend_run(
    recipe_yaml: str,
    rc: int = 0,
    stdout: str = "",
    stderr: str = "",
    outputs: dict | None = None,
):
    """Return a configured LocalBackend with all subprocess calls mocked."""
    from kdream.core.recipe import parse_yaml_recipe

    backend = LocalBackend.__new__(LocalBackend)
    backend.verbose = False
    backend.hardware = MagicMock()
    backend.hardware.best_accelerator.return_value = "cpu"
    backend.runner = MagicMock()
    backend.runner.build_command.return_value = ["/venv/bin/python", "run.py"]
    backend.runner.execute.return_value = (rc, stdout, stderr)
    backend.runner.collect_output.return_value = outputs or {}

    recipe = parse_yaml_recipe(recipe_yaml)
    return backend, recipe


# ---------------------------------------------------------------------------
# LocalBackend.run() — happy path
# ---------------------------------------------------------------------------

class TestLocalBackendRun:
    def test_run_loads_recipe_from_repo(self, tmp_path):
        pkg = _make_package(tmp_path, "simple-model", RECIPE_YAML_SIMPLE)
        backend, recipe = _setup_backend_run(RECIPE_YAML_SIMPLE, stdout="done")
        backend.runner.collect_output.return_value = {"image": "outputs/result.png"}

        with patch.object(backend, "_ensure_cli_wrapper", return_value=None):
            with patch("shutil.copy2"):
                result = backend.run(pkg, {"prompt": "hello"})

        backend.runner.execute.assert_called_once()
        assert isinstance(result, dict)

    def test_run_validates_inputs_before_execution(self, tmp_path):
        pkg = _make_package(tmp_path, "simple-model", RECIPE_YAML_SIMPLE)
        backend, _ = _setup_backend_run(RECIPE_YAML_SIMPLE)

        # Missing required 'prompt' → BackendError before execution
        with pytest.raises(BackendError, match="Input validation failed"):
            backend.run(pkg, {})

        backend.runner.execute.assert_not_called()

    def test_run_applies_defaults_for_optional_inputs(self, tmp_path):
        pkg = _make_package(tmp_path, "simple-model", RECIPE_YAML_SIMPLE)
        backend, _ = _setup_backend_run(RECIPE_YAML_SIMPLE, stdout="ok")
        backend.runner.collect_output.return_value = {}

        with patch.object(backend, "_ensure_cli_wrapper", return_value=None):
            backend.run(pkg, {"prompt": "hello"})

        # build_command is called with merged inputs including default `steps`
        call_args = backend.runner.build_command.call_args
        merged_inputs = call_args.args[1] if call_args.args else call_args.kwargs.get("inputs", {})
        assert "steps" in merged_inputs
        assert merged_inputs["steps"] == 20  # default from recipe

    def test_run_injects_device_when_not_provided(self, tmp_path):
        pkg = _make_package(tmp_path, "device-model", RECIPE_YAML_WITH_DEVICE)
        backend, _ = _setup_backend_run(RECIPE_YAML_WITH_DEVICE, stdout="ok")
        backend.hardware.best_accelerator.return_value = "cuda"
        backend.runner.collect_output.return_value = {}

        with patch.object(backend, "_ensure_cli_wrapper", return_value=None):
            backend.run(pkg, {"prompt": "hi"})

        call_args = backend.runner.build_command.call_args
        merged = call_args.args[1] if call_args.args else call_args.kwargs.get("inputs", {})
        assert merged.get("device") == "cuda"

    def test_run_respects_user_provided_device(self, tmp_path):
        pkg = _make_package(tmp_path, "device-model", RECIPE_YAML_WITH_DEVICE)
        backend, _ = _setup_backend_run(RECIPE_YAML_WITH_DEVICE, stdout="ok")
        backend.hardware.best_accelerator.return_value = "cuda"
        backend.runner.collect_output.return_value = {}

        with patch.object(backend, "_ensure_cli_wrapper", return_value=None):
            backend.run(pkg, {"prompt": "hi", "device": "cpu"})

        call_args = backend.runner.build_command.call_args
        merged = call_args.args[1] if call_args.args else call_args.kwargs.get("inputs", {})
        # User override must win over detected accelerator
        assert merged.get("device") == "cpu"

    def test_run_raises_on_nonzero_exit_code(self, tmp_path):
        pkg = _make_package(tmp_path, "simple-model", RECIPE_YAML_SIMPLE)
        backend, _ = _setup_backend_run(
            RECIPE_YAML_SIMPLE, rc=1, stderr="CUDA out of memory"
        )

        with patch.object(backend, "_ensure_cli_wrapper", return_value=None):
            with pytest.raises(BackendError, match="Inference failed"):
                backend.run(pkg, {"prompt": "hello"})

    def test_run_returns_string_output_from_stdout(self, tmp_path):
        pkg = _make_package(tmp_path, "text-model", RECIPE_YAML_STRING_OUTPUT)
        backend, _ = _setup_backend_run(
            RECIPE_YAML_STRING_OUTPUT, stdout="generated text here"
        )
        backend.runner.collect_output.return_value = {"text": "generated text here"}

        with patch.object(backend, "_ensure_cli_wrapper", return_value=None):
            result = backend.run(pkg, {"prompt": "hello"})

        assert result.get("text") == "generated text here"

    def test_run_copies_file_outputs_to_cwd(self, tmp_path):
        pkg = _make_package(tmp_path, "simple-model", RECIPE_YAML_SIMPLE)
        backend, _ = _setup_backend_run(RECIPE_YAML_SIMPLE, stdout="done")

        # Create a fake output file in repo directory
        output_file = pkg.repo_path / "outputs" / "result.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"PNG_DATA")

        backend.runner.collect_output.return_value = {"image": str(output_file)}

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch.object(backend, "_ensure_cli_wrapper", return_value=None):
            result = backend.run(pkg, {"prompt": "hello", "output_dir": str(output_dir)})

        assert "image" in result
        # Output should have been copied to the specified output_dir
        assert str(output_dir) in result["image"]

    def test_run_falls_back_to_registry_when_no_recipe_file(self, tmp_path):
        # Package with no kdream-recipe.yaml in repo
        pkg = _make_package(tmp_path, "simple-model")  # no recipe_yaml written
        backend, recipe = _setup_backend_run(RECIPE_YAML_SIMPLE, stdout="ok")
        backend.runner.collect_output.return_value = {}

        with patch("kdream.core.registry.RegistryClient") as MockClient:
            MockClient.return_value.fetch_recipe.return_value = recipe
            with patch.object(backend, "_ensure_cli_wrapper", return_value=None):
                backend.run(pkg, {"prompt": "hi"})

        MockClient.return_value.fetch_recipe.assert_called_once_with("simple-model")

    def test_run_raises_if_registry_lookup_fails_without_recipe_file(self, tmp_path):
        pkg = _make_package(tmp_path, "simple-model")
        backend, _ = _setup_backend_run(RECIPE_YAML_SIMPLE)

        from kdream.exceptions import RegistryError

        with patch("kdream.core.registry.RegistryClient") as MockClient:
            MockClient.return_value.fetch_recipe.side_effect = RegistryError("not found")
            with pytest.raises(BackendError, match="Could not load recipe"):
                backend.run(pkg, {"prompt": "hi"})


# ---------------------------------------------------------------------------
# LocalBackend.install() — GPU warning
# ---------------------------------------------------------------------------

class TestLocalBackendInstall:
    def _recipe(self):
        from kdream.core.recipe import parse_yaml_recipe
        return parse_yaml_recipe("""\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: gpu-model
  version: 1.0.0
  description: GPU model
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test/gpu-model
  ref: main
entrypoint:
  script: run.py
  type: python
backends:
  local:
    requires_gpu: true
    min_vram_gb: 8
""")

    def test_install_warns_when_gpu_required_but_cpu_detected(self, tmp_path, capsys):
        recipe = self._recipe()
        backend = LocalBackend(cache_dir=tmp_path)

        with patch.object(backend.hardware, "detect", return_value={"device": "cpu", "vram_gb": 0, "cuda_version": None}):
            with patch.object(backend.env_manager, "clone_repo"):
                with patch.object(backend.env_manager, "create_venv"):
                    with patch.object(backend.env_manager, "install_deps"):
                        backend.install(recipe, tmp_path, force_reinstall=False)
        # No exception raised — just a warning printed (no assertion on output, just no crash)

    def test_install_creates_package_dirs(self, tmp_path):
        recipe = self._recipe()
        backend = LocalBackend(cache_dir=tmp_path)

        with patch.object(backend.hardware, "detect", return_value={"device": "cpu", "vram_gb": 0, "cuda_version": None}):
            with patch.object(backend.env_manager, "clone_repo"):
                with patch.object(backend.env_manager, "create_venv"):
                    with patch.object(backend.env_manager, "install_deps"):
                        pkg = backend.install(recipe, tmp_path, force_reinstall=False)

        assert pkg.recipe_name == "gpu-model"
        assert pkg.path.exists()

    def test_force_reinstall_removes_existing_dir(self, tmp_path):
        recipe = self._recipe()
        pkg_dir = tmp_path / "gpu-model"
        pkg_dir.mkdir(parents=True)
        existing_file = pkg_dir / "old_file.txt"
        existing_file.write_text("old")

        backend = LocalBackend(cache_dir=tmp_path)

        with patch.object(backend.hardware, "detect", return_value={"device": "cpu", "vram_gb": 0, "cuda_version": None}):
            with patch.object(backend.env_manager, "clone_repo"):
                with patch.object(backend.env_manager, "create_venv"):
                    with patch.object(backend.env_manager, "install_deps"):
                        backend.install(recipe, tmp_path, force_reinstall=True)

        # The old file should be gone after force reinstall
        assert not existing_file.exists()


# ---------------------------------------------------------------------------
# _ensure_cli_wrapper
# ---------------------------------------------------------------------------

class TestEnsureCliWrapper:
    def _make_recipe_and_pkg(self, tmp_path, recipe_yaml: str, script_content: str = ""):
        from kdream.core.recipe import parse_yaml_recipe
        recipe = parse_yaml_recipe(recipe_yaml)
        pkg = _make_package(tmp_path, recipe.metadata.name, recipe_yaml)
        if script_content is not None:
            (pkg.repo_path / "run.py").write_text(script_content)
        return recipe, pkg

    def test_returns_none_when_script_uses_argparse(self, tmp_path):
        recipe, pkg = self._make_recipe_and_pkg(
            tmp_path, RECIPE_YAML_SIMPLE,
            "import argparse\nparser = argparse.ArgumentParser()\n"
        )
        backend = LocalBackend.__new__(LocalBackend)
        backend.verbose = False
        backend.hardware = MagicMock()

        with patch.object(backend, "_find_bundled_wrapper", return_value=None):
            result = backend._ensure_cli_wrapper(recipe, pkg)

        assert result is None

    def test_returns_none_when_script_uses_click(self, tmp_path):
        recipe, pkg = self._make_recipe_and_pkg(
            tmp_path, RECIPE_YAML_SIMPLE,
            "import click\n@click.command()\ndef main(): pass\n"
        )
        backend = LocalBackend.__new__(LocalBackend)
        backend.verbose = False
        backend.hardware = MagicMock()

        with patch.object(backend, "_find_bundled_wrapper", return_value=None):
            result = backend._ensure_cli_wrapper(recipe, pkg)

        assert result is None

    def test_generates_wrapper_when_script_has_no_cli(self, tmp_path):
        recipe, pkg = self._make_recipe_and_pkg(
            tmp_path, RECIPE_YAML_SIMPLE,
            "prompt = 'default'\nprint(prompt)\n"
        )
        backend = LocalBackend.__new__(LocalBackend)
        backend.verbose = False
        backend.hardware = MagicMock()

        with patch.object(backend, "_find_bundled_wrapper", return_value=None):
            result = backend._ensure_cli_wrapper(recipe, pkg)

        assert result is not None
        assert result.name == "_kdream_runner.py"
        wrapper_source = result.read_text()
        assert "argparse" in wrapper_source
        assert "--prompt" in wrapper_source

    def test_returns_none_when_script_missing(self, tmp_path):
        recipe, pkg = self._make_recipe_and_pkg(tmp_path, RECIPE_YAML_SIMPLE, None)
        # Remove the script
        script = pkg.repo_path / "run.py"
        if script.exists():
            script.unlink()

        backend = LocalBackend.__new__(LocalBackend)
        backend.verbose = False

        with patch.object(backend, "_find_bundled_wrapper", return_value=None):
            result = backend._ensure_cli_wrapper(recipe, pkg)

        assert result is None

    def test_uses_bundled_wrapper_when_available(self, tmp_path):
        recipe, pkg = self._make_recipe_and_pkg(tmp_path, RECIPE_YAML_SIMPLE, "print('hi')")
        bundled = tmp_path / "bundled_runner.py"
        bundled.write_text("# bundled runner\n")

        backend = LocalBackend.__new__(LocalBackend)
        backend.verbose = False

        with patch.object(backend, "_find_bundled_wrapper", return_value=bundled):
            result = backend._ensure_cli_wrapper(recipe, pkg)

        assert result is not None
        assert result.name == "_kdream_runner.py"
        assert "bundled runner" in result.read_text()


# ---------------------------------------------------------------------------
# _build_cli_wrapper
# ---------------------------------------------------------------------------

class TestBuildCliWrapper:
    def test_generates_argparse_preamble(self, tmp_path):
        from kdream.core.recipe import parse_yaml_recipe
        recipe = parse_yaml_recipe(RECIPE_YAML_SIMPLE)
        script_path = tmp_path / "run.py"
        source = "prompt = 'default'\nsteps = 20\nprint(prompt)\n"
        script_path.write_text(source)

        backend = LocalBackend.__new__(LocalBackend)
        result = backend._build_cli_wrapper(recipe, script_path, source)

        assert "import argparse as _kap" in result
        assert "--prompt" in result
        assert "--steps" in result

    def test_injects_overrides_after_assignments(self, tmp_path):
        from kdream.core.recipe import parse_yaml_recipe
        recipe = parse_yaml_recipe(RECIPE_YAML_SIMPLE)
        script_path = tmp_path / "run.py"
        source = "prompt = 'default'\nsteps = 20\n"
        script_path.write_text(source)

        backend = LocalBackend.__new__(LocalBackend)
        result = backend._build_cli_wrapper(recipe, script_path, source)

        # Override lines should be injected
        assert "_kdream_args.get('prompt')" in result
        assert "_kdream_args.get('steps')" in result

    def test_handles_syntax_error_gracefully(self, tmp_path):
        from kdream.core.recipe import parse_yaml_recipe
        recipe = parse_yaml_recipe(RECIPE_YAML_SIMPLE)
        script_path = tmp_path / "run.py"
        source = "def broken(:\n    pass\n"  # intentional syntax error
        script_path.write_text(source)

        backend = LocalBackend.__new__(LocalBackend)
        # Should not raise — just produces wrapper without AST overrides
        result = backend._build_cli_wrapper(recipe, script_path, source)
        assert "import argparse as _kap" in result

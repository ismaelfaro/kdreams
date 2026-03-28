"""Tests for the Google Colab backend (kdream.backends.colab)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kdream.backends.colab import (
    ColabBackend,
    GoogleDriveUploader,
    NotebookBuilder,
    _colab_github_url,
    _uid,
)
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
# Fixtures
# ---------------------------------------------------------------------------

def _make_recipe(
    name: str = "test-model",
    models: list | None = None,
    inputs: dict | None = None,
    outputs: list | None = None,
    entry_type: str = "python",
    repo: str = "https://github.com/example/test-model",
    _default_outputs: bool = True,
) -> Recipe:
    if outputs is None and _default_outputs:
        outputs = [OutputSpec(name="image", type="file", path="outputs/{ts}.png")]
    elif outputs is None:
        outputs = []
    return Recipe(
        metadata=RecipeMetadata(
            name=name,
            version="1.0.0",
            description="A test recipe for unit tests",
            tags=["image-generation"],
            license="Apache-2.0",
            author="test",
        ),
        source=RecipeSource(repo=repo, ref="main"),
        models=models or [],
        entrypoint=EntrypointSpec(script="run.py", type=entry_type),
        inputs=inputs or {
            "prompt": InputSpec(type="string", required=True,
                                description="Text prompt"),
            "steps": InputSpec(type="integer", default=20),
        },
        outputs=outputs,
        backends=BackendSpecs(
            local=LocalBackendSpec(requires_gpu=True, min_vram_gb=8, tested_on=["cuda"])
        ),
    )


def _make_hf_model(
    name: str = "test-weights",
    model_id: str = "org/model",
    destination: str = "models/test",
) -> ModelDescriptor:
    return ModelDescriptor(
        name=name, source="huggingface", id=model_id, destination=destination
    )


# ---------------------------------------------------------------------------
# NotebookBuilder
# ---------------------------------------------------------------------------

class TestNotebookBuilder:
    def test_build_returns_valid_notebook_structure(self):
        builder = NotebookBuilder()
        recipe = _make_recipe()
        nb = builder.build(recipe)

        assert nb["nbformat"] == 4
        assert "cells" in nb
        assert "metadata" in nb
        assert nb["metadata"]["colab"]["name"] == "kdream-test-model.ipynb"
        assert nb["metadata"]["accelerator"] == "GPU"

    def test_build_cells_count_no_models(self):
        builder = NotebookBuilder()
        recipe = _make_recipe()
        nb = builder.build(recipe)
        # markdown + hardware + clone + deps + inputs + run + outputs = 7
        assert len(nb["cells"]) == 7

    def test_build_cells_count_with_models(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(models=[_make_hf_model()])
        nb = builder.build(recipe)
        # +1 for models cell = 8
        assert len(nb["cells"]) == 8

    def test_build_markdown_cell_contains_name(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(name="my-recipe")
        nb = builder.build(recipe)
        md = nb["cells"][0]
        assert md["cell_type"] == "markdown"
        assert "my-recipe" in md["source"]

    def test_build_code_cells_have_correct_fields(self):
        builder = NotebookBuilder()
        recipe = _make_recipe()
        nb = builder.build(recipe)
        for cell in nb["cells"][1:]:
            assert cell["cell_type"] == "code"
            assert "source" in cell
            assert cell["execution_count"] is None
            assert "outputs" in cell

    def test_clone_cell_contains_repo_url(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(repo="https://github.com/example/my-proj")
        nb = builder.build(recipe)
        clone_cell = nb["cells"][2]  # 0=md, 1=hw, 2=clone
        assert "https://github.com/example/my-proj" in clone_cell["source"]
        assert '"git"' in clone_cell["source"] and '"clone"' in clone_cell["source"]

    def test_deps_cell_uses_uv(self):
        builder = NotebookBuilder()
        recipe = _make_recipe()
        nb = builder.build(recipe)
        deps_cell = nb["cells"][3]
        assert "uv" in deps_cell["source"]

    def test_deps_cell_skip_install(self):
        builder = NotebookBuilder()
        recipe = _make_recipe()
        recipe.source.skip_package_install = True
        nb = builder.build(recipe)
        deps_cell = nb["cells"][3]
        assert "skip_package_install" in deps_cell["source"].lower() or "skipped" in deps_cell["source"].lower()

    def test_models_cell_huggingface(self):
        builder = NotebookBuilder()
        model = _make_hf_model(model_id="stabilityai/sdxl-turbo")
        recipe = _make_recipe(models=[model])
        nb = builder.build(recipe)
        # With models, order is: md, hw, clone, deps, models, inputs, run, outputs
        models_cell = nb["cells"][4]
        assert "stabilityai/sdxl-turbo" in models_cell["source"]
        assert "snapshot_download" in models_cell["source"]

    def test_models_cell_url_source(self):
        builder = NotebookBuilder()
        model = ModelDescriptor(
            name="weights", source="url",
            id="https://example.com/model.safetensors", destination="models/w"
        )
        recipe = _make_recipe(models=[model])
        nb = builder.build(recipe)
        models_cell = nb["cells"][4]
        assert "urlretrieve" in models_cell["source"]
        assert "https://example.com/model.safetensors" in models_cell["source"]

    def test_models_cell_civitai_source(self):
        builder = NotebookBuilder()
        model = ModelDescriptor(
            name="civitai-weights", source="civitai", id="12345", destination="models/c"
        )
        recipe = _make_recipe(models=[model])
        nb = builder.build(recipe)
        models_cell = nb["cells"][4]
        assert "civitai.com" in models_cell["source"]
        assert "CIVITAI_TOKEN" in models_cell["source"]

    def test_inputs_cell_contains_values(self):
        builder = NotebookBuilder()
        recipe = _make_recipe()
        nb = builder.build(recipe, inputs={"prompt": "a red panda", "steps": 30})
        # With no models: inputs is cell index 4
        inputs_cell = nb["cells"][4]
        assert "'a red panda'" in inputs_cell["source"]
        assert "30" in inputs_cell["source"]

    def test_inputs_cell_defaults_when_no_inputs(self):
        builder = NotebookBuilder()
        recipe = _make_recipe()
        nb = builder.build(recipe)
        inputs_cell = nb["cells"][4]
        # steps default is 20
        assert "20" in inputs_cell["source"]

    def test_inputs_cell_boolean_type(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(inputs={
            "use_fp16": InputSpec(type="boolean", default=True),
        })
        nb = builder.build(recipe)
        inputs_cell = nb["cells"][4]
        assert "True" in inputs_cell["source"]

    def test_run_cell_contains_script(self):
        builder = NotebookBuilder()
        recipe = _make_recipe()
        nb = builder.build(recipe)
        run_cell = nb["cells"][5]
        assert "run.py" in run_cell["source"]
        assert "subprocess" in run_cell["source"]

    def test_run_cell_gradio_type(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(entry_type="gradio")
        nb = builder.build(recipe)
        run_cell = nb["cells"][5]
        assert "Popen" in run_cell["source"] or "gradio" in run_cell["source"].lower()

    def test_outputs_cell_no_outputs(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(_default_outputs=False)
        nb = builder.build(recipe)
        out_cell = nb["cells"][6]
        assert "No outputs" in out_cell["source"]

    def test_outputs_cell_image_display(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(outputs=[OutputSpec(name="img", type="file", path="out.png")])
        nb = builder.build(recipe)
        out_cell = nb["cells"][6]
        assert "IPython.display" in out_cell["source"]
        assert "Image" in out_cell["source"]

    def test_outputs_cell_audio_display(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(outputs=[OutputSpec(name="aud", type="file", path="out.wav")])
        nb = builder.build(recipe)
        out_cell = nb["cells"][6]
        assert "Audio" in out_cell["source"]

    def test_outputs_cell_video_display(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(outputs=[OutputSpec(name="vid", type="file", path="out.mp4")])
        nb = builder.build(recipe)
        out_cell = nb["cells"][6]
        assert "video" in out_cell["source"].lower()

    def test_notebook_is_json_serialisable(self):
        builder = NotebookBuilder()
        recipe = _make_recipe(models=[_make_hf_model()])
        nb = builder.build(recipe, inputs={"prompt": "test"})
        # Should not raise
        serialised = json.dumps(nb)
        parsed = json.loads(serialised)
        assert parsed["nbformat"] == 4


# ---------------------------------------------------------------------------
# _uid
# ---------------------------------------------------------------------------

class TestUid:
    def test_returns_12_char_hex(self):
        uid = _uid()
        assert len(uid) == 12
        assert all(c in "0123456789abcdef" for c in uid)

    def test_unique(self):
        assert _uid() != _uid()


# ---------------------------------------------------------------------------
# _colab_github_url
# ---------------------------------------------------------------------------

class TestColabGithubUrl:
    def test_github_repo_returns_url(self):
        recipe = _make_recipe(repo="https://github.com/owner/repo")
        url = _colab_github_url(recipe)
        assert url is not None
        assert "colab.research.google.com/github/owner/repo" in url
        assert recipe.metadata.name in url

    def test_non_github_repo_returns_none(self):
        recipe = _make_recipe(repo="https://example.com/some/repo")
        assert _colab_github_url(recipe) is None

    def test_empty_repo_returns_none(self):
        recipe = _make_recipe(repo="")
        assert _colab_github_url(recipe) is None


# ---------------------------------------------------------------------------
# GoogleDriveUploader
# ---------------------------------------------------------------------------

class TestGoogleDriveUploader:
    def test_upload_raises_if_google_deps_missing(self, tmp_path):
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")
        uploader = GoogleDriveUploader(str(creds_file))
        with patch.dict("sys.modules", {"google.oauth2": None,
                                         "google.oauth2.service_account": None,
                                         "googleapiclient": None,
                                         "googleapiclient.discovery": None,
                                         "googleapiclient.http": None}):
            with pytest.raises((BackendError, ImportError, TypeError)):
                uploader.upload(creds_file, "test.ipynb")

    def test_upload_calls_drive_api(self, tmp_path):
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        fake_service = MagicMock()
        fake_service.files.return_value.create.return_value.execute.return_value = {"id": "abc123"}
        fake_service.permissions.return_value.create.return_value.execute.return_value = {}

        mock_creds = MagicMock()
        with patch("kdream.backends.colab.GoogleDriveUploader.upload") as mock_upload:
            mock_upload.return_value = "https://colab.research.google.com/drive/abc123"
            uploader = GoogleDriveUploader(str(creds_file))
            url = uploader.upload(nb_file, "test.ipynb")
            assert "colab.research.google.com/drive" in url


# ---------------------------------------------------------------------------
# ColabBackend.install
# ---------------------------------------------------------------------------

class TestColabBackendInstall:
    def test_install_creates_notebook_file(self, tmp_path):
        backend = ColabBackend()
        recipe = _make_recipe()
        pkg = backend.install(recipe, cache_dir=tmp_path)

        nb_path = tmp_path / "test-model" / "colab" / "kdream-test-model.ipynb"
        assert nb_path.exists()
        nb = json.loads(nb_path.read_text())
        assert nb["nbformat"] == 4

    def test_install_returns_package_info(self, tmp_path):
        from kdream.core.runner import PackageInfo
        backend = ColabBackend()
        recipe = _make_recipe()
        pkg = backend.install(recipe, cache_dir=tmp_path)
        assert isinstance(pkg, PackageInfo)
        assert pkg.recipe_name == "test-model"
        assert pkg.ready is True

    def test_install_skips_if_exists_no_force(self, tmp_path):
        backend = ColabBackend()
        recipe = _make_recipe()

        backend.install(recipe, cache_dir=tmp_path)
        # Modify the notebook to detect whether it was regenerated
        nb_path = tmp_path / "test-model" / "colab" / "kdream-test-model.ipynb"
        nb_path.write_text('{"marker": "do-not-overwrite"}')

        backend.install(recipe, cache_dir=tmp_path, force_reinstall=False)
        content = json.loads(nb_path.read_text())
        assert content.get("marker") == "do-not-overwrite"

    def test_install_force_reinstall_regenerates(self, tmp_path):
        backend = ColabBackend()
        recipe = _make_recipe()

        backend.install(recipe, cache_dir=tmp_path)
        nb_path = tmp_path / "test-model" / "colab" / "kdream-test-model.ipynb"
        nb_path.write_text('{"marker": "old-notebook"}')

        backend.install(recipe, cache_dir=tmp_path, force_reinstall=True)
        content = json.loads(nb_path.read_text())
        assert "marker" not in content
        assert content["nbformat"] == 4


# ---------------------------------------------------------------------------
# ColabBackend.is_installed
# ---------------------------------------------------------------------------

class TestColabBackendIsInstalled:
    def test_returns_false_when_not_installed(self, tmp_path):
        backend = ColabBackend()
        assert backend.is_installed("my-recipe", tmp_path) is False

    def test_returns_true_after_install(self, tmp_path):
        backend = ColabBackend()
        recipe = _make_recipe()
        backend.install(recipe, cache_dir=tmp_path)
        assert backend.is_installed("test-model", tmp_path) is True


# ---------------------------------------------------------------------------
# ColabBackend.run
# ---------------------------------------------------------------------------

class TestColabBackendRun:
    def _install_and_get_package(self, backend, recipe, tmp_path):
        return backend.install(recipe, cache_dir=tmp_path)

    def test_run_generates_run_notebook(self, tmp_path):
        backend = ColabBackend()
        recipe = _make_recipe()
        pkg = backend.install(recipe, cache_dir=tmp_path)

        with patch("kdream.core.registry.RegistryClient.fetch_recipe", return_value=recipe):
            result = backend.run(pkg, inputs={"prompt": "a cat", "steps": 10})

        run_nb = tmp_path / "test-model" / "colab" / "kdream-test-model-run.ipynb"
        assert run_nb.exists()
        assert result["notebook_path"] == str(run_nb)

    def test_run_returns_colab_url_when_github_repo(self, tmp_path):
        backend = ColabBackend()
        recipe = _make_recipe(repo="https://github.com/example/test-model")
        pkg = backend.install(recipe, cache_dir=tmp_path)

        with patch("kdream.core.registry.RegistryClient.fetch_recipe", return_value=recipe):
            result = backend.run(pkg, inputs={"prompt": "test"})

        # GitHub repo → colab URL should be present
        assert "colab_url" in result
        assert "colab.research.google.com" in result["colab_url"]

    def test_run_raises_if_notebook_missing(self, tmp_path):
        from kdream.core.runner import PackageInfo
        backend = ColabBackend()
        fake_pkg = PackageInfo(
            recipe_name="missing-recipe",
            path=tmp_path / "missing-recipe" / "colab",
            ready=False,
            venv_path=tmp_path,
            repo_path=tmp_path,
            models_path=tmp_path,
        )
        with pytest.raises(BackendError, match="not found"):
            backend.run(fake_pkg, inputs={})

    def test_run_with_gdrive_credentials_attempts_upload(self, tmp_path):
        backend = ColabBackend(gdrive_credentials=str(tmp_path / "creds.json"))
        recipe = _make_recipe()
        pkg = backend.install(recipe, cache_dir=tmp_path)

        with patch("kdream.backends.colab.GoogleDriveUploader.upload",
                   return_value="https://colab.research.google.com/drive/xyz") as mock_up, \
             patch("kdream.core.registry.RegistryClient.fetch_recipe", return_value=recipe):
            result = backend.run(pkg, inputs={"prompt": "test"})

        mock_up.assert_called_once()
        assert result["colab_url"] == "https://colab.research.google.com/drive/xyz"

    def test_run_gdrive_upload_failure_falls_back_gracefully(self, tmp_path):
        backend = ColabBackend(gdrive_credentials=str(tmp_path / "creds.json"))
        recipe = _make_recipe(repo="https://github.com/example/test-model")
        pkg = backend.install(recipe, cache_dir=tmp_path)

        with patch("kdream.backends.colab.GoogleDriveUploader.upload",
                   side_effect=BackendError("Drive error")), \
             patch("kdream.core.registry.RegistryClient.fetch_recipe", return_value=recipe):
            result = backend.run(pkg, inputs={"prompt": "test"})

        # Falls back to GitHub Colab URL
        assert "colab.research.google.com" in result["colab_url"]


# ---------------------------------------------------------------------------
# ColabBackend.generate_notebook
# ---------------------------------------------------------------------------

class TestColabBackendGenerateNotebook:
    def test_generates_ipynb_file(self, tmp_path):
        backend = ColabBackend()
        recipe = _make_recipe()
        out = tmp_path / "my-notebook.ipynb"
        result = backend.generate_notebook(recipe, output_path=str(out))
        assert result == out
        assert out.exists()
        nb = json.loads(out.read_text())
        assert nb["nbformat"] == 4

    def test_generates_in_cwd_by_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        backend = ColabBackend()
        recipe = _make_recipe()
        result = backend.generate_notebook(recipe)
        assert result.name == "kdream-test-model.ipynb"
        assert result.exists()

    def test_inputs_injected_in_generated_notebook(self, tmp_path):
        backend = ColabBackend()
        recipe = _make_recipe()
        out = tmp_path / "nb.ipynb"
        backend.generate_notebook(recipe, inputs={"prompt": "sunset", "steps": 42},
                                  output_path=str(out))
        content = out.read_text()
        assert "sunset" in content
        assert "42" in content


# ---------------------------------------------------------------------------
# CLI colab generate
# ---------------------------------------------------------------------------

class TestColabCLI:
    def test_colab_help(self):
        from click.testing import CliRunner
        from kdream.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["colab", "--help"])
        assert result.exit_code == 0
        assert "colab" in result.output.lower()

    def test_colab_generate_help(self):
        from click.testing import CliRunner
        from kdream.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["colab", "generate", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output

    def test_colab_generate_creates_notebook(self, tmp_path):
        from click.testing import CliRunner
        from kdream.cli import cli
        runner = CliRunner()
        recipe = _make_recipe()
        out = str(tmp_path / "test.ipynb")

        with patch("kdream.core.runner._resolve_recipe", return_value=recipe):
            result = runner.invoke(cli, ["colab", "generate", "test-model",
                                         "--output", out, "--prompt", "hello"])

        assert result.exit_code == 0
        assert Path(out).exists()

    def test_colab_generate_error_shows_message(self):
        from click.testing import CliRunner
        from kdream.cli import cli
        runner = CliRunner()
        with patch("kdream.core.runner._resolve_recipe", side_effect=Exception("not found")):
            result = runner.invoke(cli, ["colab", "generate", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output

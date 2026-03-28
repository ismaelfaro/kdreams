"""Tests for kdream.hub — HuggingFace Hub exploration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kdream.hub import (
    HF_TASK_ALIASES,
    HFModel,
    _resolve_task,
    search_hf_models,
)


# ---------------------------------------------------------------------------
# _resolve_task
# ---------------------------------------------------------------------------

class TestResolveTask:
    def test_passthrough_for_native_hf_tag(self):
        assert _resolve_task("text-to-image") == "text-to-image"

    def test_resolves_friendly_alias(self):
        assert _resolve_task("image-generation") == "text-to-image"
        assert _resolve_task("chat") == "text-generation"
        assert _resolve_task("llm") == "text-generation"
        assert _resolve_task("tts") == "text-to-speech"
        assert _resolve_task("asr") == "automatic-speech-recognition"
        assert _resolve_task("transcription") == "automatic-speech-recognition"
        assert _resolve_task("audio-generation") == "text-to-audio"
        assert _resolve_task("video-generation") == "text-to-video"
        assert _resolve_task("embedding") == "feature-extraction"

    def test_case_insensitive(self):
        assert _resolve_task("IMAGE-GENERATION") == "text-to-image"
        assert _resolve_task("Text-Generation") == "text-generation"

    def test_unknown_task_passes_through(self):
        assert _resolve_task("my-custom-task") == "my-custom-task"

    def test_all_aliases_produce_non_empty_string(self):
        for alias, canonical in HF_TASK_ALIASES.items():
            assert _resolve_task(alias) == canonical
            assert canonical  # not empty


# ---------------------------------------------------------------------------
# HFModel
# ---------------------------------------------------------------------------

class TestHFModel:
    def _model(self, **kwargs) -> HFModel:
        defaults = dict(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            author="stabilityai",
            task="text-to-image",
            tags=["diffusers", "safetensors"],
            likes=5000,
            downloads=1_000_000,
            description="SDXL base model",
            license="openrail++",
            last_modified="2024-01-15",
        )
        defaults.update(kwargs)
        return HFModel(**defaults)

    def test_hf_url(self):
        m = self._model()
        assert m.hf_url == "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"

    def test_to_hf_repo_url_equals_hf_url(self):
        m = self._model()
        assert m.to_hf_repo_url() == m.hf_url

    def test_task_display_known_task(self):
        m = self._model(task="text-to-image")
        assert m.task_display == "image-generation"

    def test_task_display_unknown_task_returns_raw(self):
        m = self._model(task="my-custom-pipeline")
        assert m.task_display == "my-custom-pipeline"

    def test_task_display_text_generation(self):
        m = self._model(task="text-generation")
        assert m.task_display == "text-generation"

    def test_default_fields(self):
        m = HFModel(model_id="org/model", author="org", task="text-to-image")
        assert m.tags == []
        assert m.likes == 0
        assert m.downloads == 0
        assert m.description == ""
        assert m.license == ""
        assert m.last_modified == ""

    def test_fields_stored_correctly(self):
        m = self._model()
        assert m.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
        assert m.author == "stabilityai"
        assert m.likes == 5000
        assert m.downloads == 1_000_000
        assert m.license == "openrail++"
        assert m.last_modified == "2024-01-15"


# ---------------------------------------------------------------------------
# search_hf_models
# ---------------------------------------------------------------------------

def _make_hf_model_stub(
    model_id: str = "org/model",
    pipeline_tag: str = "text-to-image",
    likes: int = 100,
    downloads: int = 5000,
    tags: list | None = None,
    author: str | None = None,
    card_data: dict | None = None,
    last_modified: str | None = "2024-06-01",
) -> MagicMock:
    stub = MagicMock()
    stub.modelId = model_id
    stub.pipeline_tag = pipeline_tag
    stub.likes = likes
    stub.downloads = downloads
    stub.tags = tags or ["diffusers", "pytorch"]
    stub.author = author or model_id.split("/")[0]
    stub.cardData = card_data or {"license": "apache-2.0"}
    stub.lastModified = last_modified
    return stub


class TestSearchHfModels:
    def _patched_api(self, stubs: list) -> MagicMock:
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter(stubs)
        return mock_api

    def test_returns_list_of_hf_models(self):
        stubs = [
            _make_hf_model_stub("stabilityai/sdxl", "text-to-image", likes=5000),
            _make_hf_model_stub("runwayml/sd-v1-5", "text-to-image", likes=3000),
        ]
        with patch("huggingface_hub.HfApi", return_value=self._patched_api(stubs)):
            result = search_hf_models(task="text-to-image", limit=2)

        assert len(result) == 2
        assert all(isinstance(m, HFModel) for m in result)

    def test_passes_task_to_api(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            search_hf_models(task="image-generation", limit=5)

        call_kwargs = mock_api.list_models.call_args.kwargs
        assert call_kwargs["task"] == "text-to-image"  # alias resolved

    def test_passes_query_to_api(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            search_hf_models(query="flux", limit=5)

        call_kwargs = mock_api.list_models.call_args.kwargs
        assert call_kwargs["search"] == "flux"

    def test_passes_sort_to_api(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            search_hf_models(sort="downloads", limit=5)

        call_kwargs = mock_api.list_models.call_args.kwargs
        assert call_kwargs["sort"] == "downloads"
        assert call_kwargs["direction"] == -1

    def test_passes_author_to_api(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            search_hf_models(author="stabilityai", limit=5)

        call_kwargs = mock_api.list_models.call_args.kwargs
        assert call_kwargs["author"] == "stabilityai"

    def test_passes_limit_to_api(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            search_hf_models(limit=42)

        call_kwargs = mock_api.list_models.call_args.kwargs
        assert call_kwargs["limit"] == 42

    def test_model_fields_populated_correctly(self):
        stubs = [_make_hf_model_stub(
            model_id="black-forest-labs/FLUX.1-schnell",
            pipeline_tag="text-to-image",
            likes=8000,
            downloads=500_000,
            tags=["flux", "diffusers", "safetensors"],
            card_data={"license": "apache-2.0"},
            last_modified="2024-08-01",
        )]
        with patch("huggingface_hub.HfApi", return_value=self._patched_api(stubs)):
            result = search_hf_models(limit=1)

        m = result[0]
        assert m.model_id == "black-forest-labs/FLUX.1-schnell"
        assert m.author == "black-forest-labs"
        assert m.task == "text-to-image"
        assert m.likes == 8000
        assert m.downloads == 500_000
        assert m.license == "apache-2.0"
        assert m.last_modified == "2024-08-01"

    def test_tags_truncated_to_six(self):
        stubs = [_make_hf_model_stub(
            tags=["a", "b", "c", "d", "e", "f", "g", "h"],
        )]
        with patch("huggingface_hub.HfApi", return_value=self._patched_api(stubs)):
            result = search_hf_models(limit=1)

        assert len(result[0].tags) <= 6

    def test_license_tags_stripped_from_tags(self):
        stubs = [_make_hf_model_stub(
            tags=["diffusers", "license:mit", "pytorch"],
        )]
        with patch("huggingface_hub.HfApi", return_value=self._patched_api(stubs)):
            result = search_hf_models(limit=1)

        assert "license:mit" not in result[0].tags
        assert "diffusers" in result[0].tags

    def test_empty_results_returns_empty_list(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            result = search_hf_models(task="text-to-image", limit=10)

        assert result == []

    def test_no_task_passes_none_to_api(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            search_hf_models(limit=5)

        call_kwargs = mock_api.list_models.call_args.kwargs
        assert call_kwargs["task"] is None

    def test_none_query_passes_none(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            search_hf_models(query=None, limit=5)

        call_kwargs = mock_api.list_models.call_args.kwargs
        assert call_kwargs["search"] is None

    def test_missing_huggingface_hub_raises_import_error(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                search_hf_models(limit=1)

    def test_model_with_null_fields_handled_gracefully(self):
        stub = MagicMock()
        stub.modelId = "org/model"
        stub.pipeline_tag = None
        stub.likes = None
        stub.downloads = None
        stub.tags = None
        stub.author = None
        stub.cardData = None
        stub.lastModified = None

        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([stub])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            result = search_hf_models(limit=1)

        assert len(result) == 1
        m = result[0]
        assert m.model_id == "org/model"
        assert m.task == ""
        assert m.likes == 0
        assert m.downloads == 0
        assert m.tags == []
        assert m.license == ""
        assert m.last_modified == ""


# ---------------------------------------------------------------------------
# CLI: kdream explore hf
# ---------------------------------------------------------------------------

class TestExploreCLI:
    @pytest.fixture
    def runner(self):
        from click.testing import CliRunner
        return CliRunner()

    def _mock_models(self) -> list[HFModel]:
        return [
            HFModel(
                model_id="stabilityai/sdxl",
                author="stabilityai",
                task="text-to-image",
                tags=["diffusers"],
                likes=5000,
                downloads=1_000_000,
                license="openrail++",
                last_modified="2024-01-15",
            ),
            HFModel(
                model_id="runwayml/sd-v1-5",
                author="runwayml",
                task="text-to-image",
                tags=["diffusers"],
                likes=3000,
                downloads=500_000,
                license="creativeml-openrail-m",
                last_modified="2023-12-01",
            ),
        ]

    def test_explore_hf_help(self, runner):
        from kdream.cli import cli
        result = runner.invoke(cli, ["explore", "hf", "--help"])
        assert result.exit_code == 0
        assert "--task" in result.output
        assert "--query" in result.output
        assert "--limit" in result.output
        assert "--generate" in result.output

    def test_explore_hf_displays_table(self, runner):
        from kdream.cli import cli
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter([
                _make_hf_model_stub("org/model-a", "text-to-image", likes=100),
                _make_hf_model_stub("org/model-b", "text-to-image", likes=50),
            ])
            result = runner.invoke(cli, ["explore", "hf", "--task", "text-to-image"])

        assert result.exit_code == 0
        assert "org/model-a" in result.output
        assert "org/model-b" in result.output

    def test_explore_hf_shows_tip_without_generate(self, runner):
        from kdream.cli import cli
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter([
                _make_hf_model_stub("org/model", "text-to-image"),
            ])
            result = runner.invoke(cli, ["explore", "hf"])

        assert "--generate" in result.output

    def test_explore_hf_no_results_shows_warning(self, runner):
        from kdream.cli import cli
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter([])
            result = runner.invoke(cli, ["explore", "hf", "--query", "zzznomatch"])

        assert result.exit_code == 0
        assert "No models found" in result.output

    def test_explore_hf_generate_cancel(self, runner):
        from kdream.cli import cli
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter([
                _make_hf_model_stub("org/model", "text-to-image"),
            ])
            # Input "0" cancels
            result = runner.invoke(cli, ["explore", "hf", "--generate"], input="0\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_explore_hf_generate_selects_model(self, runner):
        from kdream.cli import cli
        from kdream.core.recipe import parse_yaml_recipe

        mock_recipe_yaml = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: sdxl
  version: 1.0.0
  description: SDXL
  tags: [image-generation]
  license: openrail++
  author: stabilityai
source:
  repo: https://huggingface.co/stabilityai/sdxl
  ref: main
entrypoint:
  script: run.py
  type: python
"""
        mock_recipe = parse_yaml_recipe(mock_recipe_yaml)
        mock_recipe._runner_script = None

        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter([
                _make_hf_model_stub("stabilityai/sdxl", "text-to-image"),
            ])
            with patch("kdream.generate_recipe", return_value=mock_recipe):
                # Input "1" selects the first model
                result = runner.invoke(
                    cli, ["explore", "hf", "--generate", "--output", "/tmp/test-recipe.yaml"],
                    input="1\n",
                )

        assert result.exit_code == 0
        assert "sdxl" in result.output.lower()

    def test_explore_hf_with_author_filter(self, runner):
        from kdream.cli import cli
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter([
                _make_hf_model_stub("stabilityai/sdxl", "text-to-image", author="stabilityai"),
            ])
            result = runner.invoke(cli, ["explore", "hf", "--author", "stabilityai"])

        assert result.exit_code == 0
        assert "stabilityai/sdxl" in result.output
        # Verify author was passed to API
        call_kwargs = MockApi.return_value.list_models.call_args.kwargs
        assert call_kwargs["author"] == "stabilityai"

    def test_explore_hf_sort_by_downloads(self, runner):
        from kdream.cli import cli
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter([
                _make_hf_model_stub("org/model", "text-to-image", downloads=999_000),
            ])
            result = runner.invoke(cli, ["explore", "hf", "--sort", "downloads"])

        assert result.exit_code == 0
        call_kwargs = MockApi.return_value.list_models.call_args.kwargs
        assert call_kwargs["sort"] == "downloads"

    def test_explore_hf_error_exits_with_code_1(self, runner):
        from kdream.cli import cli
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.list_models.side_effect = Exception("network error")
            result = runner.invoke(cli, ["explore", "hf"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_explore_group_help(self, runner):
        from kdream.cli import cli
        result = runner.invoke(cli, ["explore", "--help"])
        assert result.exit_code == 0
        assert "hf" in result.output

"""Tests for kdream.core.registry — RegistryClient."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kdream.core.registry import CACHE_TTL_SECONDS, RegistryClient
from kdream.exceptions import RegistryError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_RECIPE_YAML = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: flux-schnell
  version: 1.0.0
  description: Fast image generation model
  tags: [image-generation, flux]
  license: Apache-2.0
  author: black-forest-labs
source:
  repo: https://github.com/test/flux-schnell
  ref: main
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
    required: true
    description: Text prompt
outputs:
  - name: image
    type: file
    path: outputs/result.png
"""

MINIMAL_RECIPE_YAML_2 = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: whisper-transcribe
  version: 1.0.0
  description: Audio transcription with Whisper
  tags: [audio, transcription]
  license: MIT
  author: openai
source:
  repo: https://github.com/test/whisper
  ref: main
entrypoint:
  script: transcribe.py
  type: python
inputs:
  audio_file:
    type: string
    required: true
    description: Path to audio file
outputs:
  - name: transcript
    type: string
"""


def _make_metadata_dict(name: str = "flux-schnell", tags: list | None = None) -> dict:
    return {
        "name": name,
        "version": "1.0.0",
        "description": "A test recipe",
        "tags": tags or ["image-generation"],
        "license": "Apache-2.0",
        "author": "test",
    }


# ---------------------------------------------------------------------------
# _is_fresh
# ---------------------------------------------------------------------------

class TestIsFresh:
    def test_fresh_file_returns_true(self, tmp_path):
        f = tmp_path / "file.json"
        f.write_text("{}")
        assert RegistryClient._is_fresh(f) is True

    def test_stale_file_returns_false(self, tmp_path):
        f = tmp_path / "file.json"
        f.write_text("{}")
        # backdate modification time beyond TTL
        old_time = time.time() - CACHE_TTL_SECONDS - 1
        import os
        os.utime(f, (old_time, old_time))
        assert RegistryClient._is_fresh(f) is False


# ---------------------------------------------------------------------------
# fetch_recipe — local path
# ---------------------------------------------------------------------------

class TestFetchRecipeLocalPath:
    def test_fetch_local_dot_path(self, tmp_path):
        recipe_file = tmp_path / "my-recipe.yaml"
        recipe_file.write_text(MINIMAL_RECIPE_YAML)

        client = RegistryClient(cache_dir=tmp_path / "cache")
        recipe = client.fetch_recipe(str(recipe_file))
        assert recipe.metadata.name == "flux-schnell"

    def test_fetch_local_slash_path(self, tmp_path):
        recipe_file = tmp_path / "my-recipe.yaml"
        recipe_file.write_text(MINIMAL_RECIPE_YAML)

        client = RegistryClient(cache_dir=tmp_path / "cache")
        recipe = client.fetch_recipe(str(recipe_file))
        assert recipe.metadata.name == "flux-schnell"


# ---------------------------------------------------------------------------
# fetch_recipe — cache hit
# ---------------------------------------------------------------------------

class TestFetchRecipeCacheHit:
    def test_uses_fresh_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = cache_dir / "flux-schnell.yaml"
        cached.write_text(MINIMAL_RECIPE_YAML)

        client = RegistryClient(cache_dir=cache_dir)
        # With a fresh cache file the client should never hit the network
        with patch("httpx.get") as mock_get:
            recipe = client.fetch_recipe("flux-schnell")
            mock_get.assert_not_called()

        assert recipe.metadata.name == "flux-schnell"

    def test_ignores_stale_cache_and_fetches_network(self, tmp_path):
        import os

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = cache_dir / "flux-schnell.yaml"
        cached.write_text(MINIMAL_RECIPE_YAML)
        old_time = time.time() - CACHE_TTL_SECONDS - 1
        os.utime(cached, (old_time, old_time))

        client = RegistryClient(cache_dir=cache_dir)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MINIMAL_RECIPE_YAML

        with patch("httpx.get", return_value=mock_resp):
            recipe = client.fetch_recipe("flux-schnell")

        assert recipe.metadata.name == "flux-schnell"


# ---------------------------------------------------------------------------
# fetch_recipe — GitHub fetch
# ---------------------------------------------------------------------------

class TestFetchRecipeGitHub:
    def test_fetches_from_github_and_caches(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = RegistryClient(cache_dir=cache_dir)

        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.text = MINIMAL_RECIPE_YAML

        mock_resp_404 = MagicMock()
        mock_resp_404.status_code = 404

        def side_effect(url, **kwargs):
            if "image-generation" in url:
                return mock_resp_ok
            return mock_resp_404

        with patch("httpx.get", side_effect=side_effect):
            recipe = client.fetch_recipe("flux-schnell")

        assert recipe.metadata.name == "flux-schnell"
        # Should have written to cache
        assert (cache_dir / "flux-schnell.yaml").exists()

    def test_network_error_falls_back_to_bundled(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = RegistryClient(cache_dir=cache_dir)

        import httpx

        with patch("httpx.get", side_effect=httpx.RequestError("timeout")):
            # No bundled recipe with this name → RegistryError
            with pytest.raises(RegistryError):
                client.fetch_recipe("nonexistent-recipe-xyz")

    def test_all_categories_404_falls_back_to_bundled_then_raises(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = RegistryClient(cache_dir=cache_dir)

        mock_resp_404 = MagicMock()
        mock_resp_404.status_code = 404

        with patch("httpx.get", return_value=mock_resp_404):
            with pytest.raises(RegistryError, match="not found in registry"):
                client.fetch_recipe("definitely-not-a-real-recipe")


# ---------------------------------------------------------------------------
# fetch_recipe — stale cache fallback
# ---------------------------------------------------------------------------

class TestFetchRecipeStaleCacheFallback:
    def test_stale_cache_used_when_network_and_bundled_unavailable(self, tmp_path):
        import os
        import httpx

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = cache_dir / "flux-schnell.yaml"
        cached.write_text(MINIMAL_RECIPE_YAML)
        old_time = time.time() - CACHE_TTL_SECONDS - 1
        os.utime(cached, (old_time, old_time))

        client = RegistryClient(cache_dir=cache_dir)

        # All network calls fail
        with patch("httpx.get", side_effect=httpx.RequestError("offline")):
            with patch.object(client, "_find_bundled", return_value=None):
                recipe = client.fetch_recipe("flux-schnell")

        assert recipe.metadata.name == "flux-schnell"


# ---------------------------------------------------------------------------
# _find_bundled
# ---------------------------------------------------------------------------

class TestFindBundled:
    def test_returns_none_when_bundled_dir_missing(self, tmp_path):
        client = RegistryClient(cache_dir=tmp_path / "cache")
        with patch("kdream.core.registry._BUNDLED_RECIPES_DIR", tmp_path / "nonexistent"):
            result = client._find_bundled("flux-schnell")
        assert result is None

    def test_finds_recipe_in_bundled_dir(self, tmp_path):
        bundled_dir = tmp_path / "recipes" / "image-generation"
        bundled_dir.mkdir(parents=True)
        (bundled_dir / "flux-schnell.yaml").write_text(MINIMAL_RECIPE_YAML)

        client = RegistryClient(cache_dir=tmp_path / "cache")
        with patch("kdream.core.registry._BUNDLED_RECIPES_DIR", tmp_path / "recipes"):
            result = client._find_bundled("flux-schnell")

        assert result is not None
        assert result.name == "flux-schnell.yaml"

    def test_returns_none_when_recipe_not_in_bundled(self, tmp_path):
        bundled_dir = tmp_path / "recipes"
        bundled_dir.mkdir()

        client = RegistryClient(cache_dir=tmp_path / "cache")
        with patch("kdream.core.registry._BUNDLED_RECIPES_DIR", bundled_dir):
            result = client._find_bundled("no-such-recipe")

        assert result is None


# ---------------------------------------------------------------------------
# _load_bundled_metadata
# ---------------------------------------------------------------------------

class TestLoadBundledMetadata:
    def test_loads_all_bundled_yaml_files(self, tmp_path):
        bundled_dir = tmp_path / "recipes"
        cat = bundled_dir / "image-generation"
        cat.mkdir(parents=True)
        (cat / "flux-schnell.yaml").write_text(MINIMAL_RECIPE_YAML)
        (cat / "whisper.yaml").write_text(MINIMAL_RECIPE_YAML_2)

        client = RegistryClient(cache_dir=tmp_path / "cache")
        with patch("kdream.core.registry._BUNDLED_RECIPES_DIR", bundled_dir):
            metadata = client._load_bundled_metadata()

        names = [m.name for m in metadata]
        assert "flux-schnell" in names
        assert "whisper-transcribe" in names

    def test_returns_empty_when_bundled_dir_missing(self, tmp_path):
        client = RegistryClient(cache_dir=tmp_path / "cache")
        with patch("kdream.core.registry._BUNDLED_RECIPES_DIR", tmp_path / "nonexistent"):
            result = client._load_bundled_metadata()
        assert result == []

    def test_skips_invalid_yaml_files(self, tmp_path):
        bundled_dir = tmp_path / "recipes"
        bundled_dir.mkdir()
        (bundled_dir / "broken.yaml").write_text("not: valid: yaml: [[[")
        (bundled_dir / "good.yaml").write_text(MINIMAL_RECIPE_YAML)

        client = RegistryClient(cache_dir=tmp_path / "cache")
        with patch("kdream.core.registry._BUNDLED_RECIPES_DIR", bundled_dir):
            result = client._load_bundled_metadata()

        # Only the good file should be loaded; broken one silently skipped
        assert len(result) == 1
        assert result[0].name == "flux-schnell"


# ---------------------------------------------------------------------------
# _get_all_metadata — index cache
# ---------------------------------------------------------------------------

class TestGetAllMetadata:
    def test_uses_fresh_index_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        meta = _make_metadata_dict()
        index = cache_dir / "_index.json"
        index.write_text(json.dumps([meta]))

        client = RegistryClient(cache_dir=cache_dir)
        with patch.object(client, "_fetch_index_from_github") as mock_fetch:
            result = client._get_all_metadata()
            mock_fetch.assert_not_called()

        assert len(result) == 1
        assert result[0].name == "flux-schnell"

    def test_fetches_when_no_index_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        from kdream.core.recipe import RecipeMetadata
        mock_meta = RecipeMetadata(**_make_metadata_dict())

        client = RegistryClient(cache_dir=cache_dir)
        with patch.object(client, "_fetch_index_from_github", return_value=[mock_meta]) as mock_fetch:
            result = client._get_all_metadata()
            mock_fetch.assert_called_once()

        assert result[0].name == "flux-schnell"

    def test_falls_back_to_stale_index_on_network_error(self, tmp_path):
        import os

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        meta = _make_metadata_dict()
        index = cache_dir / "_index.json"
        index.write_text(json.dumps([meta]))
        old_time = time.time() - CACHE_TTL_SECONDS - 1
        os.utime(index, (old_time, old_time))

        client = RegistryClient(cache_dir=cache_dir)
        with patch.object(client, "_fetch_index_from_github", side_effect=RegistryError("offline")):
            with patch.object(client, "_load_bundled_metadata", return_value=[]):
                result = client._get_all_metadata()

        assert len(result) == 1
        assert result[0].name == "flux-schnell"

    def test_falls_back_to_bundled_when_everything_fails(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        from kdream.core.recipe import RecipeMetadata
        bundled_meta = RecipeMetadata(**_make_metadata_dict())

        client = RegistryClient(cache_dir=cache_dir)
        with patch.object(client, "_fetch_index_from_github", side_effect=RegistryError("offline")):
            with patch.object(client, "_load_bundled_metadata", return_value=[bundled_meta]):
                result = client._get_all_metadata()

        assert result[0].name == "flux-schnell"

    def test_raises_when_all_sources_exhausted(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        client = RegistryClient(cache_dir=cache_dir)
        with patch.object(client, "_fetch_index_from_github", side_effect=RegistryError("offline")):
            with patch.object(client, "_load_bundled_metadata", return_value=[]):
                with pytest.raises(RegistryError, match="Could not reach"):
                    client._get_all_metadata()


# ---------------------------------------------------------------------------
# list_recipes — filtering
# ---------------------------------------------------------------------------

class TestListRecipes:
    def _client_with_metadata(self, tmp_path, metadata_list):
        from kdream.core.recipe import RecipeMetadata
        client = RegistryClient(cache_dir=tmp_path / "cache")
        metas = [RecipeMetadata(**m) for m in metadata_list]
        with patch.object(client, "_get_all_metadata", return_value=metas):
            return client, metas

    def test_returns_all_when_no_filters(self, tmp_path):
        client = RegistryClient(cache_dir=tmp_path / "cache")
        from kdream.core.recipe import RecipeMetadata
        metas = [
            RecipeMetadata(**_make_metadata_dict("recipe-a", ["image-generation"])),
            RecipeMetadata(**_make_metadata_dict("recipe-b", ["audio"])),
        ]
        with patch.object(client, "_get_all_metadata", return_value=metas):
            result = client.list_recipes()
        assert len(result) == 2

    def test_filters_by_single_tag(self, tmp_path):
        client = RegistryClient(cache_dir=tmp_path / "cache")
        from kdream.core.recipe import RecipeMetadata
        metas = [
            RecipeMetadata(**_make_metadata_dict("recipe-a", ["image-generation"])),
            RecipeMetadata(**_make_metadata_dict("recipe-b", ["audio"])),
        ]
        with patch.object(client, "_get_all_metadata", return_value=metas):
            result = client.list_recipes(tags=["audio"])
        assert len(result) == 1
        assert result[0].name == "recipe-b"

    def test_filters_by_multiple_tags(self, tmp_path):
        client = RegistryClient(cache_dir=tmp_path / "cache")
        from kdream.core.recipe import RecipeMetadata
        metas = [
            RecipeMetadata(**_make_metadata_dict("recipe-a", ["image-generation", "flux"])),
            RecipeMetadata(**_make_metadata_dict("recipe-b", ["audio"])),
            RecipeMetadata(**_make_metadata_dict("recipe-c", ["text-generation"])),
        ]
        with patch.object(client, "_get_all_metadata", return_value=metas):
            result = client.list_recipes(tags=["image-generation", "text-generation"])
        names = [r.name for r in result]
        assert "recipe-a" in names
        assert "recipe-c" in names
        assert "recipe-b" not in names

    def test_no_match_returns_empty(self, tmp_path):
        client = RegistryClient(cache_dir=tmp_path / "cache")
        from kdream.core.recipe import RecipeMetadata
        metas = [RecipeMetadata(**_make_metadata_dict("recipe-a", ["image-generation"]))]
        with patch.object(client, "_get_all_metadata", return_value=metas):
            result = client.list_recipes(tags=["video"])
        assert result == []


# ---------------------------------------------------------------------------
# search_recipes
# ---------------------------------------------------------------------------

class TestSearchRecipes:
    def _make_client(self, tmp_path, metadata_list):
        from kdream.core.recipe import RecipeMetadata
        client = RegistryClient(cache_dir=tmp_path / "cache")
        metas = [RecipeMetadata(**m) for m in metadata_list]
        client._get_all_metadata = lambda: metas
        return client

    def test_search_by_name(self, tmp_path):
        client = self._make_client(tmp_path, [
            _make_metadata_dict("flux-schnell", ["image-generation"]),
            _make_metadata_dict("whisper-transcribe", ["audio"]),
        ])
        result = client.search_recipes("flux")
        assert len(result) == 1
        assert result[0].name == "flux-schnell"

    def test_search_by_description(self, tmp_path):
        meta = _make_metadata_dict("my-model", ["image-generation"])
        meta["description"] = "Super fast diffusion model"
        client = self._make_client(tmp_path, [meta])
        result = client.search_recipes("diffusion")
        assert len(result) == 1

    def test_search_by_tag(self, tmp_path):
        client = self._make_client(tmp_path, [
            _make_metadata_dict("recipe-a", ["image-generation"]),
            _make_metadata_dict("recipe-b", ["audio"]),
        ])
        result = client.search_recipes("audio")
        assert len(result) == 1
        assert result[0].name == "recipe-b"

    def test_search_case_insensitive(self, tmp_path):
        meta = _make_metadata_dict("flux-schnell", ["image-generation"])
        meta["description"] = "Fast FLUX image model"
        client = self._make_client(tmp_path, [meta])
        result = client.search_recipes("FLUX")
        assert len(result) == 1

    def test_search_no_match_returns_empty(self, tmp_path):
        client = self._make_client(tmp_path, [
            _make_metadata_dict("flux-schnell", ["image-generation"]),
        ])
        result = client.search_recipes("zzznomatch")
        assert result == []


# ---------------------------------------------------------------------------
# _fetch_index_from_github
# ---------------------------------------------------------------------------

class TestFetchIndexFromGitHub:
    def test_api_error_raises_registry_error(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = RegistryClient(cache_dir=cache_dir)

        mock_resp = MagicMock()
        mock_resp.status_code = 403

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(RegistryError, match="Registry API returned 403"):
                client._fetch_index_from_github(cache_dir / "_index.json")

    def test_network_error_raises_registry_error(self, tmp_path):
        import httpx

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = RegistryClient(cache_dir=cache_dir)

        with patch("httpx.get", side_effect=httpx.RequestError("connection refused")):
            with pytest.raises(RegistryError, match="Network error"):
                client._fetch_index_from_github(cache_dir / "_index.json")

    def test_successful_fetch_writes_index(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = RegistryClient(cache_dir=cache_dir)

        # API response: one category dir
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.json.return_value = [
            {"type": "dir", "url": "https://api.github.com/repos/.../image-generation"}
        ]

        # Category listing: one YAML file
        cat_resp = MagicMock()
        cat_resp.status_code = 200
        cat_resp.json.return_value = [
            {"name": "flux-schnell.yaml", "download_url": "https://raw.../flux-schnell.yaml"}
        ]

        # YAML content
        yaml_resp = MagicMock()
        yaml_resp.status_code = 200
        yaml_resp.text = MINIMAL_RECIPE_YAML

        def side_effect(url, **kwargs):
            if "api.github.com/repos" in url and "image-generation" not in url:
                return api_resp
            if "image-generation" in url and "raw" not in url:
                return cat_resp
            return yaml_resp

        with patch("httpx.get", side_effect=side_effect):
            result = client._fetch_index_from_github(cache_dir / "_index.json")

        assert len(result) >= 1
        assert (cache_dir / "_index.json").exists()

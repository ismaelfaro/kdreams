"""Registry client for fetching kdream recipes from the public GitHub registry."""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import yaml

from kdream.core.recipe import Recipe, RecipeMetadata, parse_yaml_recipe
from kdream.exceptions import RegistryError

REGISTRY_BASE_URL = (
    "https://raw.githubusercontent.com/ismaelfaro/kdreams/main/recipes"
)
REGISTRY_API_URL = (
    "https://api.github.com/repos/ismaelfaro/kdreams/contents/recipes"
)
LOCAL_CACHE = Path.home() / ".kdream" / "registry_cache"
CACHE_TTL_SECONDS = 3600  # 1 hour

# Bundled recipes shipped with the package (kdream/recipes/ inside the package)
_BUNDLED_RECIPES_DIR = Path(__file__).parent.parent / "recipes"


class RegistryClient:
    """Client for the kdream public recipe registry."""

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir or LOCAL_CACHE
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_recipes(
        self,
        tags: list[str] | None = None,
        backend: str | None = None,
    ) -> list[RecipeMetadata]:
        """List available recipes, optionally filtered by tags or backend."""
        all_recipes = self._get_all_metadata()

        if tags:
            all_recipes = [
                r for r in all_recipes
                if any(t in r.tags for t in tags)
            ]
        if backend:
            # backend filtering — recipes without backend specs pass through
            pass

        return all_recipes

    def fetch_recipe(self, name: str) -> Recipe:
        """Fetch a recipe by name — checks cache, then GitHub, then bundled package recipes."""
        # Support local paths
        if name.startswith(".") or name.startswith("/"):
            from kdream.core.recipe import load_recipe
            return load_recipe(name)

        cached = self._cache_dir / f"{name}.yaml"

        # 1. Check local download cache (fresh)
        if cached.exists() and self._is_fresh(cached):
            return parse_yaml_recipe(cached.read_text())

        # 2. Try to fetch from GitHub (online first)
        categories = [
            "image-generation", "text-generation", "audio",
            "video-generation", "3d", "multimodal",
        ]
        for cat in categories:
            url = f"{REGISTRY_BASE_URL}/{cat}/{name}.yaml"
            try:
                resp = httpx.get(url, timeout=10, follow_redirects=True)
                if resp.status_code == 200:
                    content = resp.text
                    cached.write_text(content)
                    return parse_yaml_recipe(content)
            except httpx.RequestError:
                continue

        # 3. Fall back to bundled recipes shipped with the package
        bundled = self._find_bundled(name)
        if bundled:
            return parse_yaml_recipe(bundled.read_text())

        # 4. Stale cache is better than nothing
        if cached.exists():
            return parse_yaml_recipe(cached.read_text())

        raise RegistryError(
            f"Recipe '{name}' not found in registry. "
            "Run `kdream list` to see available recipes, "
            "or provide a local file path."
        )

    def search_recipes(self, query: str) -> list[RecipeMetadata]:
        """Fuzzy text search across name, description, and tags."""
        query_lower = query.lower()
        all_recipes = self._get_all_metadata()
        return [
            r for r in all_recipes
            if (
                query_lower in r.name.lower()
                or query_lower in r.description.lower()
                or any(query_lower in t.lower() for t in r.tags)
            )
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_all_metadata(self) -> list[RecipeMetadata]:
        index_file = self._cache_dir / "_index.json"
        if index_file.exists() and self._is_fresh(index_file):
            try:
                data = json.loads(index_file.read_text())
                return [RecipeMetadata(**item) for item in data]
            except Exception:
                pass

        try:
            return self._fetch_index_from_github(index_file)
        except Exception:
            pass

        if index_file.exists():
            try:
                data = json.loads(index_file.read_text())
                return [RecipeMetadata(**item) for item in data]
            except Exception:
                pass

        # Fall back to bundled recipes shipped with the package
        bundled = self._load_bundled_metadata()
        if bundled:
            return bundled

        raise RegistryError(
            "Could not reach the recipe registry and no local cache found. "
            "Check your internet connection."
        )

    def _find_bundled(self, name: str) -> Path | None:
        """Search the bundled recipes/ directory for a recipe by name."""
        if not _BUNDLED_RECIPES_DIR.exists():
            return None
        for yaml_file in _BUNDLED_RECIPES_DIR.rglob(f"{name}.yaml"):
            return yaml_file
        return None

    def _load_bundled_metadata(self) -> list[RecipeMetadata]:
        """Parse all bundled YAML recipes and return their metadata."""
        recipes: list[RecipeMetadata] = []
        if not _BUNDLED_RECIPES_DIR.exists():
            return recipes
        for yaml_file in sorted(_BUNDLED_RECIPES_DIR.rglob("*.yaml")):
            try:
                recipe = parse_yaml_recipe(yaml_file.read_text())
                meta = recipe.metadata.model_copy(update={"repo": recipe.source.repo})
                recipes.append(meta)
            except Exception:
                pass
        return recipes

    def _fetch_index_from_github(self, index_file: Path) -> list[RecipeMetadata]:
        recipes: list[RecipeMetadata] = []

        try:
            resp = httpx.get(REGISTRY_API_URL, timeout=10, follow_redirects=True)
            if resp.status_code != 200:
                raise RegistryError(f"Registry API returned {resp.status_code}")

            categories = resp.json()
            for cat in categories:
                if cat.get("type") != "dir":
                    continue
                cat_url = cat["url"]
                cat_resp = httpx.get(cat_url, timeout=10, follow_redirects=True)
                if cat_resp.status_code != 200:
                    continue
                for item in cat_resp.json():
                    if item.get("name", "").endswith(".yaml"):
                        raw_url = item["download_url"]
                        try:
                            yaml_resp = httpx.get(raw_url, timeout=15, follow_redirects=True)
                            if yaml_resp.status_code == 200:
                                recipe = parse_yaml_recipe(yaml_resp.text)
                                meta = recipe.metadata.model_copy(
                                    update={"repo": recipe.source.repo}
                                )
                                recipes.append(meta)
                        except Exception:
                            pass
        except httpx.RequestError as e:
            raise RegistryError(f"Network error fetching registry: {e}") from e

        if recipes:
            index_file.write_text(json.dumps([r.model_dump() for r in recipes]))

        return recipes

    @staticmethod
    def _is_fresh(path: Path) -> bool:
        return (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS

"""Google Colab backend — Phase 2 (not yet implemented)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from kdream.backends.base import AbstractBackend
from kdream.core.recipe import Recipe
from kdream.exceptions import BackendError

_NOT_IMPLEMENTED_MSG = (
    "The Colab backend is not yet implemented (planned for Phase 2). "
    "Use --backend local instead."
)


class ColabBackend(AbstractBackend):
    """Google Colab compute backend (Phase 2 — coming soon)."""

    name = "colab"

    def install(self, recipe: Recipe, cache_dir: Path, force_reinstall: bool = False):
        raise BackendError(_NOT_IMPLEMENTED_MSG)

    def run(self, package: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        raise BackendError(_NOT_IMPLEMENTED_MSG)

    def is_installed(self, recipe_name: str, cache_dir: Path) -> bool:
        return False

"""Abstract base class for kdream compute backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from kdream.core.recipe import Recipe


class AbstractBackend(ABC):
    """Base class for all kdream compute backends."""

    #: Short identifier used in recipe ``backends:`` blocks and CLI ``--backend`` flag.
    name: str = "abstract"

    @abstractmethod
    def install(
        self,
        recipe: Recipe,
        cache_dir: Path,
        force_reinstall: bool = False,
    ):
        """Install a recipe package: clone repo, create venv, download models.

        Args:
            recipe:          Parsed recipe to install.
            cache_dir:       Root cache directory (``~/.kdream/cache`` by default).
            force_reinstall: If True, re-install even if already cached.

        Returns:
            :class:`~kdream.core.runner.PackageInfo`
        """

    @abstractmethod
    def run(self, package, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute inference with the given inputs.

        Args:
            package: :class:`~kdream.core.runner.PackageInfo` from :meth:`install`.
            inputs:  Dict of recipe input values.

        Returns:
            Dict mapping output names to values (file paths, strings, etc.).
        """

    @abstractmethod
    def is_installed(self, recipe_name: str, cache_dir: Path) -> bool:
        """Return True if the recipe is already installed and ready to run."""

    def validate_inputs(self, recipe: Recipe, inputs: dict[str, Any]) -> list[str]:
        """Validate *inputs* against *recipe*'s schema.

        Returns:
            List of human-readable error strings; empty means valid.
        """
        errors: list[str] = []
        for name, spec in recipe.inputs.items():
            if spec.required and name not in inputs:
                errors.append(f"Required input '{name}' is missing.")
                continue
            if name in inputs:
                val = inputs[name]
                if spec.min is not None and val < spec.min:
                    errors.append(
                        f"Input '{name}' value {val} is below minimum {spec.min}."
                    )
                if spec.max is not None and val > spec.max:
                    errors.append(
                        f"Input '{name}' value {val} exceeds maximum {spec.max}."
                    )
        return errors

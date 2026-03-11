"""Backend orchestrator — resolves recipes and dispatches to backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kdream.core.recipe import Recipe, load_recipe
from kdream.core.registry import RegistryClient
from kdream.exceptions import BackendError, RecipeError

DEFAULT_CACHE_DIR = Path.home() / ".kdream" / "cache"


@dataclass
class RunResult:
    """Result returned by kdream.run()."""
    outputs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


@dataclass
class PackageInfo:
    """Represents an installed kdream package."""
    recipe_name: str
    path: Path
    ready: bool
    venv_path: Path
    repo_path: Path
    models_path: Path


def _resolve_recipe(recipe_name_or_path: str) -> Recipe:
    """Resolve a recipe name (registry) or local file path to a Recipe object."""
    if recipe_name_or_path.startswith(".") or recipe_name_or_path.startswith("/"):
        return load_recipe(recipe_name_or_path)
    if Path(recipe_name_or_path).exists():
        return load_recipe(recipe_name_or_path)
    registry = RegistryClient()
    return registry.fetch_recipe(recipe_name_or_path)


def _get_cache_dir(cache_dir: str | None) -> Path:
    path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(
    recipe: str,
    backend: str = "local",
    cache_dir: str | None = None,
    force_reinstall: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> RunResult:
    """
    Resolve a recipe, install the package (if needed), and run inference.

    Args:
        recipe:          Registry name or local file path to a recipe.
        backend:         Compute backend — ``"local"``, ``"colab"``, ``"runpod"``.
        cache_dir:       Override the default cache directory (``~/.kdream/cache``).
        force_reinstall: Force re-clone and re-install even if the package is cached.
        verbose:         Show detailed subprocess output during install and inference.
        **kwargs:        Recipe inputs (e.g. ``prompt="..."``, ``steps=30``).

    Returns:
        :class:`RunResult` with ``outputs`` dict and ``metadata``.
    """
    import time

    from kdream.backends import get_backend

    resolved = _resolve_recipe(recipe)
    cache = _get_cache_dir(cache_dir)
    be = get_backend(backend, verbose=verbose)

    t0 = time.time()
    pkg = be.install(resolved, cache, force_reinstall=force_reinstall)
    outputs = be.run(pkg, kwargs)
    duration = time.time() - t0

    return RunResult(
        outputs=outputs,
        metadata={
            "backend": backend,
            "recipe": resolved.metadata.name,
            "duration_s": duration,
        },
        success=True,
    )


def install(
    recipe: str,
    backend: str = "local",
    cache_dir: str | None = None,
    verbose: bool = False,
) -> PackageInfo:
    """
    Pre-install a recipe package without running inference.

    Clones the repo, creates a UV virtual environment, and downloads model weights.

    Args:
        recipe:    Registry name or local file path to a recipe.
        backend:   Compute backend.
        cache_dir: Override the default cache directory.
        verbose:   Show detailed subprocess output during install.

    Returns:
        :class:`PackageInfo` describing the installed package.
    """
    from kdream.backends import get_backend

    resolved = _resolve_recipe(recipe)
    cache = _get_cache_dir(cache_dir)
    be = get_backend(backend, verbose=verbose)
    return be.install(resolved, cache)


def list_installed(cache_dir: str | None = None) -> list[PackageInfo]:
    """
    List all installed kdream packages.

    Args:
        cache_dir: Cache directory to inspect (default: ``~/.kdream/cache``).

    Returns:
        List of :class:`PackageInfo` for each installed package.
    """
    cache = _get_cache_dir(cache_dir)
    packages: list[PackageInfo] = []

    for pkg_dir in cache.iterdir():
        if not pkg_dir.is_dir():
            continue
        repo_path = pkg_dir / "repo"
        venv_path = pkg_dir / "venv"
        models_path = pkg_dir / "models"
        ready = repo_path.exists() and venv_path.exists()
        packages.append(PackageInfo(
            recipe_name=pkg_dir.name,
            path=pkg_dir,
            ready=ready,
            venv_path=venv_path,
            repo_path=repo_path,
            models_path=models_path,
        ))

    return packages

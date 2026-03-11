"""kdream — Universal AI Model Runtime & Recipe Platform.

Quick start::

    import kdream

    result = kdream.run(
        recipe="stable-diffusion-xl-base",
        prompt="a cyberpunk city at sunset",
        steps=40,
    )
    print(result.outputs["image"])  # /path/to/output.png
"""

from kdream.core.recipe import Recipe, load_recipe, validate_recipe
from kdream.core.runner import RunResult, PackageInfo, run, install, list_installed
from kdream.backends.local import detect_accelerator
from kdream.exceptions import KdreamError, RecipeError, RegistryError, BackendError

__version__ = "0.10.4"


def list_recipes(
    tags: list[str] | None = None,
    backend: str | None = None,
):
    """List available recipes from the public registry.

    Args:
        tags:    Filter by tag (e.g. ``["image-generation"]``).
        backend: Filter by backend compatibility.

    Returns:
        List of :class:`~kdream.core.recipe.RecipeMetadata` objects.
    """
    from kdream.core.registry import RegistryClient
    return RegistryClient().list_recipes(tags=tags, backend=backend)


def generate_recipe(
    repo: str,
    output: str | None = None,
    publish: bool = False,
) -> Recipe:
    """Generate a kdream recipe from a GitHub repository using AI agents.

    Requires ``ANTHROPIC_API_KEY`` environment variable.

    Args:
        repo:    GitHub repository URL to analyse.
        output:  Optional path to write the generated YAML recipe.
        publish: Open a PR to the public registry (Phase 1: not yet implemented).

    Returns:
        Generated :class:`~kdream.core.recipe.Recipe`.
    """
    from kdream.agents.recipe_generator import RecipeGeneratorAgent
    return RecipeGeneratorAgent().generate(repo=repo, output=output, publish=publish)


__all__ = [
    "run",
    "install",
    "list_recipes",
    "list_installed",
    "generate_recipe",
    "detect_accelerator",
    "load_recipe",
    "validate_recipe",
    "Recipe",
    "RunResult",
    "PackageInfo",
    "KdreamError",
    "RecipeError",
    "RegistryError",
    "BackendError",
    "__version__",
]

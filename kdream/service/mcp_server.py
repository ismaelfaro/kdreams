"""FastMCP server exposing kdream tools over the Model Context Protocol."""
from __future__ import annotations

from typing import Any, NoReturn

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

import kdream as k
from kdream.exceptions import KdreamError


def _tool_error(exc: KdreamError) -> NoReturn:
    """Re-raise a KdreamError as an MCP ToolError."""
    raise ToolError(str(exc)) from exc


def create_mcp_server(host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    """Build and return a configured FastMCP instance with all kdream tools."""

    mcp = FastMCP(
        name="kdream",
        instructions=(
            "kdream — Universal AI Model Runtime & Recipe Platform. "
            "Use these tools to run, install, list, generate, and validate AI model recipes."
        ),
        host=host,
        port=port,
    )

    # ── run_recipe ────────────────────────────────────────────────────────────

    @mcp.tool()
    def run_recipe(
        recipe: str,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        guidance_scale: float | None = None,
        output_dir: str | None = None,
        backend: str = "local",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run inference using a kdream recipe.

        Args:
            recipe:          Registry name or local file path to a recipe.
            prompt:          Text prompt for generation.
            negative_prompt: Negative text prompt.
            steps:           Number of inference steps.
            seed:            Random seed (-1 for random).
            width:           Output image width in pixels.
            height:          Output image height in pixels.
            guidance_scale:  CFG guidance scale.
            output_dir:      Directory to write output files into.
            backend:         Compute backend (local|colab|runpod).
            verbose:         Show detailed subprocess output.
        """
        kwargs: dict[str, Any] = {}
        for key, val in [
            ("prompt", prompt), ("negative_prompt", negative_prompt),
            ("steps", steps), ("seed", seed), ("width", width),
            ("height", height), ("guidance_scale", guidance_scale),
            ("output_dir", output_dir),
        ]:
            if val is not None:
                kwargs[key] = val

        try:
            result = k.run(recipe=recipe, backend=backend, verbose=verbose, **kwargs)
        except KdreamError as exc:
            _tool_error(exc)

        return {
            "success": result.success,
            "outputs": result.outputs,
            "metadata": result.metadata,
            "error": result.error,
        }

    # ── install_recipe ────────────────────────────────────────────────────────

    @mcp.tool()
    def install_recipe(
        recipe: str,
        backend: str = "local",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Pre-install a kdream recipe package without running inference.

        Args:
            recipe:  Registry name or local file path to a recipe.
            backend: Compute backend (local|colab|runpod).
            verbose: Show detailed subprocess output.
        """
        try:
            pkg = k.install(recipe=recipe, backend=backend, verbose=verbose)
        except KdreamError as exc:
            _tool_error(exc)

        return {
            "recipe_name": pkg.recipe_name,
            "path": str(pkg.path),
            "ready": pkg.ready,
        }

    # ── list_recipes ──────────────────────────────────────────────────────────

    @mcp.tool()
    def list_recipes(
        tags: list[str] | None = None,
        backend: str | None = None,
    ) -> list[dict[str, Any]]:
        """List available recipes from the kdream public registry.

        Args:
            tags:    Filter by tags (e.g. ["image-generation"]).
            backend: Filter by backend compatibility.
        """
        try:
            recipes = k.list_recipes(tags=tags, backend=backend)
        except KdreamError as exc:
            _tool_error(exc)

        return [
            {
                "name": r.name,
                "version": r.version,
                "description": r.description,
                "tags": r.tags,
                "repo": r.repo,
            }
            for r in recipes
        ]

    # ── generate_recipe ───────────────────────────────────────────────────────

    @mcp.tool()
    def generate_recipe(
        repo: str,
        output: str | None = None,
    ) -> dict[str, Any]:
        """Generate a kdream recipe from a GitHub repository using AI agents.

        Requires ANTHROPIC_API_KEY environment variable.

        Args:
            repo:   GitHub repository URL to analyse.
            output: Optional local file path to write the generated YAML recipe.
        """
        try:
            recipe = k.generate_recipe(repo=repo, output=output)
        except KdreamError as exc:
            _tool_error(exc)

        return {
            "name": recipe.metadata.name,
            "version": recipe.metadata.version,
            "description": recipe.metadata.description,
            "tags": recipe.metadata.tags,
            "repo": recipe.metadata.repo,
            "output_path": output,
        }

    # ── validate_recipe ───────────────────────────────────────────────────────

    @mcp.tool()
    def validate_recipe(recipe_file: str) -> dict[str, Any]:
        """Validate a local kdream recipe YAML or Markdown file.

        Args:
            recipe_file: Absolute or relative path to the recipe file.
        """
        try:
            recipe = k.load_recipe(recipe_file)
            errors = k.validate_recipe(recipe)
        except KdreamError as exc:
            _tool_error(exc)

        return {
            "valid": len(errors) == 0,
            "name": recipe.metadata.name,
            "version": recipe.metadata.version,
            "errors": errors,
            "input_count": len(recipe.inputs),
            "model_count": len(recipe.models),
            "output_count": len(recipe.outputs),
        }

    # ── list_installed ────────────────────────────────────────────────────────

    @mcp.tool()
    def list_installed(cache_dir: str | None = None) -> list[dict[str, Any]]:
        """List all locally installed kdream packages.

        Args:
            cache_dir: Override the default cache directory (~/.kdream/cache).
        """
        try:
            packages = k.list_installed(cache_dir=cache_dir)
        except KdreamError as exc:
            _tool_error(exc)

        return [
            {
                "recipe_name": pkg.recipe_name,
                "path": str(pkg.path),
                "ready": pkg.ready,
            }
            for pkg in packages
        ]

    # ── recipe_info ───────────────────────────────────────────────────────

    @mcp.tool()
    def recipe_info(recipe: str) -> dict[str, Any]:
        """Return full details about a recipe: inputs, outputs, models, backend requirements.

        Args:
            recipe: Registry name or local file path to a recipe.
        """
        try:
            from kdream.core.runner import _resolve_recipe
            r = _resolve_recipe(recipe)
        except Exception as exc:
            raise ToolError(str(exc)) from exc

        local = r.backends.local
        return {
            "name": r.metadata.name,
            "version": r.metadata.version,
            "description": r.metadata.description,
            "tags": r.metadata.tags,
            "license": r.metadata.license,
            "author": r.metadata.author,
            "repo": r.source.repo,
            "entrypoint": r.entrypoint.script,
            "models": [
                {
                    "name": m.name,
                    "source": m.source,
                    "id": m.id,
                    "size_gb": m.size_gb,
                    "license": m.license,
                }
                for m in r.models
            ],
            "inputs": {
                name: {
                    "type": spec.type,
                    "required": spec.required,
                    "default": spec.default,
                    "description": spec.description,
                    "min": spec.min,
                    "max": spec.max,
                }
                for name, spec in r.inputs.items()
            },
            "outputs": [
                {"name": o.name, "type": o.type, "path": o.path}
                for o in r.outputs
            ],
            "backend": {
                "requires_gpu": local.requires_gpu if local else False,
                "min_vram_gb": local.min_vram_gb if local else 0,
                "tested_on": local.tested_on if local else [],
            },
        }

    # ── detect_accelerator ────────────────────────────────────────────────

    @mcp.tool()
    def detect_accelerator() -> dict[str, Any]:
        """Detect the best available compute accelerator on the server machine.

        Priority: cuda (NVIDIA GPU) > mps (Apple Silicon) > cpu.
        """
        from kdream.backends.local import HardwareDetector
        hw = HardwareDetector().detect()
        return {
            "device": hw["device"],
            "vram_gb": hw.get("vram_gb", 0),
            "cuda_version": hw.get("cuda_version"),
        }

    return mcp

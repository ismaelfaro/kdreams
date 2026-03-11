#!/usr/bin/env python3
"""kdream CLI — Universal AI Model Runtime & Recipe Platform."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.5.0", prog_name="kdream")
def cli():
    """kdream — Run any AI model with a single command.

    \b
    Examples:
      kdream run stable-diffusion-xl-base --prompt "red panda hacker"
      kdream install llama-3-8b-instruct
      kdream list --tag image-generation
      kdream generate --repo https://github.com/Tongyi-MAI/Z-Image
    """


@cli.command()
@click.argument("recipe")
@click.option("--backend", default="local", show_default=True,
              help="Compute backend (local|colab|runpod).")
@click.option("--cache-dir", default=None,
              help="Override cache directory (default: ~/.kdream/cache).")
@click.option("--force-reinstall", is_flag=True, default=False,
              help="Force reinstall even if cached.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed subprocess output (commands, uv logs, stderr).")
@click.option("--prompt", default=None, help="Text prompt.")
@click.option("--negative-prompt", default=None, help="Negative prompt.")
@click.option("--steps", default=None, type=int, help="Inference steps.")
@click.option("--guidance-scale", default=None, type=float, help="Guidance scale.")
@click.option("--seed", default=None, type=int, help="Random seed (-1 for random).")
@click.option("--width", default=None, type=int, help="Output width.")
@click.option("--height", default=None, type=int, help="Output height.")
@click.option("--output-dir", default=None, help="Directory to save outputs.")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def run(recipe, backend, cache_dir, force_reinstall, verbose,
        prompt, negative_prompt, steps, guidance_scale,
        seed, width, height, output_dir, extra_args):
    """Run inference using a recipe.

    \b
    RECIPE can be a registry name or a local file path.

    \b
    Examples:
      kdream run stable-diffusion-xl-base --prompt "sunset over mountains"
      kdream run ./my-recipe.yaml --prompt "hello world"
    """
    kwargs: dict = {}
    if prompt is not None:
        kwargs["prompt"] = prompt
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if steps is not None:
        kwargs["steps"] = steps
    if guidance_scale is not None:
        kwargs["guidance_scale"] = guidance_scale
    if seed is not None:
        kwargs["seed"] = seed
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    if output_dir is not None:
        kwargs["output_dir"] = output_dir

    # Parse extra --key value pairs
    i, extra = 0, list(extra_args)
    while i < len(extra):
        if extra[i].startswith("--"):
            key = extra[i][2:].replace("-", "_")
            if i + 1 < len(extra) and not extra[i + 1].startswith("--"):
                kwargs[key] = extra[i + 1]
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1

    try:
        import kdream as k
        console.print(Panel(f"[bold blue]Running recipe:[/bold blue] {recipe}", expand=False))
        result = k.run(
            recipe=recipe,
            backend=backend,
            cache_dir=cache_dir,
            force_reinstall=force_reinstall,
            verbose=verbose,
            **kwargs,
        )
        console.print("\n[bold green]✓ Inference complete[/bold green]")
        if result.outputs:
            table = Table(title="Outputs", show_header=True)
            table.add_column("Name", style="cyan")
            table.add_column("Value", style="white")
            for name, value in result.outputs.items():
                table.add_row(name, str(value))
            console.print(table)
        if result.metadata:
            console.print(
                f"[dim]Backend: {result.metadata.get('backend', backend)} | "
                f"Duration: {result.metadata.get('duration_s', 0):.1f}s[/dim]"
            )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("recipe")
@click.option("--backend", default="local", show_default=True)
@click.option("--cache-dir", default=None)
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed subprocess output (commands, uv logs, stderr).")
def install(recipe, backend, cache_dir, verbose):
    """Pre-install a recipe package without running inference.

    \b
    Example:
      kdream install stable-diffusion-xl-base
    """
    try:
        import kdream as k
        console.print(Panel(f"[bold blue]Installing:[/bold blue] {recipe}", expand=False))
        pkg = k.install(recipe=recipe, backend=backend, cache_dir=cache_dir, verbose=verbose)
        console.print("\n[bold green]✓ Installed successfully[/bold green]")
        console.print(f"  Path:  [cyan]{pkg.path}[/cyan]")
        console.print(
            f"  Ready: {'[green]Yes[/green]' if pkg.ready else '[yellow]Partial[/yellow]'}"
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@cli.command(name="list")
@click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable).")
@click.option("--backend", default=None, help="Filter by backend.")
@click.option("--search", default=None, help="Search query.")
def list_recipes(tags, backend, search):
    """List available recipes from the public registry.

    \b
    Examples:
      kdream list
      kdream list --tag image-generation
      kdream list --search llama
    """
    try:
        from kdream.core.registry import RegistryClient
        registry = RegistryClient()

        if search:
            recipes = registry.search_recipes(search)
        else:
            import kdream as k
            recipes = k.list_recipes(tags=list(tags) or None, backend=backend)

        if not recipes:
            console.print("[yellow]No recipes found.[/yellow]")
            return

        table = Table(title="Available Recipes", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan", min_width=28)
        table.add_column("Tags", style="green", min_width=16)
        table.add_column("Description", style="dim", min_width=20)
        table.add_column("Repo", style="blue", min_width=30)

        for r in recipes:
            repo_display = r.repo.removeprefix("https://github.com/") if r.repo else ""
            table.add_row(
                r.name,
                ", ".join(r.tags[:3]),
                (r.description[:55] + "…") if len(r.description) > 55 else r.description,
                repo_display,
            )
        console.print(table)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@cli.command()
@click.option("--repo", required=True, help="GitHub repository URL.")
@click.option("--output", default=None, help="Output file path for the generated recipe.")
@click.option("--publish", is_flag=True, default=False,
              help="Open a PR to the public registry.")
@click.option("--format", "fmt", default="yaml",
              type=click.Choice(["yaml", "markdown"]), show_default=True)
def generate(repo, output, publish, fmt):
    """Generate a kdream recipe from a GitHub repository using AI agents.

    \b
    Requires ANTHROPIC_API_KEY environment variable.

    \b
    Examples:
      kdream generate --repo https://github.com/Tongyi-MAI/Z-Image
      kdream generate --repo https://github.com/nikopueringer/CorridorKey --output ./my-recipe.yaml
    """
    try:
        import kdream as k
        console.print(Panel(
            f"[bold blue]Generating recipe from:[/bold blue]\n{repo}",
            title="Recipe Generator",
            expand=False,
        ))
        recipe = k.generate_recipe(repo=repo, output=output, publish=publish)
        console.print(f"\n[bold green]✓ Recipe generated:[/bold green] {recipe.metadata.name}")

        # If no --output was given, save to ./recipes/<category>/<name>.yaml
        if not output:
            category = recipe.metadata.tags[0] if recipe.metadata.tags else "uncategorized"
            default_out = Path("recipes") / category / f"{recipe.metadata.name}.yaml"
            default_out.parent.mkdir(parents=True, exist_ok=True)
            from kdream.core.recipe import recipe_to_yaml
            default_out.write_text(recipe_to_yaml(recipe), encoding="utf-8")
            console.print(f"  Saved to: [cyan]{default_out}[/cyan]")
        else:
            console.print(f"  Saved to: [cyan]{output}[/cyan]")
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@cli.command()
@click.option("--cache-dir", default=None)
def packages(cache_dir):
    """List all installed kdream packages.

    \b
    Example:
      kdream packages
    """
    try:
        import kdream as k
        pkgs = k.list_installed(cache_dir=cache_dir)

        if not pkgs:
            console.print("[yellow]No packages installed yet.[/yellow]")
            console.print("Run [cyan]kdream install <recipe>[/cyan] to install one.")
            return

        table = Table(title="Installed Packages", show_header=True, header_style="bold cyan")
        table.add_column("Recipe", style="cyan")
        table.add_column("Path", style="dim")
        table.add_column("Ready")

        for pkg in pkgs:
            table.add_row(
                pkg.recipe_name,
                str(pkg.path),
                "[green]✓[/green]" if pkg.ready else "[yellow]⏳[/yellow]",
            )
        console.print(table)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("recipe_file")
def validate(recipe_file):
    """Validate a local recipe file.

    \b
    Example:
      kdream validate ./my-recipe.yaml
    """
    try:
        from kdream.core.recipe import load_recipe, validate_recipe
        recipe = load_recipe(recipe_file)
        errors = validate_recipe(recipe)

        if errors:
            console.print(
                f"[bold red]✗ Validation failed ({len(errors)} error(s)):[/bold red]"
            )
            for err in errors:
                console.print(f"  • {err}")
            sys.exit(1)
        else:
            console.print(
                f"[bold green]✓ Valid:[/bold green] "
                f"{recipe.metadata.name} v{recipe.metadata.version}"
            )
            console.print(f"  Inputs:  {len(recipe.inputs)}")
            console.print(f"  Models:  {len(recipe.models)}")
            console.print(f"  Outputs: {len(recipe.outputs)}")
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@cli.group()
def cache():
    """Manage the kdream package cache."""


@cache.command(name="clear")
@click.option("--recipe", default=None, help="Clear only this recipe's cache.")
@click.option("--cache-dir", default=None)
@click.confirmation_option(prompt="Clear cache — are you sure?")
def cache_clear(recipe, cache_dir):
    """Clear cached packages and model weights.

    \b
    Examples:
      kdream cache clear
      kdream cache clear --recipe stable-diffusion-xl-base
    """
    base = Path(cache_dir) if cache_dir else (Path.home() / ".kdream" / "cache")

    if recipe:
        target = base / recipe
        if target.exists():
            shutil.rmtree(target)
            console.print(f"[green]✓ Cleared cache for {recipe}[/green]")
        else:
            console.print(f"[yellow]No cache found for {recipe}[/yellow]")
    else:
        if base.exists():
            shutil.rmtree(base)
            console.print(f"[green]✓ Cleared all cache at {base}[/green]")
        else:
            console.print("[yellow]Cache directory does not exist.[/yellow]")


@cache.command(name="info")
@click.option("--cache-dir", default=None)
def cache_info(cache_dir):
    """Show cache directory and disk usage."""
    base = Path(cache_dir) if cache_dir else (Path.home() / ".kdream" / "cache")
    console.print(f"Cache directory: [cyan]{base}[/cyan]")

    if base.exists():
        total, used, free = shutil.disk_usage(base)
        cache_size = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
        console.print(f"Cache size:      [yellow]{cache_size / 1e9:.2f} GB[/yellow]")
        console.print(f"Disk free:       [green]{free / 1e9:.2f} GB[/green]")
    else:
        console.print("[dim]Cache is empty (directory does not exist yet)[/dim]")


if __name__ == "__main__":
    cli()

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
@click.version_option(version="0.10.1", prog_name="kdream")
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
@click.option("--repo", required=True,
              help="GitHub or HuggingFace repository URL.")
@click.option("--output", default=None, help="Output file path for the generated recipe.")
@click.option("--publish", is_flag=True, default=False,
              help="Open a PR to the public registry.")
@click.option("--format", "fmt", default="yaml",
              type=click.Choice(["yaml", "markdown"]), show_default=True)
def generate(repo, output, publish, fmt):
    """Generate a kdream recipe from a GitHub or HuggingFace repository using AI agents.

    \b
    Requires ANTHROPIC_API_KEY environment variable.
    Supports both GitHub repos and HuggingFace model URLs.

    \b
    Examples:
      kdream generate --repo https://github.com/Tongyi-MAI/Z-Image
      kdream generate --repo https://huggingface.co/stabilityai/sdxl-turbo
      kdream generate --repo https://github.com/nikopueringer/CorridorKey --output ./my-recipe.yaml
    """
    from kdream.agents.recipe_generator import is_huggingface_url, normalize_github_url

    # Normalise GitHub URLs that include /tree/<branch>
    if not is_huggingface_url(repo):
        repo = normalize_github_url(repo)

    try:
        import kdream as k
        console.print(Panel(
            f"[bold blue]Generating recipe from:[/bold blue]\n{repo}",
            title="Recipe Generator",
            expand=False,
        ))
        recipe = k.generate_recipe(repo=repo, output=output, publish=publish)
        console.print(f"\n[bold green]✓ Recipe generated:[/bold green] {recipe.metadata.name}")

        # If no --output was given, save to kdream/recipes/<category>/<name>.yaml
        if not output:
            category = recipe.metadata.tags[0] if recipe.metadata.tags else "uncategorized"
            _recipes_root = Path(__file__).parent / "recipes"
            default_out = _recipes_root / category / f"{recipe.metadata.name}.yaml"
            default_out.parent.mkdir(parents=True, exist_ok=True)
            from kdream.core.recipe import recipe_to_yaml
            default_out.write_text(recipe_to_yaml(recipe), encoding="utf-8")
            console.print(f"  Saved to: [cyan]{default_out}[/cyan]")

            if recipe._runner_script:
                script_path = default_out.parent / "run.py"
                script_path.write_text(recipe._runner_script, encoding="utf-8")
                console.print(f"  Runner:   [cyan]{script_path}[/cyan]")
        else:
            console.print(f"  Saved to: [cyan]{output}[/cyan]")
            if recipe._runner_script:
                script_path = Path(output).parent / "run.py"
                console.print(f"  Runner:   [cyan]{script_path}[/cyan]")
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
        table.add_column("Models", style="green")
        table.add_column("Size", style="yellow", justify="right")
        table.add_column("Path", style="dim")
        table.add_column("Ready")

        for pkg in pkgs:
            # Scan models directory for downloaded weight files
            model_files: list[str] = []
            total_bytes = 0
            model_exts = {".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".gguf"}
            if pkg.models_path.exists():
                for f in pkg.models_path.rglob("*"):
                    if f.is_file() and f.suffix.lower() in model_exts:
                        model_files.append(f.name)
                        total_bytes += f.stat().st_size
            models_str = "\n".join(model_files[:3]) + (
                f"\n…+{len(model_files) - 3} more" if len(model_files) > 3 else ""
            ) if model_files else "[dim]none[/dim]"
            size_str = f"{total_bytes / 1e9:.1f} GB" if total_bytes else "—"

            table.add_row(
                pkg.recipe_name,
                models_str,
                size_str,
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


@cli.command()
@click.argument("recipe")
def info(recipe):
    """Show detailed info about a recipe: inputs, outputs, models, backend requirements.

    \b
    RECIPE can be a registry name or a local file path.

    \b
    Examples:
      kdream info stable-diffusion-xl-base
      kdream info ./my-recipe.yaml
    """
    try:
        from kdream.core.runner import _resolve_recipe
        from kdream.backends.local import HardwareDetector

        r = _resolve_recipe(recipe)
        hw = HardwareDetector().detect()
        local = r.backends.local

        # ── Header ────────────────────────────────────────────────────────
        console.print(Panel(
            f"[bold cyan]{r.metadata.name}[/bold cyan]  "
            f"[dim]v{r.metadata.version}[/dim]\n"
            f"{r.metadata.description}",
            title="Recipe Info",
            expand=False,
        ))

        # ── Metadata ──────────────────────────────────────────────────────
        meta_table = Table(show_header=False, box=None, padding=(0, 2))
        meta_table.add_column(style="bold dim", no_wrap=True)
        meta_table.add_column()
        meta_table.add_row("Tags",    ", ".join(r.metadata.tags) or "—")
        meta_table.add_row("License", r.metadata.license)
        meta_table.add_row("Author",  r.metadata.author)
        if r.source.repo:
            meta_table.add_row("Repo",    f"[link]{r.source.repo}[/link]")
        meta_table.add_row("Script",  r.entrypoint.script)
        console.print(meta_table)

        # ── Hardware requirements ─────────────────────────────────────────
        if local:
            needs_gpu = local.requires_gpu
            has_gpu = hw["device"] in ("cuda", "mps")
            compat = (not needs_gpu) or has_gpu
            colour = "green" if compat else "red"
            gpu_str = (
                f"[{colour}]{'✓' if compat else '✗'}[/{colour}] "
                f"{'GPU required' if needs_gpu else 'GPU optional'}"
                + (f" · {local.min_vram_gb} GB VRAM" if local.min_vram_gb else "")
                + (f" · tested on: {', '.join(local.tested_on)}" if local.tested_on else "")
            )
            your_hw = (
                f"Your hardware: [bold]{hw['device'].upper()}[/bold]"
                + (f" ({hw['vram_gb']} GB)" if hw.get("vram_gb") else "")
            )
            console.print(f"\n[bold]Backend[/bold]  {gpu_str}")
            console.print(f"         {your_hw}")

        # ── Models ────────────────────────────────────────────────────────
        if r.models:
            console.print()
            m_table = Table(title="Models", show_header=True, header_style="bold cyan")
            m_table.add_column("Name",        style="cyan")
            m_table.add_column("Source",      style="green")
            m_table.add_column("ID / URL",    style="dim")
            m_table.add_column("Size",        justify="right")
            m_table.add_column("License",     style="dim")
            for m in r.models:
                m_table.add_row(
                    m.name,
                    m.source,
                    m.id,
                    f"{m.size_gb} GB" if m.size_gb else "—",
                    m.license or "—",
                )
            console.print(m_table)

        # ── Inputs ────────────────────────────────────────────────────────
        if r.inputs:
            console.print()
            i_table = Table(title="Inputs", show_header=True, header_style="bold cyan")
            i_table.add_column("Name",        style="cyan",    no_wrap=True)
            i_table.add_column("Type",        style="green",   no_wrap=True)
            i_table.add_column("Required",    justify="center")
            i_table.add_column("Default",     style="yellow")
            i_table.add_column("Range",       style="dim")
            i_table.add_column("Description", style="dim")
            for name, spec in r.inputs.items():
                req = "[bold red]yes[/bold red]" if spec.required else "no"
                rng = ""
                if spec.min is not None or spec.max is not None:
                    rng = f"{spec.min or ''}..{spec.max or ''}"
                i_table.add_row(
                    name,
                    spec.type,
                    req,
                    str(spec.default) if spec.default is not None else "—",
                    rng or "—",
                    spec.description or "—",
                )
            console.print(i_table)

        # ── Outputs ───────────────────────────────────────────────────────
        if r.outputs:
            console.print()
            o_table = Table(title="Outputs", show_header=True, header_style="bold cyan")
            o_table.add_column("Name", style="cyan")
            o_table.add_column("Type", style="green")
            o_table.add_column("Path", style="dim")
            for o in r.outputs:
                o_table.add_row(o.name, o.type, o.path or "—")
            console.print(o_table)

    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@cli.command()
def accelerator():
    """Detect and display the best available compute accelerator.

    \b
    Priority: cuda (NVIDIA GPU) > mps (Apple Silicon) > cpu.

    \b
    Example:
      kdream accelerator
    """
    from kdream.backends.local import HardwareDetector
    hw = HardwareDetector().detect()
    device = hw["device"]

    colour = {"cuda": "green", "mps": "blue", "cpu": "yellow"}.get(device, "white")
    label = {"cuda": "NVIDIA CUDA GPU", "mps": "Apple Silicon (MPS)", "cpu": "CPU"}.get(
        device, device
    )

    console.print(f"\n[bold {colour}]✓ Best accelerator: {device.upper()}[/bold {colour}]")
    console.print(f"  [dim]{label}[/dim]")
    if hw.get("vram_gb"):
        console.print(f"  VRAM: [yellow]{hw['vram_gb']} GB[/yellow]")
    if hw.get("cuda_version"):
        console.print(f"  CUDA: [dim]{hw['cuda_version']}[/dim]")
    console.print(
        f"\n[dim]kdream will automatically use [bold]{device}[/bold] "
        f"when running recipes (env: KDREAM_DEVICE={device})[/dim]"
    )


@cli.command()
@click.option("--port", default=8765, show_default=True,
              help="Port to bind the MCP server to.")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Host to bind (use 0.0.0.0 to listen on all interfaces).")
@click.option("--transport", default="http",
              type=click.Choice(["stdio", "http"]), show_default=True,
              help="MCP transport: stdio (for Claude Desktop) or http (streamable-http).")
@click.option("--ngrok", "use_ngrok", is_flag=True, default=False,
              help="Expose the server publicly via an ngrok tunnel.")
@click.option("--ngrok-token", default=None, envvar="NGROK_AUTHTOKEN",
              help="ngrok auth token (or set NGROK_AUTHTOKEN env var).")
def serve(port, host, transport, use_ngrok, ngrok_token):
    """Start kdream as an MCP server.

    \b
    Exposes kdream tools over the Model Context Protocol so any MCP-compatible
    client (Claude Desktop, Cursor, etc.) can invoke run_recipe, install_recipe,
    list_recipes, generate_recipe, validate_recipe, and list_installed.

    \b
    Examples:
      kdream serve --transport stdio          # stdio (Claude Desktop)
      kdream serve --transport http           # streamable-http on port 8765
      kdream serve --transport http --ngrok   # public ngrok URL
    """
    try:
        from kdream.service.mcp_server import create_mcp_server
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] MCP dependencies not installed.\n"
            "Run: [cyan]uv pip install 'kdream[service]'[/cyan]"
        )
        sys.exit(1)

    mcp = create_mcp_server(host=host, port=port)

    if transport == "stdio":
        console.print("[bold blue]Starting kdream MCP server[/bold blue] (stdio)")
        console.print("[dim]Connect via Claude Desktop or any stdio MCP client.[/dim]")
        mcp.run(transport="stdio")
        return

    tunnel = None
    try:
        if use_ngrok:
            from kdream.service.ngrok_tunnel import NgrokTunnel
            tunnel = NgrokTunnel(port=port, auth_token=ngrok_token)
            public_url = tunnel.start()
            console.print(Panel(
                f"[bold green]kdream MCP server[/bold green]\n"
                f"Local:  [cyan]http://{host}:{port}/mcp[/cyan]\n"
                f"Public: [cyan]{public_url}/mcp[/cyan]",
                title="MCP Server",
                expand=False,
            ))
        else:
            console.print(Panel(
                f"[bold green]kdream MCP server[/bold green]\n"
                f"URL: [cyan]http://{host}:{port}/mcp[/cyan]\n"
                f"[dim]Use --ngrok to expose publicly.[/dim]",
                title="MCP Server",
                expand=False,
            ))

        mcp.run(transport="streamable-http")

    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    finally:
        if tunnel is not None:
            tunnel.stop()


@cli.group()
def remote():
    """Run inference on a remote kdream MCP server."""


def _remote_client():
    """Import the MCP client, printing a helpful error if deps are missing."""
    try:
        from kdream.service.mcp_client import call_tool
        return call_tool
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] MCP dependencies not installed.\n"
            "Run: [cyan]uv pip install 'kdream[service]'[/cyan]"
        )
        sys.exit(1)


@remote.command(name="run")
@click.argument("recipe")
@click.option("--url", required=True, envvar="KDREAM_REMOTE_URL",
              help="Remote MCP server URL, e.g. http://host:8765/mcp (or set KDREAM_REMOTE_URL).")
@click.option("--prompt", default=None, help="Text prompt.")
@click.option("--negative-prompt", default=None, help="Negative prompt.")
@click.option("--steps", default=None, type=int, help="Inference steps.")
@click.option("--guidance-scale", default=None, type=float, help="Guidance scale.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--width", default=None, type=int, help="Output width.")
@click.option("--height", default=None, type=int, help="Output height.")
@click.option("--output-dir", default=None, help="Output directory on the remote machine.")
@click.option("--backend", default="local", show_default=True,
              help="Compute backend on the remote server.")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def remote_run(recipe, url, prompt, negative_prompt, steps, guidance_scale,
               seed, width, height, output_dir, backend, extra_args):
    """Run inference on a remote kdream MCP server.

    \b
    RECIPE is the recipe name or path as understood by the remote server.

    \b
    Examples:
      kdream remote run stable-diffusion-xl-base --url http://host:8765/mcp --prompt "red panda"
      kdream remote run llama-3-8b-instruct --url $KDREAM_REMOTE_URL --prompt "hello"
    """
    call_tool = _remote_client()

    args: dict = {"recipe": recipe, "backend": backend}
    for key, val in [
        ("prompt", prompt), ("negative_prompt", negative_prompt),
        ("steps", steps), ("guidance_scale", guidance_scale),
        ("seed", seed), ("width", width), ("height", height),
        ("output_dir", output_dir),
    ]:
        if val is not None:
            args[key] = val

    # Parse extra --key value pairs
    i, extra = 0, list(extra_args)
    while i < len(extra):
        if extra[i].startswith("--"):
            key = extra[i][2:].replace("-", "_")
            if i + 1 < len(extra) and not extra[i + 1].startswith("--"):
                args[key] = extra[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1

    try:
        console.print(Panel(
            f"[bold blue]Remote inference[/bold blue]\n"
            f"Server: [cyan]{url}[/cyan]\n"
            f"Recipe: {recipe}",
            expand=False,
        ))
        result = call_tool(url, "run_recipe", args)
        if result.get("error"):
            console.print(f"[bold red]Remote error:[/bold red] {result['error']}")
            sys.exit(1)
        console.print("\n[bold green]✓ Remote inference complete[/bold green]")
        outputs = result.get("outputs") or {}
        if outputs:
            table = Table(title="Outputs", show_header=True)
            table.add_column("Name", style="cyan")
            table.add_column("Value", style="white")
            for name, value in outputs.items():
                table.add_row(name, str(value))
            console.print(table)
        meta = result.get("metadata") or {}
        if meta:
            console.print(
                f"[dim]Backend: {meta.get('backend', backend)} | "
                f"Duration: {meta.get('duration_s', 0):.1f}s[/dim]"
            )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@remote.command(name="list")
@click.option("--url", required=True, envvar="KDREAM_REMOTE_URL",
              help="Remote MCP server URL (or set KDREAM_REMOTE_URL).")
@click.option("--tag", "tags", multiple=True, help="Filter by tag.")
@click.option("--backend", default=None, help="Filter by backend.")
def remote_list(url, tags, backend):
    """List recipes available on the remote kdream server.

    \b
    Example:
      kdream remote list --url http://host:8765/mcp --tag image-generation
    """
    call_tool = _remote_client()
    args: dict = {}
    if tags:
        args["tags"] = list(tags)
    if backend:
        args["backend"] = backend

    try:
        recipes = call_tool(url, "list_recipes", args)
        if not isinstance(recipes, list) or not recipes:
            console.print("[yellow]No recipes found.[/yellow]")
            return
        table = Table(title=f"Recipes on {url}", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan", min_width=28)
        table.add_column("Tags", style="green", min_width=16)
        table.add_column("Description", style="dim", min_width=20)
        for r in recipes:
            table.add_row(
                r.get("name", ""),
                ", ".join((r.get("tags") or [])[:3]),
                (r.get("description", "")[:55] + "…")
                if len(r.get("description", "")) > 55 else r.get("description", ""),
            )
        console.print(table)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


@remote.command(name="packages")
@click.option("--url", required=True, envvar="KDREAM_REMOTE_URL",
              help="Remote MCP server URL (or set KDREAM_REMOTE_URL).")
def remote_packages(url):
    """List installed packages on the remote kdream server.

    \b
    Example:
      kdream remote packages --url http://host:8765/mcp
    """
    call_tool = _remote_client()
    try:
        pkgs = call_tool(url, "list_installed", {})
        if not isinstance(pkgs, list) or not pkgs:
            console.print("[yellow]No packages installed on remote server.[/yellow]")
            return
        table = Table(title=f"Installed on {url}", show_header=True, header_style="bold cyan")
        table.add_column("Recipe", style="cyan")
        table.add_column("Path", style="dim")
        table.add_column("Ready")
        for pkg in pkgs:
            table.add_row(
                pkg.get("recipe_name", ""),
                pkg.get("path", ""),
                "[green]✓[/green]" if pkg.get("ready") else "[yellow]⏳[/yellow]",
            )
        console.print(table)
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

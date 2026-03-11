"""Multi-agent pipeline for generating kdream recipes from GitHub or HuggingFace repositories."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

console = Console()

SKILLS_DIR = Path(__file__).parent / "skills"
MODEL = "claude-sonnet-4-6"


def load_skill(name: str) -> str:
    """Load agent system prompt from a skill Markdown file, stripping frontmatter."""
    skill_path = SKILLS_DIR / f"{name}.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill file not found: {skill_path}")
    content = skill_path.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return content


def call_agent(agent_name: str, user_message: str, client: Any) -> str:
    """Call a single Claude agent using its skill file as system prompt."""
    system_prompt = load_skill(agent_name)
    console.print(f"  [dim]→ Agent: {agent_name}[/dim]")
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# GitHub repo info
# ---------------------------------------------------------------------------

def get_repo_info(repo_url: str) -> dict[str, str]:
    """Clone the repo (shallow) and collect key files for agent analysis."""
    import git  # type: ignore[import]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "repo"
        console.print(f"  [dim]Cloning {repo_url}…[/dim]")
        try:
            git.Repo.clone_from(repo_url, tmp_path, depth=1)
        except Exception as e:
            raise RuntimeError(f"Failed to clone repository: {e}") from e

        info: dict[str, str] = {
            "url": repo_url,
            "tree": "",
            "readme": "",
            "requirements": "",
            "setup_py": "",
            "pyproject": "",
            "candidate_scripts": "",
        }

        # File tree (top 2 levels, skip hidden)
        tree_lines: list[str] = []
        for p in sorted(tmp_path.rglob("*")):
            rel = p.relative_to(tmp_path)
            parts = rel.parts
            if len(parts) <= 2 and not any(part.startswith(".") for part in parts):
                indent = "  " * (len(parts) - 1)
                tree_lines.append(f"{indent}{rel.name}{'/' if p.is_dir() else ''}")
        info["tree"] = "\n".join(tree_lines[:200])

        # README
        for name in ["README.md", "README.rst", "README.txt", "readme.md"]:
            p = tmp_path / name
            if p.exists():
                info["readme"] = p.read_text(errors="replace")[:8000]
                break

        # Requirements
        for name in ["requirements.txt", "requirements/base.txt", "requirements/main.txt"]:
            p = tmp_path / name
            if p.exists():
                info["requirements"] = p.read_text(errors="replace")[:3000]
                break

        # Setup files
        for name in ["setup.py", "setup.cfg"]:
            p = tmp_path / name
            if p.exists():
                info["setup_py"] = p.read_text(errors="replace")[:3000]
                break

        p = tmp_path / "pyproject.toml"
        if p.exists():
            info["pyproject"] = p.read_text(errors="replace")[:3000]

        # Candidate inference scripts
        scripts: list[str] = []
        keywords = {"infer", "predict", "demo", "run", "generate", "sample", "inference"}
        for py_file in sorted(tmp_path.rglob("*.py")):
            if any(kw in py_file.name.lower() for kw in keywords):
                try:
                    scripts.append(f"### {py_file.relative_to(tmp_path)}\n"
                                   + py_file.read_text(errors="replace")[:2000])
                except Exception:
                    pass
        info["candidate_scripts"] = "\n\n".join(scripts[:3])

        return info


# ---------------------------------------------------------------------------
# HuggingFace repo info
# ---------------------------------------------------------------------------

_HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/([^/]+/[^/?\s]+)",
    re.IGNORECASE,
)
_GH_TREE_RE = re.compile(r"^(https?://github\.com/[^/]+/[^/]+)/tree/.*$", re.IGNORECASE)


def normalize_github_url(url: str) -> str:
    """Strip ``/tree/<branch>`` suffixes from GitHub URLs."""
    m = _GH_TREE_RE.match(url.strip())
    return m.group(1) if m else url.strip()


def is_huggingface_url(url: str) -> bool:
    """Return True if *url* points to a HuggingFace model repository."""
    return bool(_HF_URL_RE.match(url.strip()))


def hf_model_id_from_url(url: str) -> str:
    """Extract ``org/model`` from a HuggingFace URL."""
    m = _HF_URL_RE.match(url.strip())
    if not m:
        raise ValueError(f"Not a valid HuggingFace model URL: {url!r}")
    return m.group(1)


def get_hf_model_info(model_id: str) -> dict[str, str]:
    """Fetch model card, metadata and file listing from the HuggingFace Hub."""
    from huggingface_hub import HfApi, ModelCard  # type: ignore[import]

    api = HfApi()
    console.print(f"  [dim]Fetching HuggingFace model info: {model_id}…[/dim]")

    try:
        hf_info = api.model_info(model_id)
    except Exception as exc:
        raise RuntimeError(f"Could not fetch HF model info for {model_id!r}: {exc}") from exc

    # Model card text
    card_text = ""
    try:
        card = ModelCard.load(model_id)
        card_text = str(card)[:10000]
    except Exception:
        pass

    # File listing
    files: list[str] = []
    try:
        files = list(api.list_repo_files(model_id))
    except Exception:
        pass

    # Card data (YAML front-matter parsed by huggingface_hub)
    card_data = getattr(hf_info, "card_data", None)
    try:
        license_val = str(
            getattr(card_data, "get", lambda k, d: None)("license", None)
            or getattr(card_data, "license", None)
            or "unknown"
        )
    except Exception:
        license_val = "unknown"

    return {
        "model_id": model_id,
        "url": f"https://huggingface.co/{model_id}",
        "pipeline_tag": getattr(hf_info, "pipeline_tag", "") or "",
        "library_name": getattr(hf_info, "library_name", "") or "",
        "tags": ", ".join(getattr(hf_info, "tags", []) or []),
        "license": license_val or "unknown",
        "downloads": str(getattr(hf_info, "downloads", 0) or 0),
        "model_card": card_text,
        "files": "\n".join(files[:150]),
    }


def _build_hf_base_msg(hf_info: dict[str, str]) -> str:
    """Format HF model info into a base message for the agent pipeline."""
    return (
        f"SOURCE_TYPE: huggingface\n"
        f"Model ID: {hf_info['model_id']}\n"
        f"URL: {hf_info['url']}\n"
        f"Pipeline Tag: {hf_info['pipeline_tag']}\n"
        f"Library: {hf_info['library_name']}\n"
        f"Tags: {hf_info['tags']}\n"
        f"License: {hf_info['license']}\n"
        f"Downloads: {hf_info['downloads']}\n\n"
        f"## Model Card\n{hf_info['model_card'] or '(unavailable)'}\n\n"
        f"## Files in Repository\n{hf_info['files'] or '(unavailable)'}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_yaml(text: str) -> str:
    """Strip markdown code fences from a YAML block if present."""
    if "```yaml" in text:
        return text.split("```yaml", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _extract_python(text: str) -> str:
    """Strip markdown code fences from a Python block if present."""
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------

class RecipeGeneratorAgent:
    """Multi-agent pipeline: RepoInspector → EntrypointFinder → ModelLocator
    → ParameterMapper → RecipeWriter [→ HFScriptWriter for HF models]."""

    def __init__(self, api_key: str | None = None):
        import anthropic  # type: ignore[import]
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        repo: str,
        output: str | None = None,
        publish: bool = False,
    ) -> Any:
        """Run the full agent pipeline and return a parsed Recipe object.

        Supports both GitHub repository URLs and HuggingFace model URLs
        (e.g. ``https://huggingface.co/stabilityai/sdxl-turbo``).

        Args:
            repo:    GitHub repository URL or HuggingFace model URL.
            output:  Optional path to write the generated YAML recipe.
                     For HF models, a companion ``run.py`` is written next to it.
            publish: If True, open a PR to the registry (not yet implemented).
        """
        from kdream.core.recipe import parse_yaml_recipe, validate_recipe

        _is_hf = is_huggingface_url(repo)
        source_label = "HuggingFace model" if _is_hf else "GitHub repository"
        model_id: str = ""
        hf_info: dict[str, str] = {}

        console.print(Panel(
            f"[bold]Recipe Generator[/bold]\n"
            f"{source_label}: [cyan]{repo}[/cyan]\n"
            "Pipeline: RepoInspector → EntrypointFinder → ModelLocator "
            "→ ParameterMapper → RecipeWriter"
            + (" → HFScriptWriter" if _is_hf else ""),
            title="kdream Agent",
            expand=False,
        ))

        # ── Step 1: Collect source data ───────────────────────────────────
        console.print(f"\n[bold]1/{'7' if _is_hf else '6'}[/bold] Collecting source data…")
        if _is_hf:
            model_id = hf_model_id_from_url(repo)
            hf_info = get_hf_model_info(model_id)
            base_msg = _build_hf_base_msg(hf_info)
        else:
            try:
                repo_info = get_repo_info(repo)
            except Exception as exc:
                console.print(f"[yellow]Warning:[/yellow] Could not clone repo ({exc}). "
                              "Using URL-only analysis.")
                repo_info = {k: "" for k in
                             ["url", "tree", "readme", "requirements", "setup_py", "pyproject",
                              "candidate_scripts"]}
                repo_info["url"] = repo
            base_msg = (
                f"Repository URL: {repo_info['url']}\n\n"
                f"## File Tree\n{repo_info['tree'] or '(unavailable)'}\n\n"
                f"## README\n{repo_info['readme'] or '(unavailable)'}\n\n"
                f"## Requirements\n{repo_info['requirements'] or '(unavailable)'}\n\n"
                f"## Setup files\n"
                f"{repo_info['setup_py'] or repo_info['pyproject'] or '(unavailable)'}\n\n"
                f"## Candidate Inference Scripts\n"
                f"{repo_info['candidate_scripts'] or '(unavailable)'}"
            )

        # ── Step 2: RepoInspector ─────────────────────────────────────────
        console.print(f"\n[bold]2/{'7' if _is_hf else '6'}[/bold] Inspecting repository structure…")
        repo_analysis = call_agent("repo-inspector", base_msg, self.client)

        # ── Step 3: EntrypointFinder ──────────────────────────────────────
        console.print(f"\n[bold]3/{'7' if _is_hf else '6'}[/bold] Finding inference entrypoints…")
        entrypoint_info = call_agent(
            "entrypoint-finder",
            base_msg + f"\n\n## Repo Analysis\n{repo_analysis}",
            self.client,
        )

        # ── Step 4: ModelLocator ──────────────────────────────────────────
        console.print(f"\n[bold]4/{'7' if _is_hf else '6'}[/bold] Locating model weights…")
        model_info = call_agent(
            "model-locator",
            base_msg + f"\n\n## Repo Analysis\n{repo_analysis}",
            self.client,
        )

        # ── Step 5: ParameterMapper ───────────────────────────────────────
        console.print(f"\n[bold]5/{'7' if _is_hf else '6'}[/bold] Mapping parameters to kdream schema…")
        param_info = call_agent(
            "parameter-mapper",
            f"## Entrypoint Analysis\n{entrypoint_info}\n\n"
            f"## Model Information\n{model_info}\n\n"
            f"## Repository Context\n{repo_analysis}",
            self.client,
        )

        # ── Step 6: RecipeWriter ──────────────────────────────────────────
        console.print(f"\n[bold]6/{'7' if _is_hf else '6'}[/bold] Writing recipe…")
        recipe_yaml_raw = call_agent(
            "recipe-writer",
            f"Repository URL: {repo}\n\n"
            f"## Repo Analysis\n{repo_analysis}\n\n"
            f"## Entrypoint\n{entrypoint_info}\n\n"
            f"## Models\n{model_info}\n\n"
            f"## Parameters\n{param_info}\n\n"
            "Generate a complete kdream YAML recipe.",
            self.client,
        )

        yaml_content = _extract_yaml(recipe_yaml_raw)

        # ── Parse & validate ──────────────────────────────────────────────
        try:
            recipe = parse_yaml_recipe(yaml_content)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not parse recipe: {exc}")
            console.print("[dim]Raw output:[/dim]\n" + yaml_content)
            raise

        errors = validate_recipe(recipe)
        if errors:
            console.print(f"[yellow]Validation warnings ({len(errors)}):[/yellow]")
            for err in errors:
                console.print(f"  • {err}")

        # ── Step 7 (HF only): Generate companion runner script ────────────
        if _is_hf:
            console.print("\n[bold]7/7[/bold] Generating companion runner script…")
            script_raw = call_agent(
                "hf-script-writer",
                f"Model ID: {model_id}\n"
                f"Pipeline Tag: {hf_info['pipeline_tag']}\n"
                f"Library: {hf_info['library_name']}\n\n"
                f"## Model Card\n{hf_info['model_card'][:4000]}\n\n"
                f"## Parameters\n{param_info}\n\n"
                f"## Entrypoint Analysis\n{entrypoint_info}",
                self.client,
            )
            recipe._runner_script = _extract_python(script_raw)

        # ── Save if requested ─────────────────────────────────────────────
        if output:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml_content, encoding="utf-8")
            console.print(f"\n[green]✓ Recipe saved to {out_path}[/green]")

            if recipe._runner_script:
                script_path = out_path.parent / "run.py"
                script_path.write_text(recipe._runner_script, encoding="utf-8")
                console.print(f"[green]✓ Runner script saved to {script_path}[/green]")

        if publish:
            console.print(
                "[yellow]Note:[/yellow] Registry PR publishing not yet implemented "
                "in Phase 1."
            )

        return recipe

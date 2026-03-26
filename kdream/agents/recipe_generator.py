"""Multi-agent pipeline for generating kdream recipes from GitHub or HuggingFace repositories."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table

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


_GH_REPO_RE = re.compile(
    r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
)


def _extract_github_url_from_card(card_text: str) -> str:
    """Try to find a GitHub repository URL in a HuggingFace model card."""
    matches = _GH_REPO_RE.findall(card_text)
    if not matches:
        return ""
    # Normalise: strip /tree/... suffixes and pick the first unique repo URL
    seen: set[str] = set()
    for url in matches:
        cleaned = normalize_github_url(url)
        # Remove trailing punctuation that might be captured
        cleaned = cleaned.rstrip("/.,;:!?)")
        if cleaned not in seen:
            seen.add(cleaned)
    # Return the most common URL (first match is usually the primary repo)
    return next(iter(seen), "")


# ---------------------------------------------------------------------------
# Formatted file listing (with sizes)
# ---------------------------------------------------------------------------

def _format_file_listing(files: list[str], file_sizes: dict[str, int]) -> str:
    """Format HF file listing with human-readable sizes, grouped by directory."""
    if not files:
        return "(no files)"

    def _human_size(size_bytes: int) -> str:
        if size_bytes <= 0:
            return "—"
        if size_bytes >= 1024 ** 3:
            return f"{size_bytes / (1024 ** 3):.1f} GB"
        if size_bytes >= 1024 ** 2:
            return f"{size_bytes / (1024 ** 2):.1f} MB"
        if size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes} B"

    # Group by top-level directory
    groups: dict[str, list[str]] = {}
    for f in files:
        parts = f.split("/", 1)
        if len(parts) == 2:
            group = parts[0] + "/"
        else:
            group = "./"
        size = file_sizes.get(f, 0)
        line = f"  {f} ({_human_size(size)})" if size else f"  {f}"
        groups.setdefault(group, []).append(line)

    lines: list[str] = []
    for group_name in sorted(groups.keys()):
        lines.append(f"\n### {group_name}")
        lines.extend(groups[group_name])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Search HuggingFace for quantized alternatives (for GitHub URLs)
# ---------------------------------------------------------------------------

_GH_ORG_NAME_RE = re.compile(r"github\.com/([^/]+)/([^/?\s]+)", re.IGNORECASE)


def search_hf_quantized_alternatives(repo_url: str) -> dict[str, Any] | None:
    """When given a GitHub URL, search HuggingFace for quantized/unsloth alternatives.

    Tries multiple search strategies:
    1. Original project name with hyphens preserved (e.g. "LTX-2 gguf")
    2. Org/project name (e.g. "Lightricks LTX-2 gguf")
    3. Space-separated name (e.g. "LTX 2 gguf")
    4. Unsloth-specific searches

    Returns the HF model info dict if the user picks one, or None to continue with GitHub-only.
    """
    m = _GH_ORG_NAME_RE.search(repo_url)
    if not m:
        return None
    org_name = m.group(1)
    project_name_raw = m.group(2)  # e.g. "LTX-2"
    project_name_spaced = project_name_raw.replace("-", " ").replace("_", " ")

    try:
        from huggingface_hub import HfApi  # type: ignore[import]
        api = HfApi()
    except ImportError:
        return None

    console.print(f"\n  [dim]Searching HuggingFace for quantized versions of '{project_name_raw}'…[/dim]")

    # Build diverse search queries to maximize matches
    search_queries = [
        f"{project_name_raw} gguf",               # "LTX-2 gguf"
        f"{project_name_raw} unsloth",             # "LTX-2 unsloth"
        f"{org_name} {project_name_raw} gguf",     # "Lightricks LTX-2 gguf"
        f"{org_name} {project_name_raw}",          # "Lightricks LTX-2"
        f"{project_name_spaced} gguf",             # "LTX 2 gguf"
    ]
    # Deduplicate queries while preserving order
    seen_queries: set[str] = set()
    unique_queries: list[str] = []
    for q in search_queries:
        ql = q.lower()
        if ql not in seen_queries:
            seen_queries.add(ql)
            unique_queries.append(q)

    results: list[Any] = []
    for query in unique_queries:
        try:
            hits = list(api.list_models(search=query, limit=5, sort="downloads"))
            results.extend(hits)
        except Exception:
            pass

    # Deduplicate by model ID
    seen: set[str] = set()
    unique: list[Any] = []
    for r in results:
        mid = r.id if hasattr(r, "id") else str(r)
        if mid not in seen:
            seen.add(mid)
            unique.append(r)

    # Filter: prioritize models that look like quantized variants
    # (contain gguf/awq/gptq in name or tags) and models from the same org
    def _relevance_score(model: Any) -> tuple[int, int]:
        mid_lower = model.id.lower()
        tags = set(t.lower() for t in (getattr(model, "tags", None) or []))
        score = 0
        # Boost for quantized format indicators
        if "gguf" in mid_lower or "gguf" in tags:
            score += 10
        if "awq" in mid_lower or "awq" in tags:
            score += 10
        if "gptq" in mid_lower or "gptq" in tags:
            score += 10
        if "unsloth" in mid_lower:
            score += 5
        # Boost for same org
        if org_name.lower() in mid_lower:
            score += 3
        # Boost for project name match
        if project_name_raw.lower() in mid_lower:
            score += 5
        downloads = getattr(model, "downloads", 0) or 0
        return (-score, -downloads)

    unique.sort(key=_relevance_score)

    # Only show models that seem relevant (have project name in ID or tags)
    project_tokens = {t.lower() for t in project_name_raw.replace("-", " ").split() if len(t) > 1}
    relevant = [
        r for r in unique
        if any(tok in r.id.lower() for tok in project_tokens)
    ]
    if relevant:
        unique = relevant

    if not unique:
        console.print("  [dim]No quantized HuggingFace alternatives found.[/dim]")
        return None

    # Display results
    table = Table(show_header=True, header_style="bold cyan", title="HuggingFace Quantized Alternatives")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model ID")
    table.add_column("Downloads", justify="right")
    table.add_column("Tags")

    for i, r in enumerate(unique[:8], 1):
        tags = ", ".join((getattr(r, "tags", None) or [])[:4])
        downloads = str(getattr(r, "downloads", 0) or 0)
        table.add_row(str(i), r.id, downloads, tags)

    # Add "skip" option
    table.add_row(str(len(unique[:8]) + 1), "[dim]Skip — use GitHub repo only[/dim]", "", "")
    console.print(table)

    skip_idx = len(unique[:8]) + 1
    choice = IntPrompt.ask(
        "\n[bold]Select an alternative (or skip)[/bold]",
        choices=[str(i) for i in range(1, skip_idx + 1)],
        default=skip_idx,
    )

    if choice == skip_idx:
        console.print("  [dim]Continuing with GitHub repository only.[/dim]")
        return None

    selected = unique[choice - 1]
    console.print(f"  [green]Selected:[/green] {selected.id}")

    # Fetch full info for the selected model
    hf_info = get_hf_model_info(selected.id)
    return hf_info


# ---------------------------------------------------------------------------
# Quantized variant detection
# ---------------------------------------------------------------------------

_QUANT_EXTENSIONS = {".gguf", ".awq"}
_QUANT_LABEL_RE = re.compile(
    r"[-_]((?:UD-)?(?:Q[0-9]+_K_[A-Z]+|Q[0-9]+_[A-Z0-9]+|Q[0-9]+_[0-9]+|BF16|FP16|F16|F32|fp8))",
    re.IGNORECASE,
)

# Hardware compatibility: which quantization formats run on which accelerators.
# GGUF runs everywhere (llama-cpp-python supports Metal, CUDA, CPU).
# AWQ and GPTQ require CUDA — they will NOT work on Mac (MPS) or CPU.
_FORMAT_HARDWARE_COMPAT: dict[str, set[str]] = {
    "gguf": {"cuda", "mps", "cpu"},
    "awq": {"cuda"},
    "gptq": {"cuda"},
}


def _detect_accelerator() -> str:
    """Detect the local accelerator (cuda/mps/cpu).

    Re-uses the project's own ``HardwareDetector`` when available,
    then tries torch, then falls back to platform heuristics so that
    Apple Silicon Macs are detected even when torch is not installed.
    """
    try:
        from kdream.backends.local import detect_accelerator
        result = detect_accelerator()
        if result != "cpu":
            return result
        # HardwareDetector returned "cpu" — it may have failed to detect MPS
        # because torch is not installed. Fall through to platform heuristic.
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    # Fallback: detect Apple Silicon Mac without torch.
    # MPS (Metal Performance Shaders) is available on all Apple Silicon Macs
    # and macOS >= 12.3. If the user is on arm64 macOS, MPS will be usable
    # once torch is installed inside the recipe's virtual environment.
    import platform
    import sys
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def _detect_quantized_variants(
    files: list[str],
    file_sizes: dict[str, int] | None = None,
) -> list[dict]:
    """Detect quantized model variants from a HuggingFace file listing.

    Returns a list of dicts with keys: filename, quant_label, format, size_bytes.
    Returns empty list if no quantized variants are found.
    """
    file_sizes = file_sizes or {}
    variants: list[dict] = []
    for fname in files:
        lower = fname.lower()
        # Detect by extension
        fmt = ""
        if lower.endswith(".gguf"):
            fmt = "gguf"
        elif lower.endswith(".awq"):
            fmt = "awq"
        elif "gptq" in lower and lower.endswith(".safetensors"):
            fmt = "gptq"
        else:
            continue

        # Extract quantization label from filename
        m = _QUANT_LABEL_RE.search(fname)
        quant_label = m.group(1) if m else fmt.upper()

        variants.append({
            "filename": fname,
            "quant_label": quant_label,
            "format": fmt,
            "size_bytes": file_sizes.get(fname, 0),
        })

    return variants


def _prompt_variant_selection(
    variants: list[dict[str, str]], model_id: str,
) -> dict[str, str]:
    """Ask the user to pick a quantized variant, showing hardware compatibility."""
    accelerator = _detect_accelerator()
    console.print(
        f"\n[bold yellow]Multiple quantized variants found for {model_id}:[/bold yellow]"
    )
    console.print(f"  [dim]Detected accelerator: [bold]{accelerator.upper()}[/bold][/dim]")

    # Partition into compatible / incompatible
    compatible: list[tuple[int, dict[str, str]]] = []
    incompatible: list[tuple[int, dict[str, str]]] = []
    for i, v in enumerate(variants):
        supported = _FORMAT_HARDWARE_COMPAT.get(v["format"], set())
        if accelerator in supported:
            compatible.append((i, v))
        else:
            incompatible.append((i, v))

    if not compatible:
        # All variants are incompatible — warn but still let user choose
        console.print(
            "[bold red]Warning:[/bold red] None of the quantized variants are compatible "
            f"with your accelerator ({accelerator.upper()}). "
            "AWQ and GPTQ require an NVIDIA GPU (CUDA). "
            "Consider using a GGUF variant or the full-precision model instead."
        )
        # Fall through and show all variants anyway
        compatible = [(i, v) for i, v in enumerate(variants)]

    # Build the selection table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Variant")
    table.add_column("Format")
    table.add_column("Size", justify="right")
    table.add_column("Compat", width=8)
    table.add_column("File")

    # Show compatible variants first
    display_order: list[tuple[int, dict[str, str]]] = []
    for orig_idx, v in compatible:
        display_order.append((orig_idx, v))
    for orig_idx, v in incompatible:
        display_order.append((orig_idx, v))

    for display_num, (_, v) in enumerate(display_order, 1):
        supported = _FORMAT_HARDWARE_COMPAT.get(v["format"], set())
        if accelerator in supported:
            compat_str = "[green]Yes[/green]"
        else:
            compat_str = "[red]No[/red]"
        size_bytes = v.get("size_bytes", 0)
        if size_bytes > 0:
            size_gb = size_bytes / (1024 ** 3)
            size_str = f"{size_gb:.1f} GB"
        else:
            size_str = "—"
        table.add_row(
            str(display_num), v["quant_label"], v["format"].upper(),
            size_str, compat_str, v["filename"],
        )

    console.print(table)

    if incompatible and compatible:
        console.print(
            f"  [dim]Variants marked [red]No[/red] require CUDA and won't run "
            f"on {accelerator.upper()}.[/dim]"
        )

    choice = IntPrompt.ask(
        "\n[bold]Select a variant[/bold]",
        choices=[str(i) for i in range(1, len(display_order) + 1)],
        default=1,
    )
    selected_orig_idx, selected = display_order[choice - 1]

    # Warn if user picked an incompatible variant
    supported = _FORMAT_HARDWARE_COMPAT.get(selected["format"], set())
    if accelerator not in supported:
        console.print(
            f"  [bold yellow]Note:[/bold yellow] {selected['format'].upper()} is not supported "
            f"on {accelerator.upper()}. The generated recipe will target CUDA."
        )
    else:
        console.print(
            f"  [green]Selected:[/green] {selected['quant_label']} ({selected['filename']})"
        )

    return selected


def get_hf_model_info(model_id: str) -> dict[str, str]:
    """Fetch model card, metadata and file listing from the HuggingFace Hub."""
    from huggingface_hub import HfApi, ModelCard  # type: ignore[import]

    api = HfApi()
    console.print(f"  [dim]Fetching HuggingFace model info: {model_id}…[/dim]")

    try:
        hf_info = api.model_info(model_id, files_metadata=True)
    except Exception as exc:
        raise RuntimeError(f"Could not fetch HF model info for {model_id!r}: {exc}") from exc

    # Build file-size lookup from siblings (populated by files_metadata=True)
    file_sizes: dict[str, int] = {}
    for sib in getattr(hf_info, "siblings", None) or []:
        if getattr(sib, "size", None) is not None:
            file_sizes[sib.rfilename] = sib.size

    # Model card text
    card_text = ""
    try:
        card = ModelCard.load(model_id)
        card_text = str(card)[:10000]
    except Exception:
        pass

    # File listing
    files: list[str] = [sib.rfilename for sib in (getattr(hf_info, "siblings", None) or [])]

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

    # Try to find an associated GitHub repository in the model card
    github_repo = _extract_github_url_from_card(card_text)

    # Detect quantized model variants (with file sizes)
    quantized_variants = _detect_quantized_variants(files, file_sizes)

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
        "file_sizes": file_sizes,
        "github_repo": github_repo,
        "quantized_variants": quantized_variants,
    }


def _build_hf_base_msg(
    hf_info: dict[str, Any],
    selected_variant: dict[str, str] | None = None,
) -> str:
    """Format HF model info into a base message for the agent pipeline."""
    accelerator = _detect_accelerator()
    github_line = ""
    if hf_info.get("github_repo"):
        github_line = f"GitHub Repository: {hf_info['github_repo']}\n"
    variant_line = ""
    if selected_variant:
        fmt = selected_variant["format"]
        supported = _FORMAT_HARDWARE_COMPAT.get(fmt, set())
        compat_note = (
            f"compatible with {accelerator.upper()}"
            if accelerator in supported
            else f"NOT compatible with {accelerator.upper()} — requires CUDA"
        )
        variant_line = (
            f"Selected Variant: {selected_variant['filename']} "
            f"(format: {selected_variant['format']}, "
            f"quantization: {selected_variant['quant_label']}, "
            f"{compat_note})\n"
        )
    return (
        f"SOURCE_TYPE: huggingface\n"
        f"Model ID: {hf_info['model_id']}\n"
        f"URL: {hf_info['url']}\n"
        f"Accelerator: {accelerator}\n"
        f"{github_line}"
        f"{variant_line}"
        f"Pipeline Tag: {hf_info['pipeline_tag']}\n"
        f"Library: {hf_info['library_name']}\n"
        f"Tags: {hf_info['tags']}\n"
        f"License: {hf_info['license']}\n"
        f"Downloads: {hf_info['downloads']}\n\n"
        f"## Model Card\n{hf_info['model_card'] or '(unavailable)'}\n\n"
        f"## Files in Repository (with sizes)\n"
        f"{_format_file_listing(hf_info['files'].split(chr(10)) if hf_info['files'] else [], hf_info.get('file_sizes', {}))}"
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


def _sanitize_recipe_data(data: dict) -> dict:
    """Fix common agent-generated YAML issues before Pydantic parsing."""
    import copy
    data = copy.deepcopy(data)

    # Ensure tested_on is always a list
    backends = data.get("backends") or {}
    local = backends.get("local") or {}
    if isinstance(local, dict):
        if not local.get("tested_on"):
            local["tested_on"] = ["cpu"]
        backends["local"] = local
        data["backends"] = backends

    # Ensure source.repo is a string (not None)
    source = data.get("source") or {}
    if isinstance(source, dict):
        if source.get("repo") is None:
            source["repo"] = ""
        data["source"] = source

    # Normalise output types: map unknown types to nearest valid value
    _valid_output_types = {"file", "string", "base64", "json", "directory"}
    for out in data.get("outputs") or []:
        if isinstance(out, dict) and out.get("type") not in _valid_output_types:
            # directory-like → directory; else fall back to file
            t = str(out.get("type", "file")).lower()
            out["type"] = "directory" if "dir" in t or "folder" in t else "file"

    # Normalise input types: agents sometimes use "number" instead of "float"
    _input_type_map = {"number": "float", "num": "float", "int": "integer",
                       "str": "string", "bool": "boolean"}
    _valid_input_types = {"string", "integer", "float", "boolean"}
    for inp_spec in (data.get("inputs") or {}).values():
        if isinstance(inp_spec, dict):
            t = str(inp_spec.get("type", "string")).lower()
            if t not in _valid_input_types:
                inp_spec["type"] = _input_type_map.get(t, "string")

    return data


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
    """5-agent pipeline: RepoInspector → ModelLocator → InferenceMapper
    → RecipeWriter [→ HFScriptWriter for HF models]."""

    def __init__(self, api_key: str | None = None):
        import anthropic  # type: ignore[import]
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        repo: str,
        output: str | None = None,
        publish: bool = False,
    ) -> Any:
        """Run the full 5-agent pipeline and return a parsed Recipe object.

        Supports both GitHub repository URLs and HuggingFace model URLs
        (e.g. ``https://huggingface.co/stabilityai/sdxl-turbo``).

        Pipeline (5 agents):
            1. Collect source data (+ search for quantized alternatives if GitHub)
            2. RepoInspector — Deep architecture analysis
            3. ModelLocator — Deep file analysis, all component mapping
            4. InferenceMapper — Merged entrypoint + parameter mapping
            5. RecipeWriter — YAML with correct file references
            6. HFScriptWriter — Runner script (conditional: HF or GitHub+quantized)

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
        hf_info: dict[str, Any] = {}
        _needs_script = _is_hf  # May become True for GitHub repos with quantized HF models

        total_steps = 6 if _is_hf else 5

        console.print(Panel(
            f"[bold]Recipe Generator[/bold]\n"
            f"{source_label}: [cyan]{repo}[/cyan]\n"
            "Pipeline: RepoInspector → ModelLocator → InferenceMapper "
            "→ RecipeWriter"
            + (" → HFScriptWriter" if _is_hf else ""),
            title="kdream Agent",
            expand=False,
        ))

        selected_variant: dict[str, str] | None = None

        # ── Step 1: Collect source data ───────────────────────────────────
        console.print(f"\n[bold]1/{total_steps}[/bold] Collecting source data…")
        if _is_hf:
            model_id = hf_model_id_from_url(repo)
            hf_info = get_hf_model_info(model_id)

            # Prompt user to select a quantized variant if available
            variants = hf_info.get("quantized_variants", [])
            if variants:
                selected_variant = _prompt_variant_selection(variants, model_id)

            base_msg = _build_hf_base_msg(hf_info, selected_variant)
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

            # Search HuggingFace for quantized alternatives
            alt_hf_info = search_hf_quantized_alternatives(repo)
            if alt_hf_info is not None:
                # User picked a quantized HF model — switch to hybrid mode
                hf_info = alt_hf_info
                model_id = hf_info["model_id"]
                _is_hf = True
                _needs_script = True
                total_steps = 6

                # Prompt variant selection on the HF model
                variants = hf_info.get("quantized_variants", [])
                if variants:
                    selected_variant = _prompt_variant_selection(variants, model_id)

                # Build hybrid base message: GitHub context + HF model info
                base_msg = (
                    f"SOURCE_TYPE: huggingface (via GitHub hybrid)\n"
                    f"Original GitHub URL: {repo}\n"
                    f"{_build_hf_base_msg(hf_info, selected_variant)}\n\n"
                    f"## GitHub Repository Context\n"
                    f"Repository URL: {repo_info['url']}\n"
                    f"## File Tree\n{repo_info['tree'] or '(unavailable)'}\n\n"
                    f"## README\n{repo_info['readme'] or '(unavailable)'}\n\n"
                    f"## Candidate Inference Scripts\n"
                    f"{repo_info['candidate_scripts'] or '(unavailable)'}"
                )
                console.print(Panel(
                    f"[bold]Hybrid Mode[/bold]\n"
                    f"GitHub: [cyan]{repo}[/cyan]\n"
                    f"HF Model: [cyan]{model_id}[/cyan]\n"
                    "Pipeline: RepoInspector → ModelLocator → InferenceMapper "
                    "→ RecipeWriter → HFScriptWriter",
                    title="kdream Agent",
                    expand=False,
                ))
            else:
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
        console.print(f"\n[bold]2/{total_steps}[/bold] Inspecting repository structure…")
        repo_analysis = call_agent("repo-inspector", base_msg, self.client)

        # ── Step 3: ModelLocator (enhanced with component analysis) ───────
        console.print(f"\n[bold]3/{total_steps}[/bold] Locating model weights and components…")
        model_info = call_agent(
            "model-locator",
            base_msg
            + f"\n\n## Repo Analysis (including component architecture)\n{repo_analysis}",
            self.client,
        )

        # ── Step 4: InferenceMapper (merged entrypoint + parameter) ──────
        console.print(f"\n[bold]4/{total_steps}[/bold] Mapping entrypoints and parameters…")
        inference_info = call_agent(
            "inference-mapper",
            base_msg
            + f"\n\n## Repo Analysis\n{repo_analysis}"
            + f"\n\n## Model Components\n{model_info}",
            self.client,
        )

        # ── Step 5: RecipeWriter ──────────────────────────────────────────
        console.print(f"\n[bold]5/{total_steps}[/bold] Writing recipe…")
        recipe_yaml_raw = call_agent(
            "recipe-writer",
            f"Repository URL: {repo}\n\n"
            f"## Repo Analysis\n{repo_analysis}\n\n"
            f"## Inference Mapping (entrypoint + parameters)\n{inference_info}\n\n"
            f"## Model Components\n{model_info}\n\n"
            "Generate a complete kdream YAML recipe.",
            self.client,
        )

        yaml_content = _extract_yaml(recipe_yaml_raw)

        # ── Parse & validate ──────────────────────────────────────────────
        try:
            import yaml as _yaml
            raw_data = _yaml.safe_load(yaml_content)
            if isinstance(raw_data, dict):
                raw_data = _sanitize_recipe_data(raw_data)
            recipe = parse_yaml_recipe(_yaml.dump(raw_data, allow_unicode=True))
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not parse recipe: {exc}")
            console.print("[dim]Raw output:[/dim]\n" + yaml_content)
            raise

        errors = validate_recipe(recipe)
        if errors:
            console.print(f"[yellow]Validation warnings ({len(errors)}):[/yellow]")
            for err in errors:
                console.print(f"  • {err}")

        # ── Step 6 (HF / hybrid): Generate companion runner script ────────
        # (Runner script is generated before component verification so the
        # verifier can see it when checking the entrypoint.)
        if _needs_script:
            console.print(f"\n[bold]6/{total_steps}[/bold] Generating companion runner script…")
            variant_ctx = ""
            if selected_variant:
                variant_ctx = (
                    f"\n## Selected Quantized Variant\n"
                    f"Filename: {selected_variant['filename']}\n"
                    f"Format: {selected_variant['format']}\n"
                    f"Quantization: {selected_variant['quant_label']}\n"
                )
            script_raw = call_agent(
                "hf-script-writer",
                f"Model ID: {model_id}\n"
                f"Pipeline Tag: {hf_info.get('pipeline_tag', '')}\n"
                f"Library: {hf_info.get('library_name', '')}\n\n"
                f"## Model Card\n{(hf_info.get('model_card') or '')[:4000]}\n\n"
                f"## Inference Mapping\n{inference_info}\n\n"
                f"## Model Components\n{model_info}"
                f"{variant_ctx}",
                self.client,
            )
            recipe._runner_script = _extract_python(script_raw)

        # ── Component verification ────────────────────────────────────────
        console.print(f"\n[bold]{total_steps + 1}/{total_steps + 1}[/bold] Verifying components…")
        from kdream.core.verifier import RecipeVerifier
        verifier = RecipeVerifier()
        verification = verifier.verify(recipe, runner_script=recipe._runner_script)

        if verification.warnings:
            console.print(
                f"[yellow]⚠ {len(verification.warnings)} warning(s):[/yellow]"
            )
            for w in verification.warnings:
                console.print(f"  [yellow]{w}[/yellow]")

        if not verification.ok:
            console.print(
                f"\n[bold red]✗ Component verification failed "
                f"({len(verification.errors)} error(s)):[/bold red]"
            )
            for err in verification.errors:
                console.print(f"  [red]{err}[/red]")
            console.print(
                "\n[dim]The recipe has been generated but cannot be used until "
                "the above issues are resolved.[/dim]"
            )
            verification.raise_if_errors()

        console.print("[bold green]✓ All components verified.[/bold green]")

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

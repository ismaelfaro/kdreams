"""Recipe parser and validator for kdream."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
import pydantic
from pydantic import BaseModel, field_validator

from kdream.exceptions import RecipeError


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RecipeSource(BaseModel):
    repo: str
    ref: str = "main"
    install_extras: list[str] = []


class ModelDescriptor(BaseModel):
    name: str
    source: Literal["huggingface", "url", "civitai", "local"]
    id: str
    destination: str
    checksum: str | None = None
    size_gb: float | None = None
    license: str | None = None


class InputSpec(BaseModel):
    type: Literal["string", "integer", "float", "boolean"]
    required: bool = False
    default: Any = None
    description: str = ""
    min: float | None = None
    max: float | None = None


class OutputSpec(BaseModel):
    name: str
    type: Literal["file", "string", "base64", "json"]
    path: str | None = None


class LocalBackendSpec(BaseModel):
    requires_gpu: bool = False
    min_vram_gb: int = 0
    tested_on: list[str] = []


class BackendSpecs(BaseModel):
    local: LocalBackendSpec | None = None


class EntrypointSpec(BaseModel):
    script: str
    type: Literal["python", "cli", "gradio", "fastapi"] = "python"
    args_template: str | None = None
    args_mapping: dict[str, str] = {}      # recipe input name → script variable name
    generated_wrapper: bool = False        # True when kdream generated the entrypoint script


class RecipeMetadata(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str = ""
    tags: list[str] = []
    license: str = "unknown"
    author: str = "community"
    repo: str = ""   # source git repository URL (populated by registry client)

    @field_validator("name")
    @classmethod
    def name_must_be_kebab(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError(f"Recipe name must be lowercase kebab-case, got: {v!r}")
        return v


class Recipe(BaseModel):
    api_version: str = "kdream/v1"
    kind: str = "Recipe"
    metadata: RecipeMetadata
    source: RecipeSource
    models: list[ModelDescriptor] = []
    entrypoint: EntrypointSpec
    inputs: dict[str, InputSpec] = {}
    outputs: list[OutputSpec] = []
    backends: BackendSpecs = BackendSpecs()

    # Transient: generated runner script content (not serialised to YAML).
    # Set by RecipeGeneratorAgent for HuggingFace-sourced recipes.
    _runner_script: str | None = pydantic.PrivateAttr(default=None)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_yaml_recipe(content: str) -> Recipe:
    """Parse a YAML-format kdream recipe."""
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise RecipeError(f"Invalid YAML: {e}") from e

    if not isinstance(data, dict):
        raise RecipeError("Recipe YAML must be a mapping at the top level.")

    try:
        # Normalise top-level keys
        metadata = RecipeMetadata(**data.get("metadata", {}))
        source_raw = data.get("source", {})
        source = RecipeSource(**source_raw) if source_raw else RecipeSource(repo="")

        models = [ModelDescriptor(**m) for m in data.get("models", [])]
        entrypoint = EntrypointSpec(**data.get("entrypoint", {"script": "run.py"}))

        inputs: dict[str, InputSpec] = {}
        for k, v in data.get("inputs", {}).items():
            inputs[k] = InputSpec(**v)

        outputs = [OutputSpec(**o) for o in data.get("outputs", [])]

        backends_raw = data.get("backends", {})
        backends = BackendSpecs(
            local=LocalBackendSpec(**backends_raw["local"]) if "local" in backends_raw else None
        )

        return Recipe(
            api_version=data.get("apiVersion", "kdream/v1"),
            kind=data.get("kind", "Recipe"),
            metadata=metadata,
            source=source,
            models=models,
            entrypoint=entrypoint,
            inputs=inputs,
            outputs=outputs,
            backends=backends,
        )
    except Exception as e:
        raise RecipeError(f"Failed to parse recipe: {e}") from e


def parse_markdown_recipe(content: str) -> Recipe:
    """Parse a Markdown Skill format kdream recipe (YAML frontmatter)."""
    if not content.startswith("---"):
        raise RecipeError("Markdown recipe must start with '---' frontmatter.")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise RecipeError("Could not find closing '---' for frontmatter.")

    frontmatter_str = parts[1].strip()
    try:
        fm = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        raise RecipeError(f"Invalid frontmatter YAML: {e}") from e

    if not isinstance(fm, dict):
        raise RecipeError("Frontmatter must be a mapping.")

    # Build normalised recipe data from flat frontmatter
    name = fm.get("name", "unnamed-recipe")
    # Ensure kebab-case
    name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    # Parse shorthand model list: ["huggingface:org/name", ...]
    models = []
    for m in fm.get("models", []):
        if isinstance(m, str) and ":" in m:
            src, mid = m.split(":", 1)
            models.append({
                "name": mid.split("/")[-1],
                "source": src,
                "id": mid,
                "destination": f"models/{mid.split('/')[-1]}",
            })
        elif isinstance(m, dict):
            models.append(m)

    yaml_data = {
        "apiVersion": "kdream/v1",
        "kind": "Recipe",
        "metadata": {
            "name": name,
            "version": fm.get("version", "1.0.0"),
            "description": fm.get("description", ""),
            "tags": fm.get("tags", []),
            "license": fm.get("license", "unknown"),
            "author": fm.get("author", "community"),
        },
        "source": {
            "repo": fm.get("repo", ""),
            "ref": fm.get("ref", "main"),
        },
        "models": models,
        "entrypoint": {
            "script": fm.get("entrypoint", "run.py"),
            "type": fm.get("entrypoint_type", "python"),
        },
        "inputs": fm.get("inputs", {}),
        "outputs": fm.get("outputs", []),
        "backends": fm.get("backends", {}),
    }

    return parse_yaml_recipe(yaml.dump(yaml_data))


def load_recipe(path_or_name: str) -> Recipe:
    """Load a recipe from a local file path or detect format automatically."""
    path = Path(path_or_name)
    if not path.exists():
        raise RecipeError(f"Recipe file not found: {path_or_name}")

    content = path.read_text(encoding="utf-8")
    if content.strip().startswith("---"):
        return parse_markdown_recipe(content)
    return parse_yaml_recipe(content)


def validate_recipe(recipe: Recipe) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []

    if not recipe.metadata.name:
        errors.append("metadata.name is required")

    if not recipe.source.repo:
        errors.append("source.repo is required")

    if not recipe.entrypoint.script:
        errors.append("entrypoint.script is required")

    for name, spec in recipe.inputs.items():
        if spec.required and spec.default is not None:
            errors.append(
                f"Input '{name}' is marked required but also has a default value — "
                "consider using required=false"
            )

    return errors


def recipe_to_yaml(recipe: Recipe) -> str:
    """Serialise a Recipe to YAML string."""
    data: dict[str, Any] = {
        "apiVersion": recipe.api_version,
        "kind": recipe.kind,
        "metadata": recipe.metadata.model_dump(),
        "source": recipe.source.model_dump(),
        "models": [m.model_dump(exclude_none=True) for m in recipe.models],
        "entrypoint": recipe.entrypoint.model_dump(exclude_none=True),
        "inputs": {k: v.model_dump(exclude_none=True) for k, v in recipe.inputs.items()},
        "outputs": [o.model_dump(exclude_none=True) for o in recipe.outputs],
        "backends": {
            k: v
            for k, v in recipe.backends.model_dump().items()
            if v is not None
        },
    }
    return yaml.dump(data, default_flow_style=False, sort_keys=False)

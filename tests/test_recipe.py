"""Tests for kdream.core.recipe — parsing and validation."""
from __future__ import annotations

import pytest

from kdream.core.recipe import (
    InputSpec,
    Recipe,
    RecipeMetadata,
    RecipeSource,
    EntrypointSpec,
    BackendSpecs,
    load_recipe,
    parse_markdown_recipe,
    parse_yaml_recipe,
    recipe_to_yaml,
    validate_recipe,
)
from kdream.exceptions import RecipeError


class TestYAMLRecipeParsing:
    def test_parse_valid_yaml(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        assert recipe.metadata.name == "test-model"
        assert recipe.metadata.version == "1.0.0"
        assert recipe.source.repo == "https://github.com/test/test-model"
        assert len(recipe.models) == 1
        assert recipe.models[0].source == "huggingface"
        assert recipe.models[0].id == "test-org/test-model"

    def test_parse_inputs(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        assert "prompt" in recipe.inputs
        assert recipe.inputs["prompt"].type == "string"
        assert recipe.inputs["prompt"].required is True
        assert "steps" in recipe.inputs
        assert recipe.inputs["steps"].type == "integer"
        assert recipe.inputs["steps"].default == 30
        assert recipe.inputs["steps"].min == 1
        assert recipe.inputs["steps"].max == 100

    def test_parse_outputs(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        assert len(recipe.outputs) == 1
        assert recipe.outputs[0].name == "image"
        assert recipe.outputs[0].type == "file"

    def test_parse_backends(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        assert recipe.backends.local is not None
        assert recipe.backends.local.requires_gpu is True
        assert recipe.backends.local.min_vram_gb == 8
        assert "cuda" in recipe.backends.local.tested_on

    def test_parse_entrypoint(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        assert recipe.entrypoint.script == "run.py"
        assert recipe.entrypoint.type == "python"

    def test_invalid_yaml_raises(self):
        with pytest.raises((RecipeError, Exception)):
            parse_yaml_recipe("not: valid: yaml: :")

    def test_non_dict_yaml_raises(self):
        with pytest.raises(RecipeError):
            parse_yaml_recipe("- item1\n- item2\n")

    def test_api_version_preserved(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        assert recipe.api_version == "kdream/v1"


class TestMarkdownRecipeParsing:
    def test_parse_valid_markdown(self, sample_markdown_recipe):
        recipe = parse_markdown_recipe(sample_markdown_recipe)
        assert recipe.metadata.name == "test-model-md"
        assert recipe.source.repo == "https://github.com/test/test-model"

    def test_markdown_models_parsed(self, sample_markdown_recipe):
        recipe = parse_markdown_recipe(sample_markdown_recipe)
        assert len(recipe.models) >= 1
        assert recipe.models[0].source == "huggingface"
        assert recipe.models[0].id == "test-org/test-model"

    def test_no_frontmatter_raises(self):
        with pytest.raises(RecipeError):
            parse_markdown_recipe("# Just a markdown doc\nNo frontmatter here.")


class TestLoadRecipe:
    def test_load_from_yaml_file(self, sample_recipe_file):
        recipe = load_recipe(str(sample_recipe_file))
        assert recipe.metadata.name == "test-model"

    def test_load_from_markdown_file(self, tmp_path, sample_markdown_recipe):
        md_file = tmp_path / "test.md"
        md_file.write_text(sample_markdown_recipe)
        recipe = load_recipe(str(md_file))
        assert recipe.metadata.name == "test-model-md"

    def test_load_nonexistent_raises(self):
        with pytest.raises(RecipeError):
            load_recipe("/nonexistent/path/recipe.yaml")


class TestValidateRecipe:
    def test_valid_recipe_no_errors(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        errors = validate_recipe(recipe)
        assert errors == []

    def test_empty_repo_flagged(self):
        meta = RecipeMetadata(name="test")
        source = RecipeSource(repo="")
        entrypoint = EntrypointSpec(script="run.py")
        recipe = Recipe(metadata=meta, source=source, entrypoint=entrypoint)
        errors = validate_recipe(recipe)
        assert any("repo" in e for e in errors)


class TestRecipeToYAML:
    def test_roundtrip(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        yaml_out = recipe_to_yaml(recipe)
        recipe2 = parse_yaml_recipe(yaml_out)
        assert recipe2.metadata.name == recipe.metadata.name
        assert recipe2.source.repo == recipe.source.repo
        assert len(recipe2.inputs) == len(recipe.inputs)

    def test_output_is_string(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        yaml_out = recipe_to_yaml(recipe)
        assert isinstance(yaml_out, str)
        assert "apiVersion" in yaml_out


class TestInputValidation:
    """Test input validation via AbstractBackend.validate_inputs."""

    def _backend(self):
        from kdream.backends.local import LocalBackend
        return LocalBackend.__new__(LocalBackend)

    def test_passes_for_valid_inputs(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        errors = self._backend().validate_inputs(recipe, {"prompt": "test", "steps": 30})
        assert errors == []

    def test_fails_for_missing_required(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        errors = self._backend().validate_inputs(recipe, {})
        assert any("prompt" in e for e in errors)

    def test_fails_below_min(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        errors = self._backend().validate_inputs(recipe, {"prompt": "x", "steps": 0})
        assert any("steps" in e for e in errors)

    def test_fails_above_max(self, sample_yaml_recipe):
        recipe = parse_yaml_recipe(sample_yaml_recipe)
        errors = self._backend().validate_inputs(recipe, {"prompt": "x", "steps": 999})
        assert any("steps" in e for e in errors)

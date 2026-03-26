"""Tests for kdream.core.verifier — RecipeVerifier and VerificationResult."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kdream.core.verifier import (
    ComponentIssue,
    RecipeVerifier,
    VerificationResult,
    verify_recipe,
)
from kdream.exceptions import RecipeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RECIPE_YAML_FULL = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: test-verify-model
  version: 1.0.0
  description: Verification test model
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test-org/test-repo
  ref: main
entrypoint:
  script: run.py
  type: python
models:
  - name: weights
    source: huggingface
    id: test-org/test-model
    destination: models/weights
inputs:
  prompt:
    type: string
    required: true
outputs:
  - name: result
    type: string
"""

RECIPE_YAML_NO_MODELS = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: no-models-recipe
  version: 1.0.0
  description: Recipe without models
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test-org/test-repo
  ref: main
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
"""

RECIPE_YAML_URL_MODEL = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: url-model-recipe
  version: 1.0.0
  description: Recipe with URL model
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test-org/test-repo
  ref: main
entrypoint:
  script: run.py
  type: python
models:
  - name: weights
    source: url
    id: https://example.com/model.safetensors
    destination: models/weights
"""

RECIPE_YAML_LOCAL_MODEL = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: local-model-recipe
  version: 1.0.0
  description: Recipe with local model
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test-org/test-repo
  ref: main
entrypoint:
  script: run.py
  type: python
models:
  - name: weights
    source: local
    id: local-weights
    destination: models/weights
"""

RECIPE_YAML_HF_REPO = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: hf-repo-recipe
  version: 1.0.0
  description: HF-sourced recipe
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://huggingface.co/test-org/test-model
  ref: main
entrypoint:
  script: run.py
  type: python
"""

RECIPE_YAML_CIVITAI = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: civitai-recipe
  version: 1.0.0
  description: CivitAI model
  tags: [test]
  license: unknown
  author: test
source:
  repo: https://github.com/test-org/test-repo
  ref: main
entrypoint:
  script: run.py
  type: python
models:
  - name: lora
    source: civitai
    id: "12345"
    destination: models/lora
"""


def _parse(yaml_str: str):
    from kdream.core.recipe import parse_yaml_recipe
    return parse_yaml_recipe(yaml_str)


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:
    def test_ok_when_no_issues(self):
        result = VerificationResult()
        assert result.ok is True

    def test_ok_with_only_warnings(self):
        result = VerificationResult(issues=[
            ComponentIssue("warning", "models", "no models declared"),
        ])
        assert result.ok is True

    def test_not_ok_with_error(self):
        result = VerificationResult(issues=[
            ComponentIssue("error", "entrypoint", "script not found"),
        ])
        assert result.ok is False

    def test_errors_property(self):
        result = VerificationResult(issues=[
            ComponentIssue("error", "entrypoint", "missing"),
            ComponentIssue("warning", "source-repo", "slow"),
            ComponentIssue("error", "model:weights", "404"),
        ])
        assert len(result.errors) == 2
        assert all(i.severity == "error" for i in result.errors)

    def test_warnings_property(self):
        result = VerificationResult(issues=[
            ComponentIssue("error", "entrypoint", "missing"),
            ComponentIssue("warning", "source-repo", "slow"),
        ])
        assert len(result.warnings) == 1

    def test_raise_if_errors_raises_recipe_error(self):
        result = VerificationResult(issues=[
            ComponentIssue("error", "entrypoint", "script not found"),
        ])
        with pytest.raises(RecipeError, match="verification failed"):
            result.raise_if_errors()

    def test_raise_if_errors_does_not_raise_for_warnings_only(self):
        result = VerificationResult(issues=[
            ComponentIssue("warning", "models", "no models declared"),
        ])
        result.raise_if_errors()  # Should not raise

    def test_raise_if_errors_message_includes_all_errors(self):
        result = VerificationResult(issues=[
            ComponentIssue("error", "model:weights", "HF model not found"),
            ComponentIssue("error", "entrypoint", "script missing"),
        ])
        with pytest.raises(RecipeError) as exc_info:
            result.raise_if_errors()
        msg = str(exc_info.value)
        assert "model:weights" in msg
        assert "entrypoint" in msg


# ---------------------------------------------------------------------------
# ComponentIssue
# ---------------------------------------------------------------------------

class TestComponentIssue:
    def test_str_error(self):
        issue = ComponentIssue("error", "entrypoint", "script not found")
        assert "✗" in str(issue)
        assert "entrypoint" in str(issue)
        assert "script not found" in str(issue)

    def test_str_warning(self):
        issue = ComponentIssue("warning", "source-repo", "might be slow")
        assert "⚠" in str(issue)
        assert "source-repo" in str(issue)


# ---------------------------------------------------------------------------
# RecipeVerifier._github_raw_url
# ---------------------------------------------------------------------------

class TestGithubRawUrl:
    def test_converts_github_url(self):
        url = RecipeVerifier._github_raw_url(
            "https://github.com/test-org/test-repo", "main", "run.py"
        )
        assert url == "https://raw.githubusercontent.com/test-org/test-repo/main/run.py"

    def test_non_github_returns_empty(self):
        url = RecipeVerifier._github_raw_url("https://gitlab.com/org/repo", "main", "run.py")
        assert url == ""

    def test_huggingface_url_returns_empty(self):
        url = RecipeVerifier._github_raw_url(
            "https://huggingface.co/org/model", "main", "run.py"
        )
        assert url == ""

    def test_subdirectory_script(self):
        url = RecipeVerifier._github_raw_url(
            "https://github.com/org/repo", "dev", "src/inference.py"
        )
        assert "src/inference.py" in url
        assert "dev" in url


# ---------------------------------------------------------------------------
# RecipeVerifier._check_entrypoint
# ---------------------------------------------------------------------------

class TestCheckEntrypoint:
    def test_ok_when_runner_script_provided(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        issues = verifier._check_entrypoint(recipe, runner_script="print('hello')")
        assert issues == []

    def test_error_when_no_script_defined(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        recipe.entrypoint.script = ""
        issues = verifier._check_entrypoint(recipe, runner_script=None)
        assert any(i.severity == "error" and "entrypoint" in i.component for i in issues)

    def test_ok_when_script_reachable_on_github(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        with patch.object(verifier, "_head", return_value=(True, "")):
            issues = verifier._check_entrypoint(recipe, runner_script=None)
        assert issues == []

    def test_error_when_script_not_found_on_github(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        with patch.object(verifier, "_head", return_value=(False, "HTTP 404")):
            issues = verifier._check_entrypoint(recipe, runner_script=None)
        assert any(i.severity == "error" for i in issues)
        assert any("run.py" in i.message for i in issues)

    def test_warning_for_non_github_repo(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        recipe.source.repo = "https://bitbucket.org/org/repo"
        issues = verifier._check_entrypoint(recipe, runner_script=None)
        assert any(i.severity == "warning" and "entrypoint" in i.component for i in issues)

    def test_error_when_repo_empty_and_no_runner(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        recipe.source.repo = ""
        issues = verifier._check_entrypoint(recipe, runner_script=None)
        assert any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# RecipeVerifier._check_models
# ---------------------------------------------------------------------------

class TestCheckModels:
    def test_warning_when_no_models(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_NO_MODELS)
        issues = verifier._check_models(recipe)
        assert any(i.severity == "warning" and "models" in i.component for i in issues)

    def test_ok_for_existing_hf_model(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        with patch.object(verifier, "_check_hf_model", return_value=(True, "")):
            issues = verifier._check_models(recipe)
        assert not any(i.severity == "error" for i in issues)

    def test_error_for_missing_hf_model(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        with patch.object(verifier, "_check_hf_model",
                          return_value=(False, "HF model not found")):
            issues = verifier._check_models(recipe)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 1
        assert "weights" in errors[0].component

    def test_ok_for_reachable_url_model(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_URL_MODEL)
        with patch.object(verifier, "_head", return_value=(True, "")):
            issues = verifier._check_models(recipe)
        assert not any(i.severity == "error" for i in issues)

    def test_error_for_unreachable_url_model(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_URL_MODEL)
        with patch.object(verifier, "_head", return_value=(False, "HTTP 404")):
            issues = verifier._check_models(recipe)
        assert any(i.severity == "error" and "model:" in i.component for i in issues)

    def test_warning_for_local_model(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_LOCAL_MODEL)
        issues = verifier._check_models(recipe)
        assert any(
            i.severity == "warning" and "local" in i.message.lower()
            for i in issues
        )

    def test_warning_for_civitai_model_not_reachable(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_CIVITAI)
        with patch.object(verifier, "_check_civitai",
                          return_value=(False, "requires API key")):
            issues = verifier._check_models(recipe)
        assert any(i.severity == "warning" and "lora" in i.component for i in issues)

    def test_no_issue_for_reachable_civitai(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_CIVITAI)
        with patch.object(verifier, "_check_civitai", return_value=(True, "")):
            issues = verifier._check_models(recipe)
        assert not any("lora" in i.component for i in issues)


# ---------------------------------------------------------------------------
# RecipeVerifier._check_source_repo
# ---------------------------------------------------------------------------

class TestCheckSourceRepo:
    def test_error_when_repo_empty(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        recipe.source.repo = ""
        issues = verifier._check_source_repo(recipe)
        assert any(i.severity == "error" and "source-repo" in i.component for i in issues)

    def test_ok_for_reachable_github_repo(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        with patch.object(verifier, "_head", return_value=(True, "")):
            issues = verifier._check_source_repo(recipe)
        assert issues == []

    def test_warning_for_unreachable_github_repo(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        with patch.object(verifier, "_head", return_value=(False, "connection refused")):
            issues = verifier._check_source_repo(recipe)
        assert any(i.severity == "warning" and "source-repo" in i.component for i in issues)

    def test_skips_head_for_hf_repo(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_HF_REPO)
        with patch.object(verifier, "_head") as mock_head:
            issues = verifier._check_source_repo(recipe)
            mock_head.assert_not_called()
        assert issues == []


# ---------------------------------------------------------------------------
# RecipeVerifier._check_hf_model
# ---------------------------------------------------------------------------

class TestCheckHfModel:
    def test_returns_true_when_model_exists(self):
        verifier = RecipeVerifier()
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.model_info.return_value = MagicMock()
            ok, msg = verifier._check_hf_model("org/model")
        assert ok is True
        assert msg == ""

    def test_returns_false_when_model_not_found(self):
        verifier = RecipeVerifier()
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.model_info.side_effect = Exception(
                "Repository Not Found: 404"
            )
            ok, msg = verifier._check_hf_model("org/nonexistent-model")
        assert ok is False
        assert "does not exist" in msg or "not found" in msg.lower() or msg

    def test_returns_false_on_network_error(self):
        verifier = RecipeVerifier()
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.model_info.side_effect = Exception("timeout")
            ok, msg = verifier._check_hf_model("org/model")
        assert ok is False
        assert msg  # has some message


# ---------------------------------------------------------------------------
# Full verify() integration
# ---------------------------------------------------------------------------

class TestVerifyIntegration:
    def test_all_ok(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        with patch.object(verifier, "_head", return_value=(True, "")):
            with patch.object(verifier, "_check_hf_model", return_value=(True, "")):
                result = verifier.verify(recipe)
        assert result.ok is True
        assert result.errors == []

    def test_multiple_errors_collected(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_FULL)
        with patch.object(verifier, "_head", return_value=(False, "HTTP 404")):
            with patch.object(verifier, "_check_hf_model",
                              return_value=(False, "model 404")):
                result = verifier.verify(recipe)
        assert not result.ok
        assert len(result.errors) >= 2

    def test_runner_script_satisfies_entrypoint(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_HF_REPO)
        # HF repo, no models, but runner script present
        with patch.object(verifier, "_head", return_value=(True, "")):
            result = verifier.verify(recipe, runner_script="import transformers\n")
        # Entrypoint check should pass (runner script provided)
        entrypoint_errors = [
            i for i in result.issues
            if i.component == "entrypoint" and i.severity == "error"
        ]
        assert entrypoint_errors == []

    def test_warnings_do_not_fail_verification(self):
        verifier = RecipeVerifier()
        recipe = _parse(RECIPE_YAML_LOCAL_MODEL)
        with patch.object(verifier, "_head", return_value=(True, "")):
            result = verifier.verify(recipe)
        # Local model → warning, not an error
        assert result.ok is True
        assert any(i.severity == "warning" for i in result.issues)


# ---------------------------------------------------------------------------
# Module-level verify_recipe convenience
# ---------------------------------------------------------------------------

class TestVerifyRecipeConvenience:
    def test_returns_verification_result(self):
        recipe = _parse(RECIPE_YAML_FULL)
        with patch("kdream.core.verifier.RecipeVerifier.verify") as mock_verify:
            mock_verify.return_value = VerificationResult()
            result = verify_recipe(recipe)
        assert isinstance(result, VerificationResult)

    def test_passes_runner_script(self):
        recipe = _parse(RECIPE_YAML_FULL)
        with patch("kdream.core.verifier.RecipeVerifier.verify") as mock_verify:
            mock_verify.return_value = VerificationResult()
            verify_recipe(recipe, runner_script="# script")
        call_kwargs = mock_verify.call_args
        assert call_kwargs.kwargs.get("runner_script") == "# script"


# ---------------------------------------------------------------------------
# CLI: validate with verification
# ---------------------------------------------------------------------------

class TestValidateCLI:
    @pytest.fixture
    def runner(self):
        from click.testing import CliRunner
        return CliRunner()

    def test_validate_passes_with_ok_components(self, runner, sample_recipe_file):
        from kdream.cli import cli
        with patch("kdream.core.verifier.RecipeVerifier.verify") as mock_verify:
            mock_verify.return_value = VerificationResult()
            result = runner.invoke(cli, ["validate", str(sample_recipe_file)])
        assert result.exit_code == 0
        assert "verified" in result.output.lower()

    def test_validate_fails_on_component_error(self, runner, sample_recipe_file):
        from kdream.cli import cli
        err_result = VerificationResult(issues=[
            ComponentIssue("error", "model:test-weights", "HF model not found"),
        ])
        with patch("kdream.core.verifier.RecipeVerifier.verify", return_value=err_result):
            result = runner.invoke(cli, ["validate", str(sample_recipe_file)])
        assert result.exit_code == 1
        assert "Component verification failed" in result.output

    def test_validate_skip_verify_skips_network(self, runner, sample_recipe_file):
        from kdream.cli import cli
        with patch("kdream.core.verifier.RecipeVerifier.verify") as mock_verify:
            result = runner.invoke(
                cli, ["validate", str(sample_recipe_file), "--skip-verify"]
            )
        mock_verify.assert_not_called()
        assert result.exit_code == 0
        assert "skipped" in result.output.lower()

    def test_validate_shows_warnings_but_exits_0(self, runner, sample_recipe_file):
        from kdream.cli import cli
        warn_result = VerificationResult(issues=[
            ComponentIssue("warning", "models", "no models declared"),
        ])
        with patch("kdream.core.verifier.RecipeVerifier.verify", return_value=warn_result):
            result = runner.invoke(cli, ["validate", str(sample_recipe_file)])
        assert result.exit_code == 0
        assert "⚠" in result.output or "warning" in result.output.lower() or "models" in result.output

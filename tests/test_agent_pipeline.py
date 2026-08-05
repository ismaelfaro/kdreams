"""Tests for kdream.agents.recipe_generator."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

MINIMAL_RECIPE_YAML = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: z-image
  version: 1.0.0
  description: Z-Image AI image generation
  tags: [image-generation]
  license: Apache-2.0
  author: kdream-community
source:
  repo: https://github.com/Tongyi-MAI/Z-Image
  ref: main
  install_extras: []
models: []
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
    required: true
    description: Input prompt
outputs:
  - name: image
    type: file
    path: outputs/{timestamp}.png
backends:
  local:
    requires_gpu: false
    min_vram_gb: 0
    tested_on: [cpu]
"""


class TestLoadSkill:
    def test_all_skill_files_loadable(self):
        from kdream.agents.recipe_generator import load_skill
        skill_names = [
            "repo-inspector",
            "entrypoint-finder",
            "model-locator",
            "parameter-mapper",
            "recipe-writer",
        ]
        for name in skill_names:
            content = load_skill(name)
            assert len(content) > 50, f"Skill '{name}' seems too short: {len(content)} chars"

    def test_skill_strips_frontmatter(self):
        from kdream.agents.recipe_generator import load_skill
        content = load_skill("repo-inspector")
        assert not content.startswith("---")
        assert "name:" not in content.split("\n")[0]

    def test_nonexistent_skill_raises(self):
        from kdream.agents.recipe_generator import load_skill
        with pytest.raises(FileNotFoundError):
            load_skill("nonexistent-skill")


class TestExtractYAML:
    def test_strips_yaml_fence(self):
        from kdream.agents.recipe_generator import _extract_yaml
        text = "Here is your recipe:\n```yaml\napiVersion: kdream/v1\n```"
        result = _extract_yaml(text)
        assert result == "apiVersion: kdream/v1"

    def test_strips_plain_fence(self):
        from kdream.agents.recipe_generator import _extract_yaml
        text = "```\napiVersion: kdream/v1\n```"
        result = _extract_yaml(text)
        assert result == "apiVersion: kdream/v1"

    def test_no_fence_returns_stripped(self):
        from kdream.agents.recipe_generator import _extract_yaml
        text = "  apiVersion: kdream/v1  "
        result = _extract_yaml(text)
        assert result == "apiVersion: kdream/v1"


def _passing_verification():
    """Return a mock VerificationResult that reports no issues."""
    from kdream.core.verifier import VerificationResult
    result = MagicMock(spec=VerificationResult)
    result.ok = True
    result.warnings = []
    result.errors = []
    result.raise_if_errors = MagicMock()
    return result


class TestRecipeGeneratorAgent:
    def _make_agent(self):
        with patch("anthropic.Anthropic"):
            from kdream.agents.recipe_generator import RecipeGeneratorAgent
            return RecipeGeneratorAgent(api_key="test-key")

    def test_init(self):
        agent = self._make_agent()
        assert agent is not None

    def test_generate_z_image_mocked(self, tmp_path):
        """Full pipeline with mocked LLM calls — Z-Image repo."""
        with patch("anthropic.Anthropic") as mock_cls, \
             patch("kdream.core.verifier.RecipeVerifier.verify", return_value=_passing_verification()):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text=MINIMAL_RECIPE_YAML)]
            mock_client.messages.create.return_value = mock_msg

            from kdream.agents.recipe_generator import RecipeGeneratorAgent
            agent = RecipeGeneratorAgent(api_key="test-key")

            fake_repo = {
                "url": "https://github.com/Tongyi-MAI/Z-Image",
                "tree": "README.md\nrun.py\nrequirements.txt",
                "readme": "# Z-Image\nImage generation tool",
                "requirements": "torch\ntransformers",
                "setup_py": "",
                "pyproject": "",
                "candidate_scripts": "",
            }
            out = tmp_path / "z-image.yaml"

            with patch("kdream.agents.recipe_generator.get_repo_info", return_value=fake_repo):
                recipe = agent.generate(
                    repo="https://github.com/Tongyi-MAI/Z-Image",
                    output=str(out),
                )

            assert recipe.metadata.name == "z-image"
            assert out.exists()

    def test_generate_corridorkey_mocked(self, tmp_path):
        """Full pipeline with mocked LLM calls — CorridorKey repo."""
        corridorkey_yaml = MINIMAL_RECIPE_YAML.replace(
            "name: z-image", "name: corridorkey"
        ).replace(
            "repo: https://github.com/Tongyi-MAI/Z-Image",
            "repo: https://github.com/nikopueringer/CorridorKey",
        )

        with patch("anthropic.Anthropic") as mock_cls, \
             patch("kdream.core.verifier.RecipeVerifier.verify", return_value=_passing_verification()):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text=corridorkey_yaml)]
            mock_client.messages.create.return_value = mock_msg

            from kdream.agents.recipe_generator import RecipeGeneratorAgent
            agent = RecipeGeneratorAgent(api_key="test-key")

            fake_repo = {
                "url": "https://github.com/nikopueringer/CorridorKey",
                "tree": "README.md\nmain.py",
                "readme": "# CorridorKey\nA game project",
                "requirements": "",
                "setup_py": "",
                "pyproject": "",
                "candidate_scripts": "",
            }
            out = tmp_path / "corridorkey.yaml"

            with patch("kdream.agents.recipe_generator.get_repo_info", return_value=fake_repo):
                recipe = agent.generate(
                    repo="https://github.com/nikopueringer/CorridorKey",
                    output=str(out),
                )

            assert recipe.metadata.name == "corridorkey"
            assert out.exists()

    def test_generate_uses_repo_info(self, tmp_path):
        """Verify get_repo_info is called with the repo URL."""
        with patch("anthropic.Anthropic") as mock_cls, \
             patch("kdream.core.verifier.RecipeVerifier.verify", return_value=_passing_verification()):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text=MINIMAL_RECIPE_YAML)]
            mock_client.messages.create.return_value = mock_msg

            from kdream.agents.recipe_generator import RecipeGeneratorAgent
            agent = RecipeGeneratorAgent(api_key="test-key")

            fake_repo = {"url": "https://github.com/Tongyi-MAI/Z-Image",
                         "tree": "", "readme": "", "requirements": "",
                         "setup_py": "", "pyproject": "", "candidate_scripts": ""}

            with patch("kdream.agents.recipe_generator.get_repo_info",
                       return_value=fake_repo) as mock_get:
                agent.generate(repo="https://github.com/Tongyi-MAI/Z-Image")
                mock_get.assert_called_once_with("https://github.com/Tongyi-MAI/Z-Image")

    def test_generate_repo_clone_failure_continues(self, tmp_path):
        """If cloning fails, pipeline should continue with URL-only analysis."""
        with patch("anthropic.Anthropic") as mock_cls, \
             patch("kdream.core.verifier.RecipeVerifier.verify", return_value=_passing_verification()):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text=MINIMAL_RECIPE_YAML)]
            mock_client.messages.create.return_value = mock_msg

            from kdream.agents.recipe_generator import RecipeGeneratorAgent
            agent = RecipeGeneratorAgent(api_key="test-key")

            with patch("kdream.agents.recipe_generator.get_repo_info",
                       side_effect=RuntimeError("clone failed")):
                # Should not raise — should fall back to URL-only
                recipe = agent.generate(repo="https://github.com/Tongyi-MAI/Z-Image")
                assert recipe is not None

    def test_generate_target_arch_cuda(self, tmp_path):
        """target_arch='cuda' is recorded in tested_on and passed to agents."""
        cuda_recipe = MINIMAL_RECIPE_YAML + "\nbackends:\n  local:\n    requires_gpu: true\n    min_vram_gb: 8\n    tested_on: []\n"
        with patch("anthropic.Anthropic") as mock_cls, \
             patch("kdream.core.verifier.RecipeVerifier.verify", return_value=_passing_verification()), \
             patch("kdream.agents.recipe_generator._detect_accelerator", return_value="mps"):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text=cuda_recipe)]
            mock_client.messages.create.return_value = mock_msg

            from kdream.agents.recipe_generator import RecipeGeneratorAgent
            agent = RecipeGeneratorAgent(api_key="test-key")

            fake_repo = {"url": "https://github.com/Tongyi-MAI/Z-Image",
                         "tree": "", "readme": "", "requirements": "",
                         "setup_py": "", "pyproject": "", "candidate_scripts": ""}

            with patch("kdream.agents.recipe_generator.get_repo_info", return_value=fake_repo):
                recipe = agent.generate(
                    repo="https://github.com/Tongyi-MAI/Z-Image",
                    target_arch="cuda",
                )

            # tested_on should include the target arch
            assert recipe.backends.local is not None
            assert "cuda" in recipe.backends.local.tested_on

    def test_generate_invalid_arch_raises(self, tmp_path):
        """Invalid target_arch raises ValueError before any API calls."""
        with patch("anthropic.Anthropic"):
            from kdream.agents.recipe_generator import RecipeGeneratorAgent
            agent = RecipeGeneratorAgent(api_key="test-key")
            with pytest.raises(ValueError, match="Unknown target architecture"):
                agent.generate(repo="https://github.com/Tongyi-MAI/Z-Image", target_arch="tpu")

    def test_generate_help_arch_option(self):
        """CLI generate command exposes --arch option."""
        from click.testing import CliRunner

        from kdream.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--arch" in result.output


BROKEN_RECIPE_YAML = "metadata: [unclosed"

VALID_RUNNER_SCRIPT = '''\
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default=None)
    args = p.parse_args()
    print("OUTPUT:/tmp/out.png")

if __name__ == "__main__":
    main()
'''

BROKEN_RUNNER_SCRIPT = "def main(:\n    pass"


class TestAdaptiveRepairLoops:
    """Self-correcting generation: parse/validation/verification errors are
    fed back to the agents until the output is valid."""

    def _mock_client(self, responses: list[str]):
        """Build a mock Anthropic client that returns *responses* in order
        (last response repeats if more calls are made)."""
        client = MagicMock()

        def _create(**kwargs):
            text = responses.pop(0) if len(responses) > 1 else responses[0]
            msg = MagicMock()
            msg.content = [MagicMock(text=text)]
            return msg

        client.messages.create.side_effect = _create
        return client

    def test_recipe_repair_recovers_from_broken_yaml(self):
        """First RecipeWriter output is unparseable; repair attempt fixes it."""
        from kdream.agents.recipe_generator import RecipeGeneratorAgent

        with patch("anthropic.Anthropic"):
            agent = RecipeGeneratorAgent(api_key="test-key")
        agent.client = self._mock_client([BROKEN_RECIPE_YAML, MINIMAL_RECIPE_YAML])

        yaml_content, recipe = agent._generate_recipe_with_repair("write a recipe")
        assert recipe.metadata.name == "z-image"
        # 2 calls: initial + 1 repair
        assert agent.client.messages.create.call_count == 2
        # Repair prompt must contain the error and the previous attempt
        repair_msg = agent.client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
        assert "Errors To Fix" in repair_msg
        assert BROKEN_RECIPE_YAML in repair_msg

    def test_recipe_repair_gives_up_after_max_attempts(self):
        """All attempts unparseable → RecipeError, not silent success."""
        from kdream.agents.recipe_generator import (
            MAX_RECIPE_ATTEMPTS,
            RecipeGeneratorAgent,
        )
        from kdream.exceptions import RecipeError

        with patch("anthropic.Anthropic"):
            agent = RecipeGeneratorAgent(api_key="test-key")
        agent.client = self._mock_client([BROKEN_RECIPE_YAML])

        with pytest.raises(RecipeError, match="parseable recipe"):
            agent._generate_recipe_with_repair("write a recipe")
        assert agent.client.messages.create.call_count == MAX_RECIPE_ATTEMPTS

    def test_script_repair_recovers_from_syntax_error(self):
        """First runner script has a syntax error; repair fixes it."""
        from kdream.agents.recipe_generator import RecipeGeneratorAgent

        with patch("anthropic.Anthropic"):
            agent = RecipeGeneratorAgent(api_key="test-key")
        agent.client = self._mock_client([BROKEN_RUNNER_SCRIPT, VALID_RUNNER_SCRIPT])

        script = agent._generate_script_with_repair("write a script")
        assert "argparse" in script
        assert agent.client.messages.create.call_count == 2

    def test_check_runner_script_contract(self):
        from kdream.agents.recipe_generator import _check_runner_script

        assert _check_runner_script(VALID_RUNNER_SCRIPT) == []
        errors = _check_runner_script("print('hello')")
        assert any("CLI" in e or "argparse" in e for e in errors)
        assert any("OUTPUT:" in e for e in errors)
        assert _check_runner_script("")  # empty script is an error
        syntax_errors = _check_runner_script(BROKEN_RUNNER_SCRIPT)
        assert any("syntax error" in e.lower() for e in syntax_errors)

    def test_verification_failure_triggers_repair(self, tmp_path):
        """Verifier errors are fed back to the RecipeWriter and re-verified."""
        from kdream.core.verifier import ComponentIssue, VerificationResult

        failing = VerificationResult(issues=[ComponentIssue(
            severity="error", component="model:x", message="model does not exist",
        )])
        passing = VerificationResult(issues=[])
        verify_results = [failing, passing]

        with patch("anthropic.Anthropic") as mock_cls, \
             patch("kdream.core.verifier.RecipeVerifier.verify",
                   side_effect=lambda *a, **k: verify_results.pop(0)):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text=MINIMAL_RECIPE_YAML)]
            mock_client.messages.create.return_value = mock_msg

            from kdream.agents.recipe_generator import RecipeGeneratorAgent
            agent = RecipeGeneratorAgent(api_key="test-key")

            fake_repo = {"url": "https://github.com/Tongyi-MAI/Z-Image",
                         "tree": "", "readme": "", "requirements": "",
                         "setup_py": "", "pyproject": "", "candidate_scripts": ""}
            with patch("kdream.agents.recipe_generator.get_repo_info", return_value=fake_repo):
                recipe = agent.generate(repo="https://github.com/Tongyi-MAI/Z-Image")

        assert recipe.metadata.name == "z-image"
        assert not verify_results  # both verify rounds consumed
        # One of the agent calls must be a repair carrying the verifier error
        all_msgs = [c.kwargs["messages"][0]["content"]
                    for c in mock_client.messages.create.call_args_list]
        assert any("model does not exist" in m for m in all_msgs)


class TestResourceEstimate:
    def test_estimate_weight_size(self):
        from kdream.agents.recipe_generator import _estimate_weight_size_gb
        sizes = {
            "model.safetensors": 5 * 1024 ** 3,
            "model.gguf": 3 * 1024 ** 3,
            "README.md": 10_000,
            "config.json": 2_000,
        }
        assert abs(_estimate_weight_size_gb(sizes) - 8.0) < 0.01

    def test_resource_context_warns_when_too_big(self):
        from kdream.agents.recipe_generator import _build_resource_context
        with patch("kdream.agents.recipe_generator._total_system_memory_gb",
                   return_value=32.0):
            ctx = _build_resource_context(
                {"file_sizes": {"model.safetensors": 500 * 1024 ** 3}}
            )
        assert "500.0 GB" in ctx
        assert "WARNING" in ctx

    def test_resource_context_empty_without_weights(self):
        from kdream.agents.recipe_generator import _build_resource_context
        assert _build_resource_context({"file_sizes": {}}) == ""


class TestQuantizedDerivatives:
    """Discovery of quantized derivatives via base_model:quantized:<id>."""

    def _model(self, mid, tags=(), downloads=0):
        m = MagicMock()
        m.id = mid
        m.tags = list(tags)
        m.downloads = downloads
        return m

    def test_derivative_format_detection(self):
        from kdream.agents.recipe_generator import _derivative_format
        assert _derivative_format(self._model("org/M-GGUF", ["gguf"])) == "gguf"
        assert _derivative_format(self._model("org/M-MLX-4bit", ["mlx"])) == "mlx"
        assert _derivative_format(self._model("org/M-nvfp4")) == "nvfp4"
        assert _derivative_format(self._model("org/M-AWQ", ["awq"])) == "awq"
        assert _derivative_format(self._model("org/M-fp8-e4m3fn")) == "fp8"
        assert _derivative_format(self._model("org/M-W4A16-RTN")) == "int4"
        assert _derivative_format(self._model("org/M-plain")) == "unknown"

    def _run_search(self, candidates, interactive=False, accelerator="mps",
                    ram=32.0, weight_gb=(8.0, 4.0)):
        import kdream.agents.recipe_generator as rg
        api = MagicMock()
        api.list_models.return_value = candidates
        hf_api_cls = MagicMock(return_value=api)
        with patch.dict("sys.modules", {}), \
             patch("huggingface_hub.HfApi", hf_api_cls), \
             patch.object(rg, "_detect_accelerator", return_value=accelerator), \
             patch.object(rg, "_total_system_memory_gb", return_value=ram), \
             patch.object(rg, "_derivative_weight_gb", return_value=weight_gb), \
             patch.object(rg, "get_hf_model_info",
                          side_effect=lambda mid: {"model_id": mid}) as get_info:
            result = rg.search_hf_quantized_derivatives(
                "MiniMaxAI/MiniMax-H3", interactive=interactive,
            )
        api.list_models.assert_called_once_with(
            filter="base_model:quantized:MiniMaxAI/MiniMax-H3",
            sort="downloads", limit=20,
        )
        return result, get_info

    def test_auto_picks_compatible_fitting_derivative(self):
        candidates = [
            self._model("a/M-nvfp4", downloads=50_000),        # cuda-only
            self._model("b/M-GGUF", ["gguf"], downloads=80_000),  # mps ok
            self._model("c/M-AWQ", ["awq"], downloads=90_000),    # cuda-only
        ]
        result, _ = self._run_search(candidates, accelerator="mps")
        assert result is not None
        assert result["model_id"] == "b/M-GGUF"
        assert result["quantized_from"] == "MiniMaxAI/MiniMax-H3"

    def test_no_compatible_derivative_returns_none(self):
        candidates = [
            self._model("a/M-nvfp4", downloads=50_000),
            self._model("c/M-AWQ", ["awq"], downloads=90_000),
        ]
        result, get_info = self._run_search(candidates, accelerator="mps")
        assert result is None
        get_info.assert_not_called()

    def test_no_candidates_returns_none(self):
        result, get_info = self._run_search([])
        assert result is None
        get_info.assert_not_called()

    def test_search_failure_returns_none(self):
        import kdream.agents.recipe_generator as rg
        api = MagicMock()
        api.list_models.side_effect = RuntimeError("network down")
        with patch("huggingface_hub.HfApi", MagicMock(return_value=api)):
            assert rg.search_hf_quantized_derivatives("org/model",
                                                      interactive=False) is None

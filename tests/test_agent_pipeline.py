"""Tests for kdream.agents.recipe_generator."""
from __future__ import annotations

from pathlib import Path
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
        with patch("anthropic.Anthropic") as mock_cls:
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

        with patch("anthropic.Anthropic") as mock_cls:
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
        with patch("anthropic.Anthropic") as mock_cls:
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
        with patch("anthropic.Anthropic") as mock_cls:
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

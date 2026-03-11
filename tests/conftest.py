"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_YAML_RECIPE = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: test-model
  version: 1.0.0
  description: A test model recipe
  tags: [test, image-generation]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test/test-model
  ref: main
  install_extras: []
models:
  - name: test-weights
    source: huggingface
    id: test-org/test-model
    destination: models/test
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
    required: true
    description: Input prompt
  steps:
    type: integer
    default: 30
    min: 1
    max: 100
    description: Number of steps
  seed:
    type: integer
    default: -1
    description: Random seed
outputs:
  - name: image
    type: file
    path: outputs/{timestamp}.png
backends:
  local:
    requires_gpu: true
    min_vram_gb: 8
    tested_on: [cuda, mps]
"""

SAMPLE_MARKDOWN_RECIPE = """\
---
name: test-model-md
version: 1.0.0
kdream: true
repo: https://github.com/test/test-model
models:
  - huggingface:test-org/test-model
entrypoint: run.py
---

# Test Model Recipe

A test AI model for kdream testing.
"""


@pytest.fixture
def sample_yaml_recipe() -> str:
    return SAMPLE_YAML_RECIPE


@pytest.fixture
def sample_markdown_recipe() -> str:
    return SAMPLE_MARKDOWN_RECIPE


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / "kdream_cache"
    cache.mkdir()
    return cache


@pytest.fixture
def sample_recipe_file(tmp_path: Path, sample_yaml_recipe: str) -> Path:
    recipe_file = tmp_path / "test-model.yaml"
    recipe_file.write_text(sample_yaml_recipe)
    return recipe_file

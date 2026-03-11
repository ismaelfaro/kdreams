# Recipe Format Guide

## Overview

A **recipe** is a declarative YAML (or Markdown) file that fully describes an AI workload:
its source repo, dependencies, model weights, inference entrypoint, and accepted parameters.

Recipes are versioned, shareable, and backend-agnostic.

---

## YAML Format (full reference)

```yaml
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: my-model           # lowercase kebab-case, required
  version: 1.0.0
  description: What this model does
  tags: [image-generation, diffusion]
  license: Apache-2.0
  author: your-name

source:
  repo: https://github.com/owner/repo    # required
  ref: main                              # branch, tag, or SHA
  install_extras: []                     # extra requirements files

models:
  - name: my-model-weights
    source: huggingface              # huggingface | url | civitai | local
    id: org/repo-name                # HF repo ID, URL, or civitai model ID
    destination: models/my-model
    checksum: <sha256-hex>           # optional, for integrity checking

entrypoint:
  script: scripts/run.py             # relative path from repo root
  type: python                       # python | cli | gradio | fastapi

inputs:
  prompt:
    type: string                     # string | integer | float | boolean
    required: true
    description: Input text prompt
  steps:
    type: integer
    default: 30
    min: 1
    max: 150
    description: Diffusion steps

outputs:
  - name: image
    type: file                       # file | string | base64 | json
    path: outputs/{timestamp}.png    # {timestamp} is substituted at runtime

backends:
  local:
    requires_gpu: true
    min_vram_gb: 8
    tested_on: [cuda, mps]
```

---

## Markdown Skill Format

The Markdown format is optimised for readability. YAML frontmatter holds structured data;
the body is human-readable documentation.

```markdown
---
name: my-model
version: 1.0.0
repo: https://github.com/owner/repo
models:
  - huggingface:org/model-name
entrypoint: scripts/run.py
---

# My Model — kdream Recipe

Description of what this model does.

## Inputs

| Parameter | Type    | Default | Description      |
|-----------|---------|---------|------------------|
| prompt    | string  | —       | Input prompt     |
| steps     | integer | 30      | Diffusion steps  |
```

---

## Writing Your First Recipe

### Step 1 — Auto-generate (recommended)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
kdream generate --repo https://github.com/owner/ai-repo --output my-recipe.yaml
```

Example repos to try:
- `https://github.com/Tongyi-MAI/Z-Image` — image generation
- `https://github.com/nikopueringer/CorridorKey` — demonstrates agent handling of non-ML repos

### Step 2 — Validate

```bash
kdream validate my-recipe.yaml
# ✓ Valid: my-model v1.0.0
```

### Step 3 — Test run

```bash
kdream run ./my-recipe.yaml --prompt "test"
```

### Step 4 — Contribute

Place at `recipes/<category>/my-recipe.yaml` and open a PR.

---

## Input Types

| Type | Examples |
|------|---------|
| `string` | prompt, language, output_format |
| `integer` | steps, seed, width, height, max_tokens |
| `float` | guidance_scale, temperature, top_p, strength |
| `boolean` | use_refiner, fp16, verbose |

---

## Validation Rules

1. `metadata.name` must be lowercase kebab-case
2. `source.repo` is required
3. `entrypoint.script` is required
4. `required: true` inputs must not also have `default` values
5. `min` ≤ `max` for numeric inputs

---

## Contributing Recipes

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full workflow.

**TL;DR:**

```bash
kdream generate --repo <github-url> --output recipes/<category>/<name>.yaml
kdream validate recipes/<category>/<name>.yaml
# open PR
```

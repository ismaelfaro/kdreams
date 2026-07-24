# kdream

> Universal AI Model Runtime & Recipe Platform

[![PyPI version](https://badge.fury.io/py/kdream.svg)](https://pypi.org/project/kdream/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**kdream** is the `npm` for AI inference. Define a recipe, run it anywhere.

> *Clone a GitHub repo. Run it anywhere. Share the recipe.*

---

## Installation

### From PyPI

```bash
pip install kdream
```

### From GitHub (latest)

```bash
pip install git+https://github.com/ismaelfaro/kdreams.git
```

### From a local clone

```bash
git clone https://github.com/ismaelfaro/kdreams.git
cd kdreams
pip install -e .
```

### Recommended: use `uv` (faster)

```bash
# from PyPI
uv add kdream

# from GitHub
uv add git+https://github.com/ismaelfaro/kdreams.git

# from local clone
git clone https://github.com/ismaelfaro/kdreams.git
cd kdreams
uv pip install -e .
```

Verify the install:

```bash
kdream --version
# kdream, version 0.9.1
```

---

## CLI Quick Start

### Browse available recipes

```bash
kdream list
kdream list --tag image-generation
kdream list --search whisper
```

### Run a model

```bash
# Image generation (requires GPU, 8 GB+ VRAM)
kdream run stable-diffusion-xl-base --prompt "a cyberpunk city at sunset"

# Speech transcription (CPU-friendly)
kdream run whisper-large-v3 --audio-file interview.mp3

# Text generation
kdream run llama-3-8b-instruct --prompt "Explain quantum computing simply"

# Pass any recipe input as a flag
kdream run stable-diffusion-xl-base \
  --prompt "a red panda hacker" \
  --steps 40 \
  --guidance-scale 8.0 \
  --seed 42 \
  --width 1024 \
  --height 1024

# Show detailed progress (uv output, subprocess commands)
kdream run whisper-large-v3 --audio-file interview.mp3 --verbose
```

### Run from a local recipe file

```bash
kdream run ./kdream/recipes/image-generation/stable-diffusion-xl-base.yaml \
  --prompt "a red panda hacker"

# or a relative path
kdream run ./my-recipe.yaml --prompt "test"
```

### Pre-install without running

```bash
# Download repo + weights ahead of time
kdream install stable-diffusion-xl-base
kdream install whisper-large-v3

# Show detailed install progress
kdream install llama-3-8b-instruct --verbose
```

### Generate a recipe from any GitHub repo

```bash
# Requires ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=sk-ant-...

# Auto-saves to ./recipes/<category>/<name>.yaml
kdream generate --repo https://github.com/Tongyi-MAI/Z-Image

# Or specify an explicit output path
kdream generate --repo https://github.com/nikopueringer/CorridorKey --output ./my-recipe.yaml
```

Uses a 5-agent Claude pipeline: RepoInspector → EntrypointFinder → ModelLocator → ParameterMapper → RecipeWriter.

### Validate a recipe file

```bash
kdream validate ./my-recipe.yaml
# ✓ Valid: my-recipe v1.0.0
#   Inputs:  3
#   Models:  1
#   Outputs: 1
```

### Manage installed packages

```bash
kdream packages           # list installed
kdream cache info         # disk usage
kdream cache clear        # clear all
kdream cache clear --recipe stable-diffusion-xl-base  # clear one
```

### Run as an MCP server (service mode)

Expose all kdream capabilities as [Model Context Protocol](https://modelcontextprotocol.io/) tools so any MCP-compatible client (Claude Desktop, Cursor, etc.) can invoke them.

```bash
# Install service dependencies first
uv pip install 'kdream[service]'

# stdio transport — for Claude Desktop / Cursor
kdream serve --transport stdio

# HTTP transport — local MCP endpoint at http://127.0.0.1:8765/mcp
kdream serve --transport http

# HTTP + ngrok — public URL for remote access
kdream serve --transport http --ngrok
kdream serve --transport http --ngrok --ngrok-token <token>
# or: export NGROK_AUTHTOKEN=<token>
```

The server exposes six tools: `run_recipe`, `install_recipe`, `list_recipes`, `generate_recipe`, `validate_recipe`, `list_installed`.

### Run inference on a remote kdream server

Connect to a `kdream serve` instance running on another machine (or via ngrok) and run inference remotely:

```bash
# Set the remote URL once
export KDREAM_REMOTE_URL=https://abc123.ngrok.io/mcp

# Run inference on the remote machine
kdream remote run stable-diffusion-xl-base --prompt "red panda hacker"
kdream remote run whisper-large-v3 --audio-file interview.mp3

# Browse recipes available on the remote server
kdream remote list --tag image-generation

# List packages installed on the remote server
kdream remote packages

# Or pass --url per command (skips the env var)
kdream remote run llama-3-8b-instruct \
  --url http://192.168.1.10:8765/mcp \
  --prompt "Explain transformers"
```

**Typical workflow — GPU machine → laptop:**

| GPU machine | Your laptop |
|---|---|
| `kdream serve --transport http --ngrok` | `export KDREAM_REMOTE_URL=<ngrok-url>/mcp` |
| Prints public ngrok URL | `kdream remote run stable-diffusion-xl-base --prompt "..."` |

---

## Python API

```python
import kdream

# Run inference
result = kdream.run(
    recipe="stable-diffusion-xl-base",   # registry name or local path
    prompt="a hyperrealistic red panda hacker",
    steps=40,
    guidance_scale=8.0,
    seed=42,
    verbose=False,                        # set True to stream subprocess output
)
print(result.outputs["image"])   # /path/to/output.png
print(result.metadata)           # {"backend": "local", "duration_s": 12.3, ...}

# Run from a local file
result = kdream.run(
    recipe="./kdream/recipes/image-generation/stable-diffusion-xl-base.yaml",
    prompt="test",
)

# Pre-install only
pkg = kdream.install("whisper-large-v3")
print(pkg.path)    # ~/.kdream/cache/whisper-large-v3
print(pkg.ready)   # True

# Browse recipes
for r in kdream.list_recipes(tags=["audio"]):
    print(r.name, r.description)

# Generate a recipe with AI agents (auto-saves to ./recipes/<category>/<name>.yaml)
recipe = kdream.generate_recipe(
    repo="https://github.com/Tongyi-MAI/Z-Image",
)
```

---

## How It Works

```
kdream run stable-diffusion-xl-base --prompt "..."
      │
      ├─ 1. Recipe Resolution    → GitHub registry → bundled package recipes
      ├─ 2. Dependency Install   → uv venv + uv pip install (isolated per recipe)
      ├─ 3. Model Download       → HuggingFace / CIVITAI / URL
      ├─ 4. Backend Selection    → local GPU / Colab / RunPod (roadmap)
      ├─ 5. Inference Execution  → subprocess with mapped parameters
      └─ 6. Output Return        → file path / string / base64
```

Second run is fast — repo, venv, and weights are cached at `~/.kdream/cache/`.

---

## Available Recipes

| Recipe | Category | VRAM | Description |
|--------|----------|------|-------------|
| `stable-diffusion-xl-base` | image-generation | 8 GB | SDXL 1.0 text-to-image |
| `flux-1-dev` | image-generation | 16 GB | FLUX.1 [dev] by Black Forest Labs |
| `z-image-turbo` | image-generation | 8 GB | Z-Image fast text-to-image |
| `llama-3-8b-instruct` | text-generation | 16 GB | Meta Llama 3.1 8B chat |
| `mistral-7b-v03` | text-generation | 14 GB | Mistral 7B instruction-tuned |
| `whisper-large-v3` | audio | CPU | OpenAI Whisper transcription |
| `musicgen-large` | audio | 16 GB | Meta MusicGen music generation |
| `wan-2-1-t2v` | video-generation | 8 GB | Wan 2.1 text-to-video |

---

## CLI Reference

```
kdream run <recipe> [OPTIONS]
  --backend TEXT          Compute backend: local|colab|runpod  [default: local]
  --cache-dir TEXT        Override default cache (~/.kdream/cache)
  --force-reinstall       Force re-install even if cached
  --verbose, -v           Stream subprocess output (uv logs, commands, stderr)
  --prompt TEXT           Text prompt
  --negative-prompt TEXT  Negative prompt
  --steps INT             Inference steps
  --guidance-scale FLOAT  Guidance scale
  --seed INT              Random seed (-1 for random)
  --width / --height INT  Output dimensions
  -- KEY VALUE            Any additional recipe input

kdream install <recipe> [--backend TEXT] [--cache-dir TEXT] [--verbose, -v]
kdream list [--tag TAG]... [--backend TEXT] [--search TEXT]
kdream generate --repo URL [--output FILE] [--publish]
  # --output omitted: auto-saves to ./recipes/<category>/<name>.yaml
kdream validate <recipe-file>
kdream packages [--cache-dir TEXT]
kdream cache info [--cache-dir TEXT]
kdream cache clear [--recipe NAME] [--cache-dir TEXT]

kdream serve [OPTIONS]                    # start MCP server (requires kdream[service])
  --port INT              Port to bind  [default: 8765]
  --host TEXT             Host to bind  [default: 127.0.0.1]
  --transport [stdio|http]  MCP transport  [default: http]
  --ngrok                 Expose publicly via ngrok tunnel
  --ngrok-token TEXT      ngrok auth token (or NGROK_AUTHTOKEN env var)

kdream remote run <recipe> --url URL [OPTIONS]   # run inference on a remote server
  --url TEXT              Remote MCP server URL (or KDREAM_REMOTE_URL env var)
  --prompt / --steps / --seed / --width / --height / --guidance-scale / --output-dir
  --backend TEXT          Backend on the remote server  [default: local]

kdream remote list --url URL [--tag TAG]... [--backend TEXT]
kdream remote packages --url URL
```

---

## System Requirements

| Requirement | Spec |
|-------------|------|
| Python | 3.10+ |
| UV | 0.4.0+ (auto-installed if absent) |
| OS | macOS 12+, Ubuntu 20.04+, Windows 11 (WSL2) |
| Storage | 20 GB+ free (model-dependent) |
| GPU | NVIDIA 8 GB+ VRAM (CUDA) or Apple Silicon (MPS) — optional for some recipes |

---

## Architecture

```
kdream/
  ├── core/
  │   ├── recipe.py      # Recipe parser (YAML + Markdown) + Pydantic validation
  │   ├── registry.py    # Community recipe registry client
  │   └── runner.py      # Backend orchestrator
  ├── backends/
  │   ├── local.py       # ✅ Local GPU/CPU (Phase 1)
  │   ├── colab.py       # 🔜 Google Colab (Phase 2)
  │   └── runpod.py      # 🔜 RunPod.io (Phase 3)
  ├── agents/
  │   ├── recipe_generator.py  # Multi-agent recipe generator (Claude)
  │   └── skills/              # Agent system prompt Markdown files
  ├── service/           # MCP server & remote client (kdream[service])
  │   ├── mcp_server.py  # FastMCP server with all kdream tools
  │   ├── mcp_client.py  # Async MCP client (used by `kdream remote`)
  │   └── ngrok_tunnel.py  # ngrok tunnel context manager
  └── recipes/           # Bundled recipes (shipped with the package)
      ├── image-generation/
      ├── text-generation/
      ├── audio/
      └── video-generation/
```

---

## Contributing

```bash
# Dev setup
git clone https://github.com/ismaelfaro/kdreams.git
cd kdreams
uv pip install -e ".[dev]"
uv pip install "mcp>=1.6" "pyngrok>=7.0"   # optional: for service/remote features
.venv/bin/python -m pytest tests/

# Add a recipe
kdream generate --repo https://github.com/owner/repo
# Generated recipe auto-saved to kdream/recipes/<category>/name.yaml
kdream validate kdream/recipes/<category>/name.yaml
# open PR
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Recipe YAML files: Creative Commons CC-BY 4.0.

> AI model weights referenced in recipes carry their own licenses. kdream surfaces the model license in recipe metadata but does not distribute model weights.

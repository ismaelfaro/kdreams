# API Reference

## Python API

### `kdream.run()`

Run inference from a recipe.

```python
result = kdream.run(
    recipe="stable-diffusion-xl-base",   # registry name or local path
    backend="local",                      # local | colab | runpod
    cache_dir=None,                       # override ~/.kdream/cache
    force_reinstall=False,
    verbose=False,                        # stream subprocess output
    prompt="a red panda hacker",
    steps=40,
    seed=42,
)
```

**Returns:** `RunResult`

```python
result.outputs   # dict[str, str] — output_name → file path or value
result.metadata  # dict — timing, backend, recipe info
result.success   # bool
result.error     # str | None
```

---

### `kdream.install()`

Pre-install without running.

```python
pkg = kdream.install(
    recipe="stable-diffusion-xl-base",
    backend="local",
    cache_dir=None,
    verbose=False,   # stream subprocess output
)
pkg.path         # Path to installation directory
pkg.ready        # True when all models downloaded
pkg.venv_path    # Path to UV venv
pkg.repo_path    # Path to cloned repo
```

---

### `kdream.list_recipes()`

```python
recipes = kdream.list_recipes(
    tags=["image-generation"],   # optional filter
    backend="local",             # optional filter
)
for r in recipes:
    print(r.name, r.version, r.tags)
```

---

### `kdream.generate_recipe()`

Generate a recipe from a GitHub repo using AI agents.

```python
recipe = kdream.generate_recipe(
    repo="https://github.com/Tongyi-MAI/Z-Image",
    output="./recipes/z-image.yaml",
    publish=False,
)
```

Requires `ANTHROPIC_API_KEY` environment variable.

---

### `kdream.load_recipe()`

```python
recipe = kdream.load_recipe("./my-recipe.yaml")
recipe.metadata.name    # str
recipe.inputs           # dict[str, InputSpec]
recipe.models           # list[ModelDescriptor]
```

---

### `kdream.validate_recipe()`

```python
errors = kdream.validate_recipe(recipe)
# [] means valid
```

---

## CLI Reference

```
kdream run <recipe> [OPTIONS]
  --backend TEXT          local|colab|runpod [default: local]
  --cache-dir TEXT        Override cache directory
  --force-reinstall       Force re-install
  --verbose, -v           Stream subprocess output (uv logs, commands, stderr)
  --prompt TEXT           Text prompt
  --negative-prompt TEXT  Negative prompt
  --steps INT             Inference steps
  --guidance-scale FLOAT  Guidance scale
  --seed INT              Random seed
  --width/--height INT    Image dimensions
  --output-dir TEXT       Directory to save outputs
  -- KEY VALUE            Any additional recipe input

kdream install <recipe> [--verbose, -v]
kdream list [--tag TAG] [--search QUERY]
kdream generate --repo URL [--output FILE]
  # When --output is omitted, auto-saves to ./recipes/<category>/<name>.yaml
kdream validate <file>
kdream packages
kdream cache info
kdream cache clear [--recipe NAME]
```

---

## Exceptions

```python
from kdream.exceptions import (
    KdreamError,        # base
    RecipeError,        # parse/validation failure
    RegistryError,      # registry fetch failure
    BackendError,       # backend install/run failure
    ModelDownloadError, # model download failure
)
```

# Contributing to kdream

Thank you for your interest in contributing to kdream! This document describes how to get involved, whether you are submitting a new recipe, fixing a bug, or improving the library itself.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Governance](#governance)
- [Fork Workflow](#fork-workflow)
- [Contributing a Recipe](#contributing-a-recipe)
- [Contributing Code](#contributing-code)
- [Running Tests](#running-tests)
- [Style & Linting](#style--linting)
- [Opening a Pull Request](#opening-a-pull-request)

---

## Code of Conduct

This project follows a CODE_OF_CONDUCT. By participating you agree to abide by its terms. Please report unacceptable behavior to the project maintainers.

---

## Governance

kdream follows the **BDFL-N (Benevolent Dictator For Life — Named)** model, sometimes written as **BDFN**. A small group of named maintainers has final say on design decisions and merge authority. Community members are encouraged to open issues and pull requests; maintainers will review and guide contributions toward acceptance.

---

## Fork Workflow

1. **Fork** the repository on GitHub to your personal account.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/kdreams.git
   cd kdreams
   ```
3. Add the upstream remote so you can stay in sync:
   ```bash
   git remote add upstream https://github.com/kdream/kdreams.git
   ```
4. Create a **feature branch** from `main`:
   ```bash
   git checkout -b feat/my-contribution
   ```
5. Make your changes, commit, and push to your fork:
   ```bash
   git push origin feat/my-contribution
   ```
6. Open a Pull Request from your branch against `upstream/main`.

---

## Contributing a Recipe

Recipes are self-contained workflow definitions stored under `recipes/`. They describe how to run a specific AI task (image generation, text generation, audio, video, 3D, etc.) using one or more backends.

### Steps

1. **Generate a recipe scaffold** from an existing repository:
   ```bash
   kdream generate --repo <url>
   ```
   This will introspect the repository and produce a recipe YAML under the appropriate `recipes/<category>/` subdirectory.

2. **Validate** the generated recipe to ensure it conforms to the kdream schema:
   ```bash
   kdream validate recipes/<category>/<your-recipe>/recipe.yaml
   ```
   Fix any validation errors reported before proceeding.

3. **Test** the recipe end-to-end in a local environment:
   ```bash
   kdream run recipes/<category>/<your-recipe>/recipe.yaml
   ```

4. **Open a Pull Request** following the [fork workflow](#fork-workflow). In the PR description:
   - Explain what the recipe does and which model/backend it targets.
   - Include a short sample output or screenshot if possible.
   - Confirm that `kdream validate` passes with no errors.

### Recipe Directory Layout

```
recipes/
  <category>/          # e.g. image-generation, text-generation, audio, video-generation, 3d
    <recipe-name>/
      recipe.yaml      # required: main recipe definition
      README.md        # recommended: usage notes
      assets/          # optional: example inputs/outputs
```

---

## Contributing Code

### Setting Up a Development Environment

kdream uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

1. Install `uv` if you have not already:
   ```bash
   pip install uv
   # or via the official installer:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate    # Windows
   ```

3. Install kdream with all development dependencies:
   ```bash
   uv pip install -e ".[runpod,kubernetes]"
   uv pip install pytest ruff
   ```

### Project Layout

```
kdream/
  __init__.py          # package entry point
  cli.py               # click-based CLI (entry point: kdream)
  core/                # core abstractions (recipe loading, validation, execution)
  backends/            # backend integrations (runpod, kubernetes, local, ...)
  agents/              # agent definitions
    skills/            # reusable agent skill plugins
recipes/               # community recipe library
tests/                 # pytest test suite
docs/                  # documentation sources
```

---

## Running Tests

```bash
pytest
```

The test suite is configured via `[tool.pytest.ini_options]` in `pyproject.toml` and targets the `tests/` directory. All tests must pass before a PR will be merged.

To run a specific test file or test function:

```bash
pytest tests/test_core.py
pytest tests/test_core.py::test_recipe_loading
```

---

## Style & Linting

kdream uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

Check for issues:
```bash
ruff check .
```

Auto-fix safe issues:
```bash
ruff check --fix .
```

Format code:
```bash
ruff format .
```

All submitted code must pass `ruff check` with no errors. CI will enforce this automatically.

---

## Opening a Pull Request

- Keep PRs focused: one feature or fix per PR.
- Write a clear title and description explaining the motivation and approach.
- Reference any related issues with `Closes #<issue-number>`.
- Ensure all tests pass and `ruff check` reports no errors.
- A maintainer will review your PR and may request changes before merging.

Thank you for helping make kdream better!

# kdream

> Universal AI Model Runtime & Recipe Platform

**kdream** is the `npm` for AI inference. Write one YAML file describing an AI workload. Run it anywhere.

## Get Started in 5 Minutes

```bash
# Install
pip install kdream

# Run your first model
kdream run stable-diffusion-xl-base --prompt "a red panda hacker"
```

That's it. kdream handles the repo clone, UV environment, model weights, and inference.

## Guides

- [API Reference](api.md) — Python API and CLI reference
- [Recipe Format](recipes.md) — Writing and contributing recipes
- [Contributing](../CONTRIBUTING.md) — How to contribute

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Recipe** | YAML/Markdown file declaring an AI workload |
| **Backend** | Compute target: `local`, `colab`, `runpod` |
| **Registry** | Community library of recipes on GitHub |
| **Agent** | LLM-powered recipe generator from any GitHub URL |

# kdream Recipe Registry

Community-curated AI model recipes for kdream.

## Categories

| Category | Recipes |
|----------|---------|
| [image-generation](./image-generation/) | SDXL 1.0, FLUX.1 [dev] |
| [text-generation](./text-generation/) | Llama 3.1 8B, Mistral 7B |
| [audio](./audio/) | Whisper Large v3, MusicGen Large |
| [video-generation](./video-generation/) | Wan 2.1 T2V |
| [3d](./3d/) | Coming soon |

## Using a Recipe

```bash
kdream run stable-diffusion-xl-base --prompt "a red panda hacker"
kdream run whisper-large-v3 --audio-file interview.mp3
kdream run llama-3-8b-instruct --prompt "What is the capital of France?"
```

## Contributing a Recipe

```bash
# Auto-generate from a GitHub repo
kdream generate --repo https://github.com/owner/repo --output my-recipe.yaml

# Validate
kdream validate my-recipe.yaml

# Open a PR to add it here
```

Recipe YAML files are licensed under Creative Commons CC-BY 4.0.

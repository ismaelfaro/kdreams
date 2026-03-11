"""kdream-generated CLI runner for z-image-turbo.

Auto-generated because inference.py:main() accepts no parameters — all
inputs were hardcoded variables inside the function body.

This script is bundled with the kdream recipe and copied into the cloned
repo directory at run time by LocalBackend._find_bundled_wrapper().
"""
import argparse
import os
import sys
import time
import warnings

import torch

warnings.filterwarnings("ignore")

# Ensure the repo directory (where this file lives) is on sys.path so that
# `from utils import ...` and `from zimage import ...` resolve correctly.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
os.chdir(_here)

from utils import (  # noqa: E402
    AttentionBackend,
    ensure_model_weights,
    load_from_local_dir,
    set_attention_backend,
)
from zimage import generate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="z-image-turbo kdream runner")
    p.add_argument("--prompt", type=str, required=True,
                   help="Text prompt (English or Chinese)")
    p.add_argument("--width", type=int, default=1024, help="Output width in pixels")
    p.add_argument("--height", type=int, default=1024, help="Output height in pixels")
    p.add_argument("--steps", type=int, default=8,
                   help="Diffusion steps (8 is optimal for the Turbo model)")
    p.add_argument("--guidance-scale", type=float, default=0.0,
                   help="CFG scale (0.0 for Turbo, higher for base Z-Image)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", type=str, default="output.png",
                   help="Output image file path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    dtype = torch.bfloat16
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")

    if torch.cuda.is_available():
        device = "cuda"
        print("Chosen device: cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Chosen device: mps")
    else:
        device = "cpu"
        print("Chosen device: cpu")

    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=False)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    start = time.time()
    images = generate(
        prompt=args.prompt,
        **components,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device).manual_seed(args.seed),
    )
    print(f"Time taken: {time.time() - start:.2f} seconds")
    images[0].save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

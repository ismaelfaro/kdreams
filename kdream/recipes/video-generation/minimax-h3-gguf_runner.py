#!/usr/bin/env python3
"""Generated kdream runner for Abiray/MiniMax-H3-GGUF (MiniMax H3 FL2VA GGUF)."""
import argparse
import os
import sys
import json
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
_kdream_device = os.environ.get("KDREAM_DEVICE", "").strip()
if _kdream_device in ("cuda", "mps", "cpu"):
    device = _kdream_device
elif torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

REPO_ID = "Abiray/MiniMax-H3-GGUF"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_component(file_path: str, dest: str) -> Path:
    """Download a single component file from HuggingFace Hub if not already present."""
    dest_path = Path(dest)
    if dest_path.exists():
        print(f"[cache] {dest_path} already exists, skipping download.")
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching {file_path} from {REPO_ID} ...")
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=file_path,
        local_dir=str(dest_path.parent.parent),
        local_dir_use_symlinks=False,
    )
    local_path = Path(local)
    if local_path.resolve() != dest_path.resolve():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest_path)
    return dest_path


def try_comfyui_inference(args, unet_path, text_encoder_path, video_vae_path, audio_vae_path, output_path):
    """
    Attempt to run inference via ComfyUI if it is installed.
    Returns True on success, False if ComfyUI is not available or fails.
    """
    comfyui_candidates = [
        Path("ComfyUI/main.py"),
        Path(os.environ.get("COMFYUI_PATH", ""), "main.py"),
        Path("/opt/ComfyUI/main.py"),
        Path(os.path.expanduser("~/ComfyUI/main.py")),
    ]
    comfyui_main = None
    for candidate in comfyui_candidates:
        if candidate.exists():
            comfyui_main = candidate
            break

    if comfyui_main is None:
        return False

    comfyui_dir = comfyui_main.parent

    workflow_file = Path(args.workflow)
    if not workflow_file.exists():
        workflow_file = comfyui_dir / args.workflow
    if not workflow_file.exists():
        print(f"[warning] Workflow file {args.workflow} not found; skipping ComfyUI path.")
        return False

    with open(workflow_file) as f:
        workflow = json.load(f)

    for node_id, node in workflow.items():
        cls = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if "CLIPTextEncode" in cls or "text_encode" in cls.lower():
            if "text" in inputs:
                inputs["text"] = args.prompt

        if "UNETLoader" in cls or "gguf" in cls.lower():
            if "unet_name" in inputs:
                inputs["unet_name"] = str(unet_path)

        if "KSampler" in cls:
            inputs["steps"] = args.steps
            inputs["cfg"] = args.guidance_scale
            if args.seed != -1:
                inputs["seed"] = args.seed

        if "EmptyLatentVideo" in cls or "LatentVideo" in cls:
            if "width" in inputs:
                inputs["width"] = args.width
            if "height" in inputs:
                inputs["height"] = args.height
            if "length" in inputs:
                inputs["length"] = int(args.duration * 24)

        if "LoadImage" in cls or "load_image" in cls.lower():
            if args.first_frame_image and "image" in inputs:
                inputs["image"] = str(Path(args.first_frame_image).resolve())

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(workflow, tmp)
        tmp_workflow = tmp.name

    try:
        cmd = [
            sys.executable,
            str(comfyui_main),
            "--workflow", tmp_workflow,
            "--output-directory", str(output_path.parent),
        ]
        if device == "cpu":
            cmd.append("--cpu")
        print(f"[comfyui] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, timeout=3600)
        return result.returncode == 0
    except Exception as e:
        print(f"[comfyui] Error: {e}")
        return False
    finally:
        os.unlink(tmp_workflow)


def run_direct_inference(args, unet_path, text_encoder_path, video_vae_path, audio_vae_path, output_path):
    """
    Attempt direct Python inference using diffusers or a custom MiniMax H3 pipeline.
    Falls back to instructions if no suitable library is found.
    """
    # NOTE: no diffusers fallback on purpose. DiffusionPipeline.from_pretrained
    # on the base repo would download the ~498 GB full-precision model, which
    # defeats the GGUF quantization and cannot fit on machines this recipe
    # targets. The GGUF components require a ComfyUI (+ ComfyUI-GGUF nodes)
    # pipeline for inference.
    print("[warning] No suitable video inference pipeline found for MiniMax H3 GGUF.")
    print("[warning] MiniMax H3 GGUF is primarily designed for use with ComfyUI.")
    print("[warning] Please install ComfyUI and the MiniMax H3 ComfyUI nodes.")
    print("[warning] Workflows: minimax_fl2v_gguf_workflow.json / minimax_ref2va_gguf_workflow.json")
    print("")
    print("Model files have been downloaded to:")
    print(f"  UNet:          {unet_path}")
    print(f"  Text Encoder:  {text_encoder_path}")
    print(f"  Video VAE:     {video_vae_path}")
    print(f"  Audio VAE:     {audio_vae_path}")
    print("")
    print("To run inference manually with ComfyUI:")
    print("  1. Install ComfyUI: https://github.com/comfyanonymous/ComfyUI")
    print("  2. Install MiniMax H3 ComfyUI nodes")
    print("  3. Place the downloaded model files in the appropriate ComfyUI model directories")
    print("  4. Load the workflow JSON and run")

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run MiniMax H3 FL2VA GGUF inference (image-to-video / text-to-video)"
    )

    parser.add_argument("--prompt", type=str, required=True,
                        help="Text description of the video to generate")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="What to exclude from the generated video")
    parser.add_argument("--first-frame-image", type=str, default="",
                        help="Path to first frame image for image-to-video (optional)")
    parser.add_argument("--last-frame-image", type=str, default="",
                        help="Path to last frame image for first-and-last-frame mode (optional)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Output video duration in seconds (4-15)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Output video width in pixels")
    parser.add_argument("--height", type=int, default=768,
                        help="Output video height in pixels")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=6.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 for random)")
    parser.add_argument("--workflow", type=str, default="minimax_fl2v_gguf_workflow.json",
                        help="ComfyUI workflow JSON file to use")
    parser.add_argument("--variant", type=str, default="unet/MiniMax-H3-FL2VA-Q4_K_M.gguf",
                        help="UNet GGUF variant filename within the repo")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save output videos")
    parser.add_argument("--device", type=str, default="",
                        help="Override device (cuda/mps/cpu); defaults to KDREAM_DEVICE auto-detection")

    args = parser.parse_args()

    # Device override
    global device
    if args.device in ("cuda", "mps", "cpu"):
        device = args.device

    print(f"[info] Using device: {device}")
    print(f"[info] UNet variant: {args.variant}")
    print("[warning] MiniMax H3 GGUF requires ~41 GB+ RAM/VRAM. Ensure sufficient memory.")

    if args.seed != -1:
        torch.manual_seed(args.seed)

    # Validate duration
    duration = max(4.0, min(15.0, args.duration))
    if duration != args.duration:
        print(f"[info] Duration clamped to {duration}s (valid range: 4-15s)")
    args.duration = duration

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}.mp4"

    # ---------------------------------------------------------------------------
    # Download all required components
    # ---------------------------------------------------------------------------
    print("\n[phase 1] Downloading model components ...")

    # UNet (GGUF)
    unet_dest = f"models/unet/{Path(args.variant).name}"
    unet_path = download_component(args.variant, unet_dest)

    # Text Encoder (GGUF Q4_K_M — MPS compatible)
    text_encoder_path = download_component(
        "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    )

    # Video VAE
    video_vae_path = download_component(
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "models/vae/minimax_h3_video_vae_fp16.safetensors",
    )

    # Audio VAE
    audio_vae_path = download_component(
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "models/vae/minimax_h3_audio_vae_fp32.safetensors",
    )

    print("\n[phase 1] All components downloaded.")
    print(f"  UNet:          {unet_path}  ({unet_path.stat().st_size / 1e9:.1f} GB)")
    print(f"  Text Encoder:  {text_encoder_path}  ({text_encoder_path.stat().st_size / 1e9:.1f} GB)")
    print(f"  Video VAE:     {video_vae_path}  ({video_vae_path.stat().st_size / 1e9:.1f} GB)")
    print(f"  Audio VAE:     {audio_vae_path}  ({audio_vae_path.stat().st_size / 1e9:.1f} GB)")

    # ---------------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------------
    print("\n[phase 2] Running inference ...")
    print(f"  Prompt:        {args.prompt}")
    print(f"  Resolution:    {args.width}x{args.height}")
    print(f"  Duration:      {args.duration}s")
    print(f"  Steps:         {args.steps}")
    print(f"  Guidance:      {args.guidance_scale}")
    if args.first_frame_image:
        print(f"  First frame:   {args.first_frame_image}")
    if args.last_frame_image:
        print(f"  Last frame:    {args.last_frame_image}")

    # Try ComfyUI first, then direct inference
    success = try_comfyui_inference(
        args, unet_path, text_encoder_path, video_vae_path, audio_vae_path, output_path
    )

    if not success:
        success = run_direct_inference(
            args, unet_path, text_encoder_path, video_vae_path, audio_vae_path, output_path
        )

    if success and output_path.exists():
        print(f"\n[done] Video saved to: {output_path}")
        print(f"OUTPUT:{output_path}")
    else:
        print(f"\n[info] Inference did not produce output at: {output_path}")
        print("[info] Model files are downloaded and ready for use with ComfyUI.")
        print(f"OUTPUT:{output_path}")


if __name__ == "__main__":
    main()
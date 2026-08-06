#!/usr/bin/env python3
"""Pure-Python runner for MiniMax-H3 text/image-to-video+audio.

No ComfyUI, no external server — everything runs in-process with
diffusers (modular pipeline) + transformers. Single command:

    python run.py --prompt "..." [--steps 30 --duration 5 ...]

Memory strategy for 32 GB machines (two sequential phases, components
quantized on load with optimum-quanto int4):

  Phase 1  Qwen3-VL-32B text encoder (int4, ~17 GB) encodes the prompt
           (and first/last keyframes for image-to-video), then is freed.
  Phase 2  MiniMax H3 DiT (int4, ~18 GB) denoises the joint audio+video
           latent; the VAEs decode frames and stereo audio.

Weights come from the official diffusers-format repo
(MiniMaxAI/MiniMax-H3) — only the needed subfolders are downloaded.
"""
import argparse
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ID = "MiniMaxAI/MiniMax-H3"
ALLOW_PATTERNS = [
    "model_index.json", "modular_model_index.json",
    "transformer/*", "text_encoder/*", "vae/*", "audio_vae/*",
    "tokenizer/*", "processor/*", "scheduler/*", "audio_scheduler/*",
]
FPS = 24


def log(msg: str) -> None:
    print(f"[minimax-h3] {msg}", flush=True)


def detect_platform() -> tuple[str, str]:
    """Detect the execution platform. Returns (platform, torch_device):

    - ("nvidia", "cuda")  NVIDIA GPU — CUDA kernels, bnb/quanto quantization
    - ("mac", "mps")      Apple Silicon — Metal via torch-MPS (or MLX, see
                          try_mlx_runtime)
    - ("cpu", "cpu")      everything else
    """
    import torch
    env = os.environ.get("KDREAM_DEVICE", "").strip()
    if env == "cuda":
        return "nvidia", "cuda"
    if env == "mps":
        return "mac", "mps"
    if env == "cpu":
        return "cpu", "cpu"
    if torch.cuda.is_available():
        return "nvidia", "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mac", "mps"
    return "cpu", "cpu"


def try_mlx_runtime():
    """On Apple Silicon, use an MLX-native MiniMax-H3 pipeline when one is
    importable. As of writing no Python MLX port of this model exists (the
    only MLX implementation, mlx-serve, is an external server binary), so
    this probe returns None and we fall back to torch's Metal (MPS) backend —
    which is itself Apple-GPU-accelerated. The hook is here so an
    `mlx_minimax_h3` package gets picked up automatically once published.
    """
    try:
        import importlib
        importlib.import_module("mlx.core")
        return importlib.import_module("mlx_minimax_h3")
    except ImportError:
        return None


def free_memory(device: str) -> None:
    import torch
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Output muxing: frames + stereo audio -> mp4 (h264 + aac) via PyAV
# ---------------------------------------------------------------------------

def save_av(frames, fps: int, audio, sample_rate: int, path: Path) -> None:
    """frames: numpy uint8 [T, H, W, 3]; audio: numpy float [C, S] or None."""
    import av
    import numpy as np

    container = av.open(str(path), mode="w")
    vstream = container.add_stream("h264", rate=fps)
    vstream.width = frames.shape[2]
    vstream.height = frames.shape[1]
    vstream.pix_fmt = "yuv420p"

    astream = None
    if audio is not None and sample_rate:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.shape[0] > audio.shape[1]:  # [S, C] -> [C, S]
            audio = audio.T
        audio = np.clip(audio, -1.0, 1.0)
        layout = "stereo" if audio.shape[0] >= 2 else "mono"
        audio = audio[:2] if layout == "stereo" else audio[:1]
        astream = container.add_stream("aac", rate=int(sample_rate), layout=layout)

    for frame_arr in frames:
        frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
        for packet in vstream.encode(frame):
            container.mux(packet)
    for packet in vstream.encode():
        container.mux(packet)

    if astream is not None:
        aframe = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(audio), format="fltp",
            layout="stereo" if audio.shape[0] == 2 else "mono",
        )
        aframe.sample_rate = int(sample_rate)
        for packet in astream.encode(aframe):
            container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)

    container.close()


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------

def make_block_pipelines(local_path: str):
    from diffusers import MiniMaxH3Blocks
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    blocks = MiniMaxH3Blocks()

    def subset(names):
        return SequentialPipelineBlocks.from_blocks_dict(
            {k: v for k, v in blocks.sub_blocks.items() if k in names}
        )

    encode = subset(("before_encode", "text_encoder", "vae_encoder"))
    denoise = subset(("denoise", "decode"))
    return encode.init_pipeline(local_path), denoise.init_pipeline(local_path)


def component_quant_kwargs(kind: str, quantization: str, dtype, platform: str):
    """kind: 'transformers' | 'diffusers'.

    NVIDIA gets bitsandbytes NF4 (fused CUDA kernels) when requested int4;
    Mac/CPU get optimum-quanto, which runs on Metal/CPU.
    """
    kwargs = {"torch_dtype": dtype}
    if quantization not in ("int4", "int8"):
        return kwargs
    if platform == "nvidia" and quantization == "int4":
        try:
            if kind == "transformers":
                from transformers import BitsAndBytesConfig
            else:
                from diffusers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            return kwargs
        except ImportError:
            pass  # fall through to quanto
    if kind == "transformers":
        from transformers import QuantoConfig
        kwargs["quantization_config"] = QuantoConfig(weights=quantization)
    else:
        from diffusers import QuantoConfig
        kwargs["quantization_config"] = QuantoConfig(weights_dtype=quantization)
    return kwargs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MiniMax-H3 video+audio generation — pure Python (diffusers)"
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--first-frame-image", type=str, default="")
    parser.add_argument("--last-frame-image", type=str, default="")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconds at 24 fps (snapped to the model's frame grid)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--quantization", type=str, default="int4",
                        choices=["int4", "int8", "bf16"],
                        help="On-load weight quantization (quanto). bf16 = none.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    import torch
    platform, device = detect_platform()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    log(f"platform={platform} device={device} dtype={dtype} "
        f"quantization={args.quantization}")

    if platform == "mac":
        mlx_runtime = try_mlx_runtime()
        if mlx_runtime is not None:
            log("MLX runtime for MiniMax-H3 detected — using MLX execution.")
            return mlx_runtime.run(vars(args))
        log("No Python MLX pipeline for MiniMax-H3 available — using torch "
            "Metal (MPS) backend (Apple-GPU accelerated).")

    if os.environ.get("KDREAM_FORCE_IPV4") == "1":
        # Some networks have a broken/throttled IPv6 path to the HF CDN.
        import socket
        _orig_gai = socket.getaddrinfo
        socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: \
            _orig_gai(host, port, socket.AF_INET, type, proto, flags)

    from huggingface_hub import snapshot_download
    log("Ensuring model components (downloads only what is missing) ...")
    last_exc = None
    for attempt in range(1, 11):
        try:
            local_path = snapshot_download(REPO_ID, allow_patterns=ALLOW_PATTERNS)
            break
        except Exception as exc:  # transient CDN disconnects are common at this size
            last_exc = exc
            log(f"download attempt {attempt}/10 failed ({type(exc).__name__}); retrying ...")
            time.sleep(min(60, 5 * attempt))
    else:
        raise RuntimeError(f"Could not download model components: {last_exc}")

    seed = args.seed if args.seed != -1 else int.from_bytes(os.urandom(4), "big")
    generator = torch.Generator("cpu").manual_seed(seed)
    num_frames = max(5, int(args.duration * FPS))

    encode_pipe, denoise_pipe = make_block_pipelines(local_path)

    # ── Phase 1: text (+image) encoding ──────────────────────────────────
    t0 = time.time()
    log("Phase 1/2: loading text encoder "
        f"({args.quantization}) — first run quantizes ~67 GB of shards, be patient ...")
    encode_pipe.load_components(
        names=["tokenizer", "processor", "image_processor"],
    )
    encode_pipe.load_components(
        names=["text_encoder"],
        **component_quant_kwargs("transformers", args.quantization, dtype, platform),
    )

    enc_inputs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": num_frames,
        "generator": generator,
    }
    if args.first_frame_image or args.last_frame_image:
        encode_pipe.load_components(names=["vae"], torch_dtype=dtype)
        if args.first_frame_image:
            enc_inputs["image"] = load_image(args.first_frame_image)
        if args.last_frame_image:
            enc_inputs["last_image"] = load_image(args.last_frame_image)

    # On NVIDIA the encoder fits VRAM comfortably once quantized; move it.
    # On Mac/CPU keep the encode phase on CPU: unified memory means MPS gains
    # little for one forward pass, and the .to("mps") copy transiently doubles
    # the quantized encoder's footprint (observed OOM-kill on 32 GB machines).
    if platform == "nvidia":
        encode_pipe.to(device)
    state = encode_pipe(**enc_inputs)
    log(f"Phase 1 done in {time.time() - t0:.0f}s")

    # Detach encoder outputs to CPU and free the encoder before phase 2.
    for key, value in list(state.values.items()):
        if hasattr(value, "detach"):
            state.values[key] = value.detach().to("cpu")
    del encode_pipe
    free_memory(device)

    # ── Phase 2: denoise + decode ────────────────────────────────────────
    t1 = time.time()
    log(f"Phase 2/2: loading DiT ({args.quantization}) + VAEs ...")
    denoise_pipe.load_components(
        names=["scheduler", "audio_scheduler", "video_processor"],
    )
    denoise_pipe.load_components(names=["vae", "audio_vae"], torch_dtype=dtype)
    denoise_pipe.load_components(
        names=["transformer"],
        **component_quant_kwargs("diffusers", args.quantization, dtype, platform),
    )
    # transformer_ref is only needed for reference-to-video; skip it.

    denoise_pipe.to(device)
    log(f"Sampling: {args.steps} steps, {num_frames} frames "
        f"@ {args.width}x{args.height}, seed={seed}")
    out = denoise_pipe(
        state=state,
        generator=generator,
        num_inference_steps=args.steps,
        output_type="np",
        output=["videos", "audio", "sampling_rate"],
    )
    if isinstance(out, dict):
        videos, audio, sampling_rate = (
            out.get("videos"), out.get("audio"), out.get("sampling_rate"),
        )
    elif isinstance(out, (list, tuple)):
        videos, audio, sampling_rate = out
    else:  # a PipelineState
        videos, audio, sampling_rate = (
            out.get("videos"), out.get("audio"), out.get("sampling_rate"),
        )
    log(f"Phase 2 done in {time.time() - t1:.0f}s")

    import numpy as np
    frames = np.asarray(videos)
    while frames.ndim > 4:  # [B, T, H, W, C] -> [T, H, W, C]
        frames = frames[0]
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)

    audio_arr = None
    if audio is not None:
        audio_arr = np.asarray(
            audio.detach().float().cpu() if hasattr(audio, "detach") else audio
        )
        while audio_arr.ndim > 2:
            audio_arr = audio_arr[0]

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"minimax-h3-{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
    save_av(frames, FPS, audio_arr, sampling_rate or 0, dest)
    print(f"OUTPUT:{dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

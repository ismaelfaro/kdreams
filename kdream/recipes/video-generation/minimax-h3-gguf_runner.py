#!/usr/bin/env python3
"""kdream runner for Abiray/MiniMax-H3-GGUF — text/image-to-video+audio.

Runs inference through a headless ComfyUI server (native MiniMax H3 support
landed in ComfyUI core; GGUF components load via the ComfyUI-GGUF nodes).

ComfyUI resolution order:
    1. $COMFYUI_PATH (directory containing main.py)
    2. ./ComfyUI, ~/ComfyUI, /opt/ComfyUI

The ComfyUI server is started with the interpreter from $COMFYUI_PYTHON, the
ComfyUI checkout's own .venv, or this interpreter (in that order).

No diffusers fallback on purpose: DiffusionPipeline.from_pretrained on the
base repo would download the ~498 GB full-precision model, which defeats the
GGUF quantization this recipe exists for.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

REPO_ID = "Abiray/MiniMax-H3-GGUF"

COMPONENTS = {
    "text_encoder": ("text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "text_encoders"),
    "video_vae": ("vae/minimax_h3_video_vae_fp16.safetensors", "vae"),
    "audio_vae": ("vae/minimax_h3_audio_vae_fp32.safetensors", "vae"),
}

FPS = 24


# ---------------------------------------------------------------------------
# Model download (HF cache-aware: re-uses ~/.cache/huggingface)
# ---------------------------------------------------------------------------

def fetch_component(file_path: str) -> Path:
    from huggingface_hub import hf_hub_download
    print(f"[models] Ensuring {REPO_ID}/{file_path} ...", flush=True)
    return Path(hf_hub_download(repo_id=REPO_ID, filename=file_path))


# ---------------------------------------------------------------------------
# ComfyUI discovery + model wiring
# ---------------------------------------------------------------------------

def find_comfyui() -> Path | None:
    candidates = []
    if os.environ.get("COMFYUI_PATH"):
        candidates.append(Path(os.environ["COMFYUI_PATH"]))
    candidates += [Path("ComfyUI"), Path.home() / "ComfyUI", Path("/opt/ComfyUI")]
    for c in candidates:
        if (c / "main.py").exists():
            return c.resolve()
    return None


def comfy_python(comfy_dir: Path) -> str:
    if os.environ.get("COMFYUI_PYTHON"):
        return os.environ["COMFYUI_PYTHON"]
    venv_py = comfy_dir / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def link_model(src: Path, comfy_dir: Path, folder: str) -> str:
    """Symlink *src* into ComfyUI's models/<folder>/, return the filename."""
    dest_dir = comfy_dir / "models" / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists():
        dest.symlink_to(src)
    return src.name


# ---------------------------------------------------------------------------
# ComfyUI server lifecycle
# ---------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ComfyUI server exited early (code {proc.returncode}) — "
                "check the server log."
            )
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/system_stats", timeout=5,
            )
            return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise RuntimeError(f"ComfyUI server did not come up within {timeout:.0f}s")


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

def build_workflow(args, unet_name: str, te_name: str, vvae_name: str,
                   avae_name: str, seed: int, length: int) -> dict:
    """API-format ComfyUI graph: t2va / fl2va via native MiniMax H3 nodes."""
    wf = {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": unet_name}},
        "2": {"class_type": "CLIPLoaderGGUF",
              "inputs": {"clip_name": te_name, "type": "minimax"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vvae_name}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": avae_name}},
        "5": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0],
                         "prompt": args.prompt,
                         "width": args.width, "height": args.height,
                         "length": length}},
        "6": {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": ["1", 0],
                         "shift_video": 12.0, "shift_audio": 3.0}},
        "7": {"class_type": "ConditioningZeroOut",
              "inputs": {"conditioning": ["5", 0]}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["6", 0], "positive": ["5", 0],
                         "negative": ["7", 0], "latent_image": ["5", 1],
                         "seed": seed, "steps": args.steps,
                         "cfg": args.guidance_scale,
                         "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}},
        "9": {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "VAEDecodeAudio",
               "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "11": {"class_type": "CreateVideo",
               "inputs": {"images": ["9", 0], "audio": ["10", 0],
                          "fps": float(FPS)}},
        "12": {"class_type": "SaveVideo",
               "inputs": {"video": ["11", 0],
                          "filename_prefix": "minimax_h3",
                          "format": "auto", "codec": "auto"}},
    }
    if args.first_frame_image:
        img = Path(args.first_frame_image).resolve()
        wf["20"] = {"class_type": "LoadImage", "inputs": {"image": str(img)}}
        wf["5"]["inputs"]["first_frame"] = ["20", 0]
    if args.last_frame_image:
        img = Path(args.last_frame_image).resolve()
        wf["21"] = {"class_type": "LoadImage", "inputs": {"image": str(img)}}
        wf["5"]["inputs"]["last_frame"] = ["21", 0]
    return wf


def submit_and_wait(port: int, workflow: dict, proc: subprocess.Popen,
                    timeout: float) -> dict:
    payload = json.dumps(
        {"prompt": workflow, "client_id": uuid.uuid4().hex}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/prompt", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            prompt_id = json.load(resp)["prompt_id"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"ComfyUI rejected the workflow: {detail}") from e

    print(f"[comfyui] Queued prompt {prompt_id}; generating ...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ComfyUI server died mid-generation (code {proc.returncode})."
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=10,
            ) as resp:
                history = json.load(resp)
        except (urllib.error.URLError, OSError):
            time.sleep(5)
            continue
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", [])
                        if m and m[0] == "execution_error"]
                detail = json.dumps(msgs[-1][1] if msgs else status)[:2000]
                raise RuntimeError(f"ComfyUI execution failed: {detail}")
            if entry.get("outputs"):
                return entry["outputs"]
        time.sleep(10)
    raise RuntimeError(f"Generation timed out after {timeout:.0f}s")


def collect_video(outputs: dict, comfy_dir: Path) -> Path:
    for node_output in outputs.values():
        for key in ("video", "videos", "images", "gifs"):
            for item in node_output.get(key, []) or []:
                fname = item.get("filename")
                if not fname:
                    continue
                sub = item.get("subfolder", "")
                p = comfy_dir / "output" / sub / fname
                if p.exists() and p.suffix in (".mp4", ".webm", ".mov", ".mkv"):
                    return p
    raise RuntimeError(f"No video file found in ComfyUI outputs: {outputs}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MiniMax H3 GGUF video+audio generation via headless ComfyUI"
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="(unused by the CFG-zero negative path; reserved)")
    parser.add_argument("--first-frame-image", type=str, default="")
    parser.add_argument("--last-frame-image", type=str, default="")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--workflow", type=str, default="",
                        help="(reserved) custom workflow JSON override")
    parser.add_argument("--variant", type=str,
                        default="unet/MiniMax-H3-FL2VA-Q4_K_M.gguf")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    comfy_dir = find_comfyui()
    if comfy_dir is None:
        print("[error] ComfyUI not found. MiniMax H3 GGUF requires ComfyUI "
              "(native support) + the ComfyUI-GGUF custom node.")
        print("  1. git clone https://github.com/comfyanonymous/ComfyUI")
        print("  2. git clone https://github.com/city96/ComfyUI-GGUF "
              "ComfyUI/custom_nodes/ComfyUI-GGUF")
        print("  3. pip install -r ComfyUI/requirements.txt gguf")
        print("  4. export COMFYUI_PATH=/path/to/ComfyUI and re-run")
        return 1
    if not (comfy_dir / "custom_nodes" / "ComfyUI-GGUF" / "nodes.py").exists():
        print(f"[error] ComfyUI-GGUF custom node missing in {comfy_dir}/custom_nodes.")
        print("  git clone https://github.com/city96/ComfyUI-GGUF "
              f"{comfy_dir}/custom_nodes/ComfyUI-GGUF")
        return 1
    print(f"[comfyui] Using ComfyUI at {comfy_dir}", flush=True)

    # Fetch + wire models
    unet_path = fetch_component(args.variant)
    unet_name = link_model(unet_path, comfy_dir, "diffusion_models")
    te_name = link_model(fetch_component(COMPONENTS["text_encoder"][0]),
                         comfy_dir, "text_encoders")
    vvae_name = link_model(fetch_component(COMPONENTS["video_vae"][0]),
                           comfy_dir, "vae")
    avae_name = link_model(fetch_component(COMPONENTS["audio_vae"][0]),
                           comfy_dir, "vae")

    seed = args.seed if args.seed != -1 else int.from_bytes(os.urandom(4), "big")
    length = max(5, int(args.duration * FPS))

    port = free_port()
    cmd = [comfy_python(comfy_dir), "main.py",
           "--listen", "127.0.0.1", "--port", str(port)]
    if os.environ.get("KDREAM_DEVICE") == "cpu":
        cmd.append("--cpu")
    log_path = Path(args.output_dir).resolve() / "comfyui-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[comfyui] Starting server on port {port} (log: {log_path})", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=comfy_dir, stdout=log, stderr=log)
        try:
            wait_for_server(port, proc)
            workflow = build_workflow(
                args, unet_name, te_name, vvae_name, avae_name, seed, length,
            )
            timeout = float(os.environ.get("KDREAM_INFER_TIMEOUT", "7200"))
            outputs = submit_and_wait(port, workflow, proc, timeout)
            video = collect_video(outputs, comfy_dir)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    out_dir = Path(args.output_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = out_dir / f"minimax-h3-{stamp}{video.suffix}"
    shutil.copy2(video, dest)
    print(f"OUTPUT:{dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

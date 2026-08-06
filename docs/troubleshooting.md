# Big Models & Troubleshooting

Everything below is built in — no manual babysitting needed. Start with:

```bash
kdream doctor
```

It checks tooling (uv/git/git-lfs traps), hardware (accelerator, memory,
disk), and probes the network path to the HuggingFace CDN (a broken IPv6
route is a common cause of multi-hour downloads crawling at <1 MB/s). It
prints the exact environment knobs that fix what it finds.

## What kdream does automatically for large recipes

- **Memory gate** — recipes declare `backends.local.min_vram_gb`. Before
  downloading or running, kdream compares that against your GPU VRAM (CUDA)
  or unified memory (Apple Silicon / CPU, measured via the kernel's real
  reclaimable-memory metric). Models that can *never* fit are refused with
  a clear message before any download; models that fit but are temporarily
  short on memory wait (`KDREAM_MEMORY_WAIT_TIMEOUT`, default 300 s).
- **Resilient downloads** — HuggingFace downloads retry with backoff
  (`KDREAM_DOWNLOAD_RETRIES`, default 8) and resume from partial blobs.
- **LFS-safe cloning** — cloning a HuggingFace source repo never smudges
  LFS weights (some model repos carry 400+ GB of them); weights are always
  downloaded selectively.
- **Quantized-derivative discovery** — `kdream generate` on a HuggingFace
  URL surfaces the model's quantized derivatives
  (`base_model:quantized:<id>` relation) sorted best-fit-for-your-machine
  first, so the recipe targets something that actually runs locally.
- **Platform-aware runners** — generated recipes detect NVIDIA (CUDA +
  bitsandbytes NF4), Apple Silicon (Metal via torch-MPS, MLX auto-detected
  when a Python MLX pipeline for the model exists), or CPU.

## Environment knobs

All optional — `kdream doctor` shows live state:

| Variable | Effect |
|----------|--------|
| `KDREAM_FORCE_IPV4=1` | Pin downloads to IPv4 (fixes broken/throttled IPv6 routes to the HF CDN) |
| `KDREAM_DISABLE_XET=1` | Plain-HTTP HF downloads instead of the Xet backend |
| `KDREAM_DOWNLOAD_RETRIES=N` | Download retry attempts (default 8) |
| `KDREAM_MEMORY_WAIT_TIMEOUT=S` | Seconds to wait for memory to free up (default 300) |
| `KDREAM_SKIP_MEMORY_CHECK=1` | Bypass the memory gate (at your own risk) |
| `KDREAM_DEVICE=cuda\|mps\|cpu` | Force a specific accelerator |

## Example — a 100+ GB video model on a 32 GB MacBook

```bash
kdream doctor                       # check the machine + network first
kdream install minimax-h3           # venv + deps (weights download on first run)
kdream run minimax-h3 --prompt "a red panda typing on a laptop" \
  --steps 30 --width 768 --height 448 -- duration 3
```

The first run downloads weights (resumable) and quantizes them to int4 on
load; later runs skip both. If memory is tight, close heavy apps — the gate
waits and tells you what it needs.

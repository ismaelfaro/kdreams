"""Network resilience helpers for large model downloads.

Big-model downloads (tens to hundreds of GB) fail in ways small ones never
do: CDN disconnects mid-transfer, throttled/broken IPv6 paths, and backend
protocol issues (e.g. the HF Xet backend ignoring Python-level socket
overrides). These helpers make every kdream download path resilient and give
``kdream doctor`` a way to diagnose the network.

Environment knobs (all optional):

- ``KDREAM_FORCE_IPV4=1``       pin DNS resolution to A records — fixes
                                networks with a broken/throttled IPv6 path
                                to the HuggingFace CDN.
- ``KDREAM_DISABLE_XET=1``      use plain-HTTP HF downloads instead of the
                                Xet backend (Rust — unaffected by Python
                                socket overrides).
- ``KDREAM_DOWNLOAD_RETRIES=N`` retry attempts for HF downloads (default 8).
"""
from __future__ import annotations

import os
import socket
import time
from typing import Any, Callable

_ipv4_pinned = False


def pin_ipv4() -> None:
    """Force DNS resolution to IPv4 (A records) process-wide. Idempotent."""
    global _ipv4_pinned
    if _ipv4_pinned:
        return
    _orig = socket.getaddrinfo

    def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
        return _orig(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only
    _ipv4_pinned = True


def apply_network_env() -> list[str]:
    """Apply KDREAM_* network workarounds. Returns notes on what was applied."""
    notes: list[str] = []
    if os.environ.get("KDREAM_FORCE_IPV4") == "1":
        pin_ipv4()
        notes.append("IPv4-only DNS resolution (KDREAM_FORCE_IPV4=1)")
    if os.environ.get("KDREAM_DISABLE_XET") == "1":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        notes.append("HF Xet backend disabled (KDREAM_DISABLE_XET=1)")
    return notes


def with_retry(
    fn: Callable[..., Any],
    *args: Any,
    attempts: int | None = None,
    base_delay: float = 5.0,
    on_retry: Callable[[int, int, Exception], None] | None = None,
    **kwargs: Any,
) -> Any:
    """Call *fn* with retries and linear backoff (capped at 60 s).

    Transient CDN disconnects are the norm, not the exception, on
    multi-hour downloads; HF downloads resume from partial blobs, so
    retrying is always safe.
    """
    if attempts is None:
        attempts = int(os.environ.get("KDREAM_DOWNLOAD_RETRIES", "8"))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            if on_retry is not None:
                on_retry(attempt, attempts, exc)
            time.sleep(min(60.0, base_delay * attempt))
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Network diagnosis (used by `kdream doctor`)
# ---------------------------------------------------------------------------

_PROBE_URL = (
    "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
)
_PROBE_BYTES = 8_000_000


def _probe_speed(ipv4_only: bool, timeout: float = 15.0) -> float:
    """Download a small range from the HF CDN; return MB/s (0.0 on failure)."""
    try:
        import httpx

        transport = (
            httpx.HTTPTransport(local_address="0.0.0.0") if ipv4_only else None
        )
        start = time.time()
        received = 0
        with httpx.Client(transport=transport, timeout=timeout,
                          follow_redirects=True) as client:
            with client.stream(
                "GET", _PROBE_URL,
                headers={"Range": f"bytes=0-{_PROBE_BYTES}"},
            ) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes(chunk_size=65536):
                    received += len(chunk)
                    if received >= _PROBE_BYTES or time.time() - start > timeout:
                        break
        elapsed = max(time.time() - start, 0.001)
        return received / elapsed / 1e6
    except Exception:
        return 0.0


def diagnose_hf_connectivity() -> dict[str, float | str]:
    """Compare default vs IPv4-only download speed to the HuggingFace CDN.

    Returns dict with ``default_mbps``, ``ipv4_mbps`` and a ``recommendation``
    string (empty when the default path is fine).
    """
    default_speed = _probe_speed(ipv4_only=False)
    ipv4_speed = _probe_speed(ipv4_only=True)
    recommendation = ""
    if default_speed == 0.0 and ipv4_speed > 0.0:
        recommendation = (
            "Default route to the HuggingFace CDN is broken; "
            "set KDREAM_FORCE_IPV4=1 and KDREAM_DISABLE_XET=1."
        )
    elif ipv4_speed > default_speed * 2 and ipv4_speed > 1.0:
        recommendation = (
            "IPv6 path to the HuggingFace CDN is much slower than IPv4; "
            "set KDREAM_FORCE_IPV4=1 and KDREAM_DISABLE_XET=1 for large downloads."
        )
    return {
        "default_mbps": round(default_speed, 1),
        "ipv4_mbps": round(ipv4_speed, 1),
        "recommendation": recommendation,
    }

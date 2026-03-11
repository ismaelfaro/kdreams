from kdream.backends.local import LocalBackend
from kdream.backends.colab import ColabBackend
from kdream.backends.runpod import RunPodBackend

BACKENDS = {
    "local": LocalBackend,
    "colab": ColabBackend,
    "runpod": RunPodBackend,
}


def get_backend(name: str, **kwargs):
    if name not in BACKENDS:
        from kdream.exceptions import BackendError
        raise BackendError(f"Unknown backend '{name}'. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name](**kwargs)

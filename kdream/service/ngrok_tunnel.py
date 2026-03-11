"""ngrok tunnel management for kdream serve."""
from __future__ import annotations


class NgrokTunnel:
    """Context manager that opens an ngrok HTTP tunnel and closes it on exit."""

    def __init__(self, port: int, auth_token: str | None = None):
        self.port = port
        self.auth_token = auth_token
        self._tunnel = None
        self.public_url: str | None = None

    def start(self) -> str:
        """Open the tunnel and return the public HTTPS URL."""
        try:
            from pyngrok import conf, ngrok
        except ImportError as exc:
            raise RuntimeError(
                "pyngrok is required for --ngrok. Install it with: "
                "uv pip install 'kdream[service]'"
            ) from exc

        if self.auth_token:
            conf.get_default().auth_token = self.auth_token

        self._tunnel = ngrok.connect(self.port, "http")
        raw_url: str = self._tunnel.public_url or ""
        # Normalise http:// → https:// (free ngrok accounts return http)
        if raw_url.startswith("http://"):
            raw_url = "https://" + raw_url[len("http://"):]
        self.public_url = raw_url
        return self.public_url

    def stop(self) -> None:
        """Disconnect the tunnel if it is open."""
        if self._tunnel is not None:
            try:
                from pyngrok import ngrok
                ngrok.disconnect(self._tunnel.public_url)
            except Exception:
                pass
            self._tunnel = None
            self.public_url = None

    def __enter__(self) -> "NgrokTunnel":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

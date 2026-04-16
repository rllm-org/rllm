"""Reusable gateway server for tests (uvicorn in a background thread)."""

import socket
import threading
import time

import uvicorn


def _reserve_local_port(host: str) -> int:
    """Reserve an ephemeral local TCP port for a test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


class GatewayServer:
    """Run a real gateway uvicorn server in a background thread."""

    def __init__(self, app, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.app = app
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self.port == 0:
            self.port = _reserve_local_port(self.host)
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("Gateway server failed to start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)

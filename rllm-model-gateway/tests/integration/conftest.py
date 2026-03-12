"""Shared fixtures for real-vLLM integration tests."""

import httpx
import pytest
from rllm_model_gateway import GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer

VLLM_BASE = "http://localhost:4000"


def _vllm_reachable() -> bool:
    try:
        r = httpx.get(f"{VLLM_BASE}/v1/models", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


requires_vllm = pytest.mark.skipif(
    not _vllm_reachable(),
    reason="vLLM server not reachable on localhost:4000",
)


@pytest.fixture(params=["memory", "sqlite"])
def gateway_server(request, tmp_path):
    store_worker = request.param
    config_kwargs = {
        "store_worker": store_worker,
        "workers": [{"url": f"{VLLM_BASE}/v1", "worker_id": "vllm-0"}],
        "health_check_interval": 999,
        "sync_traces": True,
    }
    if store_worker == "sqlite":
        config_kwargs["db_path"] = str(tmp_path / "test_traces.db")
    config = GatewayConfig(**config_kwargs)
    app = create_app(config)
    server = GatewayServer(app, port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def gateway_server_url(gateway_server: GatewayServer) -> str:
    return gateway_server.url

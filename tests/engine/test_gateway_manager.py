"""Unit tests for GatewayManager store-backend selection and validation."""

import socket
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rllm.gateway.manager import GatewayManager, GatewayPortInUseError, _find_free_port, container_reachable_url, preflight_gateway_port


def _make_config(**gateway_overrides):
    """Build a minimal OmegaConf DictConfig with gateway overrides."""
    return OmegaConf.create({"rllm": {"gateway": gateway_overrides}})


@pytest.fixture(autouse=True)
def _unset_store_env(monkeypatch):
    # Store selection reads RLLM_GATEWAY_STORE; keep the ambient value out.
    monkeypatch.delenv("RLLM_GATEWAY_STORE", raising=False)
    monkeypatch.delenv("RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR", raising=False)


class TestGatewayStoreSelection:
    def test_default_store_is_compact(self):
        gw = GatewayManager(_make_config(), mode="thread")
        assert gw.store == "compact"
        assert gw.db_path is None

    def test_training_config_defaults_to_compact(self):
        config = OmegaConf.load(Path(__file__).parents[2] / "rllm/trainer/config/rllm/base.yaml")
        assert config.gateway.store == "compact"

    def test_explicit_memory_store(self):
        gw = GatewayManager(_make_config(store="memory"), mode="thread")
        assert gw.store == "memory"
        assert gw.db_path is None

    def test_sqlite_with_explicit_db_path(self):
        gw = GatewayManager(_make_config(store="sqlite", db_path="/tmp/x.db"), mode="thread")
        assert gw.store == "sqlite"
        assert gw.db_path == "/tmp/x.db"

    def test_sqlite_without_db_path_is_allowed(self):
        gw = GatewayManager(_make_config(store="sqlite"), mode="thread")
        assert gw.store == "sqlite"
        assert gw.db_path is None


class TestGatewayStoreValidation:
    def test_unknown_store_raises(self):
        with pytest.raises(ValueError, match="must be 'memory', 'compact' or 'sqlite'"):
            GatewayManager(_make_config(store="postgres"), mode="thread")

    def test_memory_with_db_path_raises(self):
        with pytest.raises(ValueError, match="db_path is set but store='memory'"):
            GatewayManager(_make_config(store="memory", db_path="/tmp/x.db"), mode="thread")

    def test_trace_parity_dump_requires_compact_store(self, tmp_path):
        with pytest.raises(ValueError, match="trace_parity_dump_dir requires"):
            GatewayManager(_make_config(store="memory", trace_parity_dump_dir=str(tmp_path)), mode="thread")


class TestCompactStoreDefault:
    @pytest.mark.parametrize(
        ("config_store", "env_store", "expected"),
        [
            (None, None, "compact"),
            ("compact", None, "compact"),
            (None, "compact", "compact"),
            ("memory", "compact", "compact"),
            (None, "memory", "memory"),
            ("memory", "sqlite", "sqlite"),
            ("compact", "memory", "memory"),
        ],
    )
    def test_store_env_overrides_config(self, monkeypatch, config_store, env_store, expected):
        if env_store is not None:
            monkeypatch.setenv("RLLM_GATEWAY_STORE", env_store)
        overrides = {} if config_store is None else {"store": config_store}
        assert GatewayManager(_make_config(**overrides), mode="thread").store == expected

    @pytest.mark.parametrize(("store", "expected"), [("compact", "compact"), ("memory", None), ("sqlite", None)])
    def test_trace_format_follows_store(self, store, expected):
        assert GatewayManager(_make_config(store=store), mode="thread")._trace_format == expected

    def test_compact_with_db_path_raises(self):
        with pytest.raises(ValueError, match="db_path is set but store='compact'"):
            GatewayManager(_make_config(store="compact", db_path="/tmp/x.db"), mode="thread")

    def test_trace_parity_dump_is_passed_only_to_gateway_workers(self, tmp_path):
        gw = GatewayManager(_make_config(store="compact", trace_parity_dump_dir=str(tmp_path)), mode="thread")
        assert gw.trace_parity_dump_dir == str(tmp_path.resolve())
        assert gw._gateway_cmd(9091)[-2:] == ["--trace-parity-dump-dir", str(tmp_path.resolve())]
        assert "--trace-parity-dump-dir" not in gw._gateway_cmd(9091, front=True, worker_urls=["http://127.0.0.1:9092"])


class TestContainerReachableUrl:
    @pytest.mark.parametrize(
        ("url", "backend", "expected"),
        [
            # Docker containers can't reach the host's loopback — rewrite
            # to host.docker.internal, preserving port and path.
            ("http://127.0.0.1:8000/v1", "docker", "http://host.docker.internal:8000/v1"),
            ("http://localhost:9001/sessions/x/v1", "docker", "http://host.docker.internal:9001/sessions/x/v1"),
            # Non-docker backends (and unset) pass the URL through untouched.
            ("http://127.0.0.1:8000/v1", "modal", "http://127.0.0.1:8000/v1"),
            ("http://localhost:9000/v1", "local", "http://localhost:9000/v1"),
            ("http://127.0.0.1:8000/v1", None, "http://127.0.0.1:8000/v1"),
        ],
    )
    def test_loopback_rewrite_only_for_docker_backend(self, url, backend, expected):
        assert container_reachable_url(url, backend) == expected


class TestPreflightGatewayPort:
    def test_free_port_passes(self):
        preflight_gateway_port(_find_free_port())

    def test_occupied_port_raises(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 0))
            s.listen(1)
            port = s.getsockname()[1]
            with pytest.raises(GatewayPortInUseError, match=str(port)):
                preflight_gateway_port(port)

    def test_loopback_only_listener_still_detected(self):
        # A holder bound to 127.0.0.1 (not 0.0.0.0) must still trip the
        # probe — the gateway's 0.0.0.0 bind would collide with it.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            with pytest.raises(GatewayPortInUseError):
                preflight_gateway_port(port)

    def test_daemon_upstream_conflict_names_the_tunnel(self, monkeypatch):
        import rllm.gateway.tunnel as tunnel_mod

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 0))
            s.listen(1)
            port = s.getsockname()[1]
            monkeypatch.setattr(tunnel_mod, "live_tunnel", lambda: {"backend": "ngrok", "url": "https://x.ngrok-free.app", "pid": 1, "upstream": f"http://127.0.0.1:{port}"})
            with pytest.raises(GatewayPortInUseError, match="rllm tunnel up"):
                preflight_gateway_port(port)

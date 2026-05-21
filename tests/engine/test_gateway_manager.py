"""Unit tests for GatewayManager store-backend selection and validation."""

import pytest
from omegaconf import OmegaConf

from rllm.experimental.engine.gateway_manager import GatewayManager


def _make_config(**gateway_overrides):
    """Build a minimal OmegaConf DictConfig with gateway overrides."""
    return OmegaConf.create({"rllm": {"gateway": gateway_overrides}})


class TestGatewayStoreSelection:
    def test_default_store_is_memory(self):
        gw = GatewayManager(_make_config(), mode="thread")
        assert gw.store == "memory"
        assert gw.db_path is None

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
        with pytest.raises(ValueError, match="must be 'memory' or 'sqlite'"):
            GatewayManager(_make_config(store="postgres"), mode="thread")

    def test_memory_with_db_path_raises(self):
        with pytest.raises(ValueError, match="db_path is set but store='memory'"):
            GatewayManager(_make_config(store="memory", db_path="/tmp/x.db"), mode="thread")

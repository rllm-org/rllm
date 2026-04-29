"""Tests for provider routing and route application."""

import pytest
from rllm_model_gateway.models import ProviderRoute
from rllm_model_gateway.providers import ProviderRouter, apply_route


class TestProviderRouterLookup:
    def test_exact_match(self):
        route = ProviderRoute(model_name="gpt-5-mini", backend_url="https://api.openai.com/v1")
        router = ProviderRouter([route])
        assert router.lookup("gpt-5-mini") is route

    def test_no_match_returns_none(self):
        route = ProviderRoute(model_name="gpt-5-mini", backend_url="https://api.openai.com/v1")
        router = ProviderRouter([route])
        assert router.lookup("claude-foo") is None

    def test_empty_model_returns_none(self):
        router = ProviderRouter([ProviderRoute(model_name="x", backend_url="https://x")])
        assert router.lookup(None) is None
        assert router.lookup("") is None

    def test_empty_router_truthiness(self):
        empty = ProviderRouter()
        assert not empty
        assert len(empty) == 0
        non_empty = ProviderRouter([ProviderRoute(model_name="x", backend_url="https://x")])
        assert non_empty
        assert len(non_empty) == 1


class TestApplyRouteUrl:
    def test_strips_v1_prefix_to_avoid_doubling(self):
        route = ProviderRoute(model_name="gpt-5-mini", backend_url="https://api.openai.com/v1")
        url, _, _ = apply_route(
            request_body={"model": "gpt-5-mini"},
            route=route,
            request_path="/v1/chat/completions",
        )
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_path_without_v1_passes_through(self):
        route = ProviderRoute(model_name="m", backend_url="https://example.com/api")
        url, _, _ = apply_route(
            request_body={"model": "m"},
            route=route,
            request_path="/embeddings",
        )
        assert url == "https://example.com/api/embeddings"

    def test_trailing_slash_normalization(self):
        route = ProviderRoute(model_name="m", backend_url="https://api.openai.com/v1/")
        url, _, _ = apply_route(
            request_body={"model": "m"},
            route=route,
            request_path="/v1/chat/completions",
        )
        assert url == "https://api.openai.com/v1/chat/completions"


class TestApplyRouteHeaders:
    def test_authorization_injected_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        route = ProviderRoute(
            model_name="gpt-5-mini",
            backend_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )
        _, headers, _ = apply_route(
            request_body={"model": "gpt-5-mini"},
            route=route,
            request_path="/v1/chat/completions",
        )
        assert headers["authorization"] == "Bearer sk-test-123"

    def test_x_api_key_injected_alongside_bearer(self, monkeypatch):
        """Anthropic's native API uses ``x-api-key``, not ``Authorization: Bearer``.

        Without ``x-api-key`` the gateway-forwarded request to
        ``/v1/messages`` 404s and Harbor / mini-swe-agent crashes on first
        call. We always inject both — providers that don't recognize one
        ignore it.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")
        route = ProviderRoute(
            model_name="claude-sonnet-4-6",
            backend_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
        )
        _, headers, _ = apply_route(
            request_body={"model": "claude-sonnet-4-6"},
            route=route,
            request_path="/v1/messages",
        )
        assert headers["authorization"] == "Bearer sk-ant-abc"
        assert headers["x-api-key"] == "sk-ant-abc"

    def test_missing_env_omits_both_auth_headers(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        route = ProviderRoute(
            model_name="m",
            backend_url="https://x",
            api_key_env="MISSING_KEY",
        )
        _, headers, _ = apply_route(
            request_body={"model": "m"},
            route=route,
            request_path="/v1/chat/completions",
        )
        assert "authorization" not in headers
        assert "x-api-key" not in headers

    def test_extra_headers_included(self, monkeypatch):
        monkeypatch.setenv("KEY", "k")
        route = ProviderRoute(
            model_name="m",
            backend_url="https://x",
            api_key_env="KEY",
            extra_headers={"X-Custom": "yes"},
        )
        _, headers, _ = apply_route(
            request_body={"model": "m"},
            route=route,
            request_path="/v1/chat/completions",
        )
        assert headers["x-custom"] == "yes"
        assert headers["authorization"] == "Bearer k"
        assert headers["x-api-key"] == "k"

    def test_extra_headers_override_both_auth_schemes(self, monkeypatch):
        monkeypatch.setenv("KEY", "from-env")
        route = ProviderRoute(
            model_name="m",
            backend_url="https://x",
            api_key_env="KEY",
            extra_headers={"Authorization": "Bearer from-extra", "x-api-key": "from-extra"},
        )
        _, headers, _ = apply_route(
            request_body={"model": "m"},
            route=route,
            request_path="/v1/chat/completions",
        )
        # extra_headers wins via setdefault: env-derived auth is only set if
        # not already present.
        assert headers["authorization"] == "Bearer from-extra"
        assert headers["x-api-key"] == "from-extra"


class TestApplyRouteBody:
    def test_default_backend_model_is_model_name(self):
        route = ProviderRoute(model_name="gpt-5-mini", backend_url="https://api.openai.com/v1")
        _, _, body = apply_route(
            request_body={"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}]},
            route=route,
            request_path="/v1/chat/completions",
        )
        assert body["model"] == "gpt-5-mini"
        assert body["messages"] == [{"role": "user", "content": "hi"}]

    def test_explicit_backend_model_rewrites(self):
        route = ProviderRoute(
            model_name="openai/gpt-5-mini",
            backend_url="https://api.openai.com/v1",
            backend_model="gpt-5-mini",
        )
        _, _, body = apply_route(
            request_body={"model": "openai/gpt-5-mini"},
            route=route,
            request_path="/v1/chat/completions",
        )
        assert body["model"] == "gpt-5-mini"

    def test_drop_params_strips_route_specific(self):
        route = ProviderRoute(
            model_name="m",
            backend_url="https://x",
            drop_params=["seed"],
        )
        _, _, body = apply_route(
            request_body={"model": "m", "seed": 42, "temperature": 0.7},
            route=route,
            request_path="/v1/chat/completions",
        )
        assert "seed" not in body
        assert body["temperature"] == 0.7

    def test_drop_params_strips_global(self):
        route = ProviderRoute(model_name="m", backend_url="https://x")
        _, _, body = apply_route(
            request_body={"model": "m", "logprobs": True, "return_token_ids": True},
            route=route,
            request_path="/v1/chat/completions",
            global_drop_params=["return_token_ids"],
        )
        assert "return_token_ids" not in body
        assert body["logprobs"] is True

    def test_does_not_mutate_original_body(self):
        route = ProviderRoute(
            model_name="openai/gpt",
            backend_url="https://x",
            backend_model="gpt",
            drop_params=["seed"],
        )
        original = {"model": "openai/gpt", "seed": 42}
        _, _, body = apply_route(
            request_body=original,
            route=route,
            request_path="/v1/chat/completions",
        )
        assert original == {"model": "openai/gpt", "seed": 42}
        assert body["model"] == "gpt"
        assert "seed" not in body


class TestProviderRouterDuplicates:
    def test_duplicate_routes_keeps_latest(self, caplog):
        r1 = ProviderRoute(model_name="m", backend_url="https://first")
        r2 = ProviderRoute(model_name="m", backend_url="https://second")
        with caplog.at_level("WARNING"):
            router = ProviderRouter([r1, r2])
        assert router.lookup("m").backend_url == "https://second"
        assert any("Duplicate" in r.message for r in caplog.records)


@pytest.fixture(autouse=True)
def _clear_test_keys(monkeypatch):
    """Ensure provider env vars don't leak between tests."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "KEY", "MISSING_KEY"):
        monkeypatch.delenv(key, raising=False)

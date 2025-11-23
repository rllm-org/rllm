import threading
from typing import Any

import pytest

# Skip whole module if OpenTelemetry SDK is not available
pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.sdk.trace")
pytest.importorskip("opentelemetry.baggage.propagation")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from rllm.sdk.session.opentelemetry import (
    get_current_otel_metadata,
    otel_session,
)


@pytest.fixture(autouse=True)
def otel_env():
    # Configure tracer provider and global propagators
    from opentelemetry import propagate, trace
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    resource = Resource.create({"service.name": "test-session"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    propagate.set_global_textmap(CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()]))
    instrumentor = HTTPXClientInstrumentor()
    instrumentor.instrument()
    try:
        yield
    finally:
        instrumentor.uninstrument()


def test_basic_session_usage_opentelemetry():
    # Outside any session
    assert get_current_otel_metadata() == {}

    # Inside a session
    with otel_session(user_id="12345"):
        assert get_current_otel_metadata() == {"user_id": "12345"}

    # After session exits
    assert get_current_otel_metadata() == {}


def test_nested_sessions_inheritance_opentelemetry():
    with otel_session(user_id="12345", tenant_id="acme"):
        assert get_current_otel_metadata() == {"user_id": "12345", "tenant_id": "acme"}

        with otel_session(request_id="abc-def"):
            assert get_current_otel_metadata() == {
                "user_id": "12345",
                "tenant_id": "acme",
                "request_id": "abc-def",
            }

        # Restored after inner exits
        assert get_current_otel_metadata() == {"user_id": "12345", "tenant_id": "acme"}


def test_nested_sessions_override_opentelemetry():
    with otel_session(user_id="12345", role="user"):
        assert get_current_otel_metadata() == {"user_id": "12345", "role": "user"}

        with otel_session(role="admin", operation="delete"):
            assert get_current_otel_metadata() == {
                "user_id": "12345",
                "role": "admin",
                "operation": "delete",
            }

        # Role restored, operation removed
        assert get_current_otel_metadata() == {"user_id": "12345", "role": "user"}


def test_multiple_nested_levels_opentelemetry():
    with otel_session(a=1):
        assert get_current_otel_metadata() == {"a": 1}

        with otel_session(b=2):
            assert get_current_otel_metadata() == {"a": 1, "b": 2}

            with otel_session(c=3):
                assert get_current_otel_metadata() == {"a": 1, "b": 2, "c": 3}

                with otel_session(a=999, d=4):
                    assert get_current_otel_metadata() == {
                        "a": 999,
                        "b": 2,
                        "c": 3,
                        "d": 4,
                    }

                # After deepest exits
                assert get_current_otel_metadata() == {"a": 1, "b": 2, "c": 3}

            assert get_current_otel_metadata() == {"a": 1, "b": 2}


def test_no_active_session_opentelemetry():
    # Outside any session
    assert get_current_otel_metadata() == {}

    def some_function() -> dict[str, Any]:
        return get_current_otel_metadata()

    assert some_function() == {}


def test_across_function_calls_opentelemetry():
    observed: list[dict[str, Any]] = []

    def inner_function():
        observed.append(get_current_otel_metadata())

    def outer_function():
        with otel_session(request_id="abc"):
            inner_function()

    with otel_session(user_id="12345"):
        outer_function()

    assert observed == [{"user_id": "12345", "request_id": "abc"}]


def test_parallel_sessions_opentelemetry():
    results: dict[str, dict[str, Any]] = {}

    def thread_a():
        with otel_session(thread="A", value=1):
            results["A"] = get_current_otel_metadata()

    def thread_b():
        with otel_session(thread="B", value=2):
            results["B"] = get_current_otel_metadata()

    t1 = threading.Thread(target=thread_a)
    t2 = threading.Thread(target=thread_b)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["A"] == {"thread": "A", "value": 1}
    assert results["B"] == {"thread": "B", "value": 2}


def test_empty_session_opentelemetry():
    with otel_session():
        assert get_current_otel_metadata() == {}

        with otel_session(key="value"):
            assert get_current_otel_metadata() == {"key": "value"}


def test_direct_get_metadata_opentelemetry():
    assert get_current_otel_metadata() == {}


@pytest.fixture
def fastapi_app():
    # Create and instrument FastAPI app
    from fastapi import FastAPI
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    app = FastAPI()
    FastAPIInstrumentor.instrument_app(app)
    return app


@pytest.mark.anyio
async def test_http_propagation_fastapi_instrumented_forward_only(fastapi_app):
    # Define handler that reads propagated metadata and extends it
    import httpx
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/process")
    async def process():
        with otel_session(processed_by="service-b") as server_session:
            server_meta = get_current_otel_metadata()
            return {
                "meta": server_meta,
                "chain_len": len(server_session._session_uid_chain),
            }

    fastapi_app.include_router(router)

    # Client side
    with otel_session(user_id="12345", tenant_id="acme") as client_session:
        client_meta = get_current_otel_metadata()
        assert client_meta == {"user_id": "12345", "tenant_id": "acme"}
        client_chain = list(client_session._session_uid_chain)
        assert len(client_chain) == 1

        # Real HTTP call to FastAPI app via ASGI transport
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/process")
        data = resp.json()
        server_meta = data["meta"]
        server_chain_len = data["chain_len"]

        # Server must see merged metadata
        assert server_meta["user_id"] == "12345"
        assert server_meta["tenant_id"] == "acme"
        assert server_meta["processed_by"] == "service-b"
        assert server_chain_len == len(client_chain) + 1

        # Forward-only: client context unchanged
        assert get_current_otel_metadata() == {"user_id": "12345", "tenant_id": "acme"}


@pytest.mark.anyio
async def test_http_multi_hop_propagation_fastapi_instrumented(fastapi_app):
    # Multi-hop: client -> /stage1 (server) -> /stage2 (server)
    import httpx
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/stage2")
    async def stage2():
        with otel_session(stage2="service-2") as s2:
            return {
                "meta": get_current_otel_metadata(),
                "chain_len": len(s2._session_uid_chain),
            }

    @router.get("/stage1")
    async def stage1():
        with otel_session(stage1="service-1"):
            # Outbound call to stage2 with propagated headers
            transport = httpx.ASGITransport(app=fastapi_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/stage2")
            return resp.json()

    fastapi_app.include_router(router)

    # Client root session
    with otel_session(user_id="12345", tenant_id="acme") as client_session:
        client_chain_len = len(client_session._session_uid_chain)
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/stage1")
        data = resp.json()
        meta = data["meta"]
        chain_len = data["chain_len"]

        # Meta should include client + stage1 + stage2 additions
        assert meta["user_id"] == "12345"
        assert meta["tenant_id"] == "acme"
        assert meta["stage1"] == "service-1"
        assert meta["stage2"] == "service-2"
        # Chain grew by two hops
        assert chain_len == client_chain_len + 2

        # Client remains unchanged
        assert get_current_otel_metadata() == {"user_id": "12345", "tenant_id": "acme"}


@pytest.mark.anyio
async def test_http_override_on_server_does_not_back_propagate(fastapi_app):
    import httpx
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/override")
    async def override():
        # Server overrides role and adds operation
        with otel_session(role="admin", operation="delete"):
            return {"meta": get_current_otel_metadata()}

    fastapi_app.include_router(router)

    with otel_session(user_id="12345", role="user"):
        # Client sends headers
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/override")
        server_meta = resp.json()["meta"]

        # Server sees override
        assert server_meta["user_id"] == "12345"
        assert server_meta["role"] == "admin"
        assert server_meta["operation"] == "delete"

        # Client remains user role
        assert get_current_otel_metadata() == {"user_id": "12345", "role": "user"}


@pytest.mark.anyio
async def test_http_two_hop_propagation_without_intermediate_sessions(fastapi_app):
    """Test two-hop HTTP propagation where only root has a session.

    Client -> /stage1 (no session) -> /stage2 (no session)
    Both intermediate services should receive and be able to read root session metadata.
    """
    import httpx
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/stage2")
    async def stage2():
        # No session created here - just read propagated metadata
        meta = get_current_otel_metadata()
        return {"meta": meta}

    @router.get("/stage1")
    async def stage1():
        # No session created here either - just read propagated metadata
        meta = get_current_otel_metadata()
        # Make nested HTTP call to stage2
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/stage2")
        stage2_data = resp.json()
        return {
            "stage1_meta": meta,
            "stage2_meta": stage2_data["meta"],
        }

    fastapi_app.include_router(router)

    # Client root session
    with otel_session(user_id="12345", tenant_id="acme", request_id="req-abc") as client_session:
        client_chain_len = len(client_session._session_uid_chain)
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/stage1")
        data = resp.json()
        stage1_meta = data["stage1_meta"]
        stage2_meta = data["stage2_meta"]

        # Both stages should see root session metadata
        assert stage1_meta["user_id"] == "12345"
        assert stage1_meta["tenant_id"] == "acme"
        assert stage1_meta["request_id"] == "req-abc"

        assert stage2_meta["user_id"] == "12345"
        assert stage2_meta["tenant_id"] == "acme"
        assert stage2_meta["request_id"] == "req-abc"

        # Metadata should be identical at both stages (no intermediate sessions)
        assert stage1_meta == stage2_meta

        # Client session unchanged
        assert get_current_otel_metadata() == {"user_id": "12345", "tenant_id": "acme", "request_id": "req-abc"}
        assert len(client_session._session_uid_chain) == client_chain_len

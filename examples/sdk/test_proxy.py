#!/usr/bin/env python3
"""Comprehensive test for LiteLLM proxy tracer integration with OpenAI.

This is the single comprehensive test file that validates the complete
proxy + tracer integration pipeline.

Test coverage:
1. Starts LiteLLM proxy server using ProxyManager base class
2. Tests basic chat completion with OpenAI models
3. Tests multiple requests within a single session
4. Tests concurrent sessions running in parallel
5. Tests trace persistence and flush mechanism
6. Validates trace retrieval from SQLite database
7. Automatically cleans up all resources

No manual proxy startup needed - everything is automated!

Prerequisites:
    export OPENAI_API_KEY="sk-..."

Usage:
    python examples/sdk/test_proxy_tracer_standalone.py [--db-path PATH]

Example:
    python examples/sdk/test_proxy_tracer_standalone.py \
        --db-path /tmp/test_proxy.db \
        --proxy-port 4000
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from rllm.sdk import get_chat_client_async, session
from rllm.sdk.proxy.proxy_manager import ProxyManager
from rllm.sdk.store import SqliteTraceStore


def _format_trace_summary(trace: Any) -> str:
    """Return a human-readable summary for a trace context."""

    data: dict[str, Any] = trace.data
    trace_id = trace.id
    model = data["model"]
    tokens = data["tokens"]
    latency_ms = data["latency_ms"]
    input_messages = data["input"]["messages"]
    output_choices = data["output"]["choices"]

    input_text = input_messages[-1]["content"]
    output_text = output_choices[0]["message"]["content"]
    token_text = f"prompt={tokens['prompt']}, completion={tokens['completion']}"

    lines = [
        f"    Trace ID: {trace_id}",
        f"    Model: {model}",
        f"    Latency: {latency_ms:.2f} ms",
        f"    Tokens: {token_text}",
        f"    Input: {input_text}",
        f"    Output: {output_text}",
    ]
    return "\n".join(lines)


def create_openai_config() -> dict:
    """Create OpenAI configuration for LiteLLM.

    Returns:
        Configuration dict with OpenAI model setup.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return {
        "model_list": [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": api_key,
                },
            }
        ]
    }


async def test_basic_request(proxy_manager: ProxyManager):
    """Test basic chat completion request."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Chat Completion")
    print("=" * 60)

    proxy_url = proxy_manager.get_proxy_url(include_v1=True)
    client = get_chat_client_async(base_url=proxy_url, api_key="EMPTY")

    with session(test="basic_request") as sess:
        print(f"Session UID: {sess._uid}")

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            max_tokens=5,
        )

        print("✓ Request successful")
        print(f"  - Response ID: {response.id}")
        print(f"  - Content: {response.choices[0].message.content}")

        return response.id, sess._uid, 1


async def test_multiple_requests(proxy_manager: ProxyManager):
    """Test multiple requests in one session."""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple Requests")
    print("=" * 60)

    proxy_url = proxy_manager.get_proxy_url(include_v1=True)
    client = get_chat_client_async(base_url=proxy_url, api_key="EMPTY")

    with session(test="multi_request") as sess:
        session_uid = sess._uid
        print(f"Session UID: {session_uid}")

        request_count = 0
        for i in range(3):
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Count: {i}"}],
                max_tokens=5,
            )
            request_count += 1
            print(f"  - Request {i + 1}: {response.id}")

        print(f"✓ Completed {request_count} requests")
        return session_uid, request_count


async def test_concurrent_sessions(proxy_manager: ProxyManager):
    """Test concurrent requests from multiple sessions."""
    print("\n" + "=" * 60)
    print("TEST 3: Concurrent Sessions")
    print("=" * 60)

    proxy_url = proxy_manager.get_proxy_url(include_v1=True)
    client = get_chat_client_async(base_url=proxy_url, api_key="EMPTY")

    async def run_session(session_name: str, request_count: int):
        """Run multiple requests in a single session."""
        with session(test=session_name) as sess:
            session_uid = sess._uid
            completed = 0
            for i in range(request_count):
                await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"{session_name}: {i}"}],
                    max_tokens=5,
                )
                completed += 1
            return session_uid, completed

    # Run 3 concurrent sessions with 2 requests each
    print("Running 3 concurrent sessions...")
    session_stats = await asyncio.gather(
        run_session("session_1", 2),
        run_session("session_2", 2),
        run_session("session_3", 2),
    )

    print(f"✓ Completed {len(session_stats)} concurrent sessions")
    for i, (uid, count) in enumerate(session_stats, 1):
        print(f"  - Session {i}: {uid} ({count} requests)")

    return session_stats


async def test_trace_persistence(
    db_path: str,
    proxy_manager: ProxyManager,
    expected_counts: dict[str, int],
    sample_session_uid: str | None,
):
    """Verify all session traces are persisted and counts match expected calls."""

    print("\n" + "=" * 60)
    print("TEST 4: Trace Persistence")
    print("=" * 60)

    if not expected_counts:
        print("No session expectations recorded; skipping trace validation.")
        return False

    print("Flushing tracer...")
    result = await proxy_manager.flush_tracer(timeout=30.0)
    print(f"✓ Flush result: {result}")

    store = SqliteTraceStore(db_path=db_path)
    validation_passed = True

    for session_uid, expected in expected_counts.items():
        traces = await store.get_by_session_uid(session_uid)
        actual = len(traces)
        status = "✓" if actual == expected else "✗"
        print(f"{status} Session {session_uid}: expected {expected}, found {actual}")
        if actual != expected:
            validation_passed = False

        for idx, trace in enumerate(traces, 1):
            print(f"  - Trace {idx}: {trace.id}")
            print(_format_trace_summary(trace))

    if sample_session_uid and sample_session_uid in expected_counts:
        sample_traces = await store.get_by_session_uid(sample_session_uid)
        if sample_traces:
            sample_trace = sample_traces[0]
            fetched = await store.get(sample_trace.id)
            if fetched is None:
                raise RuntimeError(f"Trace {sample_trace.id} missing when fetched directly by ID")
            print("  - Sample trace fetched by ID:")
            print(_format_trace_summary(fetched))

    return validation_passed


async def run_all_tests(db_path: str, proxy_port: int, admin_token: str):
    """Run all proxy tracer tests."""
    # Clean up existing database
    db_file = Path(db_path)
    if db_file.exists():
        print(f"Removing existing database: {db_path}")
        db_file.unlink()

    print("=" * 60)
    print("LiteLLM Proxy Tracer Standalone Test")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Proxy port: {proxy_port}")
    print()

    # Create proxy manager with base class
    proxy_manager = ProxyManager(
        proxy_host="127.0.0.1",
        proxy_port=proxy_port,
        admin_token=admin_token,
    )

    try:
        # Create OpenAI configuration
        config = create_openai_config()

        # Start proxy subprocess with config
        print(f"Starting LiteLLM proxy on port {proxy_port}...")
        print(f"  - Database: {db_path}")

        config_path = proxy_manager.start_proxy_subprocess(
            config=config,
            db_path=db_path,
            project="test-project",
        )
        print(f"  - Config: {config_path}")
        print("✓ Proxy started successfully")

        results = []
        session_uid_1 = None
        expected_counts: dict[str, int] = {}

        def record_expected(uid: str | None, count: int) -> None:
            if not uid:
                return
            expected_counts[uid] = expected_counts.get(uid, 0) + count

        # Test 1: Basic request
        try:
            response_id, session_uid_1, request_count = await test_basic_request(proxy_manager)
            record_expected(session_uid_1, request_count)
            results.append(("Basic Request", response_id is not None))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(("Basic Request", False))

        # Test 2: Multiple requests
        try:
            session_uid_2, multi_count = await test_multiple_requests(proxy_manager)
            record_expected(session_uid_2, multi_count)
            results.append(("Multiple Requests", session_uid_2 is not None))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(("Multiple Requests", False))

        # Test 3: Concurrent sessions
        try:
            concurrent_stats = await test_concurrent_sessions(proxy_manager)
            for uid, count in concurrent_stats:
                record_expected(uid, count)
            results.append(("Concurrent Sessions", len(concurrent_stats) == 3))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(("Concurrent Sessions", False))

        # Test 4: Trace persistence (use session from test 1)
        try:
            if expected_counts:
                persist_ok = await test_trace_persistence(
                    db_path=db_path,
                    proxy_manager=proxy_manager,
                    expected_counts=expected_counts,
                    sample_session_uid=session_uid_1,
                )
                results.append(("Trace Persistence", persist_ok))
            else:
                print("Trace persistence skipped: no sessions were exercised.")
                results.append(("Trace Persistence", False))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(("Trace Persistence", False))

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, success in results if success)
        total = len(results)

        for test_name, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status:<8} {test_name}")

        print()
        print(f"Results: {passed}/{total} tests passed")

        return passed == total

    finally:
        proxy_manager.shutdown_proxy()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test LiteLLM proxy tracer with OpenAI")
    parser.add_argument("--db-path", type=str, default="/tmp/test_proxy_tracer.db", help="SQLite database path")
    parser.add_argument("--proxy-port", type=int, default=4000, help="Proxy port")
    parser.add_argument("--admin-token", type=str, default="test-admin-token", help="Admin token for proxy")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    try:
        success = await run_all_tests(args.db_path, args.proxy_port, args.admin_token)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

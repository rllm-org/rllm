import threading
from typing import Any

from rllm.sdk.session.contextvar import (
    ContextVarSession,
    get_current_cv_metadata,
)


def test_basic_session_usage_contextvar():
    # Outside any session
    assert get_current_cv_metadata() == {}

    # Inside a session
    with ContextVarSession(user_id="12345"):
        assert get_current_cv_metadata() == {"user_id": "12345"}

    # After session exits
    assert get_current_cv_metadata() == {}


def test_nested_sessions_inheritance_contextvar():
    with ContextVarSession(user_id="12345", tenant_id="acme"):
        assert get_current_cv_metadata() == {"user_id": "12345", "tenant_id": "acme"}

        with ContextVarSession(request_id="abc-def"):
            assert get_current_cv_metadata() == {
                "user_id": "12345",
                "tenant_id": "acme",
                "request_id": "abc-def",
            }

        # Restored after inner exits
        assert get_current_cv_metadata() == {"user_id": "12345", "tenant_id": "acme"}


def test_nested_sessions_override_contextvar():
    with ContextVarSession(user_id="12345", role="user"):
        assert get_current_cv_metadata() == {"user_id": "12345", "role": "user"}

        with ContextVarSession(role="admin", operation="delete"):
            assert get_current_cv_metadata() == {
                "user_id": "12345",
                "role": "admin",
                "operation": "delete",
            }

        # Role restored, operation removed
        assert get_current_cv_metadata() == {"user_id": "12345", "role": "user"}


def test_multiple_nested_levels_contextvar():
    with ContextVarSession(a=1):
        assert get_current_cv_metadata() == {"a": 1}

        with ContextVarSession(b=2):
            assert get_current_cv_metadata() == {"a": 1, "b": 2}

            with ContextVarSession(c=3):
                assert get_current_cv_metadata() == {"a": 1, "b": 2, "c": 3}

                with ContextVarSession(a=999, d=4):
                    assert get_current_cv_metadata() == {
                        "a": 999,
                        "b": 2,
                        "c": 3,
                        "d": 4,
                    }

                # After deepest exits
                assert get_current_cv_metadata() == {"a": 1, "b": 2, "c": 3}


def test_no_active_session_contextvar():
    # Outside any session
    assert get_current_cv_metadata() == {}

    def some_function() -> dict[str, Any]:
        return get_current_cv_metadata()

    assert some_function() == {}


def test_across_function_calls_contextvar():
    observed: list[dict[str, Any]] = []

    def inner_function():
        observed.append(get_current_cv_metadata())

    def outer_function():
        with ContextVarSession(request_id="abc"):
            inner_function()

    with ContextVarSession(user_id="12345"):
        outer_function()

    assert observed == [{"user_id": "12345", "request_id": "abc"}]


def test_parallel_sessions_contextvar():
    results: dict[str, dict[str, Any]] = {}

    def thread_a():
        with ContextVarSession(thread="A", value=1):
            results["A"] = get_current_cv_metadata()

    def thread_b():
        with ContextVarSession(thread="B", value=2):
            results["B"] = get_current_cv_metadata()

    t1 = threading.Thread(target=thread_a)
    t2 = threading.Thread(target=thread_b)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["A"] == {"thread": "A", "value": 1}
    assert results["B"] == {"thread": "B", "value": 2}


def test_empty_session_contextvar():
    with ContextVarSession():
        assert get_current_cv_metadata() == {}

        with ContextVarSession(key="value"):
            assert get_current_cv_metadata() == {"key": "value"}

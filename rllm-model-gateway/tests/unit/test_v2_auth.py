import pytest
from rllm_model_gateway.v2.auth import GatewayAuth, bearer_token
from rllm_model_gateway.v2.types import GatewayError


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer secret", "secret"),
        ("bearer secret", "secret"),
        ("BEARER secret", "secret"),
    ],
)
def test_bearer_token_accepts_case_insensitive_scheme(header: str, expected: str) -> None:
    assert bearer_token(header) == expected


@pytest.mark.parametrize("header", [None, "", "secret", "Basic secret", "Bearer"])
def test_bearer_token_rejects_missing_or_invalid_headers(header: str | None) -> None:
    with pytest.raises(GatewayError) as error:
        bearer_token(header)

    assert error.value.status_code == 401
    assert error.value.error_type == "authentication_error"


def test_admin_and_session_credentials_have_separate_authority() -> None:
    auth = GatewayAuth("admin-secret")
    session_key = auth.issue_session_key("session-1")

    auth.require_admin("Bearer admin-secret")
    auth.require_session(f"Bearer {session_key}", "session-1")

    with pytest.raises(GatewayError, match="invalid admin key"):
        auth.require_admin(f"Bearer {session_key}")
    with pytest.raises(GatewayError, match="invalid session key"):
        auth.require_session("Bearer admin-secret", "session-1")
    with pytest.raises(GatewayError, match="invalid session key"):
        auth.require_session(f"Bearer {session_key}", "session-2")


def test_revoking_a_session_key_invalidates_it() -> None:
    auth = GatewayAuth("admin-secret")
    session_key = auth.issue_session_key("session-1")

    auth.revoke_session("session-1")

    with pytest.raises(GatewayError, match="invalid session key"):
        auth.require_session(f"Bearer {session_key}", "session-1")

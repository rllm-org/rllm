import hmac
import secrets

from rllm_model_gateway_v2.errors import GatewayError


class GatewayAuth:
    def __init__(self, admin_key: str) -> None:
        if not admin_key:
            raise ValueError("admin key must not be empty")
        self._admin_key = admin_key
        self._session_keys: dict[str, str] = {}

    def issue_session_key(self, session_id: str) -> str:
        key = secrets.token_urlsafe(32)
        self._session_keys[session_id] = key
        return key

    def revoke_session(self, session_id: str) -> None:
        self._session_keys.pop(session_id, None)

    def require_admin(self, authorization: str | None) -> None:
        supplied = bearer_token(authorization)
        if not hmac.compare_digest(supplied, self._admin_key):
            raise GatewayError("invalid admin key", 401, "authentication_error")

    def require_session(self, authorization: str | None, session_id: str) -> None:
        expected = self._session_keys.get(session_id)
        supplied = bearer_token(authorization)
        if expected is None or not hmac.compare_digest(supplied, expected):
            raise GatewayError("invalid session key", 401, "authentication_error")


def bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise GatewayError("missing bearer token", 401, "authentication_error")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise GatewayError("invalid authorization header", 401, "authentication_error")
    return token

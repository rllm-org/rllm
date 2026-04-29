"""Model-name → provider backend routing.

Provider routing is the eval-side counterpart to ``SessionRouter`` (which
selects among a pool of vLLM workers for training). Where SessionRouter
picks a backend by session stickiness, ``ProviderRouter`` picks one by
the request's ``model`` field.

The two paths coexist: the proxy first checks for a provider route; if
no route matches the requested model, it falls through to the worker
pool. Either path can be empty (eval-only deployment configures
providers; training-only deployment configures workers).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rllm_model_gateway.models import ProviderRoute

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Maps a request's ``model`` field to a configured ``ProviderRoute``.

    Lookup is exact-match on ``model_name``. To route both ``gpt-5-mini``
    and ``openai/gpt-5-mini`` to the same backend, configure two routes
    pointing at the same ``backend_url``.
    """

    def __init__(self, routes: list[ProviderRoute] | None = None) -> None:
        self._by_name: dict[str, ProviderRoute] = {}
        for r in routes or []:
            if r.model_name in self._by_name:
                logger.warning("Duplicate provider route for model %r — keeping latest", r.model_name)
            self._by_name[r.model_name] = r

    def lookup(self, model_name: str | None) -> ProviderRoute | None:
        if not model_name:
            return None
        return self._by_name.get(model_name)

    def __bool__(self) -> bool:
        return bool(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)


def apply_route(
    *,
    request_body: dict[str, Any],
    route: ProviderRoute,
    request_path: str,
    global_drop_params: list[str] | None = None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Translate a gateway request into a provider-bound request.

    Returns ``(forward_url, injected_headers, mutated_body)``:
        - ``forward_url`` is ``route.backend_url`` + the request path's
          tail after stripping the gateway's ``/v1`` prefix.
        - ``injected_headers`` carries the provider Authorization header
          (sourced from ``route.api_key_env``) and any ``extra_headers``.
          Caller is responsible for merging with forwarded client headers.
        - ``mutated_body`` is a shallow copy of ``request_body`` with the
          ``model`` field rewritten to ``route.backend_model`` (or
          ``route.model_name`` if not set) and ``drop_params`` removed.

    The function never mutates ``request_body``.
    """
    tail = _path_tail(request_path)
    forward_url = route.backend_url.rstrip("/") + tail

    headers: dict[str, str] = {k.lower(): v for k, v in route.extra_headers.items()}
    if route.api_key_env:
        key = os.environ.get(route.api_key_env, "").strip()
        if key:
            headers.setdefault("authorization", f"Bearer {key}")
        else:
            logger.warning(
                "Provider route %r references api_key_env=%r but the env var is empty — no Authorization header will be injected.",
                route.model_name,
                route.api_key_env,
            )

    body = dict(request_body)
    body["model"] = route.backend_model or route.model_name

    drops: set[str] = set(route.drop_params)
    if global_drop_params:
        drops.update(global_drop_params)
    for key in drops:
        body.pop(key, None)

    return forward_url, headers, body


def _path_tail(path: str, prefix: str = "/v1") -> str:
    """Strip the gateway's ``/v1`` prefix to get the provider-relative path tail.

    The provider's ``backend_url`` is expected to include its own ``/v1``
    (e.g. ``https://api.openai.com/v1``), so we strip the gateway's
    leading ``/v1`` to avoid doubling it.
    """
    if path.startswith(prefix):
        rest = path[len(prefix) :]
        return rest if rest else ""
    return path

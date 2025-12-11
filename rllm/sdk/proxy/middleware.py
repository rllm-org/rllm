"""FastAPI middleware for LiteLLM proxy metadata handling.

This middleware:
- Decodes /meta/{slug} and rewrites the path to the standard OpenAI-style path.
- Injects decoded metadata into the JSON body (payload["rllm_metadata"]) so downstream
  handlers and callbacks reading kwargs["data"] see it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from rllm.sdk.proxy.metadata_slug import extract_metadata_from_path

logger = logging.getLogger(__name__)


class MetadataRoutingMiddleware(BaseHTTPMiddleware):
    """Extract metadata from URL slugs, rewrite path, and inject metadata into body."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        metadata: dict[str, Any] = {}

        extracted = extract_metadata_from_path(request.url.path)
        if extracted is not None:
            clean_path, metadata = extracted
            logger.debug("MetadataRoutingMiddleware: decoded slug path=%s clean=%s metadata=%s", request.url.path, clean_path, metadata)
            request.scope["path"] = clean_path
            request.scope["raw_path"] = clean_path.encode("utf-8")

        # If we have metadata, inject it into the JSON body as rllm_metadata so LiteLLM sees it in kwargs["data"].
        if metadata:
            body_bytes = await request.body()
            if body_bytes:
                try:
                    payload = json.loads(body_bytes.decode("utf-8"))
                except Exception:
                    payload = None

                if isinstance(payload, dict):
                    # Set metadata so LiteLLM will automatically copy it to requester_metadata
                    # LiteLLM's add_litellm_data_to_request() copies payload["metadata"]
                    # to data[_metadata_variable_name]["requester_metadata"]
                    if "metadata" not in payload:
                        payload["metadata"] = {}
                    payload["metadata"].update({"rllm_metadata": metadata})

                    mutated_body = json.dumps(payload).encode("utf-8")
                    logger.debug("MetadataRoutingMiddleware: injected rllm_metadata and metadata (for requester_metadata) with keys=%s", list(metadata.keys()))

                    # Update cached body so request.json()/body() observes the mutation
                    request._body = mutated_body  # type: ignore[attr-defined]

                    # Ensure downstream ASGI stack receives the mutated body once
                    sent = False

                    async def _receive() -> dict[str, Any]:
                        nonlocal sent
                        if not sent:
                            sent = True
                            return {"type": "http.request", "body": mutated_body, "more_body": False}
                        return {"type": "http.request", "body": b"", "more_body": False}

                    request._receive = _receive  # type: ignore[attr-defined]

        return await call_next(request)

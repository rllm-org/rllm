"""Training-side client for interacting with the model gateway."""

from typing import Any

import httpx

from rllm_model_gateway.models import TraceRecord, WorkerInfo


def _expand_compact_traces(payload: dict[str, Any], *, expand_prompt_ids: bool = True) -> list[dict[str, Any]]:
    """Rebuild full per-trace message lists from a compact traces payload.

    Nodes are materialized once and *shared*: every trace whose conversation
    contains node X references the same message dict, so client memory stays
    linear in unique messages even though the expanded lists repeat them.
    Reconstruction is exact — parity-tested byte-for-byte against the default
    format on real eval dumps.

    With ``expand_prompt_ids=False`` the per-trace prompt token lists are NOT
    materialized (that materialization is the remaining O(n^2)): a trace whose
    ids arrived as a delta keeps them as the step-form marker
    ``{"__prompt_ids_delta__": [lcp, suffix]}`` relative to the previous trace
    of the same lineage, which is what the packed trainers consume natively.
    Only prompt lengths are tracked, so validation stays exact.
    """
    nodes = payload.get("nodes", {})
    out: list[dict[str, Any]] = []
    # Cache only REQUESTED leaves: caching every intermediate prefix made a
    # single deep chain cost O(M^2) allocations (review); walking to the
    # nearest cached ancestor and extending once keeps shared-prefix reuse
    # while a lone leaf costs O(M).
    paths: dict[str | None, list[dict[str, Any]]] = {None: []}

    def _path(leaf: str | None) -> list[dict[str, Any]]:
        cached = paths.get(leaf)
        if cached is not None:
            return cached
        chain: list[str] = []
        node_id = leaf
        seen: set[str] = set()
        while node_id is not None and node_id not in paths:
            if node_id in seen:
                raise ValueError(f"message node cycle at {node_id!r}")
            seen.add(node_id)
            chain.append(node_id)
            entry = nodes.get(node_id)
            if entry is None:
                raise ValueError(f"dangling message node reference {node_id!r}")
            node_id = entry["p"]
        base = list(paths[node_id] if node_id is not None else paths[None])
        base.extend(nodes[nid]["m"] for nid in reversed(chain))
        paths[leaf] = base
        return base

    ids_memo: dict[str | None, list[int]] = {None: []}
    # Delta-keeping state: full prompt lengths per trace (exact validation
    # without materialization), recorded deltas for the rare rebase walk, and
    # the previous trace id per lineage (step-form deltas resolve against the
    # lineage predecessor, which the store's chain predecessor always is —
    # verified per trace, rebased if ever not).
    len_memo: dict[str | None, int] = {None: 0}
    delta_memo: dict[str, tuple[str | None, int, list[int]]] = {}
    prev_in_lineage: dict[Any, str | None] = {}

    def _materialize(tid: str | None) -> list[int]:
        """Rebuild one trace's full prompt ids from recorded deltas (rebase
        fallback only — never runs on the store's chain order)."""
        chain: list[tuple[int, list[int]]] = []
        cursor = tid
        while cursor is not None and cursor not in ids_memo:
            prev, lcp, suffix = delta_memo[cursor]
            chain.append((lcp, suffix))
            cursor = prev
        full = list(ids_memo.get(cursor, []))
        for lcp, suffix in reversed(chain):
            full = full[:lcp] + list(suffix)
        return full

    for trace in payload.get("traces", []):
        ref = trace.get("messages_ref")
        if ref is None:
            out.append(trace)
            continue
        leaf, length = ref
        messages = _path(leaf)
        if len(messages) != length:
            raise ValueError(f"compact chain length {len(messages)} != recorded {length}")
        expanded = dict(trace)
        expanded.pop("messages_ref", None)
        expanded["messages"] = messages
        ids_ref = expanded.pop("prompt_ids_delta", None)
        tid = expanded.pop("_tid", None) or expanded.get("trace_id")
        if ids_ref is not None:
            # Delta against an earlier trace in this payload: prefix + suffix.
            prev_tid, lcp, suffix = ids_ref
            if expand_prompt_ids:
                prev_full = ids_memo.get(prev_tid)
                if prev_full is None or lcp > len(prev_full):
                    raise ValueError(f"prompt-id delta references unknown/short ancestor {prev_tid!r}")
                expanded["prompt_token_ids"] = prev_full[:lcp] + list(suffix)
            else:
                prev_len = len_memo.get(prev_tid)
                if prev_len is None or lcp > prev_len:
                    raise ValueError(f"prompt-id delta references unknown/short ancestor {prev_tid!r}")
                lineage = expanded.get("lineage_id")
                if prev_in_lineage.get(lineage) == prev_tid:
                    expanded["prompt_token_ids"] = {"__prompt_ids_delta__": [lcp, list(suffix)]}
                else:
                    # Chain predecessor is not this lineage's previous trace:
                    # rebase to a full list so step-form consumers stay exact.
                    expanded["prompt_token_ids"] = _materialize(prev_tid)[:lcp] + list(suffix)
                if tid is not None:
                    len_memo[tid] = lcp + len(suffix)
                    delta_memo[tid] = (prev_tid, lcp, list(suffix))
        if tid is not None and isinstance(expanded.get("prompt_token_ids"), list):
            ids_memo[tid] = expanded["prompt_token_ids"]
            len_memo[tid] = len(expanded["prompt_token_ids"])
        if tid is not None:
            prev_in_lineage[expanded.get("lineage_id")] = tid
        out.append(expanded)
    return out


class GatewayClient:
    """Synchronous client for the rllm-model-gateway REST API.

    Intended for use by the training framework to create sessions, retrieve
    traces, and manage workers.
    """

    def __init__(
        self,
        gateway_url: str,
        timeout: float = 30.0,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        # max_keepalive_connections=0 disables idle connection reuse so every
        # request opens a fresh TCP connection. Avoids the keepalive race where
        # client and uvicorn both expire idle connections at ~5s and the next
        # request hits a half-closed socket → httpx.ReadError. Per-request
        # handshake cost is negligible for the control-plane JSON calls this
        # client makes.
        self._http = httpx.Client(timeout=timeout, limits=httpx.Limits(max_keepalive_connections=0))

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- Session lifecycle -------------------------------------------------

    def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> str:
        """Create a session (or let the gateway generate an ID)."""
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if metadata:
            body["metadata"] = metadata
        if sampling_params:
            body["sampling_params"] = sampling_params
        resp = self._http.post(f"{self.gateway_url}/sessions", json=body)
        resp.raise_for_status()
        return resp.json()["session_id"]

    def get_session_url(self, session_id: str) -> str:
        """Return the OpenAI-compatible base URL for an agent to use."""
        return f"{self.gateway_url}/sessions/{session_id}/v1"

    def get_session_info(self, session_id: str) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json()

    def list_sessions(self, since: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        resp = self._http.get(f"{self.gateway_url}/sessions", params=params)
        resp.raise_for_status()
        return resp.json()

    def delete_session(self, session_id: str) -> int:
        resp = self._http.delete(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    # -- Trace retrieval ---------------------------------------------------

    def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
        format: str | None = None,
        expand_prompt_ids: bool = True,
    ) -> list[TraceRecord]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        if format is not None:
            params["format"] = format
        resp = self._http.get(f"{self.gateway_url}/sessions/{session_id}/traces", params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("format") == "compact":
            # model_construct skips validation, which would otherwise re-create
            # every message dict and destroy the cross-trace sharing the
            # expansion just built. The payload is gateway-produced, not user
            # input, so skipping validation is safe here.
            return [TraceRecord.model_construct(**t) for t in _expand_compact_traces(data, expand_prompt_ids=expand_prompt_ids)]
        return [TraceRecord(**t) for t in data]

    def get_trace(self, trace_id: str) -> TraceRecord:
        resp = self._http.get(f"{self.gateway_url}/traces/{trace_id}")
        resp.raise_for_status()
        return TraceRecord(**resp.json())

    # -- Worker management -------------------------------------------------

    def add_worker(
        self,
        url: str,
        api_path: str = "/v1",
        model_name: str | None = None,
        weight: int = 1,
    ) -> str:
        """Register a worker.  Returns worker_id."""
        body: dict[str, Any] = {"url": url, "api_path": api_path, "weight": weight}
        if model_name:
            body["model_name"] = model_name
        resp = self._http.post(f"{self.gateway_url}/admin/workers", json=body)
        resp.raise_for_status()
        return resp.json()["worker_id"]

    def remove_worker(self, worker_id: str) -> None:
        resp = self._http.delete(f"{self.gateway_url}/admin/workers/{worker_id}")
        resp.raise_for_status()

    def list_workers(self) -> list[WorkerInfo]:
        resp = self._http.get(f"{self.gateway_url}/admin/workers")
        resp.raise_for_status()
        return [WorkerInfo(**w) for w in resp.json()]

    # -- Weight version ----------------------------------------------------

    def set_weight_version(self, weight_version: int) -> int:
        resp = self._http.post(f"{self.gateway_url}/admin/weight_version", json={"weight_version": weight_version})
        resp.raise_for_status()
        return resp.json()["weight_version"]

    def get_weight_version(self) -> int | None:
        resp = self._http.get(f"{self.gateway_url}/admin/weight_version")
        resp.raise_for_status()
        return resp.json().get("weight_version")

    # -- Lifecycle ---------------------------------------------------------

    def flush(self, timeout: float = 30.0) -> bool:
        resp = self._http.post(f"{self.gateway_url}/admin/flush", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("status") == "flushed"

    def health(self) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/health")
        resp.raise_for_status()
        return resp.json()


class AsyncGatewayClient:
    """Async variant of :class:`GatewayClient` using ``httpx.AsyncClient``."""

    def __init__(
        self,
        gateway_url: str,
        timeout: float = 30.0,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        # See GatewayClient.__init__ for why keepalive is disabled.
        self._http = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_keepalive_connections=0))

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # -- Session lifecycle -------------------------------------------------

    async def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if metadata:
            body["metadata"] = metadata
        if sampling_params:
            body["sampling_params"] = sampling_params
        resp = await self._http.post(f"{self.gateway_url}/sessions", json=body)
        resp.raise_for_status()
        return resp.json()["session_id"]

    def get_session_url(self, session_id: str) -> str:
        return f"{self.gateway_url}/sessions/{session_id}/v1"

    async def get_session_info(self, session_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json()

    async def list_sessions(self, since: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        resp = await self._http.get(f"{self.gateway_url}/sessions", params=params)
        resp.raise_for_status()
        return resp.json()

    async def delete_session(self, session_id: str) -> int:
        resp = await self._http.delete(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    async def delete_sessions(self, session_ids: list[str]) -> int:
        """Batch-delete sessions (and their traces) in a single round-trip."""
        if not session_ids:
            return 0
        resp = await self._http.post(
            f"{self.gateway_url}/sessions/batch_delete",
            json={"session_ids": list(session_ids)},
        )
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    # -- Trace retrieval ---------------------------------------------------

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
        format: str | None = None,
        expand_prompt_ids: bool = True,
    ) -> list[TraceRecord]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        if format is not None:
            params["format"] = format
        resp = await self._http.get(f"{self.gateway_url}/sessions/{session_id}/traces", params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("format") == "compact":
            # See the sync client: model_construct preserves the shared node
            # dicts that validation would copy away.
            return [TraceRecord.model_construct(**t) for t in _expand_compact_traces(data, expand_prompt_ids=expand_prompt_ids)]
        return [TraceRecord(**t) for t in data]

    async def get_trace(self, trace_id: str) -> TraceRecord:
        resp = await self._http.get(f"{self.gateway_url}/traces/{trace_id}")
        resp.raise_for_status()
        return TraceRecord(**resp.json())

    # -- Worker management -------------------------------------------------

    async def add_worker(
        self,
        url: str,
        api_path: str = "/v1",
        model_name: str | None = None,
        weight: int = 1,
    ) -> str:
        body: dict[str, Any] = {"url": url, "api_path": api_path, "weight": weight}
        if model_name:
            body["model_name"] = model_name
        resp = await self._http.post(f"{self.gateway_url}/admin/workers", json=body)
        resp.raise_for_status()
        return resp.json()["worker_id"]

    async def remove_worker(self, worker_id: str) -> None:
        resp = await self._http.delete(f"{self.gateway_url}/admin/workers/{worker_id}")
        resp.raise_for_status()

    async def list_workers(self) -> list[WorkerInfo]:
        resp = await self._http.get(f"{self.gateway_url}/admin/workers")
        resp.raise_for_status()
        return [WorkerInfo(**w) for w in resp.json()]

    # -- Weight version ----------------------------------------------------

    async def set_weight_version(self, weight_version: int) -> int:
        resp = await self._http.post(f"{self.gateway_url}/admin/weight_version", json={"weight_version": weight_version})
        resp.raise_for_status()
        return resp.json()["weight_version"]

    async def get_weight_version(self) -> int | None:
        resp = await self._http.get(f"{self.gateway_url}/admin/weight_version")
        resp.raise_for_status()
        return resp.json().get("weight_version")

    # -- Lifecycle ---------------------------------------------------------

    async def flush(self, timeout: float = 30.0) -> bool:
        resp = await self._http.post(f"{self.gateway_url}/admin/flush", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("status") == "flushed"

    async def health(self) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/health")
        resp.raise_for_status()
        return resp.json()

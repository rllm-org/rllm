"""Settings panel — config + env-var endpoints.

* ``GET /config``                     — static, read-only paths/versions.
* ``GET /env``                        — known + extra env vars with masked values.
* ``GET /env/{key}?reveal=true``      — full value for a single key.
* ``POST /env``                       — set ``{key, value}``; persists + applies.
* ``DELETE /env/{key}``               — remove from file + ``os.environ``.

Persistence: ``~/.rllm/console.env`` (override via ``RLLM_HOME``).
"""

from __future__ import annotations

import os
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from rllm.console.panels.settings import store
from rllm.console.panels.settings.known import (
    KNOWN_ENV_VARS,
    categories,
    known_by_key,
)

router = APIRouter()

# A name is a "secret-shaped" identifier when its name carries one of
# these tokens. Used to mask unknown env vars surfaced under "Other".
_SECRET_NAME_RE = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|PASS|CREDENTIAL)", re.IGNORECASE)
_VALID_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvVarRow(BaseModel):
    key: str
    label: str
    description: str
    category: str
    secret: bool
    url: str | None = None
    is_set: bool
    in_console_file: bool
    masked_value: str | None = None


class EnvWriteRequest(BaseModel):
    key: str = Field(min_length=1, max_length=128)
    value: str = Field(max_length=8192)


def _mask(value: str) -> str:
    """Return a redacted representation safe to display.

    Short values (≤8) render as ``***`` so we don't leak prefixes of
    tiny secrets. Longer values keep first/last 4 chars to remain
    distinguishable when comparing two stored keys.
    """
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def _is_secret_shaped(key: str) -> bool:
    return bool(_SECRET_NAME_RE.search(key))


@router.get("/config")
def get_config(request: Request) -> dict[str, Any]:
    """Static, read-only config the user can't change from the UI."""
    eval_root = getattr(request.app.state, "console_eval_results_root", None)
    return {
        "version": getattr(request.app.state, "console_version", "?"),
        "url_prefix": getattr(request.app.state, "console_url_prefix", "/console"),
        "eval_results_root": str(eval_root) if eval_root else None,
        "rllm_home": os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm")),
        "gateway_db_path": os.path.expanduser(os.environ.get("RLLM_GATEWAY_DB", "~/.rllm/gateway/traces.db")),
        "console_env_file": str(store.default_env_path()),
    }


@router.get("/env")
def list_env() -> dict[str, Any]:
    """All known env vars + any "Other" set on the process."""
    file_pairs = store.read_file()
    known = known_by_key()

    rows: list[EnvVarRow] = []

    # Known vars first, in registry order (stable across reloads).
    for kv in KNOWN_ENV_VARS:
        env_val = os.environ.get(kv.key)
        rows.append(
            EnvVarRow(
                key=kv.key,
                label=kv.label,
                description=kv.description,
                category=kv.category,
                secret=kv.secret,
                url=kv.url,
                is_set=env_val is not None,
                in_console_file=kv.key in file_pairs,
                masked_value=(_mask(env_val) if (env_val and kv.secret) else env_val),
            ),
        )

    # "Other" — anything in console.env not already in the known list.
    # We deliberately do NOT enumerate every os.environ entry; that
    # would surface PATH, HOME, and tons of noise. Only file-managed
    # extras show up here.
    for k, v in file_pairs.items():
        if k in known:
            continue
        secret = _is_secret_shaped(k)
        rows.append(
            EnvVarRow(
                key=k,
                label=k,
                description="(custom)",
                category="Other",
                secret=secret,
                is_set=k in os.environ,
                in_console_file=True,
                masked_value=(_mask(v) if secret else v),
            ),
        )

    return {
        "categories": [*categories(), "Other"],
        "rows": [r.model_dump() for r in rows],
        "console_env_file": str(store.default_env_path()),
    }


@router.get("/env/{key}")
def reveal_env(
    key: str,
    reveal: bool = Query(default=False, description="When true, returns the unmasked value."),
) -> dict[str, Any]:
    """Read a single key. Default response is masked; pass ``reveal=true``
    to get the full value (intended for the edit-flow on a localhost UI)."""
    if not _VALID_KEY_RE.match(key):
        raise HTTPException(400, "Bad key name")

    val = os.environ.get(key)
    if val is None:
        raise HTTPException(404, "Not set")
    known = known_by_key().get(key)
    secret = known.secret if known else _is_secret_shaped(key)

    return {
        "key": key,
        "value": val if reveal else (_mask(val) if secret else val),
        "secret": secret,
        "revealed": bool(reveal),
        "in_console_file": key in store.read_file(),
    }


@router.post("/env")
def set_env(req: EnvWriteRequest) -> dict[str, Any]:
    """Persist + apply a key=value pair."""
    if not _VALID_KEY_RE.match(req.key):
        raise HTTPException(400, "Bad key name")
    store.write_assignment(req.key, req.value)
    os.environ[req.key] = req.value
    return {"key": req.key, "ok": True}


@router.delete("/env/{key}")
def delete_env(key: str) -> dict[str, Any]:
    """Unset in the running process and remove from the persistence file."""
    if not _VALID_KEY_RE.match(key):
        raise HTTPException(400, "Bad key name")
    removed_from_file = store.delete_assignment(key)
    removed_from_env = key in os.environ
    if removed_from_env:
        del os.environ[key]
    return {
        "key": key,
        "removed_from_file": removed_from_file,
        "removed_from_env": removed_from_env,
    }

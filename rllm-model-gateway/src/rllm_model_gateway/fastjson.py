"""Fast JSON encode/decode via orjson, with a stdlib fallback.

orjson is ~2-5x faster than the stdlib ``json`` module, and at high request
concurrency JSON (de)serialization is the gateway event loop's dominant
per-request CPU cost (see the loop-health monitor). All helpers return/accept
``bytes`` (orjson-native); callers that need ``str`` decode explicitly.

``dumps``/``dumps_sorted`` fall back to the stdlib on any type orjson can't
serialize natively (with ``default=str``), so swapping in this module never
changes *which* payloads serialize successfully — only how fast.
"""

from __future__ import annotations

import json as _json
from typing import Any

try:
    import orjson

    _OPT_SORT = orjson.OPT_SORT_KEYS

    def dumps(obj: Any) -> bytes:
        """Serialize *obj* to UTF-8 JSON bytes."""
        try:
            return orjson.dumps(obj)
        except TypeError:
            return _json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")

    def dumps_sorted(obj: Any) -> bytes:
        """Serialize *obj* with sorted keys (stable) to UTF-8 JSON bytes."""
        try:
            return orjson.dumps(obj, option=_OPT_SORT)
        except TypeError:
            return _json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")

    def loads(data: bytes | str) -> Any:
        return orjson.loads(data)

    HAVE_ORJSON = True

except ImportError:  # pragma: no cover - orjson is a declared dep; fallback for safety

    def dumps(obj: Any) -> bytes:
        return _json.dumps(obj, ensure_ascii=False).encode("utf-8")

    def dumps_sorted(obj: Any) -> bytes:
        return _json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")

    def loads(data: bytes | str) -> Any:
        return _json.loads(data)

    HAVE_ORJSON = False

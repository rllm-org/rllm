"""Fast JSON encode/decode via orjson, with a stdlib fallback.

orjson is ~2-5x faster than the stdlib ``json`` module, and at high request
concurrency JSON (de)serialization is the gateway event loop's dominant
per-request CPU cost (see the loop-health monitor). All helpers return/accept
``bytes`` (orjson-native); callers that need ``str`` decode explicitly.

The stdlib fallback accepts lone surrogate escapes that ``orjson`` rejects and
escapes them again on output. It still rejects non-finite constants and float
overflow, so only surrogate handling broadens accepted input.
"""

from __future__ import annotations

import json as _json
import math
from typing import Any


def _dumps_stdlib(obj: Any, *, sort_keys: bool = False) -> bytes:
    rendered = _json.dumps(obj, sort_keys=sort_keys, ensure_ascii=False, allow_nan=False, default=str)
    return rendered.encode("utf-8", errors="backslashreplace")


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite number is not valid JSON: {value}")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_nonfinite(value)
    return parsed


try:
    import orjson

    _OPT_SORT = orjson.OPT_SORT_KEYS

    def dumps(obj: Any) -> bytes:
        """Serialize *obj* to UTF-8 JSON bytes."""
        try:
            return orjson.dumps(obj)
        except TypeError:
            return _dumps_stdlib(obj)

    def dumps_sorted(obj: Any) -> bytes:
        """Serialize *obj* with sorted keys (stable) to UTF-8 JSON bytes."""
        try:
            return orjson.dumps(obj, option=_OPT_SORT)
        except TypeError:
            return _dumps_stdlib(obj, sort_keys=True)

    def loads(data: bytes | str) -> Any:
        try:
            return orjson.loads(data)
        except orjson.JSONDecodeError:
            return _json.loads(data, parse_constant=_reject_nonfinite, parse_float=_parse_float)

    HAVE_ORJSON = True

except ImportError:  # pragma: no cover - orjson is a declared dep; fallback for safety

    def dumps(obj: Any) -> bytes:
        return _dumps_stdlib(obj)

    def dumps_sorted(obj: Any) -> bytes:
        return _dumps_stdlib(obj, sort_keys=True)

    def loads(data: bytes | str) -> Any:
        return _json.loads(data, parse_constant=_reject_nonfinite, parse_float=_parse_float)

    HAVE_ORJSON = False

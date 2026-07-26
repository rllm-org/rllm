"""Test fixtures for plugin unit tests.

Adds `plugins/*/` to sys.path so tests can `import rllm_trace_uploader`
without needing `uv pip install` first.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

for _plugin in (
    "rllm_trace_uploader",
    "rllm_trace_sidecar",
    "rllm_sandbox_snapshot",
):
    _path = os.path.join(_ROOT, "plugins", _plugin)
    if _path not in sys.path:
        sys.path.insert(0, _path)

"""Launcher: import observability plugins then delegate to train.py.

The plugins install their monkey-patches as an import side effect:
    - rllm_trace_sidecar   → patches VerlBackend.on_{train_start,batch_start,policy_updated}
    - rllm_sandbox_snapshot → patches BwrapSandbox.close

If either is missing (dev pod without plugins installed) we log and continue —
training must never be broken by observability code.

Hydra CLI args are passed through sys.argv unchanged.
"""

from __future__ import annotations

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _try_import(name: str) -> None:
    try:
        __import__(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[observability] {name} import failed (non-fatal): {exc}", file=sys.stderr)


_try_import("rllm_trace_sidecar")
_try_import("rllm_sandbox_snapshot")

# Delegate to train.py preserving argv.
sys.argv[0] = os.path.join(_HERE, "train.py")
runpy.run_path(sys.argv[0], run_name="__main__")

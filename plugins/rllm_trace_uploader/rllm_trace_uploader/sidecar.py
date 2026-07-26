"""Read sidecar files written by rllm_trace_sidecar (VerlBackend monkey-patch).

Files under $RLLM_HOME/observability/:
    wandb_run_id.txt         — one line: "<entity>/<project>/<run_id>"
    current_step.txt         — one line: integer global step
    checkpoint_versions.jsonl — append-only, each line {"step": int, "ts": int_ns}

mtime-cache: reader keeps last mtime + cached value; only re-reads on change.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class SidecarReader:
    def __init__(self, root: str) -> None:
        self.root = root
        self._run_id_path = os.path.join(root, "wandb_run_id.txt")
        self._step_path = os.path.join(root, "current_step.txt")
        self._ckpt_path = os.path.join(root, "checkpoint_versions.jsonl")

        self._run_id: str | None = None
        self._run_id_mtime: float = 0.0
        self._step: int | None = None
        self._step_mtime: float = 0.0
        self._ckpt: dict[str, Any] | None = None
        self._ckpt_mtime: float = 0.0

    def _read_if_changed(
        self,
        path: str,
        cached_mtime: float,
    ) -> tuple[str | None, float]:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return None, cached_mtime
        except OSError as exc:
            logger.warning("stat(%s) failed: %s", path, exc)
            return None, cached_mtime
        if st.st_mtime <= cached_mtime:
            return "__unchanged__", cached_mtime
        try:
            with open(path) as f:
                return f.read().strip(), st.st_mtime
        except OSError as exc:
            logger.warning("read(%s) failed: %s", path, exc)
            return None, cached_mtime

    def get_run_id(self) -> str | None:
        data, mtime = self._read_if_changed(self._run_id_path, self._run_id_mtime)
        if data == "__unchanged__":
            return self._run_id
        self._run_id_mtime = mtime
        self._run_id = data or None
        return self._run_id

    def get_current_step(self) -> int | None:
        data, mtime = self._read_if_changed(self._step_path, self._step_mtime)
        if data == "__unchanged__":
            return self._step
        self._step_mtime = mtime
        if data:
            try:
                self._step = int(data)
            except ValueError:
                logger.warning("current_step.txt has non-int: %r", data)
                self._step = None
        else:
            self._step = None
        return self._step

    def get_latest_checkpoint(self) -> dict[str, Any] | None:
        try:
            st = os.stat(self._ckpt_path)
        except FileNotFoundError:
            return self._ckpt
        except OSError:
            return self._ckpt
        if st.st_mtime <= self._ckpt_mtime:
            return self._ckpt
        try:
            with open(self._ckpt_path) as f:
                lines = f.readlines()
        except OSError:
            return self._ckpt
        if not lines:
            return self._ckpt
        self._ckpt_mtime = st.st_mtime
        try:
            self._ckpt = json.loads(lines[-1].strip())
        except json.JSONDecodeError:
            logger.warning("checkpoint_versions.jsonl last line invalid JSON")
        return self._ckpt

"""MultimodalCodexHarness — extends CodexHarness to upload image bytes.

Kept in the cookbook (not in ``rllm/harnesses/``) so we don't touch the main
framework. The subclass overrides ``write_configs``: after the base class
writes Codex's ``auth.json`` + ``config.toml``, we stage each image blob from
``task.metadata`` to a local tmp file and call ``sandbox.upload_file`` (heredoc
is unsafe for binary PNG data).

Two payload shapes supported:

- single: ``metadata['image_bytes']`` (bytes) + ``metadata['image_file']`` (absolute path)
- multi:  ``metadata['images_bytes']`` (list[bytes]) + ``metadata['image_files']`` (list[abs path])

Absolute paths are important — Codex CLI's ``--image <path>`` consumes them
directly, so the harness doesn't need to know or match the sandbox workdir.
"""

from __future__ import annotations

import os
import tempfile

from rllm.harnesses.codex import CodexHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Task


class MultimodalCodexHarness(CodexHarness):
    """CodexHarness variant that also writes task image bytes to the sandbox."""

    name = "multimodal_codex"

    def write_configs(
        self,
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        super().write_configs(sandbox, task, config, env)
        if not task.metadata:
            return

        # Single-image shape
        single_bytes = task.metadata.get("image_bytes")
        single_name = task.metadata.get("image_file")
        if single_bytes and single_name:
            self._upload_bytes(sandbox, single_name, single_bytes)

        # Multi-image shape — bytes and paths are zipped in order
        multi_names = task.metadata.get("image_files") or []
        multi_bytes = task.metadata.get("images_bytes") or []
        for name, blob in zip(multi_names, multi_bytes):
            if not name or not blob:
                continue
            self._upload_bytes(sandbox, name, blob)

    @staticmethod
    def _upload_bytes(sandbox: Sandbox, remote_path: str, blob) -> None:
        """Stage bytes to a local tmp file, upload to sandbox, clean up.

        Accepts either:
          - raw ``bytes`` / ``bytearray`` (Arrow IPC / rLLM native path), or
          - verl's ``{"bytes": ..., "path": "..."}`` wrapper produced by
            ``DatasetRegistry._wrap_binary_columns_for_parquet`` when writing
            the ``_verl.parquet`` companion file.

        Non-bytes / empty / unrecognised shapes are silently skipped so that
        missing metadata doesn't crash the rollout.
        """
        if isinstance(blob, dict):
            blob = blob.get("bytes")
        if not isinstance(blob, (bytes, bytearray)):
            return
        if not blob:
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as f:
            f.write(blob)
            local_path = f.name
        try:
            sandbox.upload_file(local_path, remote_path)
        finally:
            os.unlink(local_path)

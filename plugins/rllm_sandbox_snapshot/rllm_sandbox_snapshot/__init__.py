"""Monkey-patch BwrapSandbox.close to rsync workdir /app + /tmp before teardown.

Import-triggered side effect. Failures are logged and swallowed. Enabled only
when RLLM_SANDBOX_SNAPSHOT_ROOT is set — otherwise the patch is a no-op.

Snapshot layout:
    $RLLM_SANDBOX_SNAPSHOT_ROOT/{seq:08d}_{sandbox_name}/
        app/  ← copy of sandbox /app (excluding pyc/pycache/git)
        tmp/  ← copy of sandbox /tmp
"""

from __future__ import annotations

import itertools
import logging
import os
import subprocess

logger = logging.getLogger("rllm_sandbox_snapshot")

_SEQ = itertools.count()


def _snapshot(name: str, workdir: str) -> None:
    root = os.environ.get("RLLM_SANDBOX_SNAPSHOT_ROOT")
    if not root:
        return
    seq_id = next(_SEQ)
    safe_name = name.replace("/", "_").replace(":", "_")
    dst = os.path.join(root, f"{seq_id:08d}_{safe_name}")
    try:
        os.makedirs(dst, exist_ok=True)
    except OSError as exc:  # noqa: BLE001
        logger.warning("snapshot: mkdir %s failed: %s", dst, exc)
        return

    for sub in ("app", "tmp"):
        src = os.path.join(workdir, sub) + "/"
        if not os.path.isdir(src):
            continue
        try:
            subprocess.run(
                [
                    "rsync", "-a",
                    "--exclude=__pycache__",
                    "--exclude=*.pyc",
                    "--exclude=.git",
                    "--max-size=10M",
                    src,
                    os.path.join(dst, sub),
                ],
                check=False,
                capture_output=True,
                timeout=60,
            )
        except Exception:  # noqa: BLE001
            logger.exception("snapshot rsync failed for %s → %s", src, dst)


def _install() -> None:
    try:
        from rllm.sandbox.backends.bwrap import BwrapSandbox
    except Exception as exc:  # noqa: BLE001
        logger.warning("BwrapSandbox import failed, snapshot plugin disabled: %s", exc)
        return

    _orig_close = BwrapSandbox.close

    def _patched_close(self):  # type: ignore[no-untyped-def]
        try:
            _snapshot(self.name, self._workdir)
        except Exception:  # noqa: BLE001
            logger.exception("snapshot failed (non-fatal)")
        return _orig_close(self)

    BwrapSandbox.close = _patched_close  # type: ignore[method-assign]
    logger.info("BwrapSandbox.close patched for workspace snapshots")


_install()

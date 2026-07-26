"""Monkey-patch VerlBackend hooks to persist wandb_run_id / current_step /
checkpoint_versions to $RLLM_HOME/observability/ for the trace uploader.

Import-triggered side effect. Failures are logged and swallowed — this plugin
must never break training.

Enabled by importing the module once before training starts, e.g.:
    python -c "import rllm_trace_sidecar; import cookbooks.multimodal_codex.train"
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger("rllm_trace_sidecar")

_OBS_DIR = os.path.join(
    os.environ.get("RLLM_HOME", "/tmp/rllm_home"),
    "observability",
)


def _atomic_write(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def _install() -> None:
    try:
        os.makedirs(_OBS_DIR, exist_ok=True)
    except OSError as exc:  # noqa: BLE001
        logger.warning("could not create %s: %s", _OBS_DIR, exc)
        return

    try:
        from rllm.trainer.verl.verl_backend import VerlBackend
    except Exception as exc:  # noqa: BLE001
        logger.warning("VerlBackend import failed, sidecar disabled: %s", exc)
        return

    _orig_on_train_start = VerlBackend.on_train_start
    _orig_on_batch_start = VerlBackend.on_batch_start
    _orig_on_policy_updated = VerlBackend.on_policy_updated

    async def _patched_on_train_start(self, trainer_state):  # type: ignore[no-untyped-def]
        await _orig_on_train_start(self, trainer_state)
        try:
            import wandb

            if wandb.run is not None:
                _atomic_write(
                    os.path.join(_OBS_DIR, "wandb_run_id.txt"),
                    f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}",
                )
        except Exception:  # noqa: BLE001
            logger.exception("sidecar: on_train_start write failed (non-fatal)")

    async def _patched_on_batch_start(self, trainer_state):  # type: ignore[no-untyped-def]
        await _orig_on_batch_start(self, trainer_state)
        try:
            _atomic_write(
                os.path.join(_OBS_DIR, "current_step.txt"),
                str(trainer_state.global_step),
            )
        except Exception:  # noqa: BLE001
            logger.exception("sidecar: on_batch_start write failed (non-fatal)")

    async def _patched_on_policy_updated(self, trainer_state):  # type: ignore[no-untyped-def]
        await _orig_on_policy_updated(self, trainer_state)
        try:
            line = json.dumps({"step": trainer_state.global_step, "ts": time.time_ns()})
            with open(os.path.join(_OBS_DIR, "checkpoint_versions.jsonl"), "a") as f:
                f.write(line + "\n")
        except Exception:  # noqa: BLE001
            logger.exception("sidecar: on_policy_updated write failed (non-fatal)")

    VerlBackend.on_train_start = _patched_on_train_start  # type: ignore[method-assign]
    VerlBackend.on_batch_start = _patched_on_batch_start  # type: ignore[method-assign]
    VerlBackend.on_policy_updated = _patched_on_policy_updated  # type: ignore[method-assign]
    logger.info("VerlBackend hooks patched (obs_dir=%s)", _OBS_DIR)


_install()

"""vLLM colocate worker extension that overrides the ZMQ IPC path.

Backports volcengine/verl#6246 for verl 0.7.1 (our pin), which keys the
weight-transfer IPC socket on the GPU UUID alone and so collides whenever
two verl jobs land on the same physical GPU. The PR adds the Ray job id
to the path; we mirror that here.

This subclass lives in its own module — NOT inlined into ``patch.py`` —
for a load-bearing reason. ``patch.py:apply_all_verl_patches`` runs as
the Ray ``runtime_env.worker_process_setup_hook``, BEFORE Ray sets the
worker's ``CUDA_VISIBLE_DEVICES``. Importing
``verl.workers.rollout.vllm_rollout.utils`` to derive a subclass at hook
time pulls in vLLM internals, which call ``torch.cuda.is_initialized()``
and pin every rank to GPU 0 — surfaces later as
``ncclInvalidUsage: Duplicate GPU detected: rank 0 and rank 1 both on
CUDA device 29000`` during FSDP init. Keeping the import isolated here,
and only resolving the FQN inside the vLLM mp-spawn worker via
``resolve_obj_by_qualname`` (after Ray has set CUDA_VISIBLE_DEVICES),
sidesteps that.
"""

from __future__ import annotations

import os

from verl.workers.rollout.vllm_rollout.utils import (
    get_device_uuid,
    vLLMColocateWorkerExtension,
)


class RLLMColocateWorkerExtension(vLLMColocateWorkerExtension):
    """``vLLMColocateWorkerExtension`` with Ray-job-id-aware IPC path."""

    def _get_zmq_handle(self) -> str:
        if not hasattr(self, "device_uuid") or not self.device_uuid:
            self.device_uuid = get_device_uuid(self.device.index)
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-{self.device_uuid}.sock"

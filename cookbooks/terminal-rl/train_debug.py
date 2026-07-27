"""Train a terminal agent on a local set of terminal-agent tasks, eval on
Terminal-Bench.

This cookbook deliberately ships no custom AgentFlow or evaluator:

* The **agent** is the in-tree ``mini-swe-agent`` harness
  (:class:`rllm.harnesses.mini_swe_agent.MiniSweAgentHarness`) — a
  :class:`~rllm.sandbox.sandboxed_flow.SandboxedAgentFlow` that runs the
  Mini-SWE CLI inside each task's sandbox. The rLLM gateway
  intercepts every LLM call, so the trainer sees full trajectories without the
  harness knowing it's being trained.
* The **evaluator** is each task's own verifier (sandbox-shell), resolved
  per-task by :class:`rllm.hooks.SandboxTaskHooks`. Both the local training
  tasks and the Terminal-Bench eval tasks ship a ``tests/test.sh`` that writes
  ``1.0``/``0.0`` to ``/logs/verifier/reward.txt``; rLLM reads that back as the
  RL reward.

Because we pass an ``agent_flow`` (and no explicit ``evaluator``/``hooks``),
:class:`AgentTrainer` runs the **rLLM-native SandboxedAgentFlow path**
(``AgentFlowEngine``) — sandboxes are created locally by ``SandboxTaskHooks``
via a pluggable ``sandbox_backend`` (``docker`` | ``local`` | ``modal`` |
``daytona``). This is NOT the remote-runtime / ``RemoteAgentFlowEngine`` path.

The sandbox backend is selected by the ``TERMINAL_SANDBOX_BACKEND`` env var
(default ``modal``). For ``modal`` install ``pip install modal`` and run
``modal token new``; for ``daytona`` install ``pip install daytona`` and set
``DAYTONA_API_KEY``. Everything else is configured by Hydra overrides on the
command line (see ``train_tinker.sh`` / ``train_verl.sh`` for working defaults).

Usage (from rllm repo root)::

    TERMINAL_SANDBOX_BACKEND=modal python cookbooks/terminal-rl/train_debug.py rllm/backend=fireworks
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import pickle
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.harnesses.mini_swe_agent import MiniSweAgentHarness
from rllm.trainer import AgentTrainer

TRAIN_DATASET = os.environ.get("TB_TRAIN_DATASET", "tb_v2_debug")

# Terminal-Bench eval version (Harbor registry). Must match prepare_data.py;
# both read TB_EVAL_VERSION so the pulled and loaded dataset names agree.
EVAL_VERSION = os.environ.get("TB_EVAL_VERSION", "2.0")
VAL_DATASET = f"terminal-bench@{EVAL_VERSION}"

# Sandbox backend for the SandboxedAgentFlow path: docker | local | modal | daytona.
SANDBOX_BACKEND = os.environ.get("TERMINAL_SANDBOX_BACKEND", "modal")

# Optional cap on the validation set size. Terminal-Bench 2.0 is 89 tasks;
# validation runs ALL of them every time it fires, which is slow. Set
# TB_VAL_MAX=N to validate on the first N tasks instead (0/unset = all).
TB_VAL_MAX = int(os.environ.get("TB_VAL_MAX", "0"))

# Per-rollout Mini-SWE model-call cap. Each step is one agent turn. The
# Fireworks debug launcher defaults this to 64.
MINISWE_MAX_TURNS = int(os.environ.get("MINISWE_MAX_TURNS", "64"))
if MINISWE_MAX_TURNS <= 0:
    raise ValueError(f"MINISWE_MAX_TURNS must be positive, got {MINISWE_MAX_TURNS}")

MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS = int(os.environ.get("MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS", "1"))
if MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS < 0:
    raise ValueError(
        "MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS must be non-negative, "
        f"got {MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS}"
    )

MINISWE_COMMAND_TIMEOUT = int(os.environ.get("MINISWE_COMMAND_TIMEOUT", "300"))
if MINISWE_COMMAND_TIMEOUT <= 0:
    raise ValueError(f"MINISWE_COMMAND_TIMEOUT must be positive, got {MINISWE_COMMAND_TIMEOUT}")


_COMPARE_DIR = os.environ.get("FIREWORKS_COMPARE_DIR")


def _jsonable(value, depth: int = 0):
    if depth > 30:
        return repr(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v, depth + 1) for v in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="python"), depth + 1)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict(), depth + 1)
        except Exception:
            pass
    if hasattr(value, "data") and not isinstance(value, str):
        try:
            return {"data": _jsonable(value.data, depth + 1), "dtype": str(getattr(value, "dtype", ""))}
        except Exception:
            pass
    return repr(value)


def _artifact_base(name: str) -> Path:
    assert _COMPARE_DIR is not None
    path = Path(_COMPARE_DIR) / re.sub(r"[^A-Za-z0-9_./-]", "_", name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_artifact(name: str, value) -> None:
    base = _artifact_base(name)
    try:
        with base.with_suffix(".pkl").open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        try:
            import cloudpickle

            with base.with_suffix(".pkl").open("wb") as f:
                cloudpickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as fallback_exc:
            base.with_suffix(".pickle_error.txt").write_text(
                f"pickle: {type(exc).__name__}: {exc}\n"
                f"cloudpickle: {type(fallback_exc).__name__}: {fallback_exc}\n"
            )
    try:
        with base.with_suffix(".json").open("w") as f:
            json.dump(_jsonable(value), f, indent=2, allow_nan=True)
    except Exception as exc:
        base.with_suffix(".json_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")


def _flatten_datums(datums) -> list:
    if isinstance(datums, dict):
        return [datum for role_datums in datums.values() for datum in role_datums]
    return list(datums)


def _field(datum, name: str) -> list:
    value = datum.loss_fn_inputs[name]
    return list(value.data)


def _token_diagnostics(datums: list, current_logprobs: list, delta: float) -> dict:
    rows = []
    loss_sum = 0.0
    active_tokens = 0.0
    dropped_tokens = 0.0
    for datum_idx, (datum, current) in enumerate(zip(datums, current_logprobs, strict=True)):
        targets = _field(datum, "target_tokens")
        rollout = _field(datum, "logprobs")
        advantages = _field(datum, "advantages")
        mask = _field(datum, "mask")
        current = list(current)
        if not (len(current) == len(targets) == len(rollout) == len(advantages) == len(mask)):
            raise ValueError(
                f"datum {datum_idx} diagnostic length mismatch: "
                f"current={len(current)} targets={len(targets)} rollout={len(rollout)} "
                f"advantages={len(advantages)} mask={len(mask)}"
            )
        tokens = []
        for token_idx, (target, curr, old, advantage, action_mask) in enumerate(
            zip(targets, current, rollout, advantages, mask, strict=True)
        ):
            curr = float(curr)
            old = float(old)
            advantage = float(advantage)
            action_mask = float(action_mask)
            # Match rllm.trainer.algorithms.loss._ratio exactly.
            ratio = math.exp(max(-20.0, min(20.0, curr - old)))
            probability_delta = math.exp(curr) - math.exp(old)
            keep = probability_delta <= delta if advantage > 0 else probability_delta >= -delta if advantage < 0 else True
            contribution = -advantage * ratio * curr * float(keep) * action_mask
            loss_sum += contribution
            active_tokens += action_mask
            dropped_tokens += action_mask * float(not keep)
            tokens.append(
                {
                    "token_index": token_idx,
                    "target_token_id": int(target),
                    "trainer_logprob": curr,
                    "rollout_logprob": old,
                    "advantage": advantage,
                    "action_mask": action_mask,
                    "ratio": ratio,
                    "probability_delta": probability_delta,
                    "dppo_keep": bool(keep),
                    "loss_contribution": contribution,
                }
            )
        rows.append({"datum_index": datum_idx, "tokens": tokens})
    return {
        "delta": delta,
        "active_tokens": active_tokens,
        "dropped_tokens": dropped_tokens,
        "dppo_mask_frac": dropped_tokens / max(active_tokens, 1.0),
        "loss_sum": loss_sum,
        "loss_token_mean": loss_sum / max(active_tokens, 1.0),
        "datums": rows,
    }


def _install_fireworks_comparison_capture() -> None:
    if _COMPARE_DIR is None:
        return

    Path(_COMPARE_DIR).mkdir(parents=True, exist_ok=True)

    from rllm.trainer.fireworks.fireworks_policy_trainer import FireworksPolicyTrainer
    from rllm.trainer.buffer import TrajectoryGroupBuffer
    from rllm.trainer.tinker.transform import transform_trajectory_groups_to_datums

    original_buffer_get = TrajectoryGroupBuffer.get
    original_run_op = FireworksPolicyTrainer._run_training_op
    original_forward_backward = FireworksPolicyTrainer.forward_backward_from_trajectory_groups
    original_optim_step = FireworksPolicyTrainer.optim_step
    op_counter = 0
    chunk_counter = 0
    consumed_slots = []

    async def capture_buffer_get(self):
        batch = await original_buffer_get(self)
        if batch is None:
            return None
        consumed_slots.append(batch.groups)
        slot_idx = len(consumed_slots) - 1
        await asyncio.to_thread(
            _save_artifact,
            f"consumed_slots/{slot_idx:03d}_trajectory_groups",
            batch.groups,
        )
        mini_batch_size = self._coordinator.config.mini_batch_size
        if len(consumed_slots) % mini_batch_size == 0:
            optimizer_batch_idx = len(consumed_slots) // mini_batch_size - 1
            start = optimizer_batch_idx * mini_batch_size
            await asyncio.to_thread(
                _save_artifact,
                f"optimizer_batches/{optimizer_batch_idx:03d}_trajectory_group_chunks",
                consumed_slots[start : start + mini_batch_size],
            )
        return batch

    async def capture_run_op(self, fn, *args, op_name: str, reconnect: bool = False, **kwargs):
        nonlocal op_counter
        op_idx = op_counter
        op_counter += 1
        if op_name in {"forward", "forward_backward", "forward_backward_custom", "optim_step"}:
            await asyncio.to_thread(
                _save_artifact,
                f"trainer_ops/{op_idx:03d}_{op_name}_request",
                {"args": args, "kwargs": kwargs},
            )
        if op_name == "forward_backward_custom" and len(args) >= 2:
            original_loss_fn = args[1]
            active_chunk = getattr(self, "_comparison_active_chunk", None)

            def capture_loss_fn(data, logprobs_list):
                actual_logprobs = [row.detach().cpu().tolist() for row in logprobs_list]
                _save_artifact(
                    f"trainer_ops/{op_idx:03d}_{op_name}_loss_fn_inputs",
                    {"data": data, "trainer_logprobs": actual_logprobs},
                )
                if active_chunk is not None:
                    chunk_idx, flat_datums, delta = active_chunk
                    diagnostics = _token_diagnostics(flat_datums, actual_logprobs, delta)
                    _save_artifact(
                        f"chunks/{chunk_idx:03d}/trainer_logprobs_forward_backward",
                        actual_logprobs,
                    )
                    _save_artifact(
                        f"chunks/{chunk_idx:03d}/per_token_diagnostics_forward_backward",
                        diagnostics,
                    )
                    actual = getattr(self, "_comparison_actual", {})
                    actual[chunk_idx] = (actual_logprobs, diagnostics)
                    self._comparison_actual = actual
                loss, metrics = original_loss_fn(data, logprobs_list)
                _save_artifact(
                    f"trainer_ops/{op_idx:03d}_{op_name}_loss_fn_output",
                    {
                        "loss": float(loss.detach().cpu()),
                        "metrics": metrics,
                    },
                )
                return loss, metrics

            args = (args[0], capture_loss_fn, *args[2:])
        try:
            result = await original_run_op(
                self,
                fn,
                *args,
                op_name=op_name,
                reconnect=reconnect,
                **kwargs,
            )
        except Exception as exc:
            if op_name in {"forward", "forward_backward", "forward_backward_custom", "optim_step"}:
                await asyncio.to_thread(
                    _save_artifact,
                    f"trainer_ops/{op_idx:03d}_{op_name}_error",
                    {"error_type": type(exc).__name__, "message": str(exc)},
                )
            raise
        if op_name in {"forward", "forward_backward", "forward_backward_custom", "optim_step"}:
            await asyncio.to_thread(
                _save_artifact,
                f"trainer_ops/{op_idx:03d}_{op_name}_response",
                result,
            )
        return result

    async def capture_forward_backward(self, trajectory_groups, algorithm_config=None):
        nonlocal chunk_counter
        chunk_idx = chunk_counter
        chunk_counter += 1
        algorithm_config = algorithm_config or self.algorithm_config

        await asyncio.to_thread(
            _save_artifact,
            f"chunks/{chunk_idx:03d}/trajectory_groups",
            trajectory_groups,
        )
        datums, transform_metrics = transform_trajectory_groups_to_datums(
            trajectory_groups,
            algorithm_config=algorithm_config,
            vocab_size=self._get_vocab_size(),
        )
        flat_datums = _flatten_datums(datums)
        await asyncio.to_thread(
            _save_artifact,
            f"chunks/{chunk_idx:03d}/fireworks_datums",
            {"datums": datums, "transform_metrics": transform_metrics},
        )

        clean_datums, _, _, _, _ = self._process_datums(flat_datums)
        trainer_logprobs = await self._compute_proximal_logprobs(clean_datums)
        await asyncio.to_thread(
            _save_artifact,
            f"chunks/{chunk_idx:03d}/trainer_logprobs_pre_update",
            trainer_logprobs,
        )
        delta = float((getattr(algorithm_config, "loss_params", None) or {}).get("delta", algorithm_config.eps_clip))
        diagnostics = _token_diagnostics(flat_datums, trainer_logprobs, delta)
        await asyncio.to_thread(
            _save_artifact,
            f"chunks/{chunk_idx:03d}/per_token_diagnostics_pre_update",
            diagnostics,
        )

        state = getattr(self, "_comparison_chunks", [])
        state.append((chunk_idx, clean_datums, flat_datums, trainer_logprobs, delta, diagnostics))
        self._comparison_chunks = state
        self._comparison_active_chunk = (chunk_idx, flat_datums, delta)

        try:
            result = await original_forward_backward(
                self,
                trajectory_groups,
                algorithm_config=algorithm_config,
            )
        finally:
            self._comparison_active_chunk = None
        await asyncio.to_thread(
            _save_artifact,
            f"chunks/{chunk_idx:03d}/forward_backward_result",
            result,
        )
        return result

    async def capture_optim_step(self, *args, **kwargs):
        chunks = getattr(self, "_comparison_chunks", [])
        actual = getattr(self, "_comparison_actual", {})
        summaries = [actual.get(chunk[0], (None, chunk[5]))[1] for chunk in chunks]
        total_loss_sum = sum(summary["loss_sum"] for summary in summaries)
        total_active_tokens = sum(summary["active_tokens"] for summary in summaries)
        total_dropped_tokens = sum(summary["dropped_tokens"] for summary in summaries)
        await asyncio.to_thread(
            _save_artifact,
            "optimizer_batch_pre_update_summary",
            {
                "loss_sum": total_loss_sum,
                "active_tokens": total_active_tokens,
                "loss_token_mean": total_loss_sum / max(total_active_tokens, 1.0),
                "dropped_tokens": total_dropped_tokens,
                "dppo_mask_frac": total_dropped_tokens / max(total_active_tokens, 1.0),
                "chunk_summaries": [
                    {k: v for k, v in summary.items() if k != "datums"}
                    for summary in summaries
                ],
            },
        )
        result = await original_optim_step(self, *args, **kwargs)
        await asyncio.to_thread(_save_artifact, "optim_step_result", result)
        for chunk_idx, clean_datums, flat_datums, pre_logprobs, delta, _ in chunks:
            post_logprobs = await self._compute_proximal_logprobs(clean_datums)
            comparison_pre = actual.get(chunk_idx, (pre_logprobs, None))[0]
            deltas = [
                [float(post) - float(pre) for pre, post in zip(pre_row, post_row, strict=True)]
                for pre_row, post_row in zip(comparison_pre, post_logprobs, strict=True)
            ]
            await asyncio.to_thread(
                _save_artifact,
                f"chunks/{chunk_idx:03d}/trainer_logprobs_post_update",
                {"logprobs": post_logprobs, "delta_from_pre_update": deltas},
            )
            await asyncio.to_thread(
                _save_artifact,
                f"chunks/{chunk_idx:03d}/per_token_diagnostics_post_update",
                _token_diagnostics(flat_datums, post_logprobs, delta),
            )
        return result

    TrajectoryGroupBuffer.get = capture_buffer_get
    FireworksPolicyTrainer._run_training_op = capture_run_op
    FireworksPolicyTrainer.forward_backward_from_trajectory_groups = capture_forward_backward
    FireworksPolicyTrainer.optim_step = capture_optim_step


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    _install_fireworks_comparison_capture()
    train_dataset = DatasetRegistry.load_dataset(TRAIN_DATASET, "train")
    val_dataset = DatasetRegistry.load_dataset(VAL_DATASET, "default")

    if train_dataset is None:
        raise RuntimeError(f"Dataset '{TRAIN_DATASET}' not found. Run: python cookbooks/terminal-rl/prepare_data.py")
    if val_dataset is None:
        raise RuntimeError(f"Dataset '{VAL_DATASET}' not found. Run: rllm dataset pull harbor:{VAL_DATASET} (or: python cookbooks/terminal-rl/prepare_data.py)")

    if TB_VAL_MAX > 0 and TB_VAL_MAX < len(val_dataset):
        val_dataset = val_dataset.select(range(TB_VAL_MAX))

    # Mini-SWE as a SandboxedAgentFlow. Passing ``agent_flow`` (with no
    # explicit evaluator/hooks) makes AgentTrainer auto-wire SandboxTaskHooks
    # for the sandbox lifecycle + per-task verifier, and route rollouts through
    # AgentFlowEngine — rLLM's own runtime, not the remote Harbor runtime.
    agent_flow = MiniSweAgentHarness(
        sandbox_backend=SANDBOX_BACKEND,
        max_turns=MINISWE_MAX_TURNS,
        max_consecutive_format_errors=MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS,
        command_timeout=MINISWE_COMMAND_TIMEOUT,
        # Only the explicit Mini-SWE submission sentinel is graded.
        capture_exit_status=True,
        verify_only_on_env_done=True,
        skipped_verifier_reward=0.0,
        cost_limit=0.0,
    )

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_flow=agent_flow,
        sandbox_backend=SANDBOX_BACKEND,
    )
    trainer.train()


if __name__ == "__main__":
    main()

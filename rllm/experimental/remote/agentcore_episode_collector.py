"""
AgentCore Episode Collector for fire-and-forget remote agent support.

Replaces the ``RemoteEpisodeCollector`` when agents run inside Amazon Bedrock
AgentCore Runtime (ACR).  Instead of holding a persistent HTTP connection per
task, it uses a **fire-and-forget** pattern:

1. **Fire** -- Invoke the ACR agent via ``boto3``'s
   ``invoke_agent_runtime`` API.  The ``@rollout_entrypoint`` decorator on
   the agent returns ``{"status": "processing"}`` immediately.
2. **Collect** -- Poll an SQS queue for completion notifications, download
   rollout data from S3, and convert each ACR rollout into an rLLM
   ``Episode``.

The class exposes the same ``execute_tasks()`` interface as
``RemoteEpisodeCollector`` / ``UnifiedWorkflowEngine`` so that existing
backends (Tinker, Verl) can use it as a drop-in replacement.

ACR Invocation
--------------
Each task is invoked via the boto3 bedrock-agentcore client::

    client.invoke_agent_runtime(
        agentRuntimeArn="arn:aws:bedrock-agentcore:...",
        runtimeSessionId=<unique session id>,
        payload=json.dumps({
            "prompt":   str,
            "answer":   str,
            ...
            "_training": {
                "exp_id":      str,
                "session_id":  str,
                "input_id":    str,
                "sqs_url":     str,
                "s3_bucket":   str,
            }
        }).encode(),
        qualifier="DEFAULT",
    )

Results are collected from S3 objects whose keys arrive via SQS
notifications.  Each S3 object contains::

    {
        "rollout_data": [...],
        "rewards":      [...],
        "input_id":     str,
        "status_code":  int,
        "stop_reason":  str,
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)


@dataclass
class AgentCoreCollectorConfig:
    """Configuration for the AgentCoreEpisodeCollector."""

    agent_runtime_arn: str = ""
    inference_api_url: str = ""
    s3_bucket: str = ""
    sqs_url: str = ""
    exp_id: str = "default"
    timeout: float = 1800.0  # global timeout for collecting all episodes
    max_concurrent: int = 128  # max concurrent fire requests
    sqs_poll_interval: float = 2.0  # seconds between SQS polls
    sqs_batch_size: int = 10  # max messages per SQS receive
    retry_limit: int = 3  # retries for fire requests


class AgentCoreEpisodeCollector:
    """Collects episodes from ACR agents using fire-and-forget + S3/SQS.

    This class mirrors the ``execute_tasks`` interface of
    ``UnifiedWorkflowEngine`` and ``RemoteEpisodeCollector`` so it can be
    used as a drop-in replacement when AgentCore mode is enabled.

    Usage::

        collector = AgentCoreEpisodeCollector(config)
        episodes = await collector.execute_tasks(tasks, task_ids)
    """

    def __init__(self, config: AgentCoreCollectorConfig):
        self.config = config
        if not config.agent_runtime_arn:
            raise ValueError("agent_runtime_arn must be configured")
        if not config.s3_bucket:
            raise ValueError("s3_bucket must be configured")
        if not config.sqs_url:
            raise ValueError("sqs_url must be configured")

        # boto3 clients (lazy import so boto3 is optional at module level)
        import boto3
        from botocore.config import Config as BotoConfig

        # Extract region from the ARN so boto3 connects to the right
        # endpoint regardless of the user's default AWS region.
        # ARN format: arn:aws:bedrock-agentcore:<region>:<account>:runtime/...
        arn_parts = config.agent_runtime_arn.split(":")
        if len(arn_parts) >= 4 and arn_parts[3]:
            acr_region = arn_parts[3]
            logger.info(f"Using region '{acr_region}' from agent runtime ARN")
        else:
            acr_region = None
            logger.warning("Could not extract region from ARN, using default AWS region")

        # Size the connection pool to match max_concurrent so urllib3
        # doesn't discard connections under high parallelism.
        pool_size = max(config.max_concurrent, 128)
        acr_boto_config = BotoConfig(max_pool_connections=pool_size)

        self._acr = boto3.client(
            "bedrock-agentcore",
            region_name=acr_region,
            config=acr_boto_config,
        )
        self._s3 = boto3.client("s3", region_name=acr_region)
        self._sqs = boto3.client("sqs", region_name=acr_region)

        # Dedicated thread pool for boto3 calls so we can force-shutdown
        # without blocking the event loop on Ctrl+C.
        self._executor = ThreadPoolExecutor(
            max_workers=min(config.max_concurrent, 256),
            thread_name_prefix="acr-boto3",
        )
        self._shutting_down = False
        self._interrupt_count = 0

        # Mirror UnifiedWorkflowEngine attributes for compatibility
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"
        self.episode_logger = None

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0):
        """Set current training step (mirrors UnifiedWorkflowEngine API)."""
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    async def execute_tasks(
        self,
        tasks: list[dict],
        task_ids: list[str] | None = None,
        is_validation: bool = False,
        **kwargs,
    ) -> list[Episode]:
        """Execute tasks by firing them to ACR and collecting results via S3/SQS.

        Args:
            tasks: List of task dictionaries to process.
            task_ids: Optional list of task identifiers.  If *None*, UUIDs
                are generated.
            is_validation: Whether the generation is for validation.
            **kwargs: Extra arguments (ignored, kept for interface compat).

        Returns:
            Ordered list of Episode objects, one per input task.
        """
        # Install a signal handler that:
        #   - 1st Ctrl+C: sets _shutting_down flag, shuts down executor
        #   - 2nd Ctrl+C: force-kills the process with os._exit
        # We use signal.signal() (not asyncio signal handlers) because it
        # works reliably from the main thread and raises KeyboardInterrupt.
        self._interrupt_count = 0
        original_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):
            self._interrupt_count += 1
            self._shutting_down = True
            self._executor.shutdown(wait=False, cancel_futures=True)
            if self._interrupt_count >= 2:
                logger.warning("Second interrupt -- force exiting")
                os._exit(1)
            logger.warning("Ctrl+C received -- shutting down gracefully (press Ctrl+C again to force exit)")

        try:
            signal.signal(signal.SIGINT, _sigint_handler)
        except (ValueError, OSError):
            # Not on main thread -- can't install signal handler
            pass

        try:
            return await self._execute_tasks_inner(tasks, task_ids, is_validation)
        finally:
            try:
                signal.signal(signal.SIGINT, original_sigint)
            except (ValueError, OSError):
                pass

    async def _execute_tasks_inner(
        self,
        tasks: list[dict],
        task_ids: list[str] | None = None,
        is_validation: bool = False,
    ) -> list[Episode]:
        """Inner implementation of execute_tasks (separated for signal handling)."""
        self._shutting_down = False

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        # Compute per-task rollout indices (same as RemoteEpisodeCollector)
        task_id_counter: dict[str, int] = defaultdict(int)

        # Build the tracking structures.
        # Each fired request gets a unique ``session_id`` used as the key
        # to match SQS notifications back to the original task index.
        pending: dict[str, int] = {}  # session_id -> result index
        fire_payloads: list[tuple[int, dict, str, str, int]] = []

        for idx, (task, task_id) in enumerate(zip(tasks, task_ids, strict=True)):
            rollout_idx = task_id_counter[task_id]
            task_id_counter[task_id] += 1
            session_id = str(uuid.uuid4())

            fire_payloads.append((idx, task, task_id, session_id, rollout_idx))
            pending[session_id] = idx

        results: list[Episode | None] = [None] * len(tasks)

        # ---- Fire phase ----
        logger.info(f"Firing {len(fire_payloads)} tasks to ACR (arn={self.config.agent_runtime_arn})")
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        fire_failures: list[int] = []  # indices of permanently failed fires

        async def _fire(idx: int, task: dict, task_id: str, session_id: str, rollout_idx: int):
            if self._shutting_down:
                fire_failures.append(idx)
                return
            async with semaphore:
                ok = await self._fire_to_acr(
                    task=task,
                    task_id=task_id,
                    session_id=session_id,
                    rollout_idx=rollout_idx,
                )
                if not ok:
                    fire_failures.append(idx)

        fire_coros = [_fire(*p) for p in fire_payloads]
        with tqdm(total=len(fire_coros), desc="Firing ACR tasks") as pbar:
            for coro in asyncio.as_completed(fire_coros):
                if self._shutting_down:
                    break
                await coro
                pbar.update(1)

        # Immediately fill failed fire slots with error episodes so
        # the collect phase doesn't wait for results that will never arrive.
        for fail_idx in fire_failures:
            _, task, task_id, session_id, rollout_idx = fire_payloads[fail_idx]
            pending.pop(session_id, None)
            if results[fail_idx] is None:
                results[fail_idx] = self._make_error_episode(
                    task_id,
                    rollout_idx,
                    task,
                    RuntimeError("Fire to ACR permanently failed"),
                )

        fire_ok = len(fire_payloads) - len(fire_failures)
        expected = len(pending)  # only tasks still pending in SQS

        if expected == 0:
            logger.warning("All fires failed -- skipping collect phase")
        else:
            logger.info(f"Fired {fire_ok}/{len(fire_payloads)} tasks. Polling SQS for {expected} results...")

        # ---- Collect phase ----
        collected = 0
        deadline = time.monotonic() + self.config.timeout

        with tqdm(total=expected, desc="Collecting ACR episodes") as pbar:
            while collected < expected and time.monotonic() < deadline and not self._shutting_down:
                new_episodes = await self._poll_sqs(
                    pending=pending,
                    results=results,
                    tasks=tasks,
                    task_ids=task_ids,
                    fire_payloads=fire_payloads,
                )
                if new_episodes > 0:
                    collected += new_episodes
                    pbar.update(new_episodes)
                else:
                    # Sleep in small increments so the _shutting_down flag
                    # is checked frequently (signal handler sets it).
                    remaining = self.config.sqs_poll_interval
                    while remaining > 0 and not self._shutting_down:
                        nap = min(remaining, 0.25)
                        await asyncio.sleep(nap)
                        remaining -= nap

        # Fill any remaining slots with error episodes
        for idx, ep in enumerate(results):
            if ep is None:
                p = fire_payloads[idx]
                task_id = p[2]
                rollout_idx = p[4]
                reason = "interrupted" if self._shutting_down else "timeout"
                logger.error(f"[{task_id}:{rollout_idx}] {reason} waiting for ACR result")
                results[idx] = self._make_error_episode(task_id, rollout_idx, tasks[idx], TimeoutError(f"ACR result not received ({reason})"))

        ordered_results: list[Episode] = results  # type: ignore[assignment]

        # Log episodes if logger is provided
        if self.episode_logger is not None:
            try:
                logger.info(f"Logging {len(ordered_results)} episodes to step={self.current_step}, mode={self.current_mode}, epoch={self.current_epoch}")
                self.episode_logger.log_episodes_batch(
                    ordered_results,
                    self.current_step,
                    self.current_mode,
                    self.current_epoch,
                )
            except Exception as e:
                logger.error(f"Failed to log episodes: {e}")

        return ordered_results

    # ------------------------------------------------------------------
    # Fire phase (boto3 invoke_agent_runtime)
    # ------------------------------------------------------------------

    async def _fire_to_acr(
        self,
        task: dict,
        task_id: str,
        session_id: str,
        rollout_idx: int,
    ) -> bool:
        """Invoke an ACR agent session via boto3.  Does not wait for result.

        Uses ``invoke_agent_runtime`` with a unique ``runtimeSessionId``
        per task so each gets its own isolated session/container.

        Returns True if the fire succeeded, False if it permanently failed.
        """
        input_id = f"{task_id}:{rollout_idx}"

        payload: dict[str, Any] = {
            **task,
            "_training": {
                "exp_id": self.config.exp_id,
                "session_id": session_id,
                "input_id": input_id,
                "sqs_url": self.config.sqs_url,
                "s3_bucket": self.config.s3_bucket,
            },
        }

        # Also pass the inference API URL so the agent knows where to
        # call for model inference.
        if self.config.inference_api_url:
            payload["_inference_api_url"] = self.config.inference_api_url

        payload_bytes = json.dumps(payload).encode("utf-8")
        loop = asyncio.get_running_loop()

        def _invoke():
            resp = self._acr.invoke_agent_runtime(
                agentRuntimeArn=self.config.agent_runtime_arn,
                runtimeSessionId=session_id,
                payload=payload_bytes,
                qualifier="DEFAULT",
            )
            # Consume the streaming response to ensure the call completes
            # and resources are released.  For fire-and-forget agents the
            # immediate reply is typically {"status": "processing"}.
            chunks = []
            for chunk in resp.get("response", []):
                if isinstance(chunk, bytes):
                    chunks.append(chunk.decode("utf-8", errors="replace"))
                else:
                    chunks.append(str(chunk))
            body = "".join(chunks)
            return body

        last_error: Exception | None = None
        for attempt in range(1, self.config.retry_limit + 1):
            if self._shutting_down:
                return False
            try:
                body = await loop.run_in_executor(self._executor, _invoke)
                logger.debug(f"[{input_id}] Fired to ACR (session={session_id}), response: {body[:200]}")
                return True
            except Exception as e:
                last_error = e
                logger.warning(f"[{input_id}] Fire failed (attempt {attempt}/{self.config.retry_limit}): {e}")
                if attempt < self.config.retry_limit:
                    await asyncio.sleep(min(2**attempt, 10))

        logger.error(f"[{input_id}] Fire permanently failed after {self.config.retry_limit} attempts: {last_error}")
        return False

    # ------------------------------------------------------------------
    # Collect phase (SQS + S3)
    # ------------------------------------------------------------------

    async def _poll_sqs(
        self,
        pending: dict[str, int],
        results: list[Episode | None],
        tasks: list[dict],
        task_ids: list[str],
        fire_payloads: list[tuple[int, dict, str, str, int]],
    ) -> int:
        """Poll SQS once and process any received notifications.

        Returns the number of new episodes collected in this poll.
        """
        if self._shutting_down:
            return 0

        loop = asyncio.get_running_loop()

        try:
            # Use WaitTimeSeconds=0 (short polling) so the executor thread
            # returns immediately.  The caller handles the polling interval
            # via asyncio.sleep() which is interruptible by Ctrl+C.
            response = await loop.run_in_executor(
                self._executor,
                lambda: self._sqs.receive_message(
                    QueueUrl=self.config.sqs_url,
                    MaxNumberOfMessages=self.config.sqs_batch_size,
                    WaitTimeSeconds=0,
                ),
            )
        except Exception as e:
            if self._shutting_down:
                return 0
            logger.warning(f"SQS poll failed: {e}")
            return 0

        messages = response.get("Messages", [])
        if not messages:
            return 0

        collected = 0
        receipt_handles = []

        for msg in messages:
            receipt_handles.append(msg["ReceiptHandle"])
            try:
                body = json.loads(msg["Body"])
                records = body.get("Records", [])
                for record in records:
                    s3_info = record.get("s3", {})
                    bucket = s3_info.get("bucket", {}).get("name", "")
                    key = s3_info.get("object", {}).get("key", "")

                    if not key:
                        continue

                    episode = await self._download_and_convert(
                        bucket=bucket,
                        key=key,
                        pending=pending,
                        results=results,
                        tasks=tasks,
                        fire_payloads=fire_payloads,
                    )
                    if episode is not None:
                        collected += 1

            except Exception as e:
                logger.warning(f"Failed to process SQS message: {e}")

        # Delete processed messages in batch
        if receipt_handles:
            try:
                entries = [{"Id": str(i), "ReceiptHandle": rh} for i, rh in enumerate(receipt_handles)]
                await loop.run_in_executor(
                    self._executor,
                    lambda: self._sqs.delete_message_batch(
                        QueueUrl=self.config.sqs_url,
                        Entries=entries,
                    ),
                )
            except Exception as e:
                logger.warning(f"Failed to delete SQS messages: {e}")

        return collected

    async def _download_and_convert(
        self,
        bucket: str,
        key: str,
        pending: dict[str, int],
        results: list[Episode | None],
        tasks: list[dict],
        fire_payloads: list[tuple[int, dict, str, str, int]],
    ) -> Episode | None:
        """Download an S3 object and convert it to an Episode.

        Returns the Episode if successful, None otherwise.
        """
        loop = asyncio.get_running_loop()

        try:
            obj = await loop.run_in_executor(
                self._executor,
                lambda: self._s3.get_object(Bucket=bucket, Key=key),
            )
            raw = json.loads(obj["Body"].read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"Failed to download S3 object s3://{bucket}/{key}: {e}")
            return None

        # Extract session_id from the key pattern: {exp_id}/{input_id}_{session_id}.json
        try:
            filename = key.rsplit("/", 1)[-1]  # e.g. "task1:0_abc123.json"
            stem = filename.rsplit(".", 1)[0]  # e.g. "task1:0_abc123"
            session_id = stem.rsplit("_", 1)[-1]  # e.g. "abc123"
        except Exception:
            logger.warning(f"Could not parse session_id from key: {key}")
            return None

        idx = pending.pop(session_id, None)
        if idx is None:
            logger.debug(f"Received notification for unknown session_id={session_id}, ignoring")
            return None

        # Look up the original task info
        _, task, task_id, _, rollout_idx = fire_payloads[idx]

        # Convert ACR rollout data -> rLLM Episode
        episode = self._convert_acr_rollout_to_episode(
            raw=raw,
            task_id=task_id,
            rollout_idx=rollout_idx,
            task=task,
        )
        results[idx] = episode
        return episode

    # ------------------------------------------------------------------
    # Data bridge: ACR rollout -> rLLM Episode
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_acr_rollout_to_episode(
        raw: dict,
        task_id: str,
        rollout_idx: int,
        task: dict,
    ) -> Episode:
        """Convert an ACR rollout JSON object to an rLLM Episode.

        The ACR rollout format is::

            {
                "rollout_data": [
                    {
                        "turn_id": int,
                        "formatted_request": {"messages": [...]},
                        "model_output": {           # present when capture_tokens=True
                            "prompt_ids": [...],
                            "completion_ids": [...],
                            "logprobs": [...],
                            "text": "...",
                            ...
                        }
                    },
                    ...
                ],
                "rewards": [float, ...],
                "status_code": int,
                "stop_reason": str,
                "input_id": str
            }
        """
        status_code = raw.get("status_code", 200)
        stop_reason = raw.get("stop_reason", "end_turn")

        # Error rollouts
        if status_code != 200:
            return Episode(
                id=f"{task_id}:{rollout_idx}",
                task=task,
                termination_reason=TerminationReason.ERROR,
                is_correct=False,
                trajectories=[],
                metrics={},
                info={"error": {"message": stop_reason}},
            )

        rollout_data = raw.get("rollout_data", [])
        rewards = raw.get("rewards", [])

        # Normalize rewards
        if not isinstance(rewards, list):
            rewards = [rewards]

        # Build Steps from each turn
        steps: list[Step] = []
        for turn in rollout_data:
            model_output_dict = turn.get("model_output")
            messages = turn.get("formatted_request", {}).get("messages", [])

            step_kwargs: dict[str, Any] = {
                "chat_completions": messages,
                "done": False,
            }

            # If token-level data is available, populate it directly
            if model_output_dict:
                step_kwargs["prompt_ids"] = model_output_dict.get("prompt_ids") or []
                step_kwargs["response_ids"] = model_output_dict.get("completion_ids") or []
                step_kwargs["logprobs"] = model_output_dict.get("logprobs") or []
                step_kwargs["model_response"] = model_output_dict.get("text") or model_output_dict.get("content") or ""

            steps.append(Step(**step_kwargs))

        # Mark the last step as done
        if steps:
            steps[-1].done = True

        # Assign per-step rewards if provided, otherwise outcome reward on last step
        if len(rewards) == len(steps):
            for step, r in zip(steps, rewards, strict=False):
                step.reward = float(r)
        elif len(rewards) == 1 and steps:
            steps[-1].reward = float(rewards[0])

        # Compute trajectory-level reward
        traj_reward = sum(s.reward for s in steps)

        trajectory = Trajectory(
            name="agent_0",
            task=task,
            steps=steps,
            reward=traj_reward,
        )

        episode = Episode(
            id=f"{task_id}:{rollout_idx}",
            task=task,
            termination_reason=None if stop_reason == "end_turn" else TerminationReason.ERROR,
            is_correct=traj_reward > 0,
            trajectories=[trajectory],
            metrics={"n_turns": len(steps)},
            info={},
        )
        return episode

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_error_episode(
        task_id: str,
        rollout_idx: int,
        task: dict,
        error: Exception | None,
    ) -> Episode:
        """Create a placeholder Episode for a failed or timed-out task."""
        return Episode(
            id=f"{task_id}:{rollout_idx}",
            task=task,
            termination_reason=TerminationReason.ERROR,
            is_correct=False,
            trajectories=[],
            metrics={},
            info={"error": {"message": str(error) if error else "unknown error"}},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Shut down the thread pool executor."""
        self._shutting_down = True
        self._executor.shutdown(wait=False, cancel_futures=True)

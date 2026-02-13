"""Client-side components for remote training.

* :class:`AgentTrainerClient` – user-facing entry-point that orchestrates a
  remote training run from a local desktop.
* :class:`RemoteAgentRunner` – internal rollout loop that calls the user's
  ``rollout_fn`` locally and submits trajectories to the remote server.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from typing import Any, Callable

import httpx
from tqdm import tqdm

from rllm.experimental.fully_async.protocol import TrajectoryGroup
from rllm.experimental.fully_async.remote.remote_client import RemoteRolloutClient

logger = logging.getLogger(__name__)


class RemoteAgentRunner:
    """Local rollout loop – replaces :class:`RolloutExecutor` for remote mode.

    Iterates through a local dataset, calls the user's ``rollout_fn``
    concurrently, and submits completed :class:`TrajectoryGroup` objects to the
    remote :class:`TrainingServer` via HTTP.

    This class is not intended to be used directly; instead, use
    :class:`AgentTrainerClient`.
    """

    def __init__(
        self,
        *,
        server_url: str,
        rollout_fn: Callable,
        client: RemoteRolloutClient,
        tokenizer,
        n: int = 1,
        max_concurrency: int = 128,
        required_samples: int = 64,
    ):
        self.server_url = server_url.rstrip("/")
        self.rollout_fn = rollout_fn
        self.client = client
        self.tokenizer = tokenizer
        self.n = n
        self.max_concurrency = max_concurrency
        self.required_samples = required_samples

        # State
        self.result_dict: dict[int, list] = defaultdict(list)
        self._trajectory_queue: asyncio.Queue | None = None
        self._sema: asyncio.Semaphore | None = None

        # HTTP client for trajectory submission
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Trajectory submission
    # ------------------------------------------------------------------

    async def _submit_trajectory_group(self, group: TrajectoryGroup) -> dict:
        """Submit a TrajectoryGroup to the remote server."""
        payload = group.to_dict()
        resp = await self._http.post(self.server_url + "/v1/trajectories", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def _drain_results(self):
        """Background task that drains the internal queue and submits to server."""
        while True:
            try:
                group = await self._trajectory_queue.get()
                result = await self._submit_trajectory_group(group)
                if not result.get("accepted", True):
                    logger.warning("Trajectory dropped by server (queue full)")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Error submitting trajectory: %s", exc)

    # ------------------------------------------------------------------
    # Single rollout
    # ------------------------------------------------------------------

    async def _generate_trajectory(self, idx: int, datum: dict):
        """Run rollout_fn for a single datum, collect n results."""
        result = None
        try:
            result = await self.rollout_fn(self.client, self.tokenizer, **datum)
        except Exception:
            import traceback
            logger.error("Trajectory %d failed:\n%s", idx, traceback.format_exc())
        finally:
            self.result_dict[idx].append(result)
            self._sema.release()
            if len(self.result_dict[idx]) >= self.n:
                group = TrajectoryGroup(
                    trajectories=[r for r in self.result_dict[idx] if r is not None]
                )
                await self._trajectory_queue.put(group)
                del self.result_dict[idx]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, dataset, total_rollout_steps: int, total_epochs: int = 1):
        """Run the rollout loop over *dataset* for *total_epochs* epochs."""
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x,
        )

        self._trajectory_queue = asyncio.Queue()
        self._sema = asyncio.Semaphore(self.max_concurrency)

        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(None),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=256),
        )

        drain_task = asyncio.create_task(self._drain_results())
        global_steps = 0
        progress = tqdm(total=total_rollout_steps, desc="Rollout")

        try:
            for epoch in range(total_epochs):
                logger.info("Starting epoch %d", epoch)
                for batch in dataloader:
                    datum = batch[0]  # batch_size=1
                    for i in range(self.n):
                        await self._sema.acquire()
                        if i == 0:
                            global_steps += 1
                            progress.update(1)
                        asyncio.create_task(self._generate_trajectory(global_steps, datum))
                    if global_steps >= total_rollout_steps:
                        break
                if global_steps >= total_rollout_steps:
                    break
        finally:
            # Wait for in-flight tasks to complete
            while self.result_dict:
                await asyncio.sleep(0.5)
            # Wait for drain queue to flush
            while not self._trajectory_queue.empty():
                await asyncio.sleep(0.5)
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
            progress.close()
            await self._http.aclose()

        logger.info("Rollout loop completed – %d steps", global_steps)


# ======================================================================
# User-facing entry-point
# ======================================================================


class AgentTrainerClient:
    """Orchestrates a remote training run from a local desktop.

    This is the primary user-facing class for the remote training workflow.
    It connects to a :class:`TrainingServer` running on a GPU cluster,
    optionally sends config overrides, then runs the user's ``rollout_fn``
    locally and streams trajectories to the server.

    Parameters
    ----------
    server_url:
        Base URL of the remote TrainingServer.
    rollout_fn:
        Async function ``async def rollout_fn(client, tokenizer, **kwargs) -> Trajectory``.
    model_name:
        HuggingFace model name/path for loading the tokenizer locally.
    dataset:
        A PyTorch-compatible dataset (or any iterable of dicts).
    n:
        Number of rollouts per prompt (GRPO group size).
    max_concurrency:
        Maximum concurrent rollout tasks.
    config_overrides:
        Dict of dot-notation config keys to override on the server.
    val_rollout_fn:
        Optional validation rollout function (not used in remote mode MVP).
    """

    def __init__(
        self,
        server_url: str,
        rollout_fn: Callable,
        model_name: str,
        dataset,
        n: int = 1,
        max_concurrency: int = 128,
        config_overrides: dict[str, Any] | None = None,
        val_rollout_fn: Callable | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.rollout_fn = rollout_fn
        self.model_name = model_name
        self.dataset = dataset
        self.n = n
        self.max_concurrency = max_concurrency
        self.config_overrides = config_overrides or {}
        self.val_rollout_fn = val_rollout_fn

        # Validate rollout_fn
        if not inspect.iscoroutinefunction(rollout_fn):
            raise TypeError(
                f"rollout_fn must be async (defined with 'async def'), "
                f"got {type(rollout_fn).__name__}"
            )

        # Set after connection
        self._tokenizer = None
        self._server_config: dict | None = None
        self._total_train_steps: int | None = None
        self._required_samples: int | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )

    def _connect_and_configure(
        self,
        max_retries: int = 60,
        retry_interval: float = 10.0,
    ):
        """Send config overrides to the server and retrieve its config.

        Retries with exponential back-off if the server is not yet reachable
        (e.g. still starting up on a remote cluster).

        Parameters
        ----------
        max_retries:
            Maximum number of connection attempts before giving up.
        retry_interval:
            Base wait time (seconds) between retries.
        """
        print(f"[AgentTrainerClient] Connecting to {self.server_url} ...")

        # /v1/configure triggers GPU component init (SGLang, model loading, etc.)
        # which can take 10-30+ minutes for large models. Use a very generous
        # timeout and retry on both connection failures and read timeouts.
        configure_timeout = httpx.Timeout(
            connect=30.0,   # 30s to establish TCP connection
            read=1800.0,    # 30min to wait for response (GPU init is slow)
            write=30.0,
            pool=30.0,
        )
        with httpx.Client(timeout=configure_timeout) as http:
            # 1. Send config overrides (with retry for connection / timeout errors)
            for attempt in range(1, max_retries + 1):
                try:
                    resp = http.post(
                        self.server_url + "/v1/configure",
                        json={"config_overrides": self.config_overrides},
                    )
                    resp.raise_for_status()
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    if attempt == max_retries:
                        raise ConnectionError(
                            f"Could not connect to server at {self.server_url} "
                            f"after {max_retries} attempts. Is the server running?"
                        ) from exc
                    wait = min(retry_interval * (1.2 ** (attempt - 1)), 60)
                    print(
                        f"[AgentTrainerClient] Server not reachable (attempt {attempt}/{max_retries}): {exc}. "
                        f"Retrying in {wait:.0f}s ..."
                    )
                    time.sleep(wait)
                except httpx.ReadTimeout as exc:
                    if attempt == max_retries:
                        raise TimeoutError(
                            f"Server at {self.server_url} accepted the request but "
                            f"did not respond in time (GPU init may be very slow). "
                            f"Try again or check server logs."
                        ) from exc
                    print(
                        f"[AgentTrainerClient] Server timed out during init (attempt {attempt}/{max_retries}). "
                        f"GPU component initialization can take 10-30 min. Retrying ..."
                    )
                    # No sleep needed – the timeout itself already waited a long time
                except httpx.HTTPStatusError as exc:
                    # Server is reachable but returned an error – don't retry
                    raise RuntimeError(
                        f"Server returned {exc.response.status_code} on /v1/configure: "
                        f"{exc.response.text}"
                    ) from exc

            configure_result = resp.json()
            self._total_train_steps = configure_result.get("total_train_steps")
            print(f"[AgentTrainerClient] Server configured – total_train_steps={self._total_train_steps}")

            # 2. Read server config
            resp = http.get(self.server_url + "/v1/config")
            resp.raise_for_status()
            self._server_config = resp.json()
            self._required_samples = self._server_config.get("required_samples", 64)
            print(f"[AgentTrainerClient] Server config: {self._server_config}")

            # 3. Start training on the server
            resp = http.post(self.server_url + "/v1/start")
            resp.raise_for_status()
            print("[AgentTrainerClient] Training started on server")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        """Run the full remote training workflow (blocking).

        1. Load tokenizer locally.
        2. Connect to server & send config overrides.
        3. Start the local rollout loop.
        4. Wait for server to finish training.
        """
        start_time = time.time()

        # Step 1: load tokenizer
        print("[AgentTrainerClient] Loading tokenizer ...")
        self._load_tokenizer()

        # Step 2: connect & configure server
        self._connect_and_configure()

        # Step 3: create rollout client & runner
        max_tokens = 0
        if self._server_config:
            max_prompt = self._server_config.get("max_prompt_length") or 0
            max_resp = self._server_config.get("max_response_length") or 0
            max_tokens = max_prompt + max_resp
        if max_tokens <= 0:
            max_tokens = 32768

        client = RemoteRolloutClient(
            server_url=self.server_url,
            tokenizer=self._tokenizer,
            max_concurrency=self.max_concurrency,
            max_tokens=max_tokens,
        )

        runner = RemoteAgentRunner(
            server_url=self.server_url,
            rollout_fn=self.rollout_fn,
            client=client,
            tokenizer=self._tokenizer,
            n=self.n,
            max_concurrency=self.max_concurrency,
            required_samples=self._required_samples or 64,
        )

        # Step 4: compute total_rollout_steps from dataset
        dataset_size = len(self.dataset)
        total_epochs = 1  # default; may be overridden via config
        if self._server_config and "total_train_steps" in self._server_config:
            total_steps = self._server_config["total_train_steps"]
            total_rollout_steps = total_steps * (self._required_samples or 64)
        else:
            total_rollout_steps = dataset_size * total_epochs

        print(
            f"[AgentTrainerClient] Starting rollout loop: "
            f"dataset_size={dataset_size}, "
            f"total_rollout_steps={total_rollout_steps}, "
            f"n={self.n}, "
            f"max_concurrency={self.max_concurrency}"
        )

        # Run the async rollout loop
        asyncio.run(
            runner.run(
                dataset=self.dataset,
                total_rollout_steps=total_rollout_steps,
                total_epochs=max(1, total_rollout_steps // max(dataset_size, 1) + 1),
            )
        )

        # Step 5: wait for training to complete
        self._wait_for_training_complete()

        elapsed = time.time() - start_time
        print(f"[AgentTrainerClient] Training complete – total time: {elapsed:.1f}s")

    def _wait_for_training_complete(self, poll_interval: float = 5.0):
        """Poll the server until training is complete."""
        print("[AgentTrainerClient] Waiting for server training to complete ...")
        with httpx.Client(timeout=httpx.Timeout(30)) as http:
            while True:
                try:
                    resp = http.get(self.server_url + "/v1/status")
                    resp.raise_for_status()
                    status = resp.json()
                    if status.get("training_complete", False):
                        print("[AgentTrainerClient] Server reports training complete")
                        return
                except httpx.HTTPError as exc:
                    logger.warning("Status check failed: %s", exc)
                time.sleep(poll_interval)

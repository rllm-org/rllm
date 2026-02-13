"""TrainingServer -- FastAPI gateway that exposes rLLM's GPU-side training
infrastructure over HTTP so that agent code can run on a separate machine.

Usage::

    from rllm.experimental.fully_async.remote import TrainingServer

    server = TrainingServer(config_path="config.yaml")
    server.run(host="0.0.0.0", port=8000)

Endpoints
---------
POST /v1/configure   – Accept client config overrides before training starts.
POST /v1/generate    – Proxy token-level generation to SGLang router.
POST /v1/trajectories – Submit a completed TrajectoryGroup (JSON).
GET  /v1/status      – Training status (version, syncing, queue, step, …).
GET  /v1/config      – Public subset of the resolved training config.
POST /v1/start       – Start training (called after /v1/configure).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from pprint import pprint
from typing import Any

import httpx
import ray
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from omegaconf import DictConfig, OmegaConf

from rllm.experimental.fully_async.protocol import TrajectoryGroup

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0, max_concurrency=10)
class _SyncFlag:
    """Tiny Ray actor used to share the ``is_syncing`` flag between the
    ParameterSynchronizer (which runs in its own actor process) and the
    FastAPI server process.
    """

    def __init__(self):
        self._value = False

    def set(self, value: bool):
        self._value = value

    def get(self) -> bool:
        return self._value

# Keys that cannot be overridden from the client (hardware-bound).
_PROTECTED_CONFIG_KEYS = frozenset(
    {
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
        "rollout.n_gpus_per_node",
        "rollout.nnodes",
        "actor_rollout_ref.model.path",
    }
)


class TrainingServer:
    """Self-contained training gateway that can be deployed on a GPU cluster.

    The server initialises all GPU-side components (SGLang inference servers,
    PPO trainer, parameter synchroniser, message queue) and exposes them via a
    FastAPI HTTP API.  Agent code on a remote machine (e.g. a local desktop)
    connects to this server using :class:`AgentTrainerClient`.

    Parameters
    ----------
    config : DictConfig | None
        OmegaConf config object.  Provide either *config* or *config_path*.
    config_path : str | None
        Path to a YAML config file.
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        config_path: str | None = None,
    ):
        if config is None and config_path is None:
            raise ValueError("Provide either config or config_path")

        if config is not None:
            self._base_config = config
        else:
            self._base_config = self._load_config(config_path)

        # Resolved config (after client overrides).  ``None`` until
        # /v1/configure or /v1/start is called.
        self._config: DictConfig | None = None

        # --- component handles (initialised in _init_components) ---
        self._tokenizer = None
        self._processor = None
        self._inference_manager = None
        self._router_url: str | None = None
        self._trainer = None
        self._message_queue = None
        self._mq_client = None
        self._param_synchronizer = None

        # --- state ---
        self._sync_flag = None  # Ray actor holding the is_syncing flag
        self._training_started = False
        self._training_complete = False
        self._trainer_future = None
        self._training_step: int = 0
        self._param_version: int = 0

        # Shared async HTTP client for proxying to SGLang
        self._proxy_client: httpx.AsyncClient | None = None

        # Total train steps (set after MQ init)
        self._total_train_steps: int | None = None

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> DictConfig:
        """Load a config file, resolving Hydra defaults if present.

        If the YAML contains ``defaults:`` (Hydra composition), we use
        Hydra's ``compose`` API to fully resolve it.  Otherwise we fall
        back to plain ``OmegaConf.load``.

        .. note::

           For configs that rely on Hydra defaults (like
           ``fully_async_ppo_trainer.yaml`` which inherits from
           ``ppo_trainer.yaml``), it is **recommended** to use
           ``@hydra.main`` in your entry-point script and pass the
           already-resolved ``DictConfig`` via the ``config=`` parameter
           instead.  See ``examples/fully_async/remote_training/server.py``
           for an example.
        """
        raw = OmegaConf.load(config_path)

        # Quick check: does this config use Hydra defaults?
        if OmegaConf.select(raw, "defaults") is not None:
            try:
                return TrainingServer._compose_hydra_config(config_path)
            except Exception as exc:
                logger.warning(
                    "Hydra compose failed for %s (%s). "
                    "Falling back to raw OmegaConf.load – some keys may be missing. "
                    "Consider using @hydra.main in your server script instead.",
                    config_path,
                    exc,
                )

        return raw

    @staticmethod
    def _compose_hydra_config(config_path: str) -> DictConfig:
        """Use Hydra's compose API to resolve a config with defaults."""
        import os

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

        abs_path = os.path.abspath(config_path)
        config_dir = os.path.dirname(abs_path)
        config_name = os.path.splitext(os.path.basename(abs_path))[0]

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)

        return cfg

    # ------------------------------------------------------------------
    # Config override / merge
    # ------------------------------------------------------------------

    def _apply_config_overrides(self, overrides: dict[str, Any]) -> DictConfig:
        """Deep-merge *overrides* into the base config, returning a new copy."""
        cfg = OmegaConf.create(OmegaConf.to_container(self._base_config, resolve=False))

        for dotkey, value in overrides.items():
            if dotkey in _PROTECTED_CONFIG_KEYS:
                raise ValueError(f"Config key '{dotkey}' is protected and cannot be overridden remotely")
            OmegaConf.update(cfg, dotkey, value)

        OmegaConf.resolve(cfg)
        return cfg

    # ------------------------------------------------------------------
    # Component initialisation (mirrors runner.py helpers)
    # ------------------------------------------------------------------

    def _init_components(self):
        """Initialise all GPU-side components using the resolved config."""
        from verl.experimental.fully_async_policy.fully_async_main import (
            create_role_worker_mapping,
        )

        from rllm.experimental.fully_async.runner import (
            init_inference_manager,
            init_message_queue,
            init_param_synchronizer,
            init_trainer,
            load_checkpoint_and_sync,
            load_tokenizer_and_processor,
        )

        config = self._config
        print(f"[TrainingServer] hostname={socket.gethostname()}, PID={os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))

        # Tokenizer / processor
        self._tokenizer, self._processor = load_tokenizer_and_processor(config)

        # Worker mapping
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

        # Inference manager + router
        self._inference_manager, self._router_url = init_inference_manager(
            config,
            self._tokenizer,
            role_worker_mapping,
            ray_worker_group_cls,
            self._processor,
        )

        # Calculate queue/step counts.
        # Without a local RolloutExecutor we derive them from config.
        required_samples = config.async_training.required_samples
        trigger_sync_step = config.async_training.trigger_parameter_sync_step
        staleness_threshold = config.async_training.get("staleness_threshold", 1)
        total_rollout_steps = config.rollout.total_rollout_steps

        max_queue_size = int(required_samples * (staleness_threshold + 1) * trigger_sync_step)
        self._total_train_steps = int(total_rollout_steps / (required_samples * trigger_sync_step))

        # Message queue
        self._message_queue, self._mq_client = init_message_queue(config, max_queue_size)

        # Trainer
        self._trainer = init_trainer(
            config,
            self._tokenizer,
            role_worker_mapping,
            ray_worker_group_cls,
            self._processor,
        )
        ray.get(self._trainer.set_total_train_steps.remote(self._total_train_steps))
        ray.get(self._trainer.set_message_queue_client.remote(self._mq_client))

        # Parameter synchroniser (remote mode – no RolloutExecutor)
        self._param_synchronizer = init_param_synchronizer(
            config,
            self._trainer,
            self._inference_manager,
            self._mq_client,
        )
        ray.get(self._param_synchronizer.set_router_url.remote(self._router_url))

        # Create shared sync flag so the API gateway can check is_syncing
        self._sync_flag = _SyncFlag.remote()
        ray.get(
            self._param_synchronizer.set_remote_mode.remote(
                sync_flag=self._sync_flag,
            )
        )

        # Load checkpoint + initial weight sync
        self._param_version = load_checkpoint_and_sync(
            self._trainer, self._param_synchronizer, config,
        )

        print("[TrainingServer] All components initialised")

    # ------------------------------------------------------------------
    # Syncing flag (shared Ray actor)
    # ------------------------------------------------------------------

    async def _is_syncing(self) -> bool:
        """Check whether the server is currently syncing weights."""
        if self._sync_flag is None:
            return False
        return await asyncio.wrap_future(self._sync_flag.get.remote().future())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _start_training(self):
        """Launch ``FullyAsyncTrainer.fit()`` in the background."""
        if self._training_started:
            return
        self._training_started = True
        self._trainer_future = self._trainer.fit.remote()
        print("[TrainingServer] Training started in background")

        # Monitor in a daemon thread
        def _monitor():
            try:
                ray.get(self._trainer_future)
            except Exception as exc:
                logger.error("Training failed: %s", exc)
            finally:
                self._training_complete = True
                print("[TrainingServer] Training loop finished")

        t = threading.Thread(target=_monitor, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self._proxy_client = httpx.AsyncClient(
                timeout=httpx.Timeout(None),
                limits=httpx.Limits(max_connections=4096, max_keepalive_connections=1000),
            )
            yield
            # Shutdown
            if self._proxy_client:
                await self._proxy_client.aclose()

        app = FastAPI(title="rLLM Training Server", lifespan=lifespan)

        # ---- POST /v1/configure ----

        @app.post("/v1/configure")
        async def configure(request: Request):
            """Accept config overrides from the client, merge, init components."""
            body = await request.json()
            overrides = body.get("config_overrides", {})
            try:
                self._config = self._apply_config_overrides(overrides)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            # Initialise components (blocking – happens once)
            self._init_components()

            return {
                "status": "configured",
                "total_train_steps": self._total_train_steps,
                "param_version": self._param_version,
                "router_url": self._router_url,
            }

        # ---- POST /v1/start ----

        @app.post("/v1/start")
        async def start_training():
            """Start the training loop (call after /v1/configure)."""
            if self._config is None:
                # No overrides – use base config as-is
                self._config = OmegaConf.create(
                    OmegaConf.to_container(self._base_config, resolve=True)
                )
                self._init_components()
            self._start_training()
            return {"status": "training_started"}

        # ---- POST /v1/generate ----

        @app.post("/v1/generate")
        async def generate(request: Request, response: Response):
            """Proxy a generation request to the SGLang router."""
            if await self._is_syncing():
                raise HTTPException(
                    status_code=503,
                    detail="Server is syncing weights – retry later",
                )
            if self._router_url is None:
                raise HTTPException(status_code=503, detail="Router not ready")

            payload = await request.json()
            try:
                proxy_resp = await self._proxy_client.post(
                    self._router_url + "/generate", json=payload,
                )
                proxy_resp.raise_for_status()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError) as exc:
                # SGLang may be temporarily unavailable during weight sync
                raise HTTPException(
                    status_code=503,
                    detail=f"Generation backend unavailable: {exc}",
                )
            data = proxy_resp.json()

            # Attach current param version so the client can track staleness
            response.headers["X-Param-Version"] = str(self._param_version)
            return data

        # ---- POST /v1/trajectories ----

        @app.post("/v1/trajectories")
        async def submit_trajectories(request: Request):
            """Accept a JSON-serialised TrajectoryGroup and enqueue it."""
            body = await request.json()
            try:
                group = TrajectoryGroup.from_dict(body)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid trajectory data: {exc}")

            # Serialise with cloudpickle for the internal MessageQueue
            # (keeps downstream FullyAsyncTrainer completely unchanged).
            serialized = ray.cloudpickle.dumps(group)
            put_ok = await asyncio.wrap_future(
                self._message_queue.put_sample.remote(serialized).future()
            )
            stats = await asyncio.wrap_future(
                self._message_queue.get_statistics.remote().future()
            )
            return {
                "accepted": put_ok,
                "queue_size": stats.get("queue_size", 0),
                "total_produced": stats.get("total_produced", 0),
            }

        # ---- GET /v1/status ----

        @app.get("/v1/status")
        async def status():
            info: dict[str, Any] = {
                "param_version": self._param_version,
                "is_syncing": await self._is_syncing(),
                "training_started": self._training_started,
                "training_complete": self._training_complete,
            }
            if self._message_queue is not None:
                stats = await asyncio.wrap_future(
                    self._message_queue.get_statistics.remote().future()
                )
                info["queue_size"] = stats.get("queue_size", 0)
                info["total_produced"] = stats.get("total_produced", 0)
                info["total_consumed"] = stats.get("total_consumed", 0)
            if self._total_train_steps is not None:
                info["total_train_steps"] = self._total_train_steps
            return info

        # ---- GET /v1/config ----

        @app.get("/v1/config")
        async def get_config():
            """Return a public subset of the resolved config."""
            if self._config is None:
                raise HTTPException(status_code=503, detail="Not configured yet")
            cfg = OmegaConf.to_container(self._config, resolve=True)
            return {
                "model_path": cfg.get("actor_rollout_ref", {}).get("model", {}).get("path"),
                "max_prompt_length": cfg.get("data", {}).get("max_prompt_length"),
                "max_response_length": cfg.get("data", {}).get("max_response_length"),
                "n": cfg.get("actor_rollout_ref", {}).get("rollout", {}).get("n", 1),
                "required_samples": cfg.get("async_training", {}).get("required_samples"),
                "total_train_steps": self._total_train_steps,
            }

        return app

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, host: str = "0.0.0.0", port: int = 8000, **uvicorn_kwargs):
        """Start the FastAPI server (blocking)."""
        app = self._build_app()
        uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)

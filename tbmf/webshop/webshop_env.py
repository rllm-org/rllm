"""WebShop environment wrapper for rllm framework.

WebShop is a simulated e-commerce website environment where agents
must navigate to find and purchase products matching user instructions.
Uses Ray remote actors for process-level isolation to ensure thread safety.
Workers are pooled and reused to avoid expensive re-initialization.
Requests are queued when all workers are busy, ensuring no request fails.
"""

import asyncio
import logging
import os
os.environ['TQDM_DISABLE'] = '1'  # Set before importing WebShop environment
import re
import sys
import threading
import uuid
from typing import ClassVar

import ray

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger(__name__)


def _ensure_webshop_pkg_on_path() -> None:
    """Make the bundled WebShop package importable inside Ray workers."""
    pkg_path = os.path.join(os.path.dirname(__file__), "webshop_pkg")
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)


def _make_webshop_text_env(**env_kwargs):
    _ensure_webshop_pkg_on_path()
    from web_agent_site.envs import WebAgentTextEnv

    return WebAgentTextEnv(**env_kwargs)


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    return value


def _read_positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.1f", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r; using %.1f", name, raw, default)
        return default
    return value


@ray.remote
class WebShopWorker:
    """Ray remote actor that holds one WebShop environment instance.
    
    Each worker runs in its own process, providing process-level isolation
    to ensure thread safety for the async execution engine.
    """
    
    def __init__(
        self,
        observation_mode: str = "text",
        num_products: int | None = None,
        seed: int | None = None,
        file_path: str | None = None,
        attr_path: str | None = None,
        human_goals: bool = False,
        **kwargs,
    ):
        """Initialize the worker with WebShop environment.
        
        Args:
            observation_mode: Observation format ("text", "html", "text_rich").
            num_products: Number of products (None for all).
            seed: Random seed for reproducibility.
            file_path: Path to product data file.
            attr_path: Path to attribute data file.
            human_goals: Whether to use human-written goals.
            **kwargs: Additional arguments for the underlying environment.
        """
        # Build environment kwargs
        env_kwargs = {
            "observation_mode": observation_mode,
            "seed": seed or 42,
            "num_products": num_products,
            "human_goals": 1 if human_goals else 0,
        }
        
        if file_path:
            env_kwargs["file_path"] = file_path
        if attr_path:
            env_kwargs["attr_path"] = attr_path
        
        # Merge additional kwargs
        env_kwargs.update(kwargs)

        self._base_env_kwargs = env_kwargs
        self._seed = env_kwargs.get("seed", 42)

        # Create environment directly
        self._env = _make_webshop_text_env(**env_kwargs)
        self._available_actions = {}
        self._instruction = None
        
        logger.info(
            f"WebShopWorker initialized with observation_mode={observation_mode}"
        )
    
    def _rebuild_env(self, seed: int | None = None) -> None:
        """Rebuild the underlying environment (used when per-episode seed changes)."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing WebShop environment before rebuild: {e}")
        env_kwargs = dict(self._base_env_kwargs)
        if seed is not None:
            env_kwargs["seed"] = seed
        self._seed = env_kwargs.get("seed", 42)
        self._env = _make_webshop_text_env(**env_kwargs)

    def reset(self, session_id: int | None = None, seed: int | None = None) -> tuple[str, dict]:
        """Reset the environment to start a new episode."""
        if seed is not None and seed != self._seed:
            self._rebuild_env(seed)
        if session_id is not None:
            obs, _ = self._env.reset(session=session_id)
        else:
            obs, _ = self._env.reset()
        
        # Get available actions
        self._available_actions = self._env.get_available_actions()
        
        # Get instruction
        self._instruction = self._env.get_instruction_text()
        
        info = {
            "available_actions": self._available_actions,
            "instruction": self._instruction,
            "has_search_bar": self._available_actions.get("has_search_bar", False),
            "clickables": self._available_actions.get("clickables", []),
        }
        
        return obs, info
    
    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute an action in the environment."""
        obs, reward, done, env_info = self._env.step(action)
        
        # Get updated available actions
        self._available_actions = self._env.get_available_actions()
        
        info = {
            "available_actions": self._available_actions,
            "instruction": self._instruction,
            "has_search_bar": self._available_actions.get("has_search_bar", False),
            "clickables": self._available_actions.get("clickables", []),
            "task_score": reward,
        }
        
        return obs, reward, done, info
    
    def get_available_actions(self) -> dict:
        """Get available actions for the current state."""
        return self._available_actions
    
    def get_instruction(self) -> str:
        """Get the current task instruction."""
        return self._instruction or ""
    
    def close(self):
        """Clean up environment resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing WebShop environment in worker: {e}")
            self._env = None


@ray.remote
class WebShopWorkerPoolActor:
    """A Ray actor that manages a pool of WebShop workers with request queuing.
    
    This actor provides centralized coordination of workers across all processes,
    ensuring that workers are properly reused and the total number is limited.
    When all workers are busy, requests are queued and processed in FIFO order
    as workers become available.
    
    IMPORTANT: This actor maintains strong references to ALL workers it creates
    (both in-use and available) to prevent Ray from garbage collecting them.
    """
    
    def __init__(self, max_workers: int = 64):
        """Initialize the worker pool.
        
        Args:
            max_workers: Maximum number of workers to create per configuration.
        """
        self._max_workers = max_workers
        self._pools: dict[str, list] = {}  # key -> list of available workers
        self._worker_counts: dict[str, int] = {}  # track total workers created per config
        # Queue for pending requests: key -> list of (request_id, asyncio.Event, result_holder)
        self._pending_requests: dict[str, list] = {}
        self._pending_count: dict[str, int] = {}  # track pending requests per config
        # CRITICAL: Registry of ALL workers to maintain strong references and prevent GC
        self._all_workers: dict[str, list] = {}  # key -> list of all workers (in-use + available)
        self._lock = threading.Lock()
        logger.info(f"WebShopWorkerPoolActor initialized with max_workers={max_workers}")
    
    def _get_pool_key(
        self,
        observation_mode: str,
        num_products: int | None,
        human_goals: bool,
        file_path: str | None,
        attr_path: str | None,
        kwargs_fingerprint: str,
    ) -> str:
        """Generate a unique key for the worker pool based on configuration."""
        return f"{observation_mode}|{num_products}|{human_goals}|{file_path}|{attr_path}|{kwargs_fingerprint}"
    
    def _init_pool_structures(self, pool_key: str):
        """Initialize pool data structures for a given config key."""
        if pool_key not in self._pools:
            self._pools[pool_key] = []
            self._worker_counts[pool_key] = 0
            self._pending_requests[pool_key] = []
            self._pending_count[pool_key] = 0
            self._all_workers[pool_key] = []  # Registry for ALL workers
    
    def _create_worker(
        self,
        pool_key: str,
        observation_mode: str,
        num_products: int | None,
        seed: int | None,
        file_path: str | None,
        attr_path: str | None,
        human_goals: bool,
        kwargs: dict,
    ):
        """Create a new worker and track it.
        
        The worker is registered in _all_workers to maintain a strong reference
        and prevent Ray from garbage collecting it while in use.
        """
        worker = WebShopWorker.remote(
            observation_mode=observation_mode,
            num_products=num_products,
            seed=seed,
            file_path=file_path,
            attr_path=attr_path,
            human_goals=human_goals,
            **kwargs,
        )
        
        # CRITICAL: Register in _all_workers to maintain strong reference
        self._all_workers[pool_key].append(worker)
        
        self._worker_counts[pool_key] += 1
        count = self._worker_counts[pool_key]
        logger.info(f"Created WebShopWorker {count}/{self._max_workers} for config: {pool_key[:50]}...")
        
        return worker
    
    async def acquire_worker(
        self,
        observation_mode: str,
        num_products: int | None,
        seed: int | None,
        file_path: str | None,
        attr_path: str | None,
        human_goals: bool,
        kwargs: dict,
        timeout: float | None = 600.0,
    ):
        """Acquire a worker from the pool, queuing if necessary.
        
        This method will:
        1. Return an available worker immediately if one exists
        2. Create a new worker if under the limit
        3. Queue the request and wait if at max capacity
        
        Args:
            timeout: Maximum time to wait for a worker (default 10 minutes).
        
        Returns:
            A Ray actor reference to a WebShopWorker.
            
        Raises:
            TimeoutError: If no worker becomes available within the timeout.
        """
        kwargs_fingerprint = WebShopWorkerPool._fingerprint_kwargs(kwargs)
        pool_key = self._get_pool_key(observation_mode, num_products, human_goals, file_path, attr_path, kwargs_fingerprint)

        with self._lock:
            self._init_pool_structures(pool_key)

            # Fast path: worker available in pool
            if self._pools[pool_key]:
                worker = self._pools[pool_key].pop()
                return worker

            # Can we create a new worker?
            if self._worker_counts[pool_key] < self._max_workers:
                return self._create_worker(
                    pool_key, observation_mode, num_products, seed,
                    file_path, attr_path, human_goals, kwargs
                )

            # At capacity - queue the request and wait
            request_id = str(uuid.uuid4())
            event = asyncio.Event()
            result_holder = {"worker": None}

            # Store the creation params for when we fulfill this request
            request_info = {
                "request_id": request_id,
                "event": event,
                "result": result_holder,
                "params": {
                    "observation_mode": observation_mode,
                    "num_products": num_products,
                    "seed": seed,
                    "file_path": file_path,
                    "attr_path": attr_path,
                    "human_goals": human_goals,
                    "kwargs": kwargs,
                }
            }

            self._pending_requests[pool_key].append(request_info)
            self._pending_count[pool_key] += 1
            queue_position = self._pending_count[pool_key]

            logger.debug(
                f"Request {request_id[:8]} queued at position {queue_position} "
                f"for config {pool_key[:30]}... (workers: {self._worker_counts[pool_key]}/{self._max_workers})"
            )
        
        try:
            # Wait for a worker to be assigned. A non-positive timeout means
            # wait indefinitely; useful for large validation waves where
            # bounded worker pools intentionally queue rollouts.
            if timeout is None or timeout <= 0:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            
            if result_holder["worker"] is None:
                raise RuntimeError(f"Request {request_id[:8]} was signaled but no worker assigned")
            
            return result_holder["worker"]
            
        except asyncio.TimeoutError:
            # Remove from pending queue if still there
            self._pending_requests[pool_key] = [
                r for r in self._pending_requests[pool_key] 
                if r["request_id"] != request_id
            ]
            raise TimeoutError(
                f"Timeout waiting for WebShop worker after {timeout}s. "
                f"Queue depth was {queue_position}, max_workers={self._max_workers}"
            )
    
    def release_worker(
        self,
        worker,
        observation_mode: str,
        num_products: int | None,
        human_goals: bool,
        file_path: str | None,
        attr_path: str | None,
        kwargs: dict,
    ):
        """Return a worker to the pool for reuse, fulfilling pending requests first."""
        kwargs_fingerprint = WebShopWorkerPool._fingerprint_kwargs(kwargs)
        pool_key = self._get_pool_key(observation_mode, num_products, human_goals, file_path, attr_path, kwargs_fingerprint)
        self._init_pool_structures(pool_key)

        with self._lock:
            # Check if there are pending requests waiting for a worker
            if self._pending_requests[pool_key]:
                # Fulfill the oldest request (FIFO)
                request_info = self._pending_requests[pool_key].pop(0)
                request_info["result"]["worker"] = worker
                request_info["event"].set()

                logger.debug(
                    f"Fulfilled pending request {request_info['request_id'][:8]} "
                    f"({len(self._pending_requests[pool_key])} still waiting)"
                )
            else:
                # No pending requests, return worker to pool
                self._pools[pool_key].append(worker)
    
    def get_stats(self) -> dict:
        """Get statistics about the worker pool."""
        return {
            "max_workers": self._max_workers,
            "worker_counts": dict(self._worker_counts),
            "available_workers": {k: len(v) for k, v in self._pools.items()},
            "pending_requests": {k: len(v) for k, v in self._pending_requests.items()},
            "total_workers_registered": {k: len(v) for k, v in self._all_workers.items()},
        }
    
    def shutdown(self):
        """Shutdown all workers in all pools."""
        # Cancel any pending requests
        for pool_key, requests in self._pending_requests.items():
            for request_info in requests:
                request_info["event"].set()  # Unblock waiters (they'll get None)
            self._pending_requests[pool_key] = []
        
        # Shutdown ALL workers (from the registry, not just available ones)
        for pool_key, workers in self._all_workers.items():
            for worker in workers:
                try:
                    ray.get(worker.close.remote())
                    ray.kill(worker)
                except Exception as e:
                    logger.warning(f"Error shutting down worker: {e}")
            self._all_workers[pool_key] = []
        
        # Clear other structures
        for pool_key in list(self._pools.keys()):
            self._pools[pool_key] = []
        self._worker_counts.clear()
        self._pending_count.clear()


class WebShopWorkerPool:
    """Client-side wrapper for accessing the Ray-based worker pool with queuing.
    
    This class provides a simple interface to acquire and release workers
    from the centralized Ray actor pool. When all workers are busy, requests
    are automatically queued and processed in FIFO order.
    """
    
    _pool_actor = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _pool_actor_name: ClassVar[str | None] = None
    _auto_run_id: ClassVar[str | None] = None
    
    # Default max workers - can be overridden via WEBSHOP_MAX_WORKERS or set_max_workers().
    _max_workers: ClassVar[int] = _read_positive_int_env("WEBSHOP_MAX_WORKERS", 64)
    
    @classmethod
    def set_max_workers(cls, max_workers: int):
        """Set the maximum number of workers before the pool is created.
        
        Args:
            max_workers: Maximum number of workers to create.
        """
        cls._max_workers = max_workers
        logger.info(f"WebShopWorkerPool max_workers set to {max_workers}")
    
    @classmethod
    def _get_pool_actor(cls):
        """Get or create the Ray actor that manages the worker pool."""
        if cls._pool_actor is None:
            with cls._lock:
                if cls._pool_actor is None:
                    pool_name = cls._get_pool_actor_name()
                    # Try to get existing named actor, or create a new one
                    try:
                        cls._pool_actor = ray.get_actor(pool_name)
                        cls._pool_actor_name = pool_name
                        logger.info(f"Connected to existing WebShopWorkerPoolActor: {pool_name}")
                    except ValueError:
                        # Actor doesn't exist, create it
                        cls._pool_actor = WebShopWorkerPoolActor.options(
                            name=pool_name,
                            max_concurrency=1000,  # Allow many concurrent calls
                        ).remote(max_workers=cls._max_workers)
                        cls._pool_actor_name = pool_name
                        logger.info(f"Created new WebShopWorkerPoolActor {pool_name} with max_workers={cls._max_workers}")
        return cls._pool_actor

    @classmethod
    def _get_pool_actor_name(cls) -> str:
        """Get a per-run unique actor name (prefers RLLM_RUN_ID)."""
        run_id = os.environ.get("RLLM_RUN_ID")
        if not run_id:
            if cls._auto_run_id is None:
                cls._auto_run_id = str(uuid.uuid4())
            run_id = cls._auto_run_id
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
        return f"webshop_worker_pool_{safe_run_id}"

    @classmethod
    def get_pool_name(cls) -> str:
        """Return the resolved pool actor name for this process."""
        return cls._pool_actor_name or cls._get_pool_actor_name()

    @staticmethod
    def _fingerprint_kwargs(kwargs: dict) -> str:
        if not kwargs:
            return "none"
        # Best-effort deterministic fingerprint
        try:
            import json
            payload = json.dumps(kwargs, sort_keys=True, default=str)
        except Exception:
            payload = repr(sorted(kwargs.items()))
        import hashlib
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    
    @classmethod
    def acquire_worker(
        cls,
        observation_mode: str,
        num_products: int | None,
        seed: int | None,
        file_path: str | None,
        attr_path: str | None,
        human_goals: bool,
        kwargs: dict,
        timeout: float | None = 600.0,
    ):
        """Acquire a worker from the pool, queuing if necessary.
        
        Requests are automatically queued when all workers are busy
        and processed in FIFO order as workers become available.
        
        Args:
            timeout: Maximum time to wait for a worker (default 10 minutes).
        
        Returns:
            A Ray actor reference to a WebShopWorker.
            
        Raises:
            TimeoutError: If no worker becomes available within the timeout.
            RuntimeError: If acquisition fails for other reasons.
        """
        pool_actor = cls._get_pool_actor()
        
        try:
            # The actor's acquire_worker is async and handles queuing internally
            worker = ray.get(pool_actor.acquire_worker.remote(
                observation_mode, num_products, seed, file_path, 
                attr_path, human_goals, kwargs, timeout
            ))
            return worker
        except ray.exceptions.RayTaskError as e:
            # Unwrap the actual exception
            if "TimeoutError" in str(e):
                raise TimeoutError(
                    f"Timeout waiting for WebShop worker. "
                    f"max_workers={cls._max_workers}. "
                    f"Consider increasing max_workers or reducing parallel environments."
                ) from e
            raise RuntimeError(f"Failed to acquire WebShop worker: {e}") from e
    
    @classmethod
    def release_worker(
        cls,
        worker,
        observation_mode: str,
        num_products: int | None,
        human_goals: bool,
        file_path: str | None,
        attr_path: str | None,
        kwargs: dict,
    ):
        """Return a worker to the pool for reuse.
        
        If there are pending requests, the worker will be immediately
        assigned to the oldest waiting request.
        """
        pool_actor = cls._get_pool_actor()
        ray.get(pool_actor.release_worker.remote(
            worker, observation_mode, num_products, human_goals, file_path, attr_path, kwargs
        ))
    
    @classmethod
    def get_stats(cls) -> dict:
        """Get statistics about the worker pool.
        
        Returns:
            Dictionary with:
                - max_workers: Maximum workers allowed
                - worker_counts: Workers created per config
                - available_workers: Workers idle per config
                - pending_requests: Requests waiting per config
        """
        pool_actor = cls._get_pool_actor()
        return ray.get(pool_actor.get_stats.remote())
    
    @classmethod
    def shutdown(cls):
        """Shutdown all workers and the pool actor."""
        if not ray.is_initialized():
            cls._pool_actor = None
            return
        pool_name = cls._pool_actor_name or cls._get_pool_actor_name()
        actor = cls._pool_actor
        if actor is None:
            try:
                actor = ray.get_actor(pool_name)
            except ValueError:
                actor = None
        if actor is not None:
            try:
                ray.get(actor.shutdown.remote())
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"Error shutting down pool actor: {e}")
        cls._pool_actor = None
        cls._pool_actor_name = None
        cls._auto_run_id = None


class WebShopEnv(BaseEnv):
    """Web shopping simulation environment with Ray-based process isolation.

    WebShop provides a text-based shopping environment where agents
    must search for products, navigate product pages, and make purchases
    that match given instructions.

    The environment uses a Ray remote actor from a shared pool to run the actual
    WebShop environment in an isolated process, ensuring thread safety for use
    with async execution engines. Workers are reused to avoid expensive re-initialization.

    Actions:
        - search[query]: Search for products with the given query
        - click[element]: Click on an element (button, link, option)

    Attributes:
        observation_mode: How to render observations ("text", "html", "text_rich").
        max_steps: Maximum steps per episode.
        num_products: Number of products in the catalog (None for all).
    """

    def __init__(
        self,
        observation_mode: str = "text",
        max_steps: int = 50,
        num_products: int | None = None,
        session_id: int | None = None,
        seed: int | None = None,
        file_path: str | None = None,
        attr_path: str | None = None,
        human_goals: bool = False,
        **kwargs,
    ):
        """Initialize WebShop environment.

        Args:
            observation_mode: Observation format ("text", "html", "text_rich").
            max_steps: Maximum steps per episode.
            num_products: Number of products (None for all).
            session_id: Specific session/goal ID to use.
            seed: Random seed for reproducibility.
            file_path: Path to product data file.
            attr_path: Path to attribute data file.
            human_goals: Whether to use human-written goals.
            **kwargs: Additional arguments for the underlying environment.
        """
        self.observation_mode = observation_mode
        self.max_steps = max_steps
        self.num_products = num_products
        self.session_id = session_id
        self.seed = seed or 42
        self.file_path = file_path
        self.attr_path = attr_path
        self.human_goals = human_goals
        self.worker_acquire_timeout = float(
            kwargs.pop(
                "worker_acquire_timeout",
                _read_positive_float_env("WEBSHOP_WORKER_ACQUIRE_TIMEOUT", 7200.0),
            )
        )
        self.kwargs = kwargs

        self._worker = None
        self._step_count = 0
        self._available_actions = {}
        self._instruction = None
        self._initialized = False

    def _lazy_init(self):
        """Lazily initialize by acquiring a worker from the pool.
        
        Workers are acquired from a shared pool. If all workers are busy,
        the request is queued and will be fulfilled when a worker becomes available.
        """
        if self._initialized:
            return

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            # Acquire a worker from the centralized Ray pool (with queuing support)
            self._worker = WebShopWorkerPool.acquire_worker(
                observation_mode=self.observation_mode,
                num_products=self.num_products,
                seed=self.seed,
                file_path=self.file_path,
                attr_path=self.attr_path,
                human_goals=self.human_goals,
                kwargs=self.kwargs,
                timeout=self.worker_acquire_timeout,
            )
            
            self._initialized = True
            logger.debug(
                f"WebShopEnv acquired worker with observation_mode={self.observation_mode}, "
                f"max_steps={self.max_steps}"
            )

        except TimeoutError as e:
            raise RuntimeError(
                f"Timeout acquiring WebShop worker: {e}. "
                "The request was queued but no worker became available in time. "
                "Consider increasing max_workers or reducing the number of parallel environments."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to acquire WebShop worker: {e}. "
                "Please ensure Ray is initialized and WebShop dependencies are installed."
            ) from e

    def _handle_worker_death(self):
        """Handle worker death by releasing the dead worker and re-acquiring a fresh one."""
        logger.warning("WebShopWorker died, re-acquiring a fresh worker...")
        
        # Mark as uninitialized so _lazy_init will acquire a new worker
        self._worker = None
        self._initialized = False
        
        # Re-acquire a fresh worker
        self._lazy_init()

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        """Reset the environment to start a new episode.

        Returns:
            A tuple of (observation, info) where:
                - observation: Text description of the current page
                - info: Dictionary containing available_actions, instruction, etc.
        """
        self._lazy_init()

        self._step_count = 0

        if seed is not None:
            self.seed = seed

        # Reset via Ray worker with actor death handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                obs, worker_info = ray.get(self._worker.reset.remote(self.session_id, self.seed))
                break
            except ray.exceptions.ActorDiedError as e:
                logger.warning(f"Worker died during reset (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self._handle_worker_death()
                else:
                    raise RuntimeError(
                        f"WebShopWorker died {max_retries} times during reset. "
                        "This may indicate resource exhaustion or configuration issues."
                    ) from e

        # Get available actions
        self._available_actions = worker_info.get("available_actions", {})

        # Get instruction
        self._instruction = worker_info.get("instruction", "")

        info = {
            "available_actions": self._available_actions,
            "instruction": self._instruction,
            "step": self._step_count,
            "max_steps": self.max_steps,
            "has_search_bar": self._available_actions.get("has_search_bar", False),
            "clickables": self._available_actions.get("clickables", []),
        }

        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute an action in the environment.

        Args:
            action: Action string in format "search[query]" or "click[element]".

        Returns:
            A tuple of (observation, reward, done, info) where:
                - observation: Text description of the new page
                - reward: 10.0 if purchase successful with perfect match, 0.0 otherwise
                - done: Whether the episode has ended
                - info: Dictionary with available_actions, task_score, etc.
        """
        self._lazy_init()

        self._step_count += 1

        # Execute action via Ray worker with actor death handling
        try:
            obs, reward, done, worker_info = ray.get(self._worker.step.remote(action))
        except ray.exceptions.ActorDiedError as e:
            logger.error(f"Worker died during step: {e}")
            # For step, we can't easily recover mid-episode, so raise with clear message
            raise RuntimeError(
                f"WebShopWorker died during step. This typically indicates resource exhaustion. "
                f"Consider reducing max_workers or concurrent trajectories."
            ) from e

        # Get updated available actions
        self._available_actions = worker_info.get("available_actions", {})

        # Process reward: 10.0 for perfect purchase, 0.0 otherwise
        task_score = worker_info.get("task_score", reward)
        if done and reward == 1.0:
            final_reward = 10.0
            won = True
        else:
            final_reward = 0.0
            won = False

        # Check max steps
        if self._step_count >= self.max_steps:
            done = True

        info = {
            "available_actions": self._available_actions,
            "instruction": self._instruction,
            "step": self._step_count,
            "max_steps": self.max_steps,
            "has_search_bar": self._available_actions.get("has_search_bar", False),
            "clickables": self._available_actions.get("clickables", []),
            "task_score": task_score,
            "won": won,
            "success": won,
            "action_is_valid": self._is_valid_action(action),
        }

        return obs, final_reward, done, info

    def _is_valid_action(self, action: str) -> bool:
        """Check if an action is valid in the current state."""
        action_lower = action.lower()

        # Check search action
        if action_lower.startswith("search["):
            return self._available_actions.get("has_search_bar", False)

        # Check click action
        if action_lower.startswith("click["):
            # Extract element from click[element]
            match = action_lower[6:-1] if action_lower.endswith("]") else ""
            clickables = self._available_actions.get("clickables", [])
            return match in [c.lower() for c in clickables]

        return False

    def get_available_actions(self) -> dict:
        """Get available actions for the current state."""
        return self._available_actions

    def get_instruction(self) -> str:
        """Get the current task instruction."""
        return self._instruction or ""

    def render(self, mode: str = "text") -> str:
        """Render the current state."""
        # Note: In the pooled version, we don't have direct access to observation
        # This would require adding a get_observation method to the worker
        return ""

    def close(self):
        """Return the worker to the pool for reuse (don't destroy it)."""
        if self._worker is not None:
            try:
                # Return worker to the centralized Ray pool for reuse
                WebShopWorkerPool.release_worker(
                    self._worker,
                    observation_mode=self.observation_mode,
                    num_products=self.num_products,
                    human_goals=self.human_goals,
                    file_path=self.file_path,
                    attr_path=self.attr_path,
                    kwargs=self.kwargs,
                )
            except Exception as e:
                logger.warning(f"Error returning WebShop worker to pool: {e}")
                # If we can't return to pool, kill it
                try:
                    ray.kill(self._worker)
                except Exception:
                    pass
            self._worker = None
        self._initialized = False

    @staticmethod
    def set_max_workers(max_workers: int):
        """Set the maximum number of Ray workers in the pool.
        
        This should be called BEFORE any environments are created.
        A good value is typically the number of concurrent trajectories you expect.
        
        Args:
            max_workers: Maximum number of workers to create.
        """
        WebShopWorkerPool.set_max_workers(max_workers)

    @staticmethod
    def shutdown_pool():
        """Shutdown the shared WebShop worker pool for this run."""
        WebShopWorkerPool.shutdown()

    @staticmethod
    def from_dict(env_args: dict) -> "WebShopEnv":
        """Create a WebShopEnv instance from a dictionary.

        Args:
            env_args: Dictionary containing environment configuration.
                Supported keys: observation_mode, max_steps, num_products,
                session_id, seed, file_path, attr_path, human_goals,
                worker_acquire_timeout, max_workers (for pool configuration).

        Returns:
            A new WebShopEnv instance.
        """
        # Handle max_workers separately - it configures the pool, not the env
        if "max_workers" in env_args:
            WebShopWorkerPool.set_max_workers(env_args["max_workers"])
        
        allowed_keys = {
            "observation_mode",
            "max_steps",
            "num_products",
            "session_id",
            "seed",
            "file_path",
            "attr_path",
            "human_goals",
            "worker_acquire_timeout",
        }
        filtered = {k: v for k, v in env_args.items() if k in allowed_keys}
        return WebShopEnv(**filtered)

    @staticmethod
    def is_multithread_safe() -> bool:
        """WebShop is multithread-safe due to Ray process-level isolation.
        
        Each WebShopEnv instance runs its environment in a separate Ray actor
        process, ensuring thread safety for concurrent access.
        """
        return True

    @classmethod
    def get_pool_stats(cls) -> dict | None:
        """Get statistics about the worker pool.
        
        Returns:
            Dictionary with pool stats, or None if pool not initialized.
        """
        try:
            return WebShopWorkerPool.get_stats()
        except Exception:
            return None
    
    @classmethod
    def print_pool_stats(cls):
        """Print a summary of the worker pool statistics."""
        stats = cls.get_pool_stats()
        if stats:
            print(f"[WebShop Worker Pool] max_workers={stats['max_workers']}")
            for config, count in stats.get('worker_counts', {}).items():
                available = stats.get('available_workers', {}).get(config, 0)
                pending = stats.get('pending_requests', {}).get(config, 0)
                print(
                    f"[WebShop Worker Pool] Config '{config[:50]}...': "
                    f"{count} workers, {available} available, {pending} pending"
                )
        else:
            print("[WebShop Worker Pool] Pool not initialized yet")

    def __repr__(self) -> str:
        return (
            f"WebShopEnv(observation_mode={self.observation_mode}, "
            f"max_steps={self.max_steps}, num_products={self.num_products})"
        )

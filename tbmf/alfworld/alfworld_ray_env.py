"""Ray-isolated ALFWorld environment sessions.

Architecture:
  TextWorldEnv    — encapsulates one TextWorld/ALFWorld gym environment
  RayEnvWorker    — Ray actor owning one env instance (generic, reusable)
  RayEnvSession   — async client wrapping one Ray actor (generic, reusable)
  create_alfworld_env_session() — public factory
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import threading
from functools import partial

import ray

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from alfworld_pkg.utils import _configure_alfworld_tmpdir, _patch_textworld_parsers

_ENV_REGISTRY_LOCK = threading.Lock()
_ENV_ID_CACHE: dict[tuple[str, int], str] = {}
_RAY_INIT_LOCK = threading.Lock()


# === Environment ===


class TextWorldEnv:
    """Single ALFWorld/TextWorld environment with init/step/close lifecycle.

    Subclass or swap this class to support other environments while keeping
    the same RayEnvWorker + RayEnvSession infrastructure.
    """

    def __init__(self, game_file: str, max_steps: int = 50):
        _configure_alfworld_tmpdir()
        _patch_textworld_parsers()

        import textworld.gym

        if not os.path.exists(game_file):
            raise FileNotFoundError(f"Game file not found: {game_file}")

        env_id = self._register(game_file, max_steps)
        self._env = textworld.gym.make(env_id)
        self._game_file = game_file

    def reset(self) -> tuple[str, list[str]]:
        obs, infos = self._env.reset()
        observation = self._unwrap(obs)
        infos = self._normalize_infos(infos)
        logger.info("ALFWorld env reset: game_file=%s", os.path.basename(self._game_file))
        return observation, infos.get("admissible_commands", [])

    def step(self, action: str) -> tuple[str, bool, bool, list[str]]:
        obs, _scores, dones, infos = self._env.step(action)
        observation = self._unwrap(obs)
        done = bool(self._unwrap(dones))
        infos = self._normalize_infos(infos)
        won = bool(infos.get("won", False))
        return observation, won, done, infos.get("admissible_commands", [])

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    @staticmethod
    def _register(game_file: str, max_steps: int) -> str:
        import textworld
        import textworld.gym

        from alfworld_pkg.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredInfos,
        )

        game_file = os.path.abspath(game_file)
        cache_key = (game_file, max_steps)

        with _ENV_REGISTRY_LOCK:
            env_id = _ENV_ID_CACHE.get(cache_key)
            if env_id is not None:
                return env_id

            request_infos = textworld.EnvInfos(
                won=True,
                admissible_commands=True,
                extras=["gamefile"],
            )
            wrappers = [partial(AlfredDemangler, shuffle=False), AlfredInfos]
            digest = hashlib.sha1(f"{game_file}:{max_steps}".encode("utf-8")).hexdigest()[:12]
            env_id = textworld.gym.register_game(
                game_file,
                request_infos,
                asynchronous=False,
                max_episode_steps=max_steps,
                wrappers=wrappers,
                name=f"alfworld-{digest}",
            )
            _ENV_ID_CACHE[cache_key] = env_id
            return env_id

    @staticmethod
    def _unwrap(value):
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value

    @staticmethod
    def _normalize_infos(infos: dict) -> dict:
        normalized = {}
        for key, value in infos.items():
            if key == "admissible_commands":
                if (
                    isinstance(value, (list, tuple))
                    and len(value) == 1
                    and isinstance(value[0], (list, tuple))
                ):
                    normalized[key] = list(value[0])
                else:
                    normalized[key] = value
            else:
                normalized[key] = TextWorldEnv._unwrap(value)
        return normalized


# === Ray Actor ===


@ray.remote(num_cpus=0)
class RayEnvWorker:
    """Ray actor owning one environment instance.

    Generic — delegates to TextWorldEnv (or any env with reset/step/close).
    """

    def __init__(self):
        self._env: TextWorldEnv | None = None

    def init(self, game_file: str, max_steps: int) -> tuple[str, list[str]]:
        self._env = TextWorldEnv(game_file, max_steps)
        return self._env.reset()

    def step(self, action: str) -> tuple[str, bool, bool, list[str]]:
        if self._env is None:
            raise RuntimeError("Environment not initialized")
        return self._env.step(action)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


# === Async Session ===


class RayEnvSession:
    """Async client wrapping one Ray env worker.

    Generic — works with any RayEnvWorker-compatible actor.
    """

    def __init__(self, worker, init_ref):
        self._worker = worker
        self._init_ref = init_ref
        self._cached_initial_state: tuple[str, list[str]] | None = None

    async def initial_state(self) -> tuple[str, list[str]]:
        if self._cached_initial_state is None:
            self._cached_initial_state = await self._await(self._init_ref)
        return self._cached_initial_state

    async def step(self, action: str) -> tuple[str, bool, bool, list[str]]:
        return await self._await(self._worker.step.remote(action))

    async def close(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        try:
            await self._await(worker.close.remote())
        except Exception:
            logger.debug("Ray env worker close failed", exc_info=True)
        finally:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                logger.debug("Ray env worker kill failed", exc_info=True)

    @staticmethod
    async def _await(ref):
        try:
            return await asyncio.wrap_future(ref.future())
        except AttributeError:
            return await asyncio.to_thread(ray.get, ref)


# === Factory ===


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name) or os.environ.get(f"RLLM_{name}")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.2f", name, raw, default)
        return default


def _ray_worker_runtime_env() -> dict:
    repo_root = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))
    pythonpath_parts = [_PKG_DIR, repo_root]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        for p in existing.split(os.pathsep):
            if p:
                pythonpath_parts.append(os.path.abspath(os.path.expanduser(p)))

    env_vars = {"PYTHONPATH": os.pathsep.join(pythonpath_parts)}
    for name in ("ALFWORLD_TMPDIR", "ALFWORLD_TMP_TARGET", "RLLM_ALFWORLD_TMPDIR",
                 "RLLM_ALFWORLD_TMP_TARGET", "TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(name)
        if value is not None:
            env_vars[name] = value
    return {"env_vars": env_vars}


def _ensure_ray_initialized():
    with _RAY_INIT_LOCK:
        if ray.is_initialized():
            return
        try:
            from rllm.trainer.ray_init_utils import get_ray_init_settings, init_ray_with_safe_cwd
            init_ray_with_safe_cwd(ignore_reinit_error=True, **get_ray_init_settings())
        except Exception:
            logger.exception("Failed to initialize Ray for env workers")
            raise


async def create_alfworld_env_session(game_file: str, max_steps: int) -> RayEnvSession:
    """Create a Ray-backed ALFWorld env session (non-blocking)."""

    def _start():
        _ensure_ray_initialized()
        worker = RayEnvWorker.options(
            num_cpus=_read_env_float("ALFWORLD_RAY_WORKER_NUM_CPUS", 0.0),
            num_gpus=_read_env_float("ALFWORLD_RAY_WORKER_NUM_GPUS", 0.0),
            runtime_env=_ray_worker_runtime_env(),
        ).remote()
        init_ref = worker.init.remote(game_file, max_steps)
        return worker, init_ref

    worker, init_ref = await asyncio.to_thread(_start)
    return RayEnvSession(worker, init_ref)

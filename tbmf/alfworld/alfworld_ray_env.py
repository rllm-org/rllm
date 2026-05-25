"""Ray-isolated ALFWorld TextWorld environment sessions."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import tempfile
import threading
from functools import partial

import ray

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_ENV_REGISTRY_LOCK = threading.Lock()
_ENV_ID_CACHE: dict[tuple[str, int], str] = {}
_TEXTWORLD_PATCH_LOCK = threading.Lock()
_TEXTWORLD_PATCHED = False
_ALFWORLD_TMPDIR_CONFIGURED = False
_ALFWORLD_TMPDIR_LOCK = threading.Lock()
_RAY_INIT_LOCK = threading.Lock()


def _patch_textworld_parsers() -> None:
    """Make TextWorld's TatSu parser entry points safe if reused in-process."""
    global _TEXTWORLD_PATCHED

    with _TEXTWORLD_PATCH_LOCK:
        if _TEXTWORLD_PATCHED:
            return
        _TEXTWORLD_PATCHED = True

    _thread_local = threading.local()

    try:
        import textworld.envs.pddl.logic as pddl_logic_mod
        from textworld.envs.pddl.logic.model import PddlLogicModelBuilderSemantics
        from textworld.envs.pddl.logic.parser import PddlLogicParser

        _OrigModelConverter = pddl_logic_mod._ModelConverter

        def _thread_safe_pddl_parse_and_convert(*args, **kwargs):
            parser = getattr(_thread_local, "pddl_parser", None)
            if parser is None:
                parser = PddlLogicParser(semantics=PddlLogicModelBuilderSemantics(), parseinfo=True)
                _thread_local.pddl_parser = parser
            model = parser.parse(*args, **kwargs)
            return _OrigModelConverter().walk(model)

        pddl_logic_mod._parse_and_convert = _thread_safe_pddl_parse_and_convert
    except Exception as e:
        logger.warning("Failed to patch pddl logic parser: %s", e)

    try:
        import textworld.envs.pddl.textgen as textgen_mod
        from textworld.envs.pddl.textgen.model import CSGModelBuilderSemantics
        from textworld.envs.pddl.textgen.parser import CSGParser

        _OrigConverter = textgen_mod._Converter

        def _thread_safe_csg_parse_and_convert(*args, **kwargs):
            parser = getattr(_thread_local, "csg_parser", None)
            if parser is None:
                parser = CSGParser(semantics=CSGModelBuilderSemantics(), parseinfo=True)
                _thread_local.csg_parser = parser
            model = parser.parse(*args, **kwargs)
            return _OrigConverter().walk(model)

        textgen_mod._parse_and_convert = _thread_safe_csg_parse_and_convert
    except Exception as e:
        logger.warning("Failed to patch CSG parser: %s", e)

    try:
        import textworld.logic as tw_logic_mod
        from textworld.logic.model import GameLogicModelBuilderSemantics
        from textworld.logic.parser import GameLogicParser

        _OrigLogicConverter = tw_logic_mod._ModelConverter

        def _thread_safe_logic_parse_and_convert(*args, **kwargs):
            parser = getattr(_thread_local, "logic_parser", None)
            if parser is None:
                parser = GameLogicParser(semantics=GameLogicModelBuilderSemantics(), parseinfo=True)
                _thread_local.logic_parser = parser
            model = parser.parse(*args, **kwargs)
            return _OrigLogicConverter().walk(model)

        tw_logic_mod._parse_and_convert = _thread_safe_logic_parse_and_convert
    except Exception as e:
        logger.warning("Failed to patch game logic parser: %s", e)


def _default_alfworld_tmpdir() -> str:
    return "/tmp/rllm_tmp"


def _default_alfworld_tmp_target() -> str:
    repo_root = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))
    return os.path.join(repo_root, "outputs", "alfworld_tmp")


def _ensure_tmpdir_symlink(link_path: str, target_path: str) -> None:
    if os.path.abspath(link_path) == os.path.abspath(target_path):
        os.makedirs(link_path, exist_ok=True)
        return

    os.makedirs(target_path, exist_ok=True)

    if os.path.islink(link_path):
        current_target = os.path.normpath(os.readlink(link_path))
        expected_target = os.path.normpath(target_path)
        if current_target == expected_target:
            return
        os.unlink(link_path)
    elif os.path.exists(link_path):
        raise RuntimeError(
            f"ALFWorld tmp link path exists and is not a symlink: {link_path}. "
            f"Remove it or set ALFWORLD_TMPDIR to another short /tmp path."
        )

    os.symlink(target_path, link_path)


def _configure_alfworld_tmpdir() -> str:
    """Force TextWorld/FastDownward temp files away from Ray's temp space."""
    global _ALFWORLD_TMPDIR_CONFIGURED

    with _ALFWORLD_TMPDIR_LOCK:
        tmpdir = (
            os.environ.get("ALFWORLD_TMPDIR")
            or os.environ.get("RLLM_ALFWORLD_TMPDIR")
            or os.environ.get("RLLM_TMPDIR")
            or _default_alfworld_tmpdir()
        )
        tmpdir = os.path.abspath(os.path.expanduser(tmpdir))
        target = (
            os.environ.get("ALFWORLD_TMP_TARGET")
            or os.environ.get("RLLM_ALFWORLD_TMP_TARGET")
            or _default_alfworld_tmp_target()
        )
        target = os.path.abspath(os.path.expanduser(target))

        if tmpdir.startswith("/tmp/"):
            _ensure_tmpdir_symlink(tmpdir, target)
        else:
            os.makedirs(tmpdir, exist_ok=True)

        os.environ["ALFWORLD_TMPDIR"] = tmpdir
        os.environ["ALFWORLD_TMP_TARGET"] = target
        os.environ["TMPDIR"] = tmpdir
        os.environ["TEMP"] = tmpdir
        os.environ["TMP"] = tmpdir
        tempfile.tempdir = tmpdir

        if not _ALFWORLD_TMPDIR_CONFIGURED:
            logger.info("ALFWorld tempfile directory configured: %s -> %s", tmpdir, target)
            _ALFWORLD_TMPDIR_CONFIGURED = True
        return tmpdir


def _unwrap_single(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


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
            continue

        normalized[key] = _unwrap_single(value)
    return normalized


def _get_registered_env_id(game_file: str, max_steps: int) -> str:
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


def _init_textworld_env(game_file: str, max_steps: int = 50):
    """Initialize one TextWorld environment for one game file."""
    _configure_alfworld_tmpdir()
    _patch_textworld_parsers()

    import textworld.gym

    if not os.path.exists(game_file):
        raise FileNotFoundError(f"Game file not found: {game_file}")

    env_id = _get_registered_env_id(game_file, max_steps)
    env = textworld.gym.make(env_id)
    obs, infos = env.reset()

    observation = _unwrap_single(obs)
    infos = _normalize_infos(infos)
    admissible_commands = infos.get("admissible_commands", [])
    logger.info("ALFWorld Ray worker initialized, game_file=%s, max_steps=%d", game_file, max_steps)

    return env, observation, admissible_commands


def _env_step(env, action: str) -> tuple[str, bool, bool, list[str]]:
    obs, _scores, dones, infos = env.step(action)

    observation = _unwrap_single(obs)
    done = bool(_unwrap_single(dones))
    infos = _normalize_infos(infos)
    won = bool(infos.get("won", False))
    admissible_commands = infos.get("admissible_commands", [])

    return observation, won, done, admissible_commands


def _close_env(env) -> None:
    env.close()


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        raw = os.environ.get(f"RLLM_{name}")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.2f", name, raw, default)
        return default


def _alfworld_ray_worker_runtime_env() -> dict:
    repo_root = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))
    pythonpath_parts = [_PKG_DIR, repo_root]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        for path in existing_pythonpath.split(os.pathsep):
            if path:
                pythonpath_parts.append(os.path.abspath(os.path.expanduser(path)))

    env_vars = {"PYTHONPATH": os.pathsep.join(pythonpath_parts)}
    for name in (
        "ALFWORLD_TMPDIR",
        "ALFWORLD_TMP_TARGET",
        "RLLM_ALFWORLD_TMPDIR",
        "RLLM_ALFWORLD_TMP_TARGET",
        "TMPDIR",
        "TEMP",
        "TMP",
    ):
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
            logger.exception("Failed to initialize Ray for ALFWorld workers")
            raise


async def _await_ray_ref(ref):
    """Await a Ray ObjectRef without blocking the asyncio event loop."""
    try:
        return await asyncio.wrap_future(ref.future())
    except AttributeError:
        return await asyncio.to_thread(ray.get, ref)


@ray.remote(num_cpus=0)
class AlfWorldRayEnvWorker:
    """Ray actor that owns exactly one ALFWorld/TextWorld environment."""

    def __init__(self):
        self._env = None
        self._initial_obs = None
        self._admissible_commands = None

    def init(self, game_file: str, max_steps: int) -> tuple[str, list[str]]:
        self._env, self._initial_obs, self._admissible_commands = _init_textworld_env(game_file, max_steps)
        return self._initial_obs, self._admissible_commands

    def step(self, action: str) -> tuple[str, bool, bool, list[str]]:
        if self._env is None:
            raise RuntimeError("ALFWorld environment is not initialized")
        return _env_step(self._env, action)

    def close(self) -> None:
        if self._env is None:
            return
        try:
            _close_env(self._env)
        finally:
            self._env = None


def _start_ray_worker(game_file: str, max_steps: int):
    _ensure_ray_initialized()
    worker = AlfWorldRayEnvWorker.options(
        num_cpus=_read_env_float("ALFWORLD_RAY_WORKER_NUM_CPUS", 0.0),
        num_gpus=_read_env_float("ALFWORLD_RAY_WORKER_NUM_GPUS", 0.0),
        runtime_env=_alfworld_ray_worker_runtime_env(),
    ).remote()
    init_ref = worker.init.remote(game_file, max_steps)
    return worker, init_ref


class AlfWorldRayEnvSession:
    """Async client wrapper around one Ray ALFWorld worker."""

    def __init__(self, worker, init_ref):
        self._worker = worker
        self._init_ref = init_ref
        self._initial_state: tuple[str, list[str]] | None = None

    async def initial_state(self) -> tuple[str, list[str]]:
        if self._initial_state is None:
            self._initial_state = await _await_ray_ref(self._init_ref)
        return self._initial_state

    async def step(self, action: str) -> tuple[str, bool, bool, list[str]]:
        return await _await_ray_ref(self._worker.step.remote(action))

    async def close(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return

        try:
            await _await_ray_ref(worker.close.remote())
        except Exception:
            logger.debug("ALFWorld Ray worker close failed", exc_info=True)
        finally:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                logger.debug("ALFWorld Ray worker kill failed", exc_info=True)


async def create_alfworld_env_session(game_file: str, max_steps: int) -> AlfWorldRayEnvSession:
    """Start a Ray env actor and queue env initialization without blocking the event loop."""
    worker, init_ref = await asyncio.to_thread(_start_ray_worker, game_file, max_steps)
    return AlfWorldRayEnvSession(worker, init_ref)

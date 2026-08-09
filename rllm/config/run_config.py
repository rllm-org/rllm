"""Config-file-driven run loading for ``rllm train`` / ``rllm eval``.

Parses a single self-contained ``.toml`` (or ``.yaml``) config file into:

1. the full OmegaConf ``DictConfig`` the trainer expects — composed from
   ``rllm/trainer/config/unified.yaml`` for the file's ``backend`` (same result
   as ``@hydra.main(config_name="unified")``), with the file's config-tree
   sections merged on top; and
2. a :class:`RunSpec` describing the *run definition* (agent, datasets,
   evaluator, sandbox, env, entrypoint) that today lives as code in each
   cookbook's ``train.py``.

Schema (see ``design/config-file-driven-train-eval.md``):

    backend = "fireworks"          # tinker | fireworks | verl
    extends = "base.toml"          # optional; merged underneath this file

    [run.agent]                    # run definition
    name = "terminus2"             # catalog name | module:Class import path
    evaluator = "..."              # optional; omit -> per-task verifier
    [run.agent.args]               # constructor kwargs / configure() overrides
    max_turns = 75
    [run.dataset]
    train = "tb-opus-pass"
    val   = "terminal-bench@2.0"
    [run.sandbox]
    backend = "modal"
    [run.env]                      # env vars exported before the run
    RLLM_HARNESS_RUN_TIMEOUT_S = "2400"

    [model]                        # everything else mirrors the config tree 1:1
    name = "..."
    [training]
    group_size = 16
    [rllm.async_training]
    enable = true

    [eval]                         # eval-only knobs (rllm eval)
    concurrency = 64
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# Bundled Hydra config root (holds unified.yaml + rllm/ + rllm/backend/).
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "trainer" / "config"

VALID_BACKENDS = ("tinker", "fireworks", "verl")
CONFIG_SUFFIXES = (".toml", ".yaml", ".yml")

# Top-level file keys consumed by the loader itself (not config-tree overrides).
_META_KEYS = frozenset({"backend", "extends", "run", "eval"})


def is_config_file(arg: str) -> bool:
    """True if ``arg`` names an existing regular file with a config suffix.

    This is the discriminator ``rllm train`` / ``rllm eval`` use to choose
    config-file mode over the benchmark-name / benchmark-directory behavior.
    Benchmarks are names or directories, so a suffixed regular file is
    unambiguous.
    """
    p = Path(os.path.expanduser(arg))
    return p.is_file() and p.suffix.lower() in CONFIG_SUFFIXES


# ---------------------------------------------------------------------------
# RunSpec — the run definition (the [run] / [eval] blocks)
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    """The declarative run definition parsed from ``[run]`` (+ ``[eval]``)."""

    backend: str = "tinker"

    # agent ([run.agent])
    agent: str | None = None
    agent_args: dict = field(default_factory=dict)
    evaluator: str | None = None
    entrypoint: str | None = None

    # dataset ([run.dataset])
    train_dataset: str | None = None
    train_split: str | None = None
    val_dataset: str | None = None
    val_split: str | None = None
    max_examples: int | None = None

    # sandbox ([run.sandbox])
    sandbox_backend: str | None = None
    sandbox_concurrency: int | None = None

    # env ([run.env]) — exported before the run
    env: dict = field(default_factory=dict)

    # eval-only knobs ([eval]) — kept raw, consumed by the eval driver
    eval: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict) -> RunSpec:
        run = dict(raw.get("run") or {})
        agent = run.get("agent")
        if isinstance(agent, str):
            agent_name, agent_block = agent, {}
        else:
            agent_block = dict(agent or {})
            agent_name = agent_block.get("name")

        dataset = dict(run.get("dataset") or {})

        sandbox = run.get("sandbox")
        if isinstance(sandbox, str):
            sandbox_backend, sandbox_block = sandbox, {}
        else:
            sandbox_block = dict(sandbox or {})
            sandbox_backend = sandbox_block.get("backend")

        return cls(
            backend=raw.get("backend", "tinker"),
            agent=agent_name,
            agent_args=dict(agent_block.get("args") or {}),
            # evaluator may sit under [run.agent] or directly under [run].
            evaluator=agent_block.get("evaluator") or run.get("evaluator"),
            entrypoint=run.get("entrypoint"),
            train_dataset=dataset.get("train"),
            train_split=dataset.get("train_split"),
            val_dataset=dataset.get("val"),
            val_split=dataset.get("val_split"),
            max_examples=dataset.get("max_examples"),
            sandbox_backend=sandbox_backend,
            sandbox_concurrency=sandbox_block.get("concurrency"),
            env=dict(run.get("env") or {}),
            eval=dict(raw.get("eval") or {}),
        )


# ---------------------------------------------------------------------------
# Parsing + extends resolution
# ---------------------------------------------------------------------------


def _parse(path: Path) -> dict:
    """Parse a single config file to a plain dict (no extends resolution)."""
    suffix = path.suffix.lower()
    if suffix == ".toml":
        import tomllib

        with open(path, "rb") as f:
            return tomllib.load(f)
    if suffix in (".yaml", ".yml"):
        loaded = OmegaConf.load(str(path))
        return OmegaConf.to_container(loaded, resolve=False)  # type: ignore[return-value]
    raise ValueError(f"Unsupported config suffix {suffix!r} (expected one of {CONFIG_SUFFIXES})")


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base`` (override wins), via OmegaConf."""
    merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(override))
    return OmegaConf.to_container(merged, resolve=False)  # type: ignore[return-value]


def _load_with_extends(path: Path, _seen: set[Path] | None = None) -> dict:
    """Parse ``path``, resolving its ``extends`` chain underneath it.

    ``extends`` is a path or list of paths (relative to the file), merged in
    order with each base *under* the file that names it (child wins).
    """
    _seen = _seen if _seen is not None else set()
    if path in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, path))
        raise ValueError(f"Circular extends chain: {chain}")
    _seen.add(path)

    raw = _parse(path)
    extends = raw.pop("extends", None)
    if not extends:
        return raw

    bases = [extends] if isinstance(extends, str) else list(extends)
    merged: dict = {}
    for base in bases:
        base_expanded = os.path.expanduser(str(base))
        base_path = Path(base_expanded) if os.path.isabs(base_expanded) else (path.parent / base_expanded)
        merged = _deep_merge(merged, _load_with_extends(base_path.resolve(), set(_seen)))
    return _deep_merge(merged, raw)


# ---------------------------------------------------------------------------
# Config composition
# ---------------------------------------------------------------------------


def merge_backend_config(backend: str, user_cfg: dict | DictConfig | None = None) -> DictConfig:
    """Compose the full trainer config for ``backend`` and merge user overrides.

    Uses Hydra's ``compose`` API against ``unified.yaml`` with
    ``rllm/backend=<backend>`` — reproducing ``@hydra.main(config_name="unified")``
    including the ``@package _global_`` split and ``${...}`` interpolations, and
    (for verl) the ``defaults: - /ppo_trainer`` composition that a plain
    ``OmegaConf.load`` cannot do. ``user_cfg`` (the file's config-tree sections)
    is merged on top.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; must be one of {VALID_BACKENDS}")

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
            cfg = compose(config_name="unified", overrides=[f"rllm/backend={backend}"])
    finally:
        GlobalHydra.instance().clear()

    # compose() omits the ``hydra`` node; guard in case that changes.
    OmegaConf.set_struct(cfg, False)
    if "hydra" in cfg:
        del cfg["hydra"]

    if user_cfg:
        # Permissive merge (additive keys allowed) — matches build_train_config.
        cfg = OmegaConf.merge(cfg, OmegaConf.create(user_cfg))
    return cfg  # type: ignore[return-value]


def _mirror_data(tree: dict) -> dict:
    """Copy a top-level ``[data]`` section into ``[rllm.data]`` (explicit ``rllm.data`` wins).

    The scripts historically set both ``data.*`` and ``rllm.data.*``; the file
    should declare ``[data]`` once. ``sync_config`` keeps the two in parity at
    runtime, but the loader-writes ensure the loader-visible config is already
    consistent.
    """
    data = tree.get("data")
    if not isinstance(data, dict):
        return tree
    tree = dict(tree)
    rllm = dict(tree.get("rllm") or {})
    rllm_data = dict(rllm.get("data") or {})
    for k, v in data.items():
        rllm_data.setdefault(k, v)  # explicit rllm.data.* takes precedence
    rllm["data"] = rllm_data
    tree["rllm"] = rllm
    return tree


def _tree_overrides(raw: dict) -> dict:
    """The config-tree sections of the file (everything but the meta keys)."""
    return {k: v for k, v in raw.items() if k not in _META_KEYS}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_run_config(
    path: str | os.PathLike,
    *,
    overrides: list[str] | None = None,
) -> tuple[DictConfig, RunSpec]:
    """Load a run config file into ``(DictConfig, RunSpec)``.

    Args:
        path: ``.toml`` / ``.yaml`` config file.
        overrides: Hydra-style ``key=value`` dotlist merged last (highest
            precedence), e.g. ``["training.learning_rate=1e-5"]``.
    """
    path = Path(os.path.expanduser(str(path))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_with_extends(path)
    run = RunSpec.from_raw(raw)
    if run.backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {run.backend!r} in {path.name}; must be one of {VALID_BACKENDS}")

    tree = _mirror_data(_tree_overrides(raw))
    cfg = merge_backend_config(run.backend, tree)

    if overrides:
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    return cfg, run


def export_env(env: dict[str, Any]) -> None:
    """Export ``[run.env]`` variables into ``os.environ`` (values stringified)."""
    for key, value in (env or {}).items():
        os.environ[str(key)] = str(value)

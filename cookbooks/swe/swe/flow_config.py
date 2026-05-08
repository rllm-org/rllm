#!/usr/bin/env python3
"""Configuration for SWEAgentFlow.

One top-level dataclass (SWEAgentFlowConfig) groups all knobs via nested
subconfigs. Scripts populate it via helpers:

* Eval CLI: ``add_flow_cli_args(parser)`` + ``flow_config_from_args(args)``.
  Scripts override defaults with ``parser.set_defaults(cost_limit=50.0, ...)``
  after registration.
* Training YAML: ``SWEAgentFlowConfig.from_config(config.swe)`` — flat keys
  with known prefixes (``val_*``, ``sandbox_retry_*``, ``compaction_*``) are
  bucketed into the matching subconfig. Unknown top-level keys are dropped
  silently. Unknown bucket keys raise TypeError — typos surface at config load.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class ValidationOverrides:
    """Overrides applied when ``AgentConfig.metadata['is_validation']`` is True.

    Each field is None by default → inherit the base (non-val) value.
    """

    step_limit: int | None = None
    agent_timeout: int | None = None
    command_timeout: int | None = None
    sandbox_timeout: int | None = None
    startup_jitter_s: float | None = None


@dataclass
class SandboxRetryConfig:
    """Retry policy for Modal sandbox creation (transient-failure handling)."""

    attempts: int = 3
    backoff_min_s: float = 5.0
    backoff_max_s: float = 10.0


@dataclass
class CompactionConfig:
    """Overrides for the ``agent.compaction_*`` section of swebench_pro.yaml.

    Each field is None by default → inherit from the YAML. Set to a concrete
    value (e.g. ``enabled=False``) to override just that field for this run.
    """

    enabled: bool | None = None
    token_trigger: int | None = None
    keep_recent_turns: int | None = None
    summary_prompt: str | None = None

    def as_agent_config_overrides(self) -> dict[str, Any]:
        """Return {compaction_<field>: value} for only non-None fields."""
        return {
            f"compaction_{name}": value
            for name, value in {
                "enabled": self.enabled,
                "token_trigger": self.token_trigger,
                "keep_recent_turns": self.keep_recent_turns,
                "summary_prompt": self.summary_prompt,
            }.items()
            if value is not None
        }


@dataclass
class ModelConfigOverrides:
    """Overrides for the ``model:`` section of swebench_pro.yaml.

    Each field is None by default → inherit from the YAML.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    return_token_ids: bool | None = None

    def as_model_config_overrides(self) -> dict[str, Any]:
        """Return model config overrides for only non-None fields."""
        kwargs = {
            name: value
            for name, value in {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "return_token_ids": self.return_token_ids,
            }.items()
            if value is not None
        }
        return {"model_kwargs": kwargs} if kwargs else {}


@dataclass
class SWEAgentFlowConfig:
    """All SWEAgentFlow knobs in one place."""

    # Base limits
    cost_limit: float = 3.0
    step_limit: int = 0
    command_timeout: int = 120
    sandbox_timeout: int = 3600
    agent_timeout: int = 0
    verbose: bool = False
    save_trajectories: bool = False
    trajectory_output_dir: str | None = None
    # Max random sleep at run() start to spread load on shared infra (Modal,
    # rate limits). 0 disables. See also validation.startup_jitter_s override.
    startup_jitter_s: float = 0.0

    validation: ValidationOverrides = field(default_factory=ValidationOverrides)
    sandbox_retry: SandboxRetryConfig = field(default_factory=SandboxRetryConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    model: ModelConfigOverrides = field(default_factory=ModelConfigOverrides)

    @classmethod
    def from_config(cls, cfg: Any) -> SWEAgentFlowConfig:
        """Build from a flat dict / OmegaConf DictConfig.

        Flat keys with known prefixes are bucketed: ``val_*`` → validation,
        ``sandbox_retry_*`` → sandbox_retry, ``compaction_*`` → compaction,
        ``model_*`` → model. Unknown top-level keys are dropped silently.
        """
        d = dict(cfg or {})
        val = {k[len("val_"):]: d.pop(k) for k in list(d) if k.startswith("val_")}
        retry = {k[len("sandbox_retry_"):]: d.pop(k) for k in list(d) if k.startswith("sandbox_retry_")}
        compaction = {k[len("compaction_"):]: d.pop(k) for k in list(d) if k.startswith("compaction_")}
        model = {k[len("model_"):]: d.pop(k) for k in list(d) if k.startswith("model_")}
        base = {f.name for f in fields(cls) if f.name not in {"validation", "sandbox_retry", "compaction", "model"}}
        return cls(
            **{k: v for k, v in d.items() if k in base and v is not None},
            validation=ValidationOverrides(**val),
            sandbox_retry=SandboxRetryConfig(**retry),
            compaction=CompactionConfig(**compaction),
            model=ModelConfigOverrides(**model),
        )


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------

def add_flow_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register all SWEAgentFlowConfig fields as flat CLI flags.

    Defaults match the dataclass defaults. Scripts override per-flag via
    ``parser.set_defaults(cost_limit=50.0, step_limit=50, ...)`` after this.
    """
    g = parser.add_argument_group("SWEAgentFlow")
    # Base
    g.add_argument("--cost_limit", type=float, default=3.0)
    g.add_argument("--step_limit", type=int, default=0)
    g.add_argument("--command_timeout", type=int, default=120)
    g.add_argument("--sandbox_timeout", type=int, default=3600)
    g.add_argument("--agent_timeout", type=int, default=0)
    g.add_argument("--startup_jitter_s", type=float, default=0.0,
                   help="Max random sleep (s) at run() start; 0 disables.")
    # Validation overrides (None = use base value)
    g.add_argument("--val_step_limit", type=int, default=None)
    g.add_argument("--val_agent_timeout", type=int, default=None)
    g.add_argument("--val_command_timeout", type=int, default=None)
    g.add_argument("--val_sandbox_timeout", type=int, default=None)
    g.add_argument("--val_startup_jitter_s", type=float, default=None)
    # Sandbox retry
    g.add_argument("--sandbox_retry_attempts", type=int, default=3)
    g.add_argument("--sandbox_retry_backoff_min_s", type=float, default=5.0)
    g.add_argument("--sandbox_retry_backoff_max_s", type=float, default=10.0)
    # Compaction overrides (None = inherit from swebench_pro.yaml)
    g.add_argument("--compaction_enabled", action=argparse.BooleanOptionalAction,
                   default=None, help="Override compaction on/off.")
    g.add_argument("--compaction_token_trigger", type=int, default=None)
    g.add_argument("--compaction_keep_recent_turns", type=int, default=None)
    g.add_argument("--compaction_summary_prompt", type=str, default=None)
    # Model config overrides (None = inherit from swebench_pro.yaml)
    g.add_argument("--model_max_tokens", type=int, default=None,
                   help="Max tokens per model response.")
    g.add_argument("--model_temperature", type=float, default=None,
                   help="Sampling temperature for the agent's chat completions calls.")
    g.add_argument("--model_top_p", type=float, default=None,
                   help="Nucleus-sampling top_p for the agent's chat completions calls.")
    g.add_argument("--model_return_token_ids", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Ask compatible vLLM/rLLM gateways to return completion token IDs.")


def flow_config_from_args(args: argparse.Namespace) -> SWEAgentFlowConfig:
    """Build SWEAgentFlowConfig from argparse Namespace populated by add_flow_cli_args()."""
    return SWEAgentFlowConfig(
        cost_limit=args.cost_limit,
        step_limit=args.step_limit,
        command_timeout=args.command_timeout,
        sandbox_timeout=args.sandbox_timeout,
        agent_timeout=args.agent_timeout,
        verbose=bool(getattr(args, "verbose", False)),
        startup_jitter_s=args.startup_jitter_s,
        validation=ValidationOverrides(
            step_limit=args.val_step_limit,
            agent_timeout=args.val_agent_timeout,
            command_timeout=args.val_command_timeout,
            sandbox_timeout=args.val_sandbox_timeout,
            startup_jitter_s=args.val_startup_jitter_s,
        ),
        sandbox_retry=SandboxRetryConfig(
            attempts=args.sandbox_retry_attempts,
            backoff_min_s=args.sandbox_retry_backoff_min_s,
            backoff_max_s=args.sandbox_retry_backoff_max_s,
        ),
        compaction=CompactionConfig(
            enabled=args.compaction_enabled,
            token_trigger=args.compaction_token_trigger,
            keep_recent_turns=args.compaction_keep_recent_turns,
            summary_prompt=args.compaction_summary_prompt,
        ),
        model=ModelConfigOverrides(
            max_tokens=args.model_max_tokens,
            temperature=args.model_temperature,
            top_p=args.model_top_p,
            return_token_ids=args.model_return_token_ids,
        ),
    )

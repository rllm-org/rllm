"""Sampling-params resolution for the ``rllm eval`` / ``rllm train`` CLIs.

A small :class:`SamplingConfig` (typed core + open ``extra`` passthrough) plus
the resolvers that turn ``--sampling-params`` (a ``"key=value,..."`` string or
``@file.yaml`` / ``@file.json``) and the ``--temperature`` / ``--top-p`` /
``--max-tokens`` shortcuts into the dict the gateway enforces on every LLM call.

Precedence (low to high): ``base.yaml rollout.{train,val}`` < ``@file`` /
string < standalone flags.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_CORE_KEYS = ("temperature", "top_p", "top_k", "max_tokens")

SAMPLING_PARAMS_HELP = (
    "Sampling params as 'key=value,...' (e.g. \"temperature=0.6,top_p=0.95,presence_penalty=0.1\") or @file.yaml / @file.json. Unknown keys (presence_penalty, min_p, ...) pass through to the backend."
)


@dataclass
class SamplingConfig:
    """Typed core sampling params plus an open ``extra`` passthrough.

    A field left ``None`` is omitted from :meth:`as_dict` and is not enforced.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Mapping | None) -> SamplingConfig:
        """Build from a mapping: core keys → fields, unknown keys → ``extra`` (``None`` values dropped)."""
        if obj is None:
            return cls()
        core: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for raw_key, value in obj.items():
            key = str(raw_key)
            if value is None:
                continue
            if key == "extra" and isinstance(value, Mapping):
                extra.update(value)
            elif key in _CORE_KEYS:
                core[key] = value
            else:
                extra[key] = value
        return cls(**core, extra=extra)

    @classmethod
    def from_string(cls, spec: str) -> SamplingConfig:
        """Parse ``"key=value,key=value"`` into a config; unknown keys go to ``extra``."""
        raw: dict[str, Any] = {}
        for token in (spec or "").split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"Invalid --sampling-params token {token!r}; expected key=value")
            key, _, value = token.partition("=")
            raw[key.strip()] = _coerce(value.strip())
        return cls.from_obj(raw)

    def merged(self, other: SamplingConfig | None) -> SamplingConfig:
        """Layer ``other`` on top of ``self`` per key (``other`` wins; its ``None`` core fields don't override)."""
        other = other or SamplingConfig()
        return SamplingConfig(
            temperature=other.temperature if other.temperature is not None else self.temperature,
            top_p=other.top_p if other.top_p is not None else self.top_p,
            top_k=other.top_k if other.top_k is not None else self.top_k,
            max_tokens=other.max_tokens if other.max_tokens is not None else self.max_tokens,
            extra={**self.extra, **other.extra},
        )

    @property
    def is_empty(self) -> bool:
        return not self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        """Flatten set keys (core non-None + extra) to a plain dict."""
        out: dict[str, Any] = {key: getattr(self, key) for key in _CORE_KEYS if getattr(self, key) is not None}
        for key, value in self.extra.items():
            out.setdefault(key, value)
        return out


def _coerce(value: str) -> Any:
    """Coerce a CLI value to int, then float, else leave as a string."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_file(path: str) -> dict[str, Any]:
    """Load a flat YAML/JSON mapping of sampling params."""
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        raise FileNotFoundError(f"sampling-params file not found: {path}")
    with open(expanded) as f:
        text = f.read()
    try:
        import yaml

        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"sampling-params file {path} must contain a mapping at the top level")
    return dict(data)


def _parse(source: str | None) -> SamplingConfig:
    """Parse a ``--sampling-params`` argument (``"k=v,..."`` string or ``@file``)."""
    source = (source or "").strip()
    if not source:
        return SamplingConfig()
    if source.startswith("@"):
        return SamplingConfig.from_obj(_load_file(source[1:]))
    return SamplingConfig.from_string(source)


def _flags(temperature: float | None, top_p: float | None, max_tokens: int | None) -> SamplingConfig:
    """The standalone ``--temperature`` / ``--top-p`` / ``--max-tokens`` shortcuts (top precedence)."""
    d: dict[str, Any] = {}
    if temperature is not None:
        d["temperature"] = temperature
    if top_p is not None:
        d["top_p"] = top_p
    if max_tokens is not None:
        d["max_tokens"] = max_tokens
    return SamplingConfig.from_obj(d)


# Default sampling params applied to every ``rllm eval`` run unless the user
# overrides them — surfaced in the eval header's Sampling row and
# gateway-enforced. Mirrors the cookbook training rollout so eval samples the
# model the same way it was trained.
_EVAL_DEFAULT_SAMPLING = SamplingConfig(temperature=1.0, top_p=1.0)


def resolve_eval_sampling(
    sampling_params: str | None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> SamplingConfig:
    """Resolve the SamplingConfig for ``rllm eval``: defaults < string/@file < flags."""
    return _EVAL_DEFAULT_SAMPLING.merged(_parse(sampling_params)).merged(_flags(temperature, top_p, max_tokens))


def resolve_train_sampling(
    sampling_params: str | None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    *,
    base_train: Mapping | None = None,
    base_val: Mapping | None = None,
) -> tuple[SamplingConfig, SamplingConfig]:
    """Resolve ``(train, val)`` configs: base.yaml rollout.{train,val} < --sampling-params < flags."""
    source = _parse(sampling_params)
    flags = _flags(temperature, top_p, max_tokens)
    train_cfg = SamplingConfig.from_obj(base_train).merged(source).merged(flags)
    val_cfg = SamplingConfig.from_obj(base_val).merged(source).merged(flags)
    return train_cfg, val_cfg

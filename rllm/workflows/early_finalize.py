from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rllm.agents.agent import Step
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START


@dataclass
class EarlyFinalizeConfig:
    enable: bool = False
    reserve_response_tokens: int = 2048
    min_phase2_tokens: int = 128
    suffix_mode: str = "auto"


@dataclass
class EarlyFinalizeResult:
    output: ModelOutput
    response_mask: list[float] | None = None
    metadata: dict[str, Any] | None = None


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


def get_early_finalize_config(rollout_engine: Any) -> EarlyFinalizeConfig:
    config = getattr(rollout_engine, "config", None)
    rllm_config = _config_get(config, "rllm", None)
    ef_cfg = _config_get(rllm_config, "early_finalize", None)
    if ef_cfg is None:
        return EarlyFinalizeConfig()

    return EarlyFinalizeConfig(
        enable=bool(_config_get(ef_cfg, "enable", False)),
        reserve_response_tokens=int(_config_get(ef_cfg, "reserve_response_tokens", 2048)),
        min_phase2_tokens=int(_config_get(ef_cfg, "min_phase2_tokens", 128)),
        suffix_mode=str(_config_get(ef_cfg, "suffix_mode", "auto")),
    )


def _supports_early_finalize(workflow: Any, messages: list[dict[str, Any]], config: EarlyFinalizeConfig) -> bool:
    if not config.enable:
        return False

    rollout_engine = workflow.rollout_engine
    if not getattr(rollout_engine, "supports_token_in_token_out", False):
        return False
    if getattr(rollout_engine, "tokenizer", None) is None:
        return False
    if getattr(rollout_engine, "chat_parser", None) is None:
        return False
    if any(msg.get("images") for msg in messages if isinstance(msg, dict)):
        return False
    return True


def _default_suffix(phase1_output: ModelOutput) -> str:
    completion_text = phase1_output.text or ""
    if THOUGHT_DELIMITER_START in completion_text and THOUGHT_DELIMITER_END not in completion_text:
        return f"{THOUGHT_DELIMITER_END}\nThe answer is: "
    return ""


def _build_suffix(workflow: Any, task: dict, messages: list[dict[str, Any]], phase1_output: ModelOutput, config: EarlyFinalizeConfig) -> str:
    builder = getattr(workflow, "build_early_finalize_suffix", None)
    if callable(builder):
        suffix = builder(task, messages, phase1_output, config=config)
        if suffix is not None:
            return str(suffix)
    return _default_suffix(phase1_output)


def _make_metadata(
    *,
    config: EarlyFinalizeConfig,
    phase1_output: ModelOutput,
    phase2_output: ModelOutput,
    suffix: str,
    suffix_ids: list[int],
    phase2_max_tokens: int,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "attempted": True,
        "suffix_mode": config.suffix_mode,
        "suffix": suffix,
        "suffix_tokens": len(suffix_ids),
        "phase1_tokens": len(phase1_output.completion_ids or []),
        "phase2_tokens": len(phase2_output.completion_ids or []),
        "phase2_max_tokens": phase2_max_tokens,
        "phase1_finish_reason": phase1_output.finish_reason,
        "final_finish_reason": phase2_output.finish_reason,
    }


def attach_model_output_to_step(
    step: Step | None,
    output: ModelOutput,
    response_mask: list[float] | None = None,
) -> None:
    if step is None:
        return

    step.model_output = output
    if output.prompt_ids is not None:
        step.prompt_ids = list(output.prompt_ids)
    if output.completion_ids is not None:
        step.response_ids = list(output.completion_ids)
    if output.logprobs is not None:
        step.logprobs = list(output.logprobs)
    if response_mask is None and output.completion_ids is not None:
        response_mask = [1.0] * len(output.completion_ids)
    if response_mask is not None:
        step.response_mask = list(response_mask)


async def maybe_generate_with_early_finalize(
    workflow: Any,
    messages: list[dict[str, Any]],
    *,
    application_id: str,
    task: dict | None = None,
    **kwargs,
) -> EarlyFinalizeResult:
    config = get_early_finalize_config(workflow.rollout_engine)
    if not _supports_early_finalize(workflow, messages, config):
        output = await workflow.timed_llm_call(messages, application_id=application_id, **kwargs)
        return EarlyFinalizeResult(output=output)

    max_response_length = int(getattr(workflow.rollout_engine, "max_response_length", 0) or 0)
    requested_max_tokens_raw = kwargs.get("max_tokens", kwargs.get("max_new_tokens"))
    requested_max_tokens = int(max_response_length if requested_max_tokens_raw is None else requested_max_tokens_raw)
    reserve_response_tokens = min(config.reserve_response_tokens, requested_max_tokens)
    phase1_max_tokens = requested_max_tokens - reserve_response_tokens

    if reserve_response_tokens < config.min_phase2_tokens or phase1_max_tokens <= 0:
        output = await workflow.timed_llm_call(messages, application_id=application_id, **kwargs)
        return EarlyFinalizeResult(output=output)

    phase1_kwargs = dict(kwargs)
    phase1_kwargs.pop("max_new_tokens", None)
    phase1_kwargs["max_tokens"] = phase1_max_tokens
    phase1_output = await workflow.timed_llm_call(messages, application_id=application_id, **phase1_kwargs)
    if phase1_output.finish_reason != "length":
        return EarlyFinalizeResult(output=phase1_output)

    if phase1_output.prompt_ids is None or phase1_output.completion_ids is None:
        return EarlyFinalizeResult(output=phase1_output)

    suffix = _build_suffix(workflow, task or {}, messages, phase1_output, config)
    suffix_ids = workflow.rollout_engine.tokenizer.encode(suffix, add_special_tokens=False)
    phase2_max_tokens = reserve_response_tokens - len(suffix_ids)
    if phase2_max_tokens < config.min_phase2_tokens:
        return EarlyFinalizeResult(output=phase1_output)

    token_input = list(phase1_output.prompt_ids) + list(phase1_output.completion_ids) + list(suffix_ids)
    phase2_kwargs = dict(kwargs)
    phase2_kwargs["enforce_max_prompt_length"] = False
    phase2_kwargs.pop("max_new_tokens", None)
    phase2_kwargs["max_tokens"] = phase2_max_tokens
    phase2_output = await workflow.timed_llm_call_from_token_input(
        token_input,
        application_id=f"{application_id}:early_finalize",
        **phase2_kwargs,
    )

    phase1_ids = list(phase1_output.completion_ids or [])
    phase2_ids = list(phase2_output.completion_ids or [])
    merged_completion_ids = phase1_ids + list(suffix_ids) + phase2_ids

    phase1_logprobs = list(phase1_output.logprobs or [])
    phase2_logprobs = list(phase2_output.logprobs or [])
    merged_logprobs = phase1_logprobs + ([0.0] * len(suffix_ids)) + phase2_logprobs

    parsed_output = workflow.rollout_engine.chat_parser.parse_completion(merged_completion_ids)
    completion_text = workflow.rollout_engine.tokenizer.decode(merged_completion_ids, skip_special_tokens=True)
    merged_output = ModelOutput(
        text=completion_text,
        content=parsed_output["content"],
        reasoning=parsed_output["reasoning"],
        tool_calls=parsed_output["tool_calls"],
        prompt_ids=list(phase1_output.prompt_ids),
        completion_ids=merged_completion_ids,
        multi_modal_inputs=phase1_output.multi_modal_inputs,
        logprobs=merged_logprobs,
        prompt_logprobs=phase1_output.prompt_logprobs,
        prompt_length=phase1_output.prompt_length,
        completion_length=len(merged_completion_ids),
        finish_reason=phase2_output.finish_reason,
    )
    response_mask = ([1.0] * len(phase1_ids)) + ([0.0] * len(suffix_ids)) + ([1.0] * len(phase2_ids))
    metadata = _make_metadata(
        config=config,
        phase1_output=phase1_output,
        phase2_output=phase2_output,
        suffix=suffix,
        suffix_ids=list(suffix_ids),
        phase2_max_tokens=phase2_max_tokens,
    )
    return EarlyFinalizeResult(
        output=merged_output,
        response_mask=response_mask,
        metadata=metadata,
    )

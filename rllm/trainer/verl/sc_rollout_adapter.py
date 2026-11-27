"""Sampler-head extraction for the pinned verl/vLLM server.

The native TokenOutput.extra_fields is the wire format. Sampling stays
untruncated at temperature=1; vLLM must return processed_logprobs. We never
renormalize the head: its missing mass is the sampler tail Q_T.
"""

from __future__ import annotations

import inspect
import textwrap

from rllm.trainer.verl.score_centering import SC_FIELDS, validate_head


def validate_sampling_params(params):
    required = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
    defaults = {"min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
    for key, expected in {**required, **defaults}.items():
        if params.get(key, expected) != expected:
            raise ValueError(f"Score centering requires {key}={expected}")
    allowed = (
        set(required)
        | set(defaults)
        | {
            "max_tokens",
            "max_new_tokens",
            "logprobs",
            "seed",
            "stop",
            "stop_token_ids",
            "ignore_eos",
            "include_stop_str_in_output",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "detokenize",
            "prompt_logprobs",
            "rllm_sc_topk",
        }
    )
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unsupported score centering sampling parameters: {sorted(unknown)}")
    # Stop-string stripping can remove token IDs independently of logprob rows.
    if params.get("stop"):
        raise ValueError("Score centering currently supports token-based stops only")


def sampler_fields(completion, k):
    ids, lps = [], []
    if completion.logprobs is None or len(completion.logprobs) != len(completion.token_ids):
        raise RuntimeError("vLLM did not return token-aligned sampler candidate logprobs")
    for head in completion.logprobs:
        # vLLM may append the sampled token outside the requested top-k.
        ordered = sorted(head.items(), key=lambda item: item[1].logprob, reverse=True)[:k]
        ids.append([int(token) for token, _ in ordered])
        lps.append([float(value.logprob) for _, value in ordered])
    validate_head(ids, lps, len(completion.token_ids))
    return {**dict(zip(SC_FIELDS, (ids, lps), strict=True)), "rllm_sc_distribution": "processed_untruncated_v1"}


def extract_sc_fields(token_output):
    fields = token_output.extra_fields
    if fields.get("rllm_sc_distribution") != "processed_untruncated_v1" or not all(key in fields for key in SC_FIELDS):
        raise RuntimeError("Score centering sampler adapter did not return an actual sampling distribution")
    result = {key: fields[key] for key in SC_FIELDS}
    validate_head(*(result[key] for key in SC_FIELDS), len(token_output.token_ids))
    if token_output.log_probs is None or len(token_output.log_probs) != len(token_output.token_ids):
        raise ValueError("Score centering requires token-aligned sampled log probabilities")
    return result


def patch_generate_source(source):
    """Apply two checked edits to the pinned upstream method, preserving its body.

    Fail instead of guessing if upstream changed the expected call sites.
    The method runs with its original module globals and annotations.
    """
    old = 'sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None'
    new = """sc_topk = sampling_params.pop("rllm_sc_topk", None)
        if sc_topk is not None:
            if self.config.logprobs_mode != "processed_logprobs":
                raise ValueError("Score centering requires rollout.logprobs_mode=processed_logprobs")
            sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
            _rllm_sc_validate(sampling_params)
            if not isinstance(sc_topk, int) or sc_topk < 1:
                raise ValueError("Score centering candidate budget must be a positive integer")
        sampling_params["logprobs"] = sc_topk if sc_topk is not None else (0 if sampling_params.pop("logprobs", False) else None)"""
    # The original key remains in the SC branch; overwrite is intentional.
    hook = 'extra_fields = {"global_steps": self.global_steps}'
    replacement = """extra_fields = {"global_steps": self.global_steps}
        if sc_topk is not None:
            extra_fields.update(_rllm_sc_fields(final_res.outputs[0], sc_topk))"""
    for anchor, updated in ((old, new), (hook, replacement)):
        if source.count(anchor) != 1:
            raise RuntimeError("Unsupported verl vLLM server source; score centering adapter not installed")
        source = source.replace(anchor, updated)
    return source


def build_sc_generate(server_cls, module):
    """Compile the adapted ``generate`` with the upstream module's globals."""
    from importlib.metadata import version

    if version("verl") != "0.8.0":
        raise RuntimeError("rLLM score centering sampler adapter requires verl==0.8.0")
    original = server_cls.generate
    source = textwrap.dedent(patch_generate_source(inspect.getsource(original)))
    namespace = dict(module.__dict__, _rllm_sc_validate=validate_sampling_params, _rllm_sc_fields=sampler_fields)
    exec(compile(source, inspect.getsourcefile(original), "exec"), namespace)
    return namespace["generate"]


def install_rollout_adapter():
    """Make verl's vLLM replicas launch the SC server class.

    Call in the process that creates rollout replicas, before they are created:
    ``vLLMReplica.__init__`` resolves ``vLLMHttpServer`` from its module at call
    time. Ray pickles the subclass by reference, so each vLLM server actor
    imports ``rllm.trainer.verl.sc_vllm_server`` and gets the adapted method.
    """
    from verl.workers.rollout.vllm_rollout import vllm_async_server as mod

    from rllm.trainer.verl.sc_vllm_server import SCvLLMHttpServer

    mod.vLLMHttpServer = SCvLLMHttpServer

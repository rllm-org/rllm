"""Fail-fast boundaries for the first score-centering integration."""

import logging
import math

logger = logging.getLogger(__name__)


def validate_score_centering_config(config):
    algorithm = config.rllm.algorithm
    sc = algorithm.get("score_centering", {})
    if not sc.get("enabled", False):
        return
    for key, default in (("record_topk", 32), ("chunk_size", 128)):
        value = sc.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"score_centering.{key} must be a positive integer")
    actor, rollout, model = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
    if actor.get("strategy") not in ("fsdp", "fsdp2") or rollout.get("name") != "vllm":
        raise ValueError("Score centering requires FSDP/FSDP2 and vLLM")
    if actor.get("policy_loss", {}).get("loss_mode", "vanilla") != "vanilla" or algorithm.get("loss_fn") not in (None, "vanilla"):
        raise ValueError("Score centering cannot combine with a different policy loss")
    if actor.get("entropy_coeff", 0) != 0 or actor.get("use_kl_loss", False) or algorithm.get("kl_beta", 0) != 0:
        raise ValueError("Score centering currently requires entropy_coeff=0 and KL loss disabled")
    if actor.get("ulysses_sequence_parallel_size", 1) != 1 or actor.get("fsdp_config", {}).get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("Score centering does not support sequence parallelism")
    if not model.get("use_remove_padding", True) or model.get("use_fused_kernels", False) or model.get("use_liger", False):
        raise ValueError("Score centering requires remove_padding=true, fused kernels and Liger disabled")
    if (model.get("mtp") or {}).get("enable", False) or (rollout.get("mtp") or {}).get("enable", False):
        raise ValueError("Score centering does not support speculative decoding/MTP")
    if rollout.get("dtype", "bfloat16") not in ("bfloat16", "bf16") or rollout.get("quantization") is not None:
        raise ValueError("Score centering currently requires BF16 unquantized rollout")
    if rollout.get("logprobs_mode", "processed_logprobs") != "processed_logprobs":
        raise ValueError("Score centering requires processed_logprobs")
    vllm = rollout.get("engine_kwargs", {}).get("vllm", {}) or {}
    if vllm.get("max_logprobs", 20) < sc.get("record_topk", 32):
        raise ValueError("Set actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs >= score_centering.record_topk")
    if vllm.get("logprobs_mode", "processed_logprobs") != "processed_logprobs":
        raise ValueError("Score centering requires processed_logprobs in vLLM engine overrides too")
    if any(key in vllm for key in ("logits_processors", "speculative_config", "quantization")):
        raise ValueError("Score centering does not support custom logits processors, speculation or quantization")
    if config.rllm.get("async_training", {}).get("partial_rollout", False):
        raise ValueError("Score centering currently requires partial_rollout=false")
    if algorithm.get("router_replay", "disabled") != "disabled":
        raise ValueError("Score centering currently supports dense models without router replay")
    if config.get("distillation", {}).get("enabled", False) or config.rllm.get("distill", {}).get("enable", False) or algorithm.get("adv_estimator") == "distill":
        raise ValueError("Score centering does not support distillation/OPD")
    correction = algorithm.get("rollout_correction", {})
    if correction.get("bypass_mode") is False:
        raise ValueError("Score centering uses sampler probabilities; set bypass_mode=true or null")
    if correction.get("tis_mode") not in (None, "token"):
        raise ValueError("Score centering supports only token-level TIS")
    cap = correction.get("tis_cap", 2.0)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("Score centering TIS cap must be finite and positive")
    # SC uses an unclipped PG objective; bound how far updates drift from the sampler.
    if actor.get("ppo_epochs", 1) != 1:
        raise ValueError("Score centering uses an unclipped PG objective; set actor.ppo_epochs=1")
    mini_batch, train_batch = actor.get("ppo_mini_batch_size"), config.get("data", {}).get("train_batch_size")
    if mini_batch and train_batch and mini_batch < train_batch:
        if correction.get("tis_mode") is None:
            raise ValueError(
                "Score centering without TIS needs one optimizer step per rollout batch "
                "(actor.ppo_mini_batch_size >= data.train_batch_size); later steps would be uncorrected off-policy. "
                "Alternatively enable rollout_correction.tis_mode=token."
            )
        logger.warning("Score centering: %d optimizer steps per rollout batch without clipping; token TIS (cap=%s) is the only off-policy correction", math.ceil(train_batch / mini_batch), cap)
    # Request overrides are checked a second time in VerlEngine and server.
    from rllm.trainer.verl.sc_rollout_adapter import validate_sampling_params

    validate_sampling_params(
        {
            key: rollout.get(key, default)
            for key, default in (
                ("temperature", 1.0),
                ("top_p", 1.0),
                ("top_k", -1),
                ("repetition_penalty", 1.0),
            )
        }
    )
    if rollout.get("do_sample", True) is False:
        raise ValueError("Score centering requires stochastic sampling")

"""Narrow, fail-fast adapter for verl 0.8.0's FSDP EngineWorker path."""

from __future__ import annotations

import math

import torch

from rllm.trainer.verl.score_centering import SC_FIELDS, candidate_log_probs, centered_token_loss


def prepare_sc_outputs(engine, output, output_args, micro_batch, loss_function):
    """Gather sampled and head log-probs without verl's inplace backward kernel."""
    from verl.utils import tensordict_utils as tu

    if not tu.get_non_tensor_data(micro_batch, "use_remove_padding", default=True):
        raise ValueError("Score centering currently requires model.use_remove_padding=true")
    if tu.get_non_tensor_data(micro_batch, "use_fused_kernels", default=False) or engine.use_ulysses_sp:
        raise ValueError("Score centering does not support fused kernels or sequence parallelism")
    if tu.get_non_tensor_data(micro_batch, "calculate_entropy", default=False):
        raise ValueError("Score centering currently requires entropy_coeff=0")
    if not all(key in micro_batch for key in SC_FIELDS):
        raise ValueError("Score centering forward requires sampler candidate fields")
    hf_config = engine.model_config.hf_config
    if getattr(hf_config, "vision_config", None) is not None or any(getattr(hf_config, key, 0) for key in ("num_experts", "num_local_experts", "n_routed_experts")):
        raise ValueError("Score centering currently supports dense text models only")
    logits = output.logits.squeeze(0)
    ids = micro_batch["rollout_topk_ids"].values()
    labels = output_args["input_ids_rmpad_rolled"]
    if not torch.all(output_args["temperature_rmpad"] == 1):
        raise ValueError("Score centering requires training temperature=1")
    label_logp, head_logp, tail_logp = candidate_log_probs(logits, labels, ids, loss_function.chunk_size)
    offsets = micro_batch["input_ids"].offsets()
    return {
        "log_probs": torch.nested.nested_tensor_from_jagged(label_logp, offsets),
        "sc_head_log_probs": torch.nested.nested_tensor_from_jagged(head_logp, offsets),
        "sc_tail_log_mass": torch.nested.nested_tensor_from_jagged(tail_logp, offsets),
    }


def install_fsdp_adapter():
    from importlib.metadata import version

    if version("verl") != "0.8.0":
        raise RuntimeError("rLLM score centering adapter requires verl==0.8.0")
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    original = FSDPEngineWithLMHead.prepare_model_outputs
    if getattr(original, "_rllm_sc_adapter", False):
        return

    def prepare(self, output, output_args, micro_batch, logits_processor_func):
        from verl.utils import tensordict_utils as tu

        if tu.get_non_tensor_data(micro_batch, "score_centering", default=False):
            if not isinstance(logits_processor_func, ScoreCenteringLoss):
                raise RuntimeError("Score centering enabled but SC loss was replaced")
            return prepare_sc_outputs(self, output, output_args, micro_batch, logits_processor_func)
        return original(self, output, output_args, micro_batch, logits_processor_func)

    prepare._rllm_sc_adapter = True
    FSDPEngineWithLMHead.prepare_model_outputs = prepare


class ScoreCenteringLoss:
    """SC policy gradient, optionally with current-policy token TIS.

    Weights are sg[min(p_theta/q, cap)], evaluated inside each forward. There
    is no additional proximal-policy ratio or PPO clipping. Outer correction
    must be disabled. This is serialized to the worker by set_loss_fn.
    """

    def __init__(self, actor_config, sc_config, tis_cap=None):
        self.loss_agg_mode = actor_config.loss_agg_mode
        self.loss_scale_factor = actor_config.get("loss_scale_factor", None)
        self.chunk_size = sc_config.get("chunk_size", 128)
        self.tis_cap = tis_cap

    def __setstate__(self, state):
        self.__dict__.update(state)
        install_fsdp_adapter()

    def __call__(self, model_output, data, dp_group=None):
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.utils import tensordict_utils as tu
        from verl.utils.metric import AggregationType, Metric
        from verl.workers.utils.padding import no_padding_2_padding

        if "sc_head_log_probs" not in model_output:
            raise RuntimeError("Score centering FSDP forward adapter did not run")
        if "rollout_is_weights" in data or tu.get(data, "policy_loss_mode_override", default=None) is not None:
            raise ValueError("Score centering cannot combine with outer IS weights or per-role loss overrides")
        selected = data.select("response_mask", "advantages", "rollout_log_probs").to_padded_tensor()
        sample_logp = no_padding_2_padding(model_output["log_probs"], data)
        head_logp = no_padding_2_padding(model_output["sc_head_log_probs"], data)
        head_logq = no_padding_2_padding(data["rollout_topk_log_probs"], data)
        tail_logp = no_padding_2_padding(model_output["sc_tail_log_mass"], data)
        mask = selected["response_mask"].bool()
        # Response padding rows hold zero q; give them a valid dummy distribution so the
        # mass checks pass. They are masked out of the aggregated loss.
        dummy = -math.log(head_logq.shape[-1])
        head_logq = torch.where(mask.unsqueeze(-1), head_logq, dummy)
        losses, q_mass, p_mass = centered_token_loss(
            sample_logp,
            selected["rollout_log_probs"],
            head_logp,
            head_logq,
            tail_logp,
            selected["advantages"],
            self.tis_cap,
        )
        loss = agg_loss(
            loss_mat=losses,
            loss_mask=mask,
            loss_agg_mode=self.loss_agg_mode,
            dp_size=tu.get(data, "dp_size"),
            batch_num_tokens=tu.get(data, "batch_num_tokens"),
            global_batch_size=tu.get(data, "global_batch_size"),
            loss_scale_factor=self.loss_scale_factor,
        )
        count = mask.sum().clamp_min(1)

        def masked_mean(x):
            return Metric(value=(x.float() * mask).sum().detach() / count, aggregation=AggregationType.MEAN)

        # Trainer/sampler mismatch on sampled tokens; with TIS this also shows how often the cap binds.
        log_ratio = (sample_logp.detach().float() - selected["rollout_log_probs"].float()).masked_fill(~mask, 0)
        ratio = log_ratio.exp()
        mismatch = {
            "score_centering/sample_ratio_mean": masked_mean(ratio),
            "score_centering/sample_abs_log_ratio_mean": masked_mean(log_ratio.abs()),
        }
        if self.tis_cap is not None:
            mismatch["score_centering/tis_weight_mean"] = masked_mean(ratio.clamp(max=self.tis_cap))
            mismatch["score_centering/tis_capped_fraction"] = masked_mean(ratio > self.tis_cap)
        metrics = {
            **mismatch,
            "actor/pg_loss": Metric(value=loss.detach(), aggregation=AggregationType.SUM),
            "score_centering/sampler_head_mass": Metric(value=(q_mass * mask).sum().detach() / count, aggregation=AggregationType.MEAN),
            "score_centering/train_head_mass": Metric(value=(p_mass * mask).sum().detach() / count, aggregation=AggregationType.MEAN),
            "score_centering/sampler_tail_mass": Metric(value=((1 - q_mass).clamp_min(0) * mask).sum().detach() / count, aggregation=AggregationType.MEAN),
        }
        return loss, metrics

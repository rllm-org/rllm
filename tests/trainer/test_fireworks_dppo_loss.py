import importlib.util
import math
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_DPP0_PATH = os.path.join(os.path.dirname(__file__), "../../rllm/trainer/fireworks/dppo_loss.py")
_spec = importlib.util.spec_from_file_location("fireworks_dppo_loss", _DPP0_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _datum(weights):
    return SimpleNamespace(loss_fn_inputs={"weights": SimpleNamespace(data=weights)})


def test_extract_logprobs_tolerates_custom_forward_backward_without_logprobs():
    from rllm.trainer.fireworks.fireworks_policy_trainer import FireworksPolicyTrainer

    result = SimpleNamespace(loss_fn_outputs=[{"loss": SimpleNamespace(data=[1.0])}])

    assert FireworksPolicyTrainer._extract_logprobs(result) == []


def test_dppo_mask_blocks_only_unsafe_directions():
    behavior_logprobs = torch.log(torch.tensor([[0.1, 0.1, 0.1]]))
    policy_logprobs = torch.log(torch.tensor([[0.5, 0.5, 0.5]])).requires_grad_(True)
    ratio = torch.exp(policy_logprobs - behavior_logprobs)
    advantages = torch.tensor([[1.0, -1.0, 0.0]])
    response_mask = torch.ones_like(policy_logprobs, dtype=torch.bool)

    mask, divergence = _mod.compute_dppo_mask(
        policy_logprobs=policy_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        ratio=ratio,
        response_mask=response_mask,
        divergence_type="tv",
        divergence_threshold=0.05,
    )

    assert torch.all(divergence > 0.05)
    assert not mask.requires_grad
    torch.testing.assert_close(mask, torch.tensor([[0.0, 1.0, 1.0]]))


def test_dppo_custom_loss_uses_rollout_anchor_and_masks_gradients():
    blocked_policy_lp = torch.log(torch.tensor([0.5])).requires_grad_(True)
    kept_policy_lp = torch.log(torch.tensor([0.5])).requires_grad_(True)
    rollout_lp = [math.log(0.1)]

    loss_fn = _mod.make_dppo_loss_fn(
        advantages=[1.0, -1.0],
        rollout_logprobs=[rollout_lp, rollout_lp],
        divergence_type="tv",
        divergence_threshold=0.05,
    )

    loss, metrics = loss_fn(
        [_datum([1.0]), _datum([1.0])],
        [blocked_policy_lp, kept_policy_lp],
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(5.0))
    torch.testing.assert_close(blocked_policy_lp.grad, torch.tensor([0.0]))
    torch.testing.assert_close(kept_policy_lp.grad, torch.tensor([5.0]))
    assert metrics["dppo_mask_frac_kept"] == pytest.approx(0.5)
    assert metrics["dppo_divergence_mean"] == pytest.approx(0.4)
    assert metrics["dppo_ratio_mean"] == pytest.approx(5.0)
    assert metrics["offpolicy/ratio/mean"] == pytest.approx(5.0)
    assert metrics["offpolicy/logprob_abs_diff/mean"] == pytest.approx(math.log(5.0))
    assert metrics["offpolicy/training_ppl"] == pytest.approx(2.0)
    assert metrics["offpolicy/rollout_ppl"] == pytest.approx(10.0)


def test_merge_remote_metrics_preserves_offpolicy_namespace():
    from rllm.trainer.fireworks.fireworks_policy_trainer import FireworksPolicyTrainer

    dummy = SimpleNamespace(_METRIC_SKIP_KEYS=set())
    result = SimpleNamespace(metrics={"offpolicy/kl": 1.25, "dppo_ratio_mean": 2.0})
    metrics = {}

    FireworksPolicyTrainer._merge_remote_metrics(dummy, result, metrics)

    assert metrics["offpolicy/kl"] == 1.25
    assert metrics["train/dppo_ratio_mean"] == 2.0

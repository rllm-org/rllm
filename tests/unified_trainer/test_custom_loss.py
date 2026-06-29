"""Unit tests for the unified custom-loss abstraction (rllm.trainer.algorithms.loss).

Backend-agnostic: pure torch on toy tensors, no GPU / verl / tinker server needed.
"""

import pytest

torch = pytest.importorskip("torch")

from rllm.trainer.algorithms.config import AlgorithmConfig
from rllm.trainer.algorithms.loss import (
    RLLM_LOSS_REGISTRY,
    LossContext,
    dppo_tv,
    get_loss,
    is_custom_loss,
    register_loss,
    resolve_loss_terms,
)


def _ctx(pi, mu, adv, action_mask=None, obs_mask=None, **params):
    pi = torch.tensor(pi, dtype=torch.float32, requires_grad=True)
    mu = torch.tensor(mu, dtype=torch.float32)
    adv = torch.tensor(adv, dtype=torch.float32)
    n = pi.shape[-1]
    am = torch.tensor(action_mask if action_mask is not None else [1.0] * n)
    om = torch.tensor(obs_mask if obs_mask is not None else [0.0] * n)
    return LossContext(pi=pi, mu=mu, advantages=adv, action_mask=am, obs_mask=om, params=params)


# --------------------------------------------------------------------------- top-level decorator
def test_top_level_decorator_exposed():
    import rllm

    # @rllm.register_loss / rllm.LossContext mirror @rllm.rollout / @rllm.evaluator,
    # and are the same objects as the algorithms-module exports.
    assert rllm.register_loss is register_loss
    assert rllm.LossContext is LossContext

    @rllm.register_loss("_test_top_level")
    def _t(ctx: rllm.LossContext):
        return -ctx.pi, ctx.action_mask, {}

    assert get_loss("_test_top_level") is _t


# --------------------------------------------------------------------------- registry
def test_builtins_registered():
    for name in ("dppo_tv", "dppo_kl", "ppo_clip", "env_prediction"):
        assert name in RLLM_LOSS_REGISTRY
        assert is_custom_loss(name)
    assert not is_custom_loss("vanilla")  # verl-native
    assert not is_custom_loss(None)


def test_register_and_get_roundtrip():
    @register_loss("_test_noop")
    def _noop(ctx):
        return -ctx.pi, ctx.action_mask, {}

    assert get_loss("_test_noop") is _noop
    with pytest.raises(ValueError):
        get_loss("_does_not_exist")


# --------------------------------------------------------------------------- DPPO math
def test_dppo_tv_mask_matches_paper_rule():
    # mu ~ behavior probs via log; pick probs then take log.
    # token0: adv>0, pi far above mu  -> moving away, |dp|>delta -> MASKED
    # token1: adv>0, pi just above mu  -> within delta          -> kept
    # token2: adv<0, pi below mu       -> moving away, |dp|>delta -> MASKED
    # token3: adv<0, pi above mu       -> moving toward          -> kept
    probs_pi = [0.90, 0.52, 0.10, 0.70]
    probs_mu = [0.50, 0.50, 0.50, 0.50]
    pi = [float(torch.tensor(p).log()) for p in probs_pi]
    mu = [float(torch.tensor(p).log()) for p in probs_mu]
    adv = [1.0, 1.0, -1.0, -1.0]
    ctx = _ctx(pi, mu, adv, delta=0.2)
    per_token, mask, metrics = dppo_tv(ctx)

    # keep is folded into per_token via the -adv*tr*pi*keep form: masked tokens contribute 0.
    masked = (per_token == 0.0).tolist()
    assert masked == [True, False, True, False]
    # mask_frac = 2 masked out of 4 action tokens
    assert metrics["dppo_tv/mask_frac"] == pytest.approx(0.5)
    assert torch.equal(mask, ctx.action_mask)


def test_dppo_tv_matches_verl_formula():
    """Reproduce verl 0.8 compute_policy_loss_dppo_tv per-token loss and compare."""
    torch.manual_seed(0)
    n = 64
    pi = (torch.rand(n) * -2).requires_grad_(True)  # logprobs in (-2,0)
    mu = torch.rand(n) * -2
    adv = torch.randn(n)
    delta = 0.2
    ctx = LossContext(pi=pi, mu=mu, advantages=adv, action_mask=torch.ones(n), obs_mask=torch.zeros(n), params={"delta": delta})
    ours, _, _ = dppo_tv(ctx)

    # verl reference (from core_algos.compute_policy_loss_dppo_tv)
    neg_approx_kl = torch.clamp(pi.detach() - mu, min=-20.0, max=20.0)
    ratio = torch.exp(neg_approx_kl)
    truncated_ratio = torch.clamp(ratio, max=20.0).detach()
    prob, old_prob = pi.detach().exp(), mu.exp()
    valid_pos = (prob - old_prob) <= delta
    valid_neg = (prob - old_prob) >= -delta
    valid = torch.where(adv > 0, valid_pos, valid_neg).detach().float()
    verl = -adv * truncated_ratio * pi.detach() * valid

    assert torch.allclose(ours.detach(), verl, atol=1e-6)


def test_dppo_tv_gradient_flows_and_is_zero_when_masked():
    probs_pi = [0.90, 0.52]  # token0 masked (far), token1 kept
    pi = torch.tensor([float(torch.tensor(p).log()) for p in probs_pi], requires_grad=True)
    mu = torch.tensor([float(torch.tensor(0.5).log())] * 2)
    adv = torch.tensor([1.0, 1.0])
    ctx = LossContext(pi=pi, mu=mu, advantages=adv, action_mask=torch.ones(2), obs_mask=torch.zeros(2), params={"delta": 0.2})
    per_token, _, _ = dppo_tv(ctx)
    per_token.sum().backward()
    assert pi.grad is not None
    assert pi.grad[0].item() == 0.0  # masked token gets no gradient
    assert pi.grad[1].item() != 0.0  # kept token does


# --------------------------------------------------------------------------- ECHO term
def test_env_prediction_is_ce_over_obs_mask():
    pi = torch.tensor([-0.5, -1.0, -2.0], requires_grad=True)
    om = [0.0, 1.0, 1.0]
    ctx = _ctx([-0.5, -1.0, -2.0], [-0.5, -1.0, -2.0], [0.0, 0.0, 0.0], obs_mask=om)
    ctx.pi = pi
    per_token, mask, _ = get_loss("env_prediction")(ctx)
    assert torch.equal(per_token, -pi)
    assert mask.tolist() == om


# --------------------------------------------------------------------------- config resolution
def test_resolve_explicit_losses_list():
    cfg = AlgorithmConfig(losses=[{"type": "dppo_tv", "coef": 1.0, "delta": 0.3}, {"type": "env_prediction", "coef": 0.05}])
    terms = resolve_loss_terms(cfg)
    assert [t.name for t in terms] == ["dppo_tv", "env_prediction"]
    assert terms[0].coef == 1.0 and terms[0].params["delta"] == 0.3
    assert terms[1].coef == 0.05


def test_resolve_backcompat_loss_fn_plus_echo():
    cfg = AlgorithmConfig(loss_fn="dppo_tv", env_loss_coef=0.05)
    terms = resolve_loss_terms(cfg)
    names = [t.name for t in terms]
    assert names[0] == "dppo_tv" and terms[0].coef == 1.0
    assert "env_prediction" in names  # ECHO folded in as a term


def test_resolve_native_loss_returns_empty():
    # A backend-native loss (not rLLM-registered) -> empty -> caller uses native path.
    cfg = AlgorithmConfig(loss_fn="vanilla")
    assert resolve_loss_terms(cfg) == []
    cfg2 = AlgorithmConfig(loss_fn=None)
    assert resolve_loss_terms(cfg2) == []


# --------------------------------------------------------------------------- managed adapter (tinker/fireworks forward_backward_custom)
def _make_datum(target, logprobs, adv, mask):
    tinker = pytest.importorskip("tinker")
    from tinker.types.tensor_data import TensorData

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1] + list(target[:-1])),
        loss_fn_inputs={
            "target_tokens": TensorData(data=list(target), dtype="int64"),
            "logprobs": TensorData(data=list(logprobs), dtype="float32"),
            "advantages": TensorData(data=list(adv), dtype="float32"),
            "mask": TensorData(data=list(mask), dtype="float32"),
        },
    )


def test_managed_closure_builds_ctx_and_backprops():
    pytest.importorskip("tinker")
    from rllm.trainer.tinker.custom_loss import build_custom_loss

    # token0 obs (mask 0 -> ECHO only); tokens1-3 action (DPPO).
    mu = [float(torch.tensor(0.5).log())] * 4
    d = _make_datum(target=[2, 3, 4, 5], logprobs=mu, adv=[0.0, 1.0, 1.0, -1.0], mask=[0.0, 1.0, 1.0, 1.0])
    cfg = AlgorithmConfig(losses=[{"type": "dppo_tv", "coef": 1.0, "delta": 0.2}, {"type": "env_prediction", "coef": 0.05}])
    terms = resolve_loss_terms(cfg)
    stripped, loss_fn = build_custom_loss(terms, [d])

    # stripped datum carries only target_tokens (forward_backward_custom requirement)
    assert set(stripped[0].loss_fn_inputs.keys()) == {"target_tokens"}

    # current-policy logprobs the server would return (requires_grad); push token1 far above mu.
    probs_pi = [0.5, 0.95, 0.52, 0.5]
    pi = torch.tensor([float(torch.tensor(p).log()) for p in probs_pi], requires_grad=True)
    loss, metrics = loss_fn(stripped, [pi])
    assert loss.dim() == 0
    loss.backward()
    assert pi.grad is not None
    # obs token (0) gets gradient from ECHO; action token1 is DPPO-masked (far from mu, adv>0).
    assert pi.grad[0].item() != 0.0  # ECHO trains the observation token
    assert metrics["custom_loss/num_datums"] == 1.0


def test_managed_dppo_masks_action_token_moving_away():
    pytest.importorskip("tinker")
    from rllm.trainer.tinker.custom_loss import build_custom_loss

    mu = [float(torch.tensor(0.5).log())] * 2
    d = _make_datum(target=[2, 3], logprobs=mu, adv=[1.0, 1.0], mask=[1.0, 1.0])  # both action
    cfg = AlgorithmConfig(losses=[{"type": "dppo_tv", "coef": 1.0, "delta": 0.2}])
    terms = resolve_loss_terms(cfg)
    stripped, loss_fn = build_custom_loss(terms, [d])
    pi = torch.tensor([float(torch.tensor(0.95).log()), float(torch.tensor(0.52).log())], requires_grad=True)
    loss, _ = loss_fn(stripped, [pi])
    loss.backward()
    assert pi.grad[0].item() == 0.0  # token0: far above mu, adv>0 -> masked
    assert pi.grad[1].item() != 0.0  # token1: within delta -> trained

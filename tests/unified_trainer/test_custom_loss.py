"""Unit tests for the verl-style custom-loss abstraction (rllm.trainer.algorithms.loss).

A loss is a single function selected by name that returns a scalar via ``ctx.aggregate``.
Backend-agnostic: pure torch on toy tensors, no GPU / verl / tinker server needed.
"""

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from rllm.trainer.algorithms.config import AlgorithmConfig
from rllm.trainer.algorithms.loss import (
    RLLM_LOSS_REGISTRY,
    LossContext,
    ResolvedLoss,
    dppo_tv,
    get_loss,
    is_custom_loss,
    ppo_clip,
    ppo_clip_env,
    register_loss,
    resolve_loss,
)


def _agg_sum(per_token, mask):
    """Test aggregator: sum over masked tokens (lets us inspect the raw per-token loss)."""
    return (per_token * mask).sum()


def _ctx(pi, mu, adv, action_mask=None, obs_mask=None, aggregate=_agg_sum, **params):
    pi = torch.tensor(pi, dtype=torch.float32, requires_grad=True)
    mu = torch.tensor(mu, dtype=torch.float32)
    adv = torch.tensor(adv, dtype=torch.float32)
    n = pi.shape[-1]
    am = torch.tensor(action_mask if action_mask is not None else [1.0] * n)
    om = torch.tensor(obs_mask if obs_mask is not None else [0.0] * n)
    return LossContext(pi=pi, mu=mu, advantages=adv, action_mask=am, obs_mask=om, aggregate=aggregate, params=params)


def _alg(**kw):
    base = {"adv_estimator": "grpo"}
    base.update(kw)
    return AlgorithmConfig.from_config(OmegaConf.create(base))


# --------------------------------------------------------------------------- registry / api
def test_builtins_registered():
    for name in ("ppo_clip", "dppo_tv", "dppo_kl", "ppo_clip_env"):
        assert name in RLLM_LOSS_REGISTRY and is_custom_loss(name)
    assert not is_custom_loss("vanilla")  # verl-native
    assert not is_custom_loss(None)


def test_top_level_decorator_exposed():
    import rllm

    assert rllm.register_loss is register_loss
    assert rllm.LossContext is LossContext

    @rllm.register_loss("_test_top_level")
    def _t(ctx):
        return ctx.aggregate(-ctx.pi, ctx.action_mask), {}

    assert get_loss("_test_top_level") is _t


# --------------------------------------------------------------------------- DPPO math
def test_dppo_tv_matches_verl_formula():
    torch.manual_seed(0)
    n = 64
    pi = (torch.rand(n) * -2).requires_grad_(True)
    mu = torch.rand(n) * -2
    adv = torch.randn(n)
    delta = 0.2
    ctx = LossContext(pi=pi, mu=mu, advantages=adv, action_mask=torch.ones(n), obs_mask=torch.zeros(n), aggregate=_agg_sum, params={"delta": delta})
    ours, _ = dppo_tv(ctx)  # = sum over all tokens of the per-token pg

    # verl reference (core_algos.compute_policy_loss_dppo_tv), summed.
    ratio = torch.exp(torch.clamp(pi.detach() - mu, -20.0, 20.0))
    tr = torch.clamp(ratio, max=20.0).detach()
    valid = torch.where(adv > 0, (pi.detach().exp() - mu.exp()) <= delta, (pi.detach().exp() - mu.exp()) >= -delta).float()
    verl = (-adv * tr * pi.detach() * valid).sum()
    assert torch.allclose(ours.detach(), verl, atol=1e-5)


def test_dppo_tv_gradient_masked_tokens_get_no_grad():
    probs_pi = [0.90, 0.52]  # token0 far above mu -> masked; token1 within delta -> kept
    pi = torch.tensor([float(torch.tensor(p).log()) for p in probs_pi], requires_grad=True)
    mu = torch.tensor([float(torch.tensor(0.5).log())] * 2)
    adv = torch.tensor([1.0, 1.0])
    ctx = LossContext(pi=pi, mu=mu, advantages=adv, action_mask=torch.ones(2), obs_mask=torch.zeros(2), aggregate=_agg_sum, params={"delta": 0.2})
    loss, _ = dppo_tv(ctx)
    loss.backward()
    assert pi.grad[0].item() == 0.0
    assert pi.grad[1].item() != 0.0


# --------------------------------------------------------------------------- ECHO as one loss function
def test_ppo_clip_env_zero_coef_equals_ppo_clip():
    args = dict(pi=[-0.5, -0.6, -0.7], mu=[-0.5, -0.6, -0.7], adv=[1.0, 1.0, 0.0], action_mask=[1.0, 1.0, 0.0], obs_mask=[0.0, 0.0, 1.0])
    base, _ = ppo_clip(_ctx(**args, eps_clip=0.2))
    same, _ = ppo_clip_env(_ctx(**args, eps_clip=0.2, env_loss_coef=0.0))
    assert torch.allclose(base, same)


def test_ppo_clip_env_adds_observation_ce():
    # obs token (idx2) is non-action; ECHO must put gradient on it, ppo_clip must not.
    args = dict(mu=[-0.5, -0.6, -0.7], adv=[1.0, 1.0, 0.0], action_mask=[1.0, 1.0, 0.0], obs_mask=[0.0, 0.0, 1.0])
    ctx_echo = _ctx(pi=[-0.5, -0.6, -0.7], **args, eps_clip=0.2, env_loss_coef=0.5)
    loss, metrics = ppo_clip_env(ctx_echo)
    loss.backward()
    assert ctx_echo.pi.grad[2].item() != 0.0  # ECHO trains the observation token
    assert metrics["echo/coef"] == 0.5

    ctx_pg = _ctx(pi=[-0.5, -0.6, -0.7], **args, eps_clip=0.2)
    ppo_clip(ctx_pg)[0].backward()
    assert ctx_pg.pi.grad[2].item() == 0.0  # plain PPO never touches it


# --------------------------------------------------------------------------- config resolution (single selector)
def test_resolve_loss_custom_vs_native():
    r = resolve_loss(_alg(loss_fn="dppo_tv", eps_clip=0.3))
    assert r is not None and r.name == "dppo_tv" and r.fn is get_loss("dppo_tv")
    assert r.params["eps_clip"] == 0.3
    assert resolve_loss(_alg(loss_fn="vanilla")) is None  # verl-native
    assert resolve_loss(_alg(loss_fn=None)) is None


def test_loss_params_merged():
    r = resolve_loss(_alg(loss_fn="dppo_tv", loss_params={"delta": 0.15}))
    assert r.params["delta"] == 0.15


def test_echo_estimator_defaults_to_ppo_clip_env():
    alg = _alg(adv_estimator="echo")
    assert alg.loss_fn == "ppo_clip_env" and alg.env_loss_coef == 0.05
    r = resolve_loss(alg)
    assert r.name == "ppo_clip_env" and r.params["env_loss_coef"] == 0.05


# --------------------------------------------------------------------------- managed adapter (forward_backward_custom)
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


def test_managed_closure_runs_single_loss_and_backprops():
    pytest.importorskip("tinker")
    from rllm.trainer.tinker.custom_loss import build_custom_loss

    mu = [float(torch.tensor(0.5).log())] * 2
    d = _make_datum(target=[2, 3], logprobs=mu, adv=[1.0, 1.0], mask=[1.0, 1.0])
    resolved = ResolvedLoss(name="dppo_tv", fn=get_loss("dppo_tv"), params={"delta": 0.2})
    stripped, loss_fn = build_custom_loss(resolved, [d])
    assert set(stripped[0].loss_fn_inputs.keys()) == {"target_tokens"}

    pi = torch.tensor([float(torch.tensor(0.95).log()), float(torch.tensor(0.52).log())], requires_grad=True)
    loss, metrics = loss_fn(stripped, [pi])
    assert loss.dim() == 0
    loss.backward()
    assert pi.grad[0].item() == 0.0  # masked (moved far, adv>0)
    assert pi.grad[1].item() != 0.0  # kept
    assert metrics["custom_loss/num_datums"] == 1.0


def test_managed_ppo_clip_env_trains_observation_tokens():
    pytest.importorskip("tinker")
    from rllm.trainer.tinker.custom_loss import build_custom_loss

    mu = [0.0, -0.5, -0.5, -0.5]
    d = _make_datum(target=[2, 3, 4, 5], logprobs=mu, adv=[0.0, 1.0, 1.0, 0.0], mask=[0.0, 1.0, 1.0, 0.0])  # idx0,3 observation
    resolved = ResolvedLoss(name="ppo_clip_env", fn=get_loss("ppo_clip_env"), params={"eps_clip": 0.2, "env_loss_coef": 0.5})
    stripped, loss_fn = build_custom_loss(resolved, [d])
    pi = torch.tensor([-0.4, -0.5, -0.6, -0.7], requires_grad=True)
    loss, _ = loss_fn(stripped, [pi])
    loss.backward()
    assert pi.grad[0].item() != 0.0 and pi.grad[3].item() != 0.0  # ECHO trains observation tokens

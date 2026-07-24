"""Unit tests for the verl-style custom-loss abstraction (rllm.trainer.algorithms.loss).

A loss is a single function selected by name that returns a scalar via ``ctx.aggregate``.
Backend-agnostic: pure torch on toy tensors, no GPU / verl / tinker server needed.
"""

import math

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from rllm.trainer.algorithms.config import AlgorithmConfig  # noqa: E402
from rllm.trainer.algorithms.loss import (  # noqa: E402
    RLLM_LOSS_REGISTRY,
    LossContext,
    ResolvedLoss,
    dppo_tv,
    echo,
    get_loss,
    icepop,
    is_custom_loss,
    ppo_clip,
    register_loss,
    resolve_loss,
)


def _agg_sum(per_token, mask, mode=None):
    """Test aggregator: sum over masked tokens (lets us inspect the raw per-token loss)."""
    return (per_token * mask).sum()


def _ctx(logp_curr, logp_old, adv, action_mask=None, obs_mask=None, aggregate=_agg_sum, **params):
    logp_curr = torch.tensor(logp_curr, dtype=torch.float32, requires_grad=True)
    logp_old = torch.tensor(logp_old, dtype=torch.float32)
    adv = torch.tensor(adv, dtype=torch.float32)
    n = logp_curr.shape[-1]
    am = torch.tensor(action_mask if action_mask is not None else [1.0] * n)
    om = torch.tensor(obs_mask if obs_mask is not None else [0.0] * n)
    return LossContext(logp_curr=logp_curr, logp_old=logp_old, advantages=adv, action_mask=am, obs_mask=om, aggregate=aggregate, params=params)


def _alg(**kw):
    base = {"adv_estimator": "grpo"}
    base.update(kw)
    return AlgorithmConfig.from_config(OmegaConf.create(base))


# --------------------------------------------------------------------------- registry / api
def test_builtins_registered():
    for name in ("ppo_clip", "dppo_tv", "dppo_kl", "echo"):
        assert name in RLLM_LOSS_REGISTRY and is_custom_loss(name)
    assert not is_custom_loss("vanilla")  # verl-native
    assert not is_custom_loss(None)


def test_entry_point_discovery_on_registry_miss(monkeypatch):
    # A loss advertised by an installed package (rllm.losses entry point) is discovered
    # lazily on a registry miss — no explicit import / loss_plugins needed.
    import rllm.trainer.algorithms.loss as L

    class _EP:
        name = "_ep_demo_loss"

        def load(self):
            @L.register_loss("_ep_demo_loss")
            def _f(ctx):
                return ctx.aggregate(-ctx.logp_curr, ctx.action_mask), {}

            return _f

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [_EP()] if group == "rllm.losses" else [])
    monkeypatch.setattr(L, "_ENTRY_POINTS_LOADED", False)
    try:
        assert "_ep_demo_loss" not in L.RLLM_LOSS_REGISTRY
        assert is_custom_loss("_ep_demo_loss")  # triggers discovery
        assert get_loss("_ep_demo_loss") is not None
    finally:
        L.RLLM_LOSS_REGISTRY.pop("_ep_demo_loss", None)


def test_top_level_decorator_exposed():
    import rllm

    assert rllm.register_loss is register_loss
    assert rllm.LossContext is LossContext

    @rllm.register_loss("_test_top_level")
    def _t(ctx):
        return ctx.aggregate(-ctx.logp_curr, ctx.action_mask), {}

    assert get_loss("_test_top_level") is _t


# --------------------------------------------------------------------------- DPPO math
def test_dppo_tv_matches_verl_formula():
    torch.manual_seed(0)
    n = 64
    logp_curr = (torch.rand(n) * -2).requires_grad_(True)
    logp_old = torch.rand(n) * -2
    adv = torch.randn(n)
    # Force one kept token to a large importance ratio (logp_curr-logp_old >> 0 -> exp() >> 20) to exercise
    # DPPO's C=inf: the ratio must pass through untruncated (paper Eq. 23). Negative adv keeps it
    # unmasked. If truncation ever regresses in, `ours` would cap this token and diverge.
    logp_old.data[0] = -6.0  # logp_curr[0]-logp_old[0] in [4,6] -> ratio in ~[55,400], well above the old cap of 20
    adv.data[0] = -1.0
    delta = 0.2
    ctx = LossContext(logp_curr=logp_curr, logp_old=logp_old, advantages=adv, action_mask=torch.ones(n), obs_mask=torch.zeros(n), aggregate=_agg_sum, params={"delta": delta})
    ours, _ = dppo_tv(ctx)  # = sum over all tokens of the per-token pg

    # Reference formula (DPPO, C=inf): untruncated ratio as a detached weight on the score. The
    # +/-20 clamp on the log-ratio is pure inf protection, not variance truncation.
    ratio = torch.exp(torch.clamp(logp_curr.detach() - logp_old, -20.0, 20.0))
    tr = ratio.detach()
    valid = torch.where(adv > 0, (logp_curr.detach().exp() - logp_old.exp()) <= delta, (logp_curr.detach().exp() - logp_old.exp()) >= -delta).float()
    ref = (-adv * tr * logp_curr.detach() * valid).sum()
    assert torch.allclose(ours.detach(), ref, atol=1e-5)


def test_dppo_tv_gradient_masked_tokens_get_no_grad():
    probs_pi = [0.90, 0.52]  # token0 far above logp_old -> masked; token1 within delta -> kept
    logp_curr = torch.tensor([float(torch.tensor(p).log()) for p in probs_pi], requires_grad=True)
    logp_old = torch.tensor([float(torch.tensor(0.5).log())] * 2)
    adv = torch.tensor([1.0, 1.0])
    ctx = LossContext(logp_curr=logp_curr, logp_old=logp_old, advantages=adv, action_mask=torch.ones(2), obs_mask=torch.zeros(2), aggregate=_agg_sum, params={"delta": 0.2})
    loss, _ = dppo_tv(ctx)
    loss.backward()
    assert logp_curr.grad[0].item() == 0.0
    assert logp_curr.grad[1].item() != 0.0


# --------------------------------------------------------------------------- CISPO / GPG
def test_builtins_include_cispo_reinforce():
    for name in ("cispo", "reinforce"):
        assert name in RLLM_LOSS_REGISTRY


def test_cispo_matches_verl_formula():
    torch.manual_seed(1)
    n = 64
    logp_curr = (torch.rand(n) * -2).requires_grad_(True)
    logp_old = torch.rand(n) * -2
    adv = torch.randn(n)
    eps = 0.2
    from rllm.trainer.algorithms.loss import cispo

    ctx = LossContext(logp_curr=logp_curr, logp_old=logp_old, advantages=adv, action_mask=torch.ones(n), obs_mask=torch.zeros(n), aggregate=_agg_sum, params={"eps_clip": eps})
    ours, _ = cispo(ctx)
    # verl reference (core_algos.compute_policy_loss_cispo), summed.
    ratio = torch.exp(torch.clamp(logp_curr.detach() - logp_old, -20.0, 20.0))
    clipped_sg = torch.clamp(ratio, 1 - eps, 1 + eps).detach()
    verl = (-clipped_sg * adv * logp_curr.detach()).sum()
    assert torch.allclose(ours.detach(), verl, atol=1e-5)


def test_cispo_keeps_gradient_where_ppo_clip_drops_it():
    # token with ratio > 1+eps and adv > 0: PPO clip zeros its gradient; CISPO does not.
    from rllm.trainer.algorithms.loss import cispo

    logp_curr = lambda: torch.tensor([float(torch.tensor(0.9).log())], requires_grad=True)  # noqa: E731
    logp_old = [float(torch.tensor(0.5).log())]  # ratio = 1.8 > 1.2
    a, b = logp_curr(), logp_curr()
    ppo_clip(
        LossContext(logp_curr=a, logp_old=torch.tensor(logp_old), advantages=torch.tensor([1.0]), action_mask=torch.ones(1), obs_mask=torch.zeros(1), aggregate=_agg_sum, params={"eps_clip": 0.2})
    )[0].backward()
    cispo(LossContext(logp_curr=b, logp_old=torch.tensor(logp_old), advantages=torch.tensor([1.0]), action_mask=torch.ones(1), obs_mask=torch.zeros(1), aggregate=_agg_sum, params={"eps_clip": 0.2}))[
        0
    ].backward()
    assert a.grad[0].item() == 0.0  # PPO clip: clipped, no gradient
    assert b.grad[0].item() != 0.0  # CISPO: gradient still flows through log_prob


def test_reinforce_is_plain_policy_gradient():
    from rllm.trainer.algorithms.loss import reinforce

    logp_curr = torch.tensor([-0.5, -0.6], requires_grad=True)
    ctx = LossContext(logp_curr=logp_curr, logp_old=torch.tensor([-0.5, -0.6]), advantages=torch.tensor([2.0, -1.0]), action_mask=torch.ones(2), obs_mask=torch.zeros(2), aggregate=_agg_sum, params={})
    loss, _ = reinforce(ctx)
    assert torch.allclose(loss.detach(), (-torch.tensor([2.0, -1.0]) * logp_curr.detach()).sum())


# --------------------------------------------------------------------------- GSPO (sequence-level)
def test_seq_reduce_per_row_broadcast():
    ctx = LossContext(logp_curr=torch.zeros(2, 3), logp_old=torch.zeros(2, 3), advantages=torch.zeros(2, 3), action_mask=torch.ones(2, 3), obs_mask=torch.zeros(2, 3), aggregate=_agg_sum)
    vals = torch.tensor([[1.0, 2.0, 9.0], [3.0, 3.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    out = ctx.seq_reduce(vals, mask, "mean")
    assert out[0].tolist() == [1.5, 1.5, 1.5]  # (1+2)/2, masked token excluded, broadcast
    assert out[1].tolist() == [3.0, 3.0, 3.0]


def test_gspo_matches_verl_formula():
    from rllm.trainer.algorithms.loss import gspo

    torch.manual_seed(2)
    logp_curr = (torch.rand(2, 4) * -1).requires_grad_(True)
    logp_old = torch.rand(2, 4) * -1
    adv = torch.randn(2, 1).expand(2, 4).contiguous()  # GRPO: one advantage per sequence
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    eps = 0.2
    ctx = LossContext(logp_curr=logp_curr, logp_old=logp_old, advantages=adv, action_mask=mask, obs_mask=torch.zeros(2, 4), aggregate=_agg_sum, params={"eps_clip": eps})
    ours, _ = gspo(ctx)

    # verl reference (core_algos.compute_policy_loss_gspo), summed over masked tokens.
    seq_len = mask.sum(-1).clamp(min=1)
    neg_kl_seq = ((logp_curr.detach() - logp_old) * mask).sum(-1) / seq_len
    log_s = torch.clamp(logp_curr.detach() - logp_curr.detach() + neg_kl_seq.detach().unsqueeze(-1), max=10.0)
    s = torch.exp(log_s)
    pg = torch.maximum(-adv * s, -adv * torch.clamp(s, 1 - eps, 1 + eps))
    ref = (pg * mask).sum()
    assert torch.allclose(ours.detach(), ref, atol=1e-5)


def test_gspo_gradient_flows_per_token():
    from rllm.trainer.algorithms.loss import gspo

    logp_curr = torch.tensor([[-0.5, -0.6, -0.7]], requires_grad=True)
    logp_old = torch.tensor([[-0.4, -0.5, -0.6]])
    adv = torch.tensor([[1.0, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    ctx = LossContext(logp_curr=logp_curr, logp_old=logp_old, advantages=adv, action_mask=mask, obs_mask=torch.zeros(1, 3), aggregate=_agg_sum, params={"eps_clip": 0.2})
    gspo(ctx)[0].backward()
    assert logp_curr.grad is not None and (logp_curr.grad != 0).all()  # every token in the sequence gets a gradient


# --------------------------------------------------------------------------- IcePop discrepancy mask
def test_icepop_masks_out_of_band_and_reports_ratio():
    """d = exp(logp_curr - logp_rollout) per token = [0.3, 1.0, 10.0]; band [0.5, 5.0] keeps only
    the middle. Masked tokens get no gradient; ratio min/max/mean reflect the raw batch ratios."""
    import math

    logp_curr = torch.tensor([-0.5, -0.5, -0.5], requires_grad=True)
    # rollout = logp_curr - ln(d) so exp(logp_curr - rollout) hits the target ratios exactly.
    rollout = torch.tensor([-0.5 - math.log(0.3), -0.5 - math.log(1.0), -0.5 - math.log(10.0)])
    ctx = LossContext(
        logp_curr=logp_curr,
        logp_old=rollout,
        logp_rollout=rollout,
        advantages=torch.tensor([1.0, 1.0, 1.0]),
        action_mask=torch.ones(3),
        obs_mask=torch.zeros(3),
        aggregate=_agg_sum,
        params={"icepop_alpha": 0.5, "icepop_beta": 5.0},
    )
    loss, m = icepop(ctx)
    loss.backward()

    assert m["icepop/mask_frac"] == pytest.approx(2 / 3)  # tokens 0 (d<α) and 2 (d>β) masked
    assert m["icepop/ratio_min"] == pytest.approx(0.3, abs=1e-4)
    assert m["icepop/ratio_max"] == pytest.approx(10.0, abs=1e-4)
    assert m["icepop/ratio_mean"] == pytest.approx((0.3 + 1.0 + 10.0) / 3, abs=1e-4)
    assert logp_curr.grad[0].item() == 0.0 and logp_curr.grad[2].item() == 0.0  # out-of-band → no grad
    assert logp_curr.grad[1].item() != 0.0  # in-band token trains


def test_icepop_requires_logp_rollout():
    """No silent fall back to logp_old (the proximal): a missing rollout log-prob is a loud error."""
    ctx = LossContext(
        logp_curr=torch.tensor([-0.5]),
        logp_old=torch.tensor([-0.5]),
        logp_rollout=None,
        advantages=torch.tensor([1.0]),
        action_mask=torch.ones(1),
        obs_mask=torch.zeros(1),
        aggregate=_agg_sum,
        params={},
    )
    with pytest.raises(ValueError, match="rollout"):
        icepop(ctx)


# --------------------------------------------------------------------------- REINFORCE + KL-to-sampler
def test_reinforce_kl_matches_doc_formula_approx():
    """kl_approx=True: L = -sg[r]·A·logp + bwd·(-log r + r - 1) + fwd·(r·log r - r + 1), with
    r = p/q against the SAMPLER (logp_rollout) — logp_old (the proximal) is ignored entirely."""
    from rllm.trainer.algorithms.loss import reinforce_kl

    torch.manual_seed(4)
    n = 16
    logp_curr = (torch.rand(n) * -2).requires_grad_(True)
    logp_rollout = torch.rand(n) * -2
    adv = torch.randn(n)
    ctx = LossContext(
        logp_curr=logp_curr,
        logp_old=torch.zeros(n),  # garbage on purpose: q is the sampler, not the proximal
        logp_rollout=logp_rollout,
        advantages=adv,
        action_mask=torch.ones(n),
        obs_mask=torch.zeros(n),
        aggregate=_agg_sum,
        params={"fwd_kl_coef": 0.3, "bwd_kl_coef": 0.7, "kl_approx": True},
    )
    ours, m = reinforce_kl(ctx)

    log_r = logp_curr.detach() - logp_rollout
    r = torch.exp(torch.clamp(log_r, -20.0, 20.0))
    pg = -r * adv * logp_curr.detach()
    kl_bwd = -log_r + r - 1.0
    kl_fwd = r * log_r - r + 1.0
    ref = (pg + 0.7 * kl_bwd + 0.3 * kl_fwd).sum()
    assert torch.allclose(ours.detach(), ref, atol=1e-5)
    # metrics: masked means of the per-token estimators, each sample-wise non-negative
    assert m["reinforce_kl/kl_bwd"] == pytest.approx(kl_bwd.mean().item(), abs=1e-5)
    assert m["reinforce_kl/kl_fwd"] == pytest.approx(kl_fwd.mean().item(), abs=1e-5)
    assert m["reinforce_kl/ratio_mean"] == pytest.approx(r.mean().item(), abs=1e-5)
    assert m["reinforce_kl/kl_bwd"] >= 0.0 and m["reinforce_kl/kl_fwd"] >= 0.0


def test_reinforce_kl_default_kl_is_exact_gradient_form():
    """kl_approx defaults to False: backward KL is plain CE on the sampled tokens (-logp,
    grad = -1 — the exact ∇KL(q‖p) under q), NOT the k3 form's (r-1). Forward KL is
    sg[r·log r]·logp — same gradient as the approx form (r·log r) but a different loss value.
    Metrics always report the non-negative estimator values regardless of the knob."""
    from rllm.trainer.algorithms.loss import reinforce_kl

    logp_rollout = torch.tensor([-1.0, -0.5])

    def _make(logp_curr, **params):
        return LossContext(
            logp_curr=logp_curr, logp_old=torch.zeros(2), logp_rollout=logp_rollout, advantages=torch.zeros(2), action_mask=torch.ones(2), obs_mask=torch.zeros(2), aggregate=_agg_sum, params=params
        )

    log_r = torch.tensor([0.5, -0.5])  # logp_curr - logp_rollout below
    r = log_r.exp()

    # backward, default (exact): grad = -coef per token, independent of r
    bwd = torch.tensor([-0.5, -1.0], requires_grad=True)
    loss_bwd, m = reinforce_kl(_make(bwd, bwd_kl_coef=1.0, fwd_kl_coef=0.0))
    loss_bwd.backward()
    assert torch.allclose(bwd.grad, torch.tensor([-1.0, -1.0]), atol=1e-5)
    # default == explicit kl_approx=False; != the approx variant
    bwd2 = torch.tensor([-0.5, -1.0], requires_grad=True)
    explicit, _ = reinforce_kl(_make(bwd2, bwd_kl_coef=1.0, fwd_kl_coef=0.0, kl_approx=False))
    assert torch.allclose(loss_bwd.detach(), explicit.detach())
    approx, _ = reinforce_kl(_make(torch.tensor([-0.5, -1.0], requires_grad=True), bwd_kl_coef=1.0, fwd_kl_coef=0.0, kl_approx=True))
    assert not torch.allclose(loss_bwd.detach(), approx.detach())

    # forward, default (exact): grad = r·log r (identical to approx), value differs from approx
    fwd = torch.tensor([-0.5, -1.0], requires_grad=True)
    loss_fwd, _ = reinforce_kl(_make(fwd, bwd_kl_coef=0.0, fwd_kl_coef=1.0))
    loss_fwd.backward()
    assert torch.allclose(fwd.grad, r * log_r, atol=1e-5)
    approx_fwd, _ = reinforce_kl(_make(torch.tensor([-0.5, -1.0], requires_grad=True), bwd_kl_coef=0.0, fwd_kl_coef=1.0, kl_approx=True))
    assert not torch.allclose(loss_fwd.detach(), approx_fwd.detach())

    # metrics stay the estimator values (comparable across the knob)
    assert m["reinforce_kl/kl_bwd"] == pytest.approx((-log_r + r - 1.0).mean().item(), abs=1e-5)
    assert m["reinforce_kl/kl_fwd"] == pytest.approx((r * log_r - r + 1.0).mean().item(), abs=1e-5)


def test_reinforce_kl_default_coefs_are_zero():
    """Both KL coefficients default to 0.0: the default loss is pure IS-weighted REINFORCE,
    and a KL term only activates when its coef is set explicitly (e.g. bwd_kl_coef=0.0025)."""
    from rllm.trainer.algorithms.loss import reinforce_kl

    def _make(**params):
        return LossContext(
            logp_curr=torch.tensor([-0.5, -1.2], requires_grad=True),
            logp_old=torch.zeros(2),
            logp_rollout=torch.tensor([-0.9, -0.4]),
            advantages=torch.tensor([1.0, -1.0]),
            action_mask=torch.ones(2),
            obs_mask=torch.zeros(2),
            aggregate=_agg_sum,
            params=params,
        )

    default_loss, _ = reinforce_kl(_make())
    pg_only, _ = reinforce_kl(_make(fwd_kl_coef=0.0, bwd_kl_coef=0.0))
    with_bwd, _ = reinforce_kl(_make(bwd_kl_coef=0.0025))
    assert torch.allclose(default_loss, pg_only)
    assert not torch.allclose(default_loss, with_bwd)


def test_reinforce_kl_keeps_gradient_where_ppo_clip_drops_it():
    """The doc's REINFORCE-vs-GRPO/PPO contrast: at ratio 1.8 (>1+eps) with adv>0 PPO clip
    zeros the gradient; here the token still trains, weighted by the full detached IS ratio."""
    from rllm.trainer.algorithms.loss import reinforce_kl

    logp_q = [float(torch.tensor(0.5).log())]
    a = torch.tensor([float(torch.tensor(0.9).log())], requires_grad=True)
    ppo_clip(LossContext(logp_curr=a, logp_old=torch.tensor(logp_q), advantages=torch.tensor([1.0]), action_mask=torch.ones(1), obs_mask=torch.zeros(1), aggregate=_agg_sum, params={"eps_clip": 0.2}))[
        0
    ].backward()

    b = torch.tensor([float(torch.tensor(0.9).log())], requires_grad=True)
    reinforce_kl(
        LossContext(
            logp_curr=b,
            logp_old=torch.zeros(1),
            logp_rollout=torch.tensor(logp_q),
            advantages=torch.tensor([1.0]),
            action_mask=torch.ones(1),
            obs_mask=torch.zeros(1),
            aggregate=_agg_sum,
            params={"fwd_kl_coef": 0.0, "bwd_kl_coef": 0.0},
        )
    )[0].backward()

    assert a.grad[0].item() == 0.0  # PPO clip: out-of-band, no gradient
    assert b.grad[0].item() == pytest.approx(-1.8, abs=1e-4)  # -sg[r]·A: unclipped weight, grad via log p only


def test_reinforce_kl_kl_gradients_match_analytic_forms():
    """kl_approx=True: the estimator terms must keep the ratio differentiable — with A=0,
    backward KL's gradient is (r-1)·∇logp and forward KL's is r·log r·∇logp, both pulling p
    toward the sampler q. (Detaching r inside them would break this and de-regularize.)"""
    from rllm.trainer.algorithms.loss import reinforce_kl

    logp_rollout = torch.tensor([-1.0, -0.5])

    def _make(logp_curr, **params):
        return LossContext(
            logp_curr=logp_curr, logp_old=torch.zeros(2), logp_rollout=logp_rollout, advantages=torch.zeros(2), action_mask=torch.ones(2), obs_mask=torch.zeros(2), aggregate=_agg_sum, params=params
        )

    log_r = torch.tensor([0.5, -0.5])  # logp_curr - logp_rollout below
    r = log_r.exp()

    bwd = torch.tensor([-0.5, -1.0], requires_grad=True)
    reinforce_kl(_make(bwd, bwd_kl_coef=1.0, fwd_kl_coef=0.0, kl_approx=True))[0].backward()
    assert torch.allclose(bwd.grad, r - 1.0, atol=1e-5)

    fwd = torch.tensor([-0.5, -1.0], requires_grad=True)
    reinforce_kl(_make(fwd, bwd_kl_coef=0.0, fwd_kl_coef=1.0, kl_approx=True))[0].backward()
    assert torch.allclose(fwd.grad, r * log_r, atol=1e-5)


def test_reinforce_kl_ratio_clamp_bounds_pg_weight():
    """ratio_clamp_min/max bound the detached PG weight CISPO-style: with adv=1 and KL off,
    grad = -clamp(r) — clamped tokens still train (unlike a mask). Default None = raw ratio."""
    import math

    from rllm.trainer.algorithms.loss import reinforce_kl

    # rollout = logp_curr - ln(d) so r = exp(logp_curr - rollout) hits [0.2, 5.0] exactly.
    rollout = torch.tensor([-0.5 - math.log(0.2), -0.5 - math.log(5.0)])

    def _run(**params):
        logp_curr = torch.tensor([-0.5, -0.5], requires_grad=True)
        ctx = LossContext(
            logp_curr=logp_curr,
            logp_old=torch.zeros(2),
            logp_rollout=rollout,
            advantages=torch.tensor([1.0, 1.0]),
            action_mask=torch.ones(2),
            obs_mask=torch.zeros(2),
            aggregate=_agg_sum,
            params={"fwd_kl_coef": 0.0, "bwd_kl_coef": 0.0, **params},
        )
        loss, m = reinforce_kl(ctx)
        loss.backward()
        return logp_curr.grad, m

    grad, m = _run(ratio_clamp_min=0.5, ratio_clamp_max=2.0)
    assert torch.allclose(grad, torch.tensor([-0.5, -2.0]), atol=1e-4)  # both bounded, both still train
    assert m["reinforce_kl/clamp_frac"] == pytest.approx(1.0)

    grad, m = _run()  # default None/None: raw detached ratio, faithful to the doc
    assert torch.allclose(grad, torch.tensor([-0.2, -5.0]), atol=1e-4)
    assert m["reinforce_kl/clamp_frac"] == 0.0

    grad, m = _run(ratio_clamp_max=2.0)  # one-sided: min stays unbounded
    assert torch.allclose(grad, torch.tensor([-0.2, -2.0]), atol=1e-4)
    assert m["reinforce_kl/clamp_frac"] == pytest.approx(0.5)


def test_reinforce_kl_ratio_clamp_leaves_kl_terms_alone():
    """The clamp applies to the PG weight only: the KL estimators must keep the TRUE r, or a
    tight clamp would silently weaken the regularizer. With A=0 and a clamp band around 1,
    backward KL's gradient must still be (r-1) with the unclamped r."""
    from rllm.trainer.algorithms.loss import reinforce_kl

    logp_curr = torch.tensor([-0.5, -1.0], requires_grad=True)
    ctx = LossContext(
        logp_curr=logp_curr,
        logp_old=torch.zeros(2),
        logp_rollout=torch.tensor([-1.0, -0.5]),  # log r = [0.5, -0.5]
        advantages=torch.zeros(2),
        action_mask=torch.ones(2),
        obs_mask=torch.zeros(2),
        aggregate=_agg_sum,
        params={"bwd_kl_coef": 1.0, "fwd_kl_coef": 0.0, "ratio_clamp_min": 0.9, "ratio_clamp_max": 1.1, "kl_approx": True},
    )
    reinforce_kl(ctx)[0].backward()
    r = torch.tensor([0.5, -0.5]).exp()
    assert torch.allclose(logp_curr.grad, r - 1.0, atol=1e-5)  # NOT clamp(r) - 1 = [0.1, -0.1]


def test_reinforce_kl_requires_logp_rollout():
    """q is the sampler distribution: a missing rollout log-prob is a loud error, not a
    silent fallback to logp_old (same contract as icepop)."""
    from rllm.trainer.algorithms.loss import reinforce_kl

    ctx = LossContext(
        logp_curr=torch.tensor([-0.5]),
        logp_old=torch.tensor([-0.5]),
        logp_rollout=None,
        advantages=torch.tensor([1.0]),
        action_mask=torch.ones(1),
        obs_mask=torch.zeros(1),
        aggregate=_agg_sum,
        params={},
    )
    with pytest.raises(ValueError, match="rollout"):
        reinforce_kl(ctx)


# --------------------------------------------------------------------------- ECHO as one loss function
def test_echo_zero_coef_equals_ppo_clip():
    args = dict(logp_curr=[-0.5, -0.6, -0.7], logp_old=[-0.5, -0.6, -0.7], adv=[1.0, 1.0, 0.0], action_mask=[1.0, 1.0, 0.0], obs_mask=[0.0, 0.0, 1.0])
    base, _ = ppo_clip(_ctx(**args, eps_clip=0.2))
    same, _ = echo(_ctx(**args, eps_clip=0.2, env_loss_coef=0.0))
    assert torch.allclose(base, same)


def test_echo_coef_defaults_to_paper_value():
    """env_loss_coef is a normal loss_params entry: absent from params → echo uses 0.05."""
    args = dict(logp_curr=[-0.5, -0.6, -0.7], logp_old=[-0.5, -0.6, -0.7], adv=[1.0, 1.0, 0.0], action_mask=[1.0, 1.0, 0.0], obs_mask=[0.0, 0.0, 1.0])
    default_loss, metrics = echo(_ctx(**args, eps_clip=0.2))  # no env_loss_coef in params
    explicit_loss, _ = echo(_ctx(**args, eps_clip=0.2, env_loss_coef=0.05))
    assert metrics["echo/coef"] == 0.05
    assert torch.allclose(default_loss, explicit_loss)


def test_echo_adds_observation_ce():
    # obs token (idx2) is non-action; ECHO must put gradient on it, ppo_clip must not.
    args = dict(logp_old=[-0.5, -0.6, -0.7], adv=[1.0, 1.0, 0.0], action_mask=[1.0, 1.0, 0.0], obs_mask=[0.0, 0.0, 1.0])
    ctx_echo = _ctx(logp_curr=[-0.5, -0.6, -0.7], **args, eps_clip=0.2, env_loss_coef=0.5)
    loss, metrics = echo(ctx_echo)
    loss.backward()
    assert ctx_echo.logp_curr.grad[2].item() != 0.0  # ECHO trains the observation token
    assert metrics["echo/coef"] == 0.5

    ctx_pg = _ctx(logp_curr=[-0.5, -0.6, -0.7], **args, eps_clip=0.2)
    ppo_clip(ctx_pg)[0].backward()
    assert ctx_pg.logp_curr.grad[2].item() == 0.0  # plain PPO never touches it


# --------------------------------------------------------------------------- config resolution (single selector)
def test_resolve_loss_custom_vs_native():
    r = resolve_loss(_alg(loss_fn="dppo_tv", eps_clip=0.3))
    assert r is not None and r.name == "dppo_tv" and r.fn is get_loss("dppo_tv")
    assert r.params["eps_clip"] == 0.3
    assert resolve_loss(_alg(loss_fn="vanilla")) is None  # verl-native
    assert resolve_loss(_alg(loss_fn=None)) is None


def test_native_loss_names_registry():
    from rllm.trainer.algorithms.loss import native_loss_names

    # Derived from each backend's own source of truth (present in this venv: tinker, fireworks).
    assert {"cross_entropy", "importance_sampling", "ppo", "cispo", "dro"} <= native_loss_names("tinker")
    assert native_loss_names("fireworks") == {"grpo", "importance_sampling", "dapo", "dro", "gspo", "cispo"}
    # A backend not importable here (verl) or unknown → empty (→ everything uses the custom path).
    assert native_loss_names("nonexistent_backend") == set()


def test_resolve_loss_prefers_native_kernel():
    # A loss the backend has a native fused kernel for → route native (None), even though
    # it's also rLLM-registered (e.g. verl-native dppo_tv, Fireworks builtin cispo/gspo).
    assert resolve_loss(_alg(loss_fn="cispo"), native_losses={"cispo", "grpo"}) is None
    assert resolve_loss(_alg(loss_fn="gspo"), native_losses={"gspo"}) is None
    # An rLLM loss the backend can't run natively → custom path.
    r = resolve_loss(_alg(loss_fn="dppo_tv"), native_losses={"cispo", "grpo"})
    assert r is not None and r.name == "dppo_tv"
    # No native set (or empty) → every rLLM loss takes the custom path.
    assert resolve_loss(_alg(loss_fn="cispo")) is not None
    assert resolve_loss(_alg(loss_fn="cispo"), native_losses=set()) is not None


def test_loss_params_merged():
    r = resolve_loss(_alg(loss_fn="dppo_tv", loss_params={"delta": 0.15}))
    assert r.params["delta"] == 0.15


def test_echo_estimator_defaults_to_echo():
    alg = _alg(adv_estimator="echo")
    assert alg.loss_fn == "echo"  # estimator default
    r = resolve_loss(alg)
    assert r.name == "echo"
    assert "env_loss_coef" not in r.params  # not a special param; defaults in the loss body
    # env_loss_coef flows like any loss param when set:
    r2 = resolve_loss(_alg(loss_fn="echo", loss_params={"env_loss_coef": 0.1}))
    assert r2.params["env_loss_coef"] == 0.1


# --------------------------------------------------------------------------- loss_agg_mode
def test_agg_mode_resolution_default_config_and_pin():
    from rllm.trainer.algorithms.loss import DEFAULT_LOSS_AGG_MODE

    # default: no config → canonical default (token-mean)
    assert resolve_loss(_alg(loss_fn="dppo_tv")).agg_mode == DEFAULT_LOSS_AGG_MODE == "token-mean"
    # config value flows through
    assert resolve_loss(_alg(loss_fn="dppo_tv", loss_agg_mode="seq-mean-token-sum")).agg_mode == "seq-mean-token-sum"
    # a loss that PINS its mode (GSPO) overrides even an explicit config
    assert resolve_loss(_alg(loss_fn="gspo", loss_agg_mode="token-mean")).agg_mode == "seq-mean-token-mean"


def test_register_loss_rejects_bad_agg_mode():
    with pytest.raises(ValueError):
        register_loss("_bad_mode_loss", agg_mode="not-a-mode")


# --------------------------------------------------------------------------- managed adapter normalization
def test_managed_server_normalized_is_accumulation_invariant():
    """Fireworks path (server_normalized=True): the raw-sum client loss must satisfy
    sum-of-per-pass-losses == single-pass-loss, so the server's one division over the whole
    window yields the same gradient no matter how the mini-batch is split into passes."""
    pytest.importorskip("tinker")
    from rllm.trainer.tinker.custom_loss import build_custom_loss

    torch.manual_seed(3)
    logp_old = [float(torch.tensor(0.5).log())] * 2
    datums = [_make_datum(target=[2, 3], logprobs=logp_old, adv=[1.0, -1.0], mask=[1.0, 1.0]) for _ in range(4)]
    pis = [torch.tensor([float(torch.tensor(0.55).log()), float(torch.tensor(0.48).log())]) for _ in range(4)]

    for mode in ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum"):
        resolved = ResolvedLoss(name="dppo_tv", fn=get_loss("dppo_tv"), params={"delta": 0.2}, agg_mode=mode)

        # single pass over all 4 datums
        _, loss_fn_all = build_custom_loss(resolved, datums, server_normalized=True)
        one, _ = loss_fn_all(datums, [p.clone() for p in pis])

        # 4 separate passes, summed (what the server accumulates under NONE-free NUM_* norm)
        acc = 0.0
        for d, p in zip(datums, pis, strict=True):
            _, lf = build_custom_loss(resolved, [d], server_normalized=True)
            li, _ = lf([d], [p.clone()])
            acc = acc + li
        assert torch.allclose(one, acc, atol=1e-6), mode  # raw sums compose exactly


def test_managed_client_normalized_matches_agg_mode():
    """Tinker path (server_normalized=False, single pass): the client divisor follows the
    aggregation mode — token count for token-mean, sequence count for seq-mean-*."""
    pytest.importorskip("tinker")
    from rllm.trainer.tinker.custom_loss import build_custom_loss

    # 2 datums, 2 action tokens each → 4 tokens, 2 sequences. reinforce → loss_i = -adv·logp_curr summed.
    d = [_make_datum(target=[2, 3], logprobs=[0.0, 0.0], adv=[1.0, 1.0], mask=[1.0, 1.0]) for _ in range(2)]
    logp_curr = [torch.tensor([-0.5, -0.5]), torch.tensor([-0.5, -0.5])]
    raw_sum = 4 * 0.5  # Σ -adv·logp_curr over 4 tokens = 4 * (-(1.0)*(-0.5)) = 2.0

    r_tok = ResolvedLoss(name="reinforce", fn=get_loss("reinforce"), params={}, agg_mode="token-mean")
    loss_tok, _ = build_custom_loss(r_tok, d, server_normalized=False)[1](d, [p.clone() for p in logp_curr])
    assert torch.allclose(loss_tok, torch.tensor(raw_sum / 4.0))  # ÷ 4 tokens

    r_seq = ResolvedLoss(name="reinforce", fn=get_loss("reinforce"), params={}, agg_mode="seq-mean-token-mean")
    loss_seq, _ = build_custom_loss(r_seq, d, server_normalized=False)[1](d, [p.clone() for p in logp_curr])
    # seq-mean-token-mean: within-seq mean (÷2) per seq = 0.5 each, summed = 1.0, ÷ 2 seqs = 0.5
    assert torch.allclose(loss_seq, torch.tensor(0.5))


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

    logp_old = [float(torch.tensor(0.5).log())] * 2
    d = _make_datum(target=[2, 3], logprobs=logp_old, adv=[1.0, 1.0], mask=[1.0, 1.0])
    resolved = ResolvedLoss(name="dppo_tv", fn=get_loss("dppo_tv"), params={"delta": 0.2})
    stripped, loss_fn = build_custom_loss(resolved, [d])
    assert set(stripped[0].loss_fn_inputs.keys()) == {"target_tokens"}

    logp_curr = torch.tensor([float(torch.tensor(0.95).log()), float(torch.tensor(0.52).log())], requires_grad=True)
    loss, metrics = loss_fn(stripped, [logp_curr])
    assert loss.dim() == 0
    loss.backward()
    assert logp_curr.grad[0].item() == 0.0  # masked (moved far, adv>0)
    assert logp_curr.grad[1].item() != 0.0  # kept
    assert metrics["custom_loss/num_datums"] == 1.0


def test_managed_logp_rollout_populated_and_distinct_from_old():
    """logp_rollout always exposes the datum's raw sampling log-probs; with a proximal mu_arrays
    override it differs from logp_old (the non-bypass case a custom loss may need to see)."""
    pytest.importorskip("tinker")
    from rllm.trainer.tinker.custom_loss import build_custom_loss

    captured = {}

    @register_loss("_capture_rollout")
    def _capture(ctx):
        captured["rollout"] = ctx.logp_rollout.tolist()
        captured["old"] = ctx.logp_old.tolist()
        return ctx.aggregate(-ctx.advantages * ctx.logp_curr, ctx.action_mask), {}

    rollout_lp, proximal_lp = [-0.7, -0.3], [-0.1, -0.9]
    d = _make_datum(target=[2, 3], logprobs=rollout_lp, adv=[1.0, 1.0], mask=[1.0, 1.0])
    resolved = ResolvedLoss(name="_capture_rollout", fn=get_loss("_capture_rollout"), params={})
    stripped, loss_fn = build_custom_loss(resolved, [d], mu_arrays=[proximal_lp])
    loss_fn(stripped, [torch.tensor([-0.5, -0.5])])

    assert captured["rollout"] == pytest.approx(rollout_lp)  # raw inference log-probs, not the override
    assert captured["old"] == pytest.approx(proximal_lp)  # the proximal override
    assert captured["rollout"] != captured["old"]  # distinct when a proximal is supplied (non-bypass)


def test_managed_echo_trains_observation_tokens():
    pytest.importorskip("tinker")
    from rllm.trainer.tinker.custom_loss import build_custom_loss

    logp_old = [0.0, -0.5, -0.5, -0.5]
    d = _make_datum(target=[2, 3, 4, 5], logprobs=logp_old, adv=[0.0, 1.0, 1.0, 0.0], mask=[0.0, 1.0, 1.0, 0.0])  # idx0,3 observation
    resolved = ResolvedLoss(name="echo", fn=get_loss("echo"), params={"eps_clip": 0.2, "env_loss_coef": 0.5})
    stripped, loss_fn = build_custom_loss(resolved, [d])
    logp_curr = torch.tensor([-0.4, -0.5, -0.6, -0.7], requires_grad=True)
    loss, _ = loss_fn(stripped, [logp_curr])
    loss.backward()
    assert logp_curr.grad[0].item() != 0.0 and logp_curr.grad[3].item() != 0.0  # ECHO trains observation tokens


def test_reinforce_kl_reports_ratio_extremes():
    from rllm.trainer.algorithms.loss import reinforce_kl

    def _rk_ctx(curr, rollout, action_mask):
        n = len(curr)
        return LossContext(
            logp_curr=torch.tensor(curr, requires_grad=True),
            logp_old=torch.zeros(n),  # ignored: q is the sampler
            logp_rollout=torch.tensor(rollout),
            advantages=torch.ones(n),
            action_mask=torch.tensor(action_mask),
            obs_mask=torch.zeros(n),
            aggregate=_agg_sum,
            params={},
        )

    # token 0: curr << rollout (ratio e^-2, sampler-overconfident); token 2: curr >> rollout (e^1)
    _, m = reinforce_kl(_rk_ctx([-3.0, -1.0, -0.5], [-1.0, -1.0, -1.5], [1.0, 1.0, 1.0]))
    assert m["reinforce_kl/ratio_min"] == pytest.approx(math.exp(-2.0), abs=1e-5)
    assert m["reinforce_kl/ratio_max"] == pytest.approx(math.exp(1.0), abs=1e-5)

    # masked token must not contribute to the extremes
    _, m = reinforce_kl(_rk_ctx([-9.0, -1.0], [-1.0, -1.0], [0.0, 1.0]))
    assert m["reinforce_kl/ratio_min"] == pytest.approx(1.0, abs=1e-5)
    assert m["reinforce_kl/ratio_max"] == pytest.approx(1.0, abs=1e-5)


def test_offpolicy_metrics_report_both_ratio_tails():
    from rllm.trainer.algorithms.loss import offpolicy_metrics

    curr = [torch.tensor([-5.0, -1.0, -1.0])]
    rollout = [torch.tensor([-1.0, -1.0, -2.0])]
    masks = [torch.ones(3)]
    m = offpolicy_metrics(curr, rollout, masks)
    assert m["offpolicy/ratio/min"] == pytest.approx(math.exp(-4.0), abs=1e-6)
    assert m["offpolicy/ratio/max"] == pytest.approx(math.exp(1.0), abs=1e-6)

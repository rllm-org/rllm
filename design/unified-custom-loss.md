# Custom losses (verl-style, single selector)

One way to define a policy loss that runs on all three backends (verl / tinker / fireworks),
modeled on verl: a **single** loss selected by name (`algorithm.loss_fn`), not a list. There
is **no** auxiliary-loss framework — a loss that wants an extra term (e.g. ECHO) adds it
inside its own body, exactly as a verl `POLICY_LOSS_REGISTRY` function would.

Motivating example: DPPO ([arXiv:2602.04879](https://arxiv.org/abs/2602.04879)) — a GRPO
variant that replaces ratio-clipping with a per-token divergence mask.

## The contract

A loss is one function that returns the **complete scalar objective** and does its own
masking + aggregation via the backend-injected `ctx.aggregate`:

```python
import rllm

@rllm.register_loss("my_dppo")                       # same style as @rllm.rollout
def my_dppo(ctx: rllm.LossContext):
    ratio = (ctx.pi - ctx.mu).exp()                  # pi: current logprobs (grad); mu: behavior
    keep  = (...).detach()
    pg    = -ctx.advantages * ratio.clamp(max=20).detach() * ctx.pi * keep
    return ctx.aggregate(pg, ctx.action_mask), {"mask_frac": ...}   # (scalar, metrics)
```

`LossContext`: `pi, mu, advantages, action_mask, obs_mask, ref, params, aggregate, backend`.
`ctx.aggregate(per_token, mask) -> scalar` is supplied by each backend (verl: `agg_loss`
with global-batch normalization; managed: seq-mean-token-mean), so the body is
backend-agnostic. Built-ins (`rllm/trainer/algorithms/loss.py`): `ppo_clip`, `dppo_tv`,
`dppo_kl`, `ppo_clip_env`.

## Config

```yaml
algorithm:
  loss_fn: dppo_tv              # single selector → verl's policy_loss.loss_mode
  loss_params: {delta: 0.2}     # loss-specific params (verl-style) → ctx.params
  loss_plugins: ["my_pkg.losses"]   # imported at startup → fires @register_loss
  eps_clip: 0.2                 # standard params also reach ctx.params
```

A backend-native `loss_fn` (verl `vanilla`/`gspo`, tinker `ppo`, fireworks `grpo`) runs the
native kernel unchanged; an rLLM-registered name runs the rLLM loss. `resolve_loss()`
returns the single loss or None.

## ECHO (how an additive term is done now)

There is no separate aux loss. ECHO is the `ppo_clip_env` loss — PPO/GRPO plus a
length-normalized cross-entropy on observation tokens, composed **inside the loss body**:

```python
@register_loss("ppo_clip_env")
def ppo_clip_env(ctx):
    loss, m = ppo_clip(ctx)
    coef = ctx.params.get("env_loss_coef", 0.05)
    if coef:
        loss = loss + coef * ctx.aggregate(-ctx.pi, ctx.obs_mask)   # CE on observation tokens
    return loss, m
```

`adv_estimator: echo` keeps working: it uses GRPO advantages and defaults `loss_fn` to
`ppo_clip_env` with `env_loss_coef=0.05`. To add ECHO to a different surrogate, write one
loss that adds the same term (the verl way) — no config list.

## How each backend runs it

| Backend | Mechanism |
|---|---|
| **verl** | the loss runs in-process in `CustomPPOLoss._rllm_loss` over a `LossContext` (native kernel still used for non-rLLM `loss_fn`). `aggregate = agg_loss` with global-batch norm; `mu = old_log_probs`. |
| **tinker** | `forward_backward_custom` closure (`tinker/custom_loss.py`) — one pass; `mu = sampling log-probs`; `aggregate` = seq-mean-token-mean. |
| **fireworks** | same `forward_backward_custom`; `optim_step` forced to `GradAccNormalization.NONE` (the closure normalizes). |

Cost: the managed (`forward_backward_custom`) path adds a forward pass (~1.5× FLOPs, up to
~3× wall-time); verl runs in-process (cheap). The loss math itself is pointwise over
log-probs — runs on the host, fine on CPU.

## Removed

The standalone auxiliary-loss framework (`aux_loss.py`, `AuxiliaryLoss`,
`@register_aux_loss`, `EnvPredictionLoss`, `build_aux_losses`) and the `algorithm.losses` /
`algorithm.aux_losses` config are gone. ECHO migrated to `ppo_clip_env`.

## Limits / follow-ups

- `mu_source: proximal` not yet wired on the fireworks custom path (falls back to inference).
- Per-role custom losses (multi-agent) flatten to the global loss on the managed path; verl
  per-role routing is unchanged.
- End-to-end GPU/training validation of the managed and verl trainer wiring (normalization
  vs optim_step, the verl `select(...).to_padded_tensor()` extraction) is pending; the loss
  math and adapters are unit-tested.

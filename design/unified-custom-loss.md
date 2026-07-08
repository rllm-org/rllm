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
`ctx.aggregate(per_token, mask, mode=None) -> scalar` is supplied by each backend (verl:
`agg_loss` with global-batch normalization; managed: seq-mean-token-mean); `mode` overrides
the aggregation (e.g. GSPO forces `seq-mean-token-mean`). For sequence-level losses,
`ctx.seq_reduce(values, mask, reduction)` reduces per sequence (per row on verl `(B,T)`;
the whole datum on the managed path) and broadcasts back to tokens.

Built-ins (`rllm/trainer/algorithms/loss.py`): `ppo_clip` (=verl `vanilla`), `cispo`,
`gpg`, `gspo`, `dppo_tv`, `dppo_kl`, `ppo_clip_env` (ECHO). Each is verified against verl
0.8's kernel where one exists.

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

## Native-loss registry

`native_loss_names(backend)` is the central map of what each backend has a **native fused
kernel** for. It's *derived from each backend's own source of truth* (not hardcoded), so it
never drifts from the installed version:

| backend | source | native losses |
|---|---|---|
| **verl** (0.8) | `verl.trainer.ppo.core_algos.POLICY_LOSS_REGISTRY` | `vanilla`, `dppo_tv`, `dppo_kl`, `gspo`, `sapo`, `gpg`, `clip_cov`, `kl_cov`, `geo_mean`, `cispo`, `bypass_mode` |
| **tinker** | `tinker.types.LossFnType` | `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro` |
| **fireworks** | `training.utils.rl.builtin_losses.BUILTIN_LOSSES` | `grpo`, `importance_sampling`, `dapo`, `dro`, `gspo`, `cispo` |

Returns `∅` if the backend isn't importable in the current process → everything falls back to
the rLLM custom path. (verl 0.7 would return a smaller set, e.g. no `dppo_tv` — hence derivation
over hardcoding.)

## Routing: native-first

`resolve_loss(config, native_losses)` is **native-first**. For a given `loss_fn`, each backend
passes its own fused-kernel menu; if the name is in it, the backend runs its **native** kernel
(fast) — even when an rLLM loss of the same name exists. Only losses the backend can't run
natively fall back to the rLLM custom path. So the same `loss_fn` can take different paths per
backend:

| `loss_fn` | verl (native: `POLICY_LOSS_REGISTRY`) | tinker (`importance_sampling/ppo/cispo/dro/cross_entropy`) | fireworks (builtin `grpo/importance_sampling/dapo/dro/gspo/cispo`) |
|---|---|---|---|
| `cispo`, `gspo` | native | tinker: `cispo` native / `gspo` custom | native |
| `dppo_tv`, `dppo_kl` | native (verl 0.8) | custom | custom |
| `ppo_clip_env` (ECHO) | custom | custom | custom |

(A loss routed to a native kernel takes its hyperparameters from the backend-native config,
e.g. `eps_clip`→`clip_ratio`, not `loss_params`.)

## How each backend runs the custom path

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
- `geo_mean` (GMPO) is now expressible via `ctx.seq_reduce` (like `gspo`); `clip_cov`/`kl_cov`
  need batch-level covariance statistics — verl-clean but the managed per-datum closure would
  need a batch-level variant first.
- End-to-end GPU/training validation of the managed and verl trainer wiring (normalization
  vs optim_step, the verl `select(...).to_padded_tensor()` extraction) is pending; the loss
  math and adapters are unit-tested.

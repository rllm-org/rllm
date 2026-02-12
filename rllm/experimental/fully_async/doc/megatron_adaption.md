# Megatron Backend Adaptation for Fully Async Training

This document summarizes the findings, bugs, and fixes discovered while adapting the fully async RL training pipeline from FSDP to Megatron backend.

## Architecture Overview

The fully async pipeline has two independent parallel loops connected by a MessageQueue:

```
Runner
├── InferenceManager (SGLang rollout servers)
│   ├── rollout_wg (DetachAsyncRolloutWorker)
│   └── SGLang router (subprocess)
├── RolloutExecutor (dataset, generation, staleness control)
├── FullyAsyncTrainer
│   ├── actor_wg (DetachActorWorker)
│   ├── critic_wg (CriticWorker)
│   └── ref_policy_wg (DetachActorWorker)
├── ParameterSynchronizer (NCCL weight broadcast between actor ↔ rollout)
└── MessageQueue (trajectory groups from RolloutExecutor → Trainer)
```

Both FSDP and Megatron share the same rllm runner code. The backend-specific code lives in verl:
- `verl/experimental/fully_async_policy/fsdp_workers.py` — FSDP workers
- `verl/experimental/fully_async_policy/megatron_worker.py` — Megatron workers
- `verl/experimental/fully_async_policy/base_detach_sync.py` — Shared base class (NCCL sync, SGLang weight update)

Worker selection is in `verl/experimental/fully_async_policy/fully_async_main.py:create_role_worker_mapping()`, which dispatches based on `config.actor_rollout_ref.actor.strategy`.

---

## Bugs Found and Fixed

### 1. Missing `self.rollout_device_mesh` in Megatron Worker

**File:** `verl/workers/megatron_workers.py` — `_build_rollout()`

**Severity:** Critical — causes `AttributeError` on first weight sync

**Root cause:** FSDP's `_build_rollout()` stores the device mesh on `self`:
```python
# fsdp_workers.py:634
self.rollout_device_mesh = rollout_device_mesh  # ✅
```

Megatron's `_build_rollout()` only used it as a local variable:
```python
# megatron_workers.py:522
rollout_device_mesh = init_device_mesh(...)   # ❌ never stored on self
```

But `base_detach_sync.py` references `self.rollout_device_mesh` in three places:
- Line 207: `_sync_sglang_weights()` — KV cache resume after weight sync
- Line 234: `update_weights()` — SGLang weight update device mesh
- Line 237: `update_weights()` — cache flush coordination

**Fix:** Added `self.rollout_device_mesh = rollout_device_mesh` after `init_device_mesh()`.

---

### 2. SGLang Router Race Condition (ConnectError)

**File:** `rllm/experimental/fully_async/inference_manager.py` — `launch_router()`

**Severity:** High — intermittent crash on startup, worse with Megatron

**Root cause:** `launch_router()` used `subprocess.Popen()` and returned immediately without waiting for the router to become healthy. The first `sync_weights()` call (via `rollout_executor.pause()` → `abort_async(router_url)`) would hit an `httpx.ConnectError` because the router wasn't ready yet.

**Why it's worse with Megatron:** Megatron worker initialization takes significantly longer than FSDP (TransformerEngine kernel compilation, `mpu.initialize_model_parallel()`, distributed optimizer setup), so the time window between `launch_router()` returning and the first `sync_weights()` call is shorter.

**Fix (two-part):**
1. Added health check loop to `launch_router()` — polls `GET {router_url}/health` every 2s until 200 response (120s timeout). Detects crashed router process.
2. Added retry logic to `get()` in `utils.py` — 10 retries, 2s interval (defense in depth).
3. Added cleanup of previous router process before launching a new one (prevents Prometheus port conflict).

---

### 3. `ppo_mini_batch_size` Assertion Failure

**File:** `verl/workers/actor/megatron_actor.py` — `make_minibatch_iterator()`

**Error:** `AssertionError: 540 % 8192 != 0`

**Severity:** Critical — always crashes during training

**Root cause:** Fundamental difference in how FSDP and Megatron handle mini-batching:

| | FSDP (`dp_actor.py`) | Megatron (`megatron_actor.py`) |
|---|---|---|
| **Mini-batch split** | `data.split(65536)` → always 1 batch (no-op) | `data.make_iterator(ppo_mini_batch_size)` → asserts divisibility |
| **Micro-batch split** | `prepare_dynamic_batch(max_token_len)` | `rearrange_micro_batches(max_token_len)` |
| **`ppo_mini_batch_size` role** | Dead code when `use_dynamic_bsz=True` | Always enforced as mini-batch size |

In fully async training, the batch size is variable (determined by `required_samples` × responses per prompt, minus filtered ones, sharded across workers). This variable batch size is almost never divisible by the configured `ppo_mini_batch_size`.

**Fix:** Changed `make_minibatch_iterator()` to cap `mini_batch_size` at the actual batch size:
```python
actual_batch_size = data.batch.batch_size[0]
mini_batch_size = min(self.config.ppo_mini_batch_size, actual_batch_size)
```

This ensures exactly one mini-batch when batch size < `ppo_mini_batch_size` (async case), while preserving original behavior for synchronous training where batch size > `ppo_mini_batch_size`.

---

### 4. Empty `response_mask` Guard

**File:** `verl/workers/actor/megatron_actor.py` — `loss_func()` inside `forward_backward_batch()`

**Severity:** Medium — potential NaN/crash with dynamic batching

**Root cause:** Dynamic micro-batching can produce micro-batches with all-padding response masks. Computing `rollout_corr_metrics_from_logprobs` on an empty mask causes division by zero or NaN.

The same bug existed in FSDP's `dp_actor.py` and was fixed by the `verl_dp_actor.patch`.

**Fix:** Added `and response_mask.any()` guard:
```python
# Before:
if loss_mode != "bypass_mode" and rollout_log_prob is not None:
# After:
if loss_mode != "bypass_mode" and rollout_log_prob is not None and response_mask.any():
```

---

## Loss Scaling Analysis: FSDP vs Megatron

### The Problem

When using gradient accumulation across multiple micro-batches, the loss from each micro-batch must be correctly scaled so that the aggregated gradient equals the gradient computed over the full batch.

### FSDP Approach (patched)

FSDP handles loss scaling **manually** in `dp_actor.py`:

```python
for micro_batch in dynamic_micro_batches:
    # Compute per-token loss for this micro-batch
    loss = compute_pg_loss(micro_batch)

    # Scale by fraction of total tokens (patch fix)
    micro_tokens = micro_batch.batch["response_mask"].sum()
    loss = loss * (micro_tokens / total_mini_batch_tokens)

    loss.backward()  # accumulate gradients
optimizer.step()
```

**Before the patch**, the scale factor was `response_mask.shape[0] / ppo_mini_batch_size` (sequence count / configured batch size) — broken when batch size varies.

**After the patch**, the scale factor is `micro_batch_num_tokens / mini_batch_num_tokens` — correct token-ratio scaling regardless of batch size.

### Megatron Approach

Megatron handles loss scaling **differently** through two mechanisms:

1. **Inside `loss_func`:** `agg_loss()` with `"token-mean"` mode computes:
   ```python
   loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
   ```
   When `global_batch_info` is empty (the non-fused-kernel path), `batch_num_tokens` falls back to `loss_mask.sum()` = tokens in **this micro-batch only**.

2. **Inside Megatron's pipeline engine (`forward_backward_func`):** The engine divides the returned loss by `num_microbatches` before accumulating gradients.

The effective computation is:
```
gradient = Σ_i [ mean_per_token(micro_batch_i) / num_microbatches ]
         = average of per-micro-batch token-means
```

### Correctness Analysis

The correct token-mean gradient should be:
```
correct = Σ_i [ sum_tokens(micro_batch_i) ] / total_tokens
        = token-weighted average
```

What Megatron computes:
```
actual  = Σ_i [ mean_tokens(micro_batch_i) ] / num_microbatches
        = unweighted average of per-micro-batch means
```

**These are mathematically equivalent ONLY when all micro-batches have the same number of valid tokens.** When micro-batches have unequal token counts (common with dynamic batching), the Megatron approach slightly over-weights shorter micro-batches and under-weights longer ones.

In practice, `rearrange_micro_batches()` tries to pack micro-batches to similar token budgets, so the difference is typically small. But it's technically not a perfect token-mean.

**To fix this properly**, the non-fused-kernel path's `loss_func` should populate `global_batch_info` with the total `batch_num_tokens` across all micro-batches (like the fused-kernel path in `verl/workers/utils/losses.py:103-104` already does). This would make `agg_loss` normalize by total tokens rather than per-micro-batch tokens.

---

## Configuration Changes

### Shell Config Changes (both `mega_stale02_cf.sh` and `mega_8b_s05.sh`)

| Setting | Before | After | Reason |
|---------|--------|-------|--------|
| `override_transformer_config.fp16=True` | Used for both actor and ref | Removed | `fp16` is not a valid key in the struct config |
| `actor.megatron.dtype` | Not set | `float16` | Use the existing `dtype` field instead of struct override |
| `ref.megatron.dtype` | Not set | `float16` | Same as above |
| `actor.optim.total_training_steps` | Not set (null) | `50000` | Required by Megatron's `OptimizerParamScheduler` for `lr_decay_steps` |
| `ppo_mini_batch_size` | `2048` | `128` | Config-level safety; code fix in `megatron_actor.py` also handles this |
| `--config-name` | Missing in `mega_8b_s05.sh` | `fully_async_ppo_megatron_trainer.yaml` | Was using wrong default config |

### Install Script Changes (`install_megatron_sglang.sh`)

| Component | Before | After | Reason |
|-----------|--------|-------|--------|
| TransformerEngine | `v2.6`, no cache flags | `v2.10` with `--no-cache --no-build-isolation` | ABI compatibility with PyTorch 2.9.1 |
| cuDNN | `9.10.x` (torch default) | `9.16.0.29` with `sed` workaround | Avoids cudnn9.10+torch2.9.1 bug (from stable Dockerfile) |

---

## Files Modified

### In verl (`<verl-repo-path>`)

| File | Change |
|------|--------|
| `verl/workers/megatron_workers.py` | Added `self.rollout_device_mesh = rollout_device_mesh` in `_build_rollout()` |
| `verl/workers/actor/megatron_actor.py` | 1. `make_minibatch_iterator()`: cap `mini_batch_size` at actual batch size<br>2. `loss_func()`: added `response_mask.any()` guard |

### In rllm (`<rllm-repo-path>`)

| File | Change |
|------|--------|
| `rllm/experimental/fully_async/inference_manager.py` | 1. Health check loop in `launch_router()`<br>2. Print worker URLs<br>3. Kill previous router on re-launch |
| `rllm/experimental/fully_async/utils.py` | Retry logic (10 retries, 2s interval) in `get()` |

---

## Remaining Known Issues

1. **Megatron token-mean loss scaling with dynamic batching** — Per-micro-batch token-mean averaged equally rather than weighted by token count. Low impact in practice but technically imprecise. Fix: populate `global_batch_info` in the non-fused `loss_func` path.

2. **Prometheus metrics port conflict** — `sglang_router` uses a fixed Prometheus exporter port. If a previous router didn't cleanly exit, the new one fails with "Address already in use". Mitigated by killing previous router process before launching.

3. **`launch_router()` health check timeout** — Set to 120s default. May need adjustment for very large models with slow SGLang server startup.
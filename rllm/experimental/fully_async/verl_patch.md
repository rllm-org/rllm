# Verl Patches for Fully Async Training

This document describes patches that need to be applied to the verl repository for fully async training to work correctly.

## verl_patch.patch

**Purpose:** All verl modifications required for fully async training with both FSDP and Megatron backends.

### How to Apply

```bash
cd /path/to/verl
git apply rllm/experimental/fully_async/verl_patch.patch
```

### How to Revert

```bash
cd /path/to/verl
git checkout verl/trainer/ppo/core_algos.py verl/workers/actor/dp_actor.py verl/workers/actor/megatron_actor.py verl/workers/megatron_workers.py
```

---

### File 1: `verl/trainer/ppo/core_algos.py`

**Purpose:** Prevent NaN from division by zero in `agg_loss()` when a micro-batch has no valid tokens.

With dynamic micro-batching (`rearrange_micro_batches` / `prepare_dynamic_batch`), it is possible for a micro-batch to have an all-zero `response_mask` (e.g., DP sync padding, VPP rounding, or very small async batches). Without this fix, `loss_mask.sum() = 0` causes `0 / 0 = NaN`, which corrupts gradients.

The fix uses `torch.clamp(..., min=1)` on the denominator, matching Megatron's own guard in `forward_step_calc_loss` (`output_tensor /= torch.clamp(num_tokens, min=1)`). When mask is all-zero, `masked_sum = 0 / 1 = 0` — the loss stays connected to the model output (autograd graph preserved), so all TP/DP collectives still fire during backward, producing zero gradients.

| Branch | Change |
|---|---|
| `token-mean` | `batch_num_tokens = loss_mask.sum()` → `torch.clamp(loss_mask.sum(), min=1)` |
| `seq-mean-token-sum` | `global_batch_size = seq_mask.sum()` → `torch.clamp(seq_mask.sum(), min=1)` |
| `seq-mean-token-mean` | `global_batch_size = seq_mask.sum()` → `torch.clamp(seq_mask.sum(), min=1)` |

---

### File 2: `verl/workers/actor/dp_actor.py`

**Purpose:** Modifications to `DataParallelPPOActor.update_policy()` for single mini-batch enforcement, correct token-mean loss scaling, and empty mask safety.

1. **Force single mini-batch** (line 541)
   - Changed: `data.split(self.config.ppo_mini_batch_size)` → `data.split(data.batch.batch_size[0])`
   - Added: `assert len(mini_batches) == 1`
   - Reason: In fully async training, batch size is variable and may not be divisible by `ppo_mini_batch_size`. Splitting by actual batch size guarantees exactly one mini-batch. Micro-batching within the mini-batch is handled by `prepare_dynamic_batch`.

2. **Track mini-batch token count** (line 555)
   - Added: `mini_batch_num_tokens = mini_batch.batch["response_mask"].sum().item()`
   - Reason: Needed as the global denominator for correct token-mean loss scaling across micro-batches.

3. **Token-ratio loss scaling** (lines 578-579)
   - Changed: `loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size`
   - To: `loss_scale_factor = micro_batch_num_tokens / mini_batch_num_tokens`
   - Reason: The old formula used sequence count / configured batch size, which breaks when batch size varies. The new formula uses actual token counts, so each micro-batch's contribution is weighted by its fraction of total tokens. This cancels the per-micro-batch denominator in `agg_loss(token-mean)` and replaces it with the global denominator, producing correct token-weighted gradients.

4. **Guard against empty response_mask** (line 624)
   - Added: `and response_mask.any()` to the `rollout_corr_metrics` condition
   - Reason: `compute_rollout_corr_metrics_from_logprobs` divides by mask count; all-zero mask causes NaN.

---

### File 3: `verl/workers/actor/megatron_actor.py`

**Purpose:** Modifications to `MegatronPPOActor` for variable batch size support and empty mask safety.

1. **Force single mini-batch in `make_minibatch_iterator()`** (line 385-389)
   - Changed: `mini_batch_size=self.config.ppo_mini_batch_size`
   - To: `mini_batch_size = data.batch.batch_size[0]`
   - Reason: Same as FSDP — `DataProto.make_iterator()` asserts `batch_size % mini_batch_size == 0` (protocol.py:827). With variable async batch sizes, this assertion always fails. Using actual batch size guarantees exactly one mini-batch, while micro-batching is handled by `forward_backward_batch` via `rearrange_micro_batches`.

2. **Guard against empty response_mask in `loss_func()`** (line 529)
   - Added: `and response_mask.any()` to the `rollout_corr_metrics` condition
   - Reason: Same as FSDP — prevents NaN from empty-mask micro-batches during dynamic batching.

---

### File 4: `verl/workers/megatron_workers.py`

**Purpose:** Store `rollout_device_mesh` on `self` for SGLang weight synchronization.

1. **Store device mesh in `_build_rollout()`** (line 525)
   - Added: `self.rollout_device_mesh = rollout_device_mesh`
   - Reason: `base_detach_sync.py` references `self.rollout_device_mesh` in `_sync_sglang_weights()` (line 207), `update_weights()` (lines 234, 237) for NCCL weight broadcast between actor and rollout workers. FSDP's `_build_rollout()` already stores it (`fsdp_workers.py:634`), but Megatron's only used it as a local variable. Without this fix, the first `sync_weights()` call crashes with `AttributeError`.
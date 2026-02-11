# Verl Patches for Fully Async Training

This document describes patches that need to be applied to the verl repository for fully async training to work correctly.

## dp_actor_functional_changes.patch

**File:** `verl/workers/actor/dp_actor.py`

**Purpose:** Modifications to `DataParallelPPOActor.update_actor()` for token-mean loss scaling and single mini-batch enforcement.

### Changes

1. **Force single mini-batch** (line 541)
   - Changed: `data.split(self.config.ppo_mini_batch_size)` → `data.split(65536)`
   - Added: `assert len(mini_batches) == 1`
   - Reason: Fully async training expects all data in a single mini-batch

2. **Track mini-batch token count** (line 555)
   - Added: `mini_batch_num_tokens = mini_batch.batch["response_mask"].sum().item()`
   - Reason: Needed for accurate token-mean loss scaling across micro-batches

3. **Token-mean loss scaling** (lines 578-579)
   - Changed: `loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size`
   - To: `micro_batch_num_tokens = micro_batch.batch["response_mask"].sum().item()` and `loss_scale_factor = micro_batch_num_tokens / mini_batch_num_tokens`
   - Reason: Scale loss by actual token count rather than batch size for more accurate gradient weighting

4. **Guard against empty response_mask** (line 624)
   - Added: `and response_mask.any()` to condition
   - Reason: Skip rollout_corr_metrics computation when response_mask is empty to avoid errors

### How to Apply

```bash
cd /path/to/verl
git apply /path/to/fully_async/dp_actor_functional_changes.patch
```

### How to Revert

```bash
cd /path/to/verl
git checkout verl/workers/actor/dp_actor.py
```
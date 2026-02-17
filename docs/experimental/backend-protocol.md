# Backend Protocol

> **Module**: `rllm.experimental.protocol`

The `BackendProtocol` is the abstract interface that decouples the
[Unified Trainer](unified-trainer.md) from any specific training infrastructure.
By implementing this protocol, you can plug in any model-serving, optimization, and
checkpointing system while reusing the trainer's episode generation, data
transformation, rejection sampling, and logging machinery.

---

## Class Signature

```python
class BackendProtocol(ABC, Generic[TDataset, TBatch]):
    name: str = "base_backend"
    requires_loop: bool = False

    def __init__(self, config: DictConfig, **kwargs): ...
```

The two type parameters let you declare your backend-specific types:

- `TDataset` -- the iterable type returned by `get_dataloader` (e.g. `torch.utils.data.DataLoader`)
- `TBatch` -- the batch type consumed by your pipeline methods (e.g. `list[tinker.Datum]`)

---

## What a Backend Provides

A backend implementation is responsible for four categories of functionality:

```
BackendProtocol
  |
  |-- Setup & teardown
  |     init_rollout_engine()   -- create the RolloutEngine for inference
  |     validate_config()       -- check backend-specific config
  |     get_dataloader()        -- wrap a Dataset into an iterable
  |     shutdown()              -- release resources
  |
  |-- Pipeline methods (called per batch, in order)
  |     generate_episodes()         -- stage 1: run workflows
  |     transform_to_backend_batch()-- stage 4: convert to native format
  |     process_backend_batch()     -- stage 5: forward/backward pass
  |     compute_advantages()        -- stage 6: advantage computation
  |     update_policy()             -- stage 7: optimizer step
  |
  |-- Lifecycle hooks (optional overrides)
  |     on_train_start / on_train_end
  |     on_epoch_start / on_epoch_end
  |     on_batch_start / on_batch_end
  |     on_validation_start / on_validation_end
```

---

## Setup Methods

### `init_rollout_engine(**kwargs) -> RolloutEngine`

Called once during trainer initialization. The backend must create and return a
`RolloutEngine` that the workflow engine will use for model inference. The trainer
passes the parsed config objects as keyword arguments:

```python
def init_rollout_engine(self, **kwargs) -> RolloutEngine:
    cf_config = kwargs.get("cf_config")           # CompactFilteringConfig
    transform_config = kwargs.get("transform_config")  # TransformConfig
    rs_config = kwargs.get("rs_config")            # RejectionSamplingConfig
    algorithm_config = kwargs.get("algorithm_config")  # AlgorithmConfig
    # ... create and return your engine
```

### `validate_config() -> None`

Called during trainer initialization to validate backend-specific configuration.
Raise or warn on invalid settings.

### `get_dataloader(dataset, trainer_state) -> TDataset`

Called at the start of each epoch (training) and at each validation round. Use
`trainer_state.is_training` to distinguish between training and validation and
return the appropriate dataloader.

### `shutdown() -> None`

Called when the trainer is torn down. Release GPU memory, close connections, etc.

---

## Pipeline Methods

These are called by the trainer in a fixed order during each training batch.
The `TrainerState` object is the shared mutable context throughout a batch.

### Stage 1: `generate_episodes(batch, agent_workflow_engine, is_validation) -> list[Episode]`

Produce episodes by running workflows on the input batch. A typical implementation:

1. Prepares the batch (e.g. repeat each task `group_size` times for GRPO)
2. Sets the current model on the rollout engine
3. Delegates to `agent_workflow_engine.execute_tasks(...)`

### Stage 4: `transform_to_backend_batch(trainer_state) -> TBatch`

Convert the framework's `TrajectoryGroup` objects into your backend-native format.
This is a sync method since it is typically pure data transformation.

Some backends defer transformation to `process_backend_batch` and return a
placeholder here.

### Stage 5: `process_backend_batch(trainer_state) -> None`

The main computational stage. Common operations:

- Run a forward pass to compute training logprobs
- Run a backward pass to compute gradients
- Store results in `trainer_state.backend_batch` and `trainer_state.extra_info`

This method updates `trainer_state` in place (no return value).

### Stage 6: `compute_advantages(trainer_state, algorithm_config) -> None`

Compute per-step advantages and store them on the `Step` objects within each
trajectory. The base class provides a default implementation using rLLM-native
advantage estimators (GRPO, REINFORCE):

```python
# Default implementation in BackendProtocol
async def compute_advantages(self, trainer_state, algorithm_config, **kwargs):
    adv_metrics = collect_reward_and_advantage_from_trajectory_groups(
        trainer_state.trajectory_groups, algorithm_config
    )
    trainer_state.metrics.update(adv_metrics)
```

**Pre-computed advantages:** If advantages are already set on the `Step` objects
(e.g. computed during episode generation via a workflow decorator), the default
implementation detects this and skips re-computation.

### Stage 7: `update_policy(trainer_state) -> None`

Run the optimizer step to update model weights. Some backends fuse this into
`process_backend_batch` and make `update_policy` a no-op (see
[Flexible Stage Organization](#flexible-stage-organization) below).

---

## Lifecycle Hooks

All hooks are `async def` methods with default no-op implementations. Override only
what you need.

### Training hooks

```
on_train_start(state)  -- called once before the first epoch
  |
  |  on_epoch_start(state)  -- called at the start of each epoch
  |    |
  |    |  on_batch_start(state)  -- called before each batch pipeline
  |    |  [... 8-stage pipeline ...]
  |    |  on_batch_end(state)    -- called after pipeline, before logging
  |    |
  |    |  (repeat for each batch)
  |    |
  |  on_epoch_end(state)  -- called at the end of each epoch
  |
  |  (repeat for each epoch)
  |
on_train_end(state)  -- called once after all epochs
```

### Validation hooks

```
on_validation_start(state) -> bool  -- return False to skip validation
  [... validation loop ...]
on_validation_end(state)
```

### Common uses for hooks

| Hook | Common use |
|------|------------|
| `on_train_start` | Initialize training client, load checkpoint, set initial `global_step` |
| `on_batch_end` | Save checkpoint, update sampling client, compute derived metrics, print metrics table |
| `on_train_end` | Save final checkpoint |
| `on_validation_start` | Toggle model to eval mode; return `False` to skip |
| `on_validation_end` | Toggle model back to train mode |

**Important:** `on_batch_end` runs **after** the pipeline but **before**
`logger.log(...)`. This makes it the right place to inject derived metrics
(e.g. KL divergence, learning rate) into `trainer_state.metrics`.

---

## Flexible Stage Organization

The protocol defines stages 4-7 as separate methods, but backends are free to
redistribute work across them. The trainer always calls them in the same order --
it is the backend's responsibility to decide what each stage does internally.

### Example: TinkerBackend's fused mode

The `TinkerBackend` demonstrates this flexibility. When
`fuse_forward_backward_and_optim_step` is enabled:

```
                       Default (non-fused)            Fused
                       -------------------            -----
transform_to_backend   returns [] placeholder          same
process_backend_batch  forward + backward              forward + backward + optim step
compute_advantages     stores algorithm_config         same
update_policy          optimizer step                  no-op (already done)
```

Both modes produce the same end result, but the fused path reduces round-trips
to the training server. The trainer does not need to know which path is active --
it simply calls all four methods in order.

### Example: Pre-computed advantages (OPSD)

For On-Policy Self-Distillation, advantages are computed during episode generation
(stage 1) via a workflow decorator. By the time `compute_advantages` (stage 6) runs,
every `Step` already has its `.advantage` field set. The default implementation in
`BackendProtocol` detects this and skips re-computation, collecting only metrics.

This means OPSD can work with the standard `TinkerBackend` -- no custom backend
subclass is needed.

---

## Implementing a Custom Backend

### Step 1: Subclass `BackendProtocol`

```python
from rllm.experimental.protocol import BackendProtocol

class MyBackend(BackendProtocol[MyDataLoader, MyBatch]):
    name = "my_backend"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # ... initialize your infrastructure
```

### Step 2: Implement required methods

At minimum, you must implement these abstract methods:

```python
# Setup
def init_rollout_engine(self, **kwargs) -> RolloutEngine: ...
def validate_config(self) -> None: ...
def get_dataloader(self, dataset, trainer_state) -> MyDataLoader: ...
def shutdown(self) -> None: ...

# Pipeline
async def generate_episodes(self, batch, agent_workflow_engine, is_validation=False, **kwargs) -> list[Episode]: ...
def transform_to_backend_batch(self, trainer_state, **kwargs) -> MyBatch: ...
async def process_backend_batch(self, trainer_state, **kwargs) -> None: ...
async def compute_advantages(self, trainer_state, algorithm_config, **kwargs) -> None: ...
async def update_policy(self, trainer_state, **kwargs) -> None: ...
```

### Step 3: Override lifecycle hooks as needed

```python
async def on_train_start(self, trainer_state):
    # Load checkpoint, initialize model
    ...

async def on_batch_end(self, trainer_state):
    # Save checkpoint, update metrics
    ...
```

### Step 4: Wire it up

```python
trainer = UnifiedTrainer(
    backend_cls=MyBackend,
    config=config,
    workflow_class=MyWorkflow,
    train_dataset=train_ds,
    val_dataset=val_ds,
)
trainer.fit()
```

---

## Reference: TinkerBackend

The `TinkerBackend` (`rllm.experimental.tinker.tinker_backend`) is the primary
production backend. It serves as a comprehensive reference implementation. Below
is a summary of how it implements each part of the protocol.

### Setup

| Method | Implementation |
|---|---|
| `init_rollout_engine` | Creates a `TinkerPolicyTrainer` and a `TinkerEngine` (rollout engine backed by a Tinker sampling server) |
| `validate_config` | Warns if sampling temperature/top_p deviate from 1.0 |
| `get_dataloader` | Returns a `torch.utils.data.DataLoader` with backend-specific batch sizes |
| `shutdown` | Delegates to parent (no-op currently) |

### Pipeline

| Stage | Method | Implementation |
|---|---|---|
| 1 | `generate_episodes` | Builds an interleaved batch (`N` repeats per task for GRPO grouping), sets the sampling client on the rollout engine, and calls `agent_workflow_engine.execute_tasks(...)` |
| 4 | `transform_to_backend_batch` | Returns an empty list (placeholder). The actual datum construction is deferred to stage 5 |
| 5 | `process_backend_batch` | Converts trajectory groups to Tinker `Datum` objects, runs forward-backward, stores training logprobs. Optionally fuses the optimizer step |
| 6 | `compute_advantages` | Stores the `AlgorithmConfig` for use by stage 5's datum construction (advantage computation is embedded in the forward-backward call) |
| 7 | `update_policy` | Runs the optimizer step (or no-op if fused into stage 5) |

### Lifecycle hooks

| Hook | Implementation |
|---|---|
| `on_train_start` | Initializes the training client, loads checkpoint, sets `trainer_state.global_step` from the checkpoint's batch index |
| `on_train_end` | Saves final checkpoint if not already saved |
| `on_batch_end` | Saves sampler checkpoint, updates `self.sampling_client`, injects `optim/lr` and KL/entropy metrics into `trainer_state.metrics`, prints metrics table |
| `on_epoch_start/end` | Logging only |
| `on_validation_start/end` | Toggles `trainer_state.is_training` flag |

### Key patterns to note

1. **Deferred transformation.** `transform_to_backend_batch` returns a placeholder;
   the real work happens in `process_backend_batch`. This is valid because the trainer
   only checks `trainer_state.has_backend_batch` *after* `process_backend_batch` runs.

2. **Checkpoint-driven sampling client.** The `sampling_client` (used by workflows
   for inference) is updated in `on_batch_end` after each checkpoint save. This
   ensures workflows always sample from the latest policy.

3. **Metrics injection in `on_batch_end`.** Since `on_batch_end` runs after the
   pipeline but before `logger.log(...)`, it is the natural place to compute derived
   metrics (KL divergence, entropy, learning rate) and add them to
   `trainer_state.metrics`.

---

## Data Flow Summary

```
  Dataset
    |
    v
  get_dataloader() --> batch
    |
    v
  generate_episodes(batch) --> list[Episode]
    |
    v
  [framework] transform to TrajectoryGroups
    |
    v
  [framework] rejection sampling & filtering
    |
    v
  transform_to_backend_batch() --> TBatch (stored in trainer_state.backend_batch)
    |
    v
  process_backend_batch()  -- forward/backward, logprobs
    |
    v
  compute_advantages()     -- advantage computation
    |
    v
  update_policy()          -- optimizer step
    |
    v
  [framework] visualization, metrics collection
    |
    v
  on_batch_end()           -- checkpoint, derived metrics
    |
    v
  logger.log(metrics)      -- wandb / tracking
```

# Unified Trainer

> **Module**: `rllm.experimental.unified_trainer`

The `UnifiedTrainer` is the central orchestrator for backend-agnostic training in rLLM.
It manages the full training loop -- episode generation, data transformation, advantage
computation, policy updates, validation, and logging -- while delegating all
backend-specific operations to a pluggable `BackendProtocol` implementation.

---

## Architecture Overview

```
                         UnifiedTrainer
                +-----------+-----------+
                |                       |
        BackendProtocol         UnifiedWorkflowEngine
        (e.g. TinkerBackend)    (manages Workflow pool)
                |                       |
        Backend-specific        Workflow instances
        infra (model, optim)    (rollout logic)
```

The trainer itself is **backend-agnostic**: it knows nothing about model weights,
optimizers, or inference servers. All of that is encapsulated behind the
`BackendProtocol` interface (see [backend-protocol.md](backend-protocol.md)).

---

## Entry Points

There are two ways to start training:

### 1. `AgentTrainer` (recommended for standard backends)

A convenience wrapper that selects the correct `TrainerLauncher` for the backend
string (`"verl"` or `"tinker"`) and handles environment setup.

```python
from rllm.experimental.unified_trainer import AgentTrainer

trainer = AgentTrainer(
    config=config,
    workflow_class=MyWorkflow,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    backend="tinker",
)
trainer.train()
```

### 2. `UnifiedTrainer` (direct, for custom backends)

When using a custom backend class, instantiate the trainer directly:

```python
from rllm.experimental.unified_trainer import UnifiedTrainer

trainer = UnifiedTrainer(
    backend_cls=MyCustomBackend,
    config=config,
    workflow_class=MyWorkflow,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    workflow_args={"my_param": 42},
    backend_args={"device": "cuda:0"},
)
trainer.fit()
```

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend_cls` | `type[BackendProtocol]` | The backend class to instantiate |
| `config` | `DictConfig` | Full Hydra config (must contain `config.rllm`) |
| `workflow_class` | `type[Workflow]` | The workflow class for episode generation |
| `train_dataset` | `Dataset | None` | Training dataset |
| `val_dataset` | `Dataset | None` | Validation dataset |
| `workflow_args` | `dict | None` | Extra kwargs passed to each workflow instance |
| `backend_args` | `dict | None` | Extra kwargs passed to the backend constructor |

---
## `TrainerState`

A mutable dataclass that serves as the shared context between the trainer and the
backend throughout a training step. It is **reset at the start of each batch** via
`reset_batch()`.

```python
@dataclass
class TrainerState:
    # Progress
    global_step: int = 0
    epoch: int = 0
    total_steps: int = 0
    is_training: bool = True

    # Timing and metrics (reset per batch)
    timing_dict: dict          # populated by simple_timer context managers
    metrics: dict              # logged to wandb/tracking after each batch
    extra_info: dict           # backend-private scratchpad (e.g. logprobs, LR)

    # Pipeline data (reset per batch)
    episodes: list[Episode] | None
    trajectory_groups: list[TrajectoryGroup] | None
    backend_batch: Any | None  # backend-specific format
```

**Convenience properties:** `has_episodes`, `has_trajectory_groups`, `has_backend_batch`
-- used by the trainer to detect early-return conditions (e.g. all episodes filtered).

---

## Initialization Sequence

When the `UnifiedTrainer` is constructed, the following happens in order:

| Step | Method | Description |
|------|--------|-------------|
| 1 | `_setup_event_loop()` | Spawn a background asyncio event loop thread |
| 2 | `backend_cls(config, ...)` | Instantiate the backend |
| 3 | `_validate_and_setup_configs()` | Build AlgorithmConfig, TransformConfig, etc. |
| 4 | `_setup_logging()` | Init wandb / episode logger |
| 5 | `backend.init_rollout_engine()` | Backend creates its RolloutEngine |
| 6 | `UnifiedWorkflowEngine(...)` | Create the workflow engine with a pool of workflow instances (workflow_cls + workflow_args) |
| 7 | `initialize_pool()` | Pre-create N workflow instances |

If any step fails, `_cleanup_on_init_failure()` tears down all resources that were
partially initialized (event loop, logger, backend, workflow engine).

---

## Training Loop

Calling `trainer.fit()` runs the async loop in the background thread. The high-level
flow is shown in the below Mermaid diagram:

??? note "Training Loop Diagram (click to expand)"

    ``` mermaid
    graph TD
      n1["fit()"] --> n2["_fit_entry_async()"]
      n2 --> n3["backend.on_train_start(state)"]
      n3 --> d1{"optional: _validate_async(state)?"}

      d1 -->|yes| v1["_validate_async(state)"]
      d1 -->|no| f1["_fit_async(state)"]
      v1 --> f1

      f1 --> e1{"for each epoch"}
      e1 --> es1["backend.on_epoch_start(state)"]
      es1 --> b1{"for each batch"}

      b1 --> r1["state.reset_batch()"]
      r1 --> bs1["backend.on_batch_start(state)"]
      bs1 --> tb1["_train_batch_async(batch, state) (8-stage pipeline)"]
      tb1 --> be1["backend.on_batch_end(state)"]
      be1 --> lg1["logger.log(state.metrics, step)"]
      lg1 --> d2{"optional: _validate_async(state)?"}

      d2 -->|yes| v2["_validate_async(state)"]
      d2 -->|no| c1["continue"]
      v2 --> c1
      c1 --> b1

      b1 --> ee1["backend.on_epoch_end(state)"]
      ee1 --> e1

      e1 --> te1["backend.on_train_end(state)"]
    ```

### The 8-Stage Batch Pipeline

Each call to `_train_batch_async` executes the following stages. Stages 1-3 are
framework-managed. Stages 4-7 are delegated to the backend.

| Stage | Method / Owner | Sync/Async | Description |
|---|---|---|---|
| 1 | `backend.generate_episodes()` | async | Run workflows to produce Episode objects |
| 2 | `transform_episodes_to_trajectory_groups` | sync | Group episodes into TrajectoryGroups |
| 3 | `apply_rejection_sampling_and_filtering` | sync | Filter groups (solve-all / solve-none / etc.) |
| 4 | `backend.transform_to_backend_batch()` | sync | Convert to backend-native format |
| 5 | `backend.process_backend_batch()` | async | Forward/backward pass, compute logprobs, etc. |
| 6 | `backend.compute_advantages()` | async | Compute per-step advantages |
| 7 | `backend.update_policy()` | async | Optimizer step |
| 8 | (framework) visualization + metrics | sync | Print trajectories, collect workflow metrics |

**Early returns:** The pipeline returns early (skipping stages 4-8) if no episodes
are generated in stage 1, or if all trajectory groups are filtered out in stage 3.
The lifecycle hooks (`on_batch_end`, `logger.log`) still execute even after an early
return.

---

## Validation Loop

Validation is triggered:

- Before training (if `config.rllm.trainer.val_before_train` is true)
- Periodically during training (every `config.rllm.trainer.test_freq` steps)
- After training completes

The validation loop calls `backend.generate_episodes(..., is_validation=True)`,
transforms the results, and computes reward metrics (no advantage computation or
policy updates). Pass@1 and pass@K metrics are computed per data source and logged.

The backend can control validation via hooks:

- `on_validation_start` returns a `bool` -- return `False` to skip validation entirely
- `on_validation_end` is called after validation completes

---

## Configuration

The trainer reads configuration from `config.rllm` (the rLLM sub-config within the
full Hydra config). Key config groups:

| Config path | Built config | Used for |
|---|---|---|
| `rllm.compact_filtering` | `CompactFilteringConfig` | Filter invalid episodes |
| `rllm.stepwise_advantage` | `TransformConfig` | Episode-to-group transform mode |
| `rllm.rejection_sample` | `RejectionSamplingConfig` | Rejection sampling settings |
| `rllm.algorithm` | `AlgorithmConfig` | Advantage estimator, loss fn, LR schedule |
| `rllm.workflow` | *(direct)* | `n_parallel_tasks`, `retry_limit`, `raise_on_error` |
| `rllm.trainer` | *(direct)* | `total_epochs`, `total_batches`, `test_freq`, `save_freq`, logger |
| `rllm.rollout` | *(direct)* | `n` (group size), `n_val` (val samples per task) |

### `AlgorithmConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `estimator` | `rLLMAdvantageEstimator` | `GRPO` | Advantage estimator (`GRPO`, `REINFORCE`) |
| `stepwise_advantage_mode` | `"broadcast"` | `"broadcast"` | How advantages map to steps |
| `norm_adv_by_std_in_grpo` | `bool` | `True` | Normalize advantages by std in GRPO |
| `loss_fn` | `str \| None` | `None` | Backend loss function (e.g. `"importance_sampling"`) |
| `lr_schedule` | `str` | `"constant"` | LR schedule: `"constant"`, `"linear"`, `"cosine"` |
| `warmup_steps_ratio` | `float` | `0.0` | Fraction of total steps for LR warmup |

---

## Async Design

The trainer uses an **async-prioritized** design:

- A dedicated `asyncio` event loop runs in a background daemon thread
- The public `fit()` method is synchronous -- it submits the async entry point to the
  loop via `asyncio.run_coroutine_threadsafe` and blocks on the result
- All backend pipeline methods are `async def`, which naturally accommodates backends
  with async APIs (like Tinker) while allowing sync backends to simply `await` their
  sync operations
- The `_run_async(coro)` helper is available for running one-off coroutines from
  synchronous code

---

## Shutdown

Always call `trainer.shutdown()` when done (or use a `try/finally` block). This:

1. Stops the background event loop thread
2. Shuts down the workflow engine (including its thread pool)
3. Calls `backend.shutdown()` for backend-specific cleanup
4. Calls `logger.finish()` to flush and close the tracking backend (e.g. wandb)

```python
try:
    trainer = UnifiedTrainer(...)
    trainer.fit()
finally:
    trainer.shutdown()
```

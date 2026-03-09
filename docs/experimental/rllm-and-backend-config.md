# rLLM Configuration System

The rLLM framework provides a unified configuration system that separates backend-agnostic settings from backend-specific configurations. This design allows you to switch between different RL backends (Tinker, Verl) while maintaining consistent core training logic.

## Configuration Structure

The configuration system is organized into three main components:

1. **rLLM Backend-Agnostic Configs**: Core training settings shared across all backends
2. **Backend-Specific Configs**: Settings specific to Tinker or Verl backends
3. **Forwarding Mechanism**: Allows backend-specific configs to override rLLM configs for backward compatibility

All configuration files are located in `rllm/experimental/config/`:

- `rllm/experimental/config/rllm/base.yaml`: Backend-agnostic rLLM configurations
- `rllm/experimental/config/rllm/backend/tinker.yaml`: Tinker-specific configurations
- `rllm/experimental/config/rllm/backend/verl.yaml`: Verl-specific configurations
- `rllm/experimental/config/unified.yaml`: Main entry point that combines all configs

---

## rLLM Backend-Agnostic Configurations

These configurations are defined in `rllm/base.yaml` and are used across different backends.

### Agent Configuration

Settings for the agent that interacts with the environment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `math_agent` | Name of the agent |
| `max_steps` | `int` | `20` | Maximum number of steps per trajectory |
| `trajectory_timeout` | `int | null` | `null` | Timeout for trajectory execution (seconds) |
| `overlong_filter` | `bool` | `False` | Whether to filter out overlong trajectories |
| `agent_args` | `dict` | `{}` | Additional agent-specific arguments |
| `engine_args` | `dict` | `{}` | Additional engine-specific arguments |

### Environment Configuration

Settings for the environment where the agent operates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `custom` | Name of the environment |
| `env_args` | `dict` | `{}` | Additional environment-specific arguments |

### Workflow Configuration

Settings for workflow-based training (alternative to agent-based training).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_workflow` | `bool` | `False` | Whether to use workflow mode instead of agent mode |
| `name` | `str` | `single_turn_workflow` | Name of the workflow |
| `workflow_args.agent_cls` | `str | null` | `null` | Agent class to use in workflow |
| `workflow_args.agent_args` | `dict` | `{}` | Agent arguments in workflow |
| `workflow_args.env_cls` | `str | null` | `null` | Environment class to use in workflow |
| `workflow_args.env_args` | `dict` | `{}` | Environment arguments in workflow |
| `workflow_args.timeout` | `float` | `1e6` | Workflow execution timeout |
| `workflow_args.gamma` | `float` | `0.0` | Discount factor (0.0 = no discounting) |
| `workflow_args.reward_bonus_coeff` | `float` | `0.0` | Reward shaping coefficient |
| `n_parallel_tasks` | `int` | `256` | Number of parallel tasks to run |
| `retry_limit` | `int` | `3` | Maximum number of retries on failure |
| `raise_on_error` | `bool` | `True` | Whether to raise exceptions on errors |

### Rollout Configuration

Settings for trajectory rollouts during training and validation.

!!! note
    These settings are primarily for logging purposes. The actual rollout behavior is determined by backend-specific configurations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `8` | Number of rollouts per prompt during training |
| `n_val` | `int` | `1` | Number of rollouts per prompt during validation |

### Trainer Configuration

Core training loop settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_epochs` | `int` | `10` | Total number of training epochs |
| `total_batches` | `int` | `-1` | Total number of training batches (-1 = use epochs) |
| `logger` | `list[str]` | `['console']` | Logging backends (options: `console`, `wandb`, `tensorboard`) |
| `project_name` | `str` | `rllm-training` | Project name for logging |
| `experiment_name` | `str` | `default` | Experiment name for logging |
| `test_freq` | `int` | `5` | Frequency of validation (in epochs) |
| `save_freq` | `int` | `20` | Frequency of checkpoint saving (in epochs) |
| `val_before_train` | `bool` | `True` | Whether to run validation before training starts |
| `val_only` | `bool` | `False` | Whether to only run validation (no training) |

### Algorithm Configuration

RL algorithm and advantage estimation settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adv_estimator` | `str` | `grpo` | Advantage estimator (options: `grpo`, `reinforce`, `reinforce_plus_plus_baseline`, `rloo`, `gae`) |
| `gamma` | `float` | `1.0` | Discount factor for future rewards |
| `lam` | `float` | `0.95` | Lambda for GAE (Generalized Advantage Estimation) |
| `norm_adv_by_std_in_grpo` | `bool` | `True` | Whether to normalize advantages by standard deviation in GRPO |
| `use_rllm` | `bool` | `False` | Whether to use rLLM-specific features |
| `loss_fn` | `str | null` | `null` | Loss function for Tinker backend (options: `importance_sampling`, `ppo`, `cispo`, `dro`, `cross_entropy`) |

### Stepwise Advantage Configuration

Settings for computing advantages at each step in multi-step trajectories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | `bool` | `False` | Whether to enable stepwise advantage computation |
| `mode` | `str` | `broadcast` | Advantage computation mode (options: `broadcast`, `per_step`) |
| `normalize_by_steps` | `bool` | `False` | Whether to normalize advantages by number of steps |

### Trajectory Processing Flags

Top-level flags for trajectory processing and filtering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable_thinking` | `bool` | `False` | Whether to disable thinking tokens in responses |
| `accumulate_reasoning` | `bool` | `False` | Whether to accumulate reasoning across steps |
| `mask_truncated_samples` | `bool` | `False` | Whether to mask trajectories that were truncated |
| `filter_token_mismatch` | `bool` | `True` | Whether to filter out trajectories with token mismatches |

### Compact Filtering Configuration

Fine-grained filtering of trajectories based on various termination conditions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | `bool` | `False` | Whether to enable compact filtering |
| `mask_max_prompt_length_exceeded` | `bool` | `True` | Mask trajectories that exceed max prompt length |
| `mask_max_response_length_exceeded` | `bool` | `True` | Mask trajectories that exceed max response length |
| `mask_env_done` | `bool` | `False` | Mask trajectories where environment signaled done |
| `mask_max_turns_exceeded` | `bool` | `True` | Mask trajectories that exceed max turns |
| `mask_timeout` | `bool` | `True` | Mask trajectories that timed out |
| `mask_unknown` | `bool` | `False` | Mask trajectories with unknown termination reasons |
| `mask_error` | `bool` | `True` | Mask trajectories that encountered errors |

### Rejection Sampling Configuration

Settings for rejection sampling to improve training data quality.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | `bool` | `False` | Whether to enable rejection sampling |
| `multiplier` | `int` | `1` | Multiplier for number of rollouts to generate |
| `min_partial_solve_tasks` | `int` | `1` | Minimum number of tasks that must be partially solved |
| `min_trajs_per_group` | `int` | `2` | Minimum number of trajectories per group to keep |

### SDK Configuration

Settings for the rLLM SDK, including trace storage and proxy server.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store.path` | `str` | `~/.rllm/traces.db` | Path to trace database |
| `processing.groupby_key` | `str | null` | `null` | Key to group trajectories by |
| `processing.traj_name_key` | `str | null` | `null` | Key to use as trajectory name |
| `proxy.host` | `str` | `127.0.0.1` | Proxy server host |
| `proxy.port` | `int` | `4000` | Proxy server port |
| `proxy.mode` | `str` | `subprocess` | Proxy mode (options: `subprocess`, `external`) |
| `proxy.admin_token` | `str` | `my-shared-secret` | Admin token for proxy authentication |

### Episode Logging Configuration

Settings for logging full episode trajectories to disk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_episodes` | `bool` | `false` | Whether to log full episodes to disk |
| `episode_log_dir` | `str` | `logs/${rllm.trainer.project_name}/${rllm.trainer.experiment_name}` | Directory for episode logs |

---

## Backend-Specific Configurations

### Tinker Backend Configuration

Tinker-specific settings live in `rllm/experimental/config/rllm/backend/tinker.yaml`.

This file contains:

1. Tinker service and execution settings
2. Model/LoRA training settings
3. Sampling and rollout-engine settings
4. Tinker-native training/data blocks
5. Forwarding into `rllm.*` common config keys

#### Top-level Tinker-specific keys

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tinker_base_url` | `str | null` | `null` | Tinker service URL (`null` for local/default) |
| `fuse_forward_backward_and_optim_step` | `bool` | `false` | Whether to fuse train-step internals in backend |

#### Model block

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model.name` | `str` | `Qwen/Qwen3-8B` | Base model name |
| `model.lora_rank` | `int` | `32` | LoRA rank |
| `model.train_unembed` | `bool` | `true` | Train LoRA on output embedding layer |
| `model.train_attn` | `bool` | `true` | Train LoRA on attention layers |
| `model.train_mlp` | `bool` | `true` | Train LoRA on MLP layers |

#### Training block (Tinker-native)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `training.group_size` | `int` | `???` | Number of rollouts per prompt |
| `training.learning_rate` | `float` | `2e-5` | Learning rate |
| `training.lr_schedule` | `str` | `constant` | LR schedule (`constant`, `linear`, `cosine`) |
| `training.warmup_steps_ratio` | `float` | `0.0` | Warmup ratio in `[0, 1]` |
| `training.beta1` | `float` | `0.9` | Adam beta1 |
| `training.beta2` | `float` | `0.95` | Adam beta2 |
| `training.eps` | `float` | `1e-8` | Adam epsilon |
| `training.max_length` | `int` | `32768` | Max model context length |
| `training.num_minibatches` | `int` | `1` | Number of minibatches |
| `training.default_local_dir` | `str` | `/tmp/rllm-tinker-checkpoints` | Local checkpoint directory |
| `training.resume_from_tinker_id` | `str | null` | `null` | Optional checkpoint/model ID to resume |

#### Validation/sampling/rollout/data blocks

| Parameter | Type | Default | Description |
|---|---|---|---|
| `validation.group_size` | `int` | `???` | Rollouts per prompt for validation |
| `sampling.train.temperature` | `float` | `1.0` | Train sampling temperature |
| `sampling.train.top_p` | `float` | `1.0` | Train nucleus sampling threshold |
| `sampling.train.top_k` | `int` | `-1` | Train top-k |
| `sampling.val.temperature` | `float` | `1.0` | Val sampling temperature |
| `sampling.val.top_p` | `float` | `1.0` | Val nucleus sampling threshold |
| `sampling.val.top_k` | `int` | `-1` | Val top-k |
| `rollout_engine.reasoning_effort` | `str` | `medium` | Reasoning effort mode |
| `rollout_engine.accumulate_reasoning` | `bool` | `false` | Whether to accumulate reasoning across steps |
| `rollout_engine.disable_thinking` | `bool` | `false` | Whether to disable thinking tokens |
| `rollout_engine.bypass_render_with_parser` | `bool` | `false` | Whether to bypass render parsing |
| `rollout_engine.renderer_name` | `str | null` | `null` | Optional renderer name |
| `data.max_prompt_length` | `int` | `2048` | Max prompt length |
| `data.max_response_length` | `int` | `2048` | Max response length |
| `data.train_batch_size` | `int` | `64` | Train batch size |
| `data.val_batch_size` | `int` | `32` | Validation batch size |

#### OPSD block

| Parameter | Type | Default | Description |
|---|---|---|---|
| `opsd.kl_penalty_coef` | `float` | `1.0` | KL penalty coefficient |
| `opsd.kl_discount_factor` | `float` | `0.0` | KL discount factor |
| `opsd.teacher_messages_key` | `str` | `teacher_messages` | Key for teacher messages |
| `opsd.teacher_policy_update_freq` | `int` | `-1` | Teacher refresh frequency (`-1` = initial teacher only) |

#### Forwarding to common `rllm.*`

Tinker backend forwards group-size settings into backend-agnostic rollout config:

- `rllm.rollout.n <- training.group_size`
- `rllm.rollout.n_val <- validation.group_size`

### Verl Backend Configuration

Verl-specific settings live in `rllm/experimental/config/rllm/backend/verl.yaml`.

This file is intentionally thin and composes Verl's native PPO config via:

```yaml
defaults:
  - /ppo_trainer
  - _self_
```

For detailed semantics of Verl-native fields, see:

- https://verl.readthedocs.io/en/latest/examples/config.html

In rLLM, the `verl.yaml` mainly does three things:

1. Sets a few required overrides for unified-trainer compatibility
2. Marks selected native fields as required (`???`) so they can be provided by user/config composition
3. Forwards native Verl fields into `rllm.*` common config for backward compatibility

#### Key fields in `verl.yaml`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `actor_rollout_ref.rollout.mode` | `str` | `async` | Required mode for unified Verl backend |
| `actor_rollout_ref.rollout.agent.num_workers` | `int` | `0` | Agent worker count |
| `actor_rollout_ref.rollout.val_kwargs.n` | `int` | `???` | Validation rollout count |
| `data.gen_batch_size` | `int` | `${mul:${data.train_batch_size},${rllm.rejection_sample.multiplier}}` | Generated batch size |
| `data.return_multi_modal_inputs` | `bool` | `False` | Include multimodal inputs in data path |
| `algorithm.adv_estimator` | `str` | `???` | Native algorithm estimator |
| `algorithm.gamma` | `float` | `???` | Native discount factor |
| `algorithm.lam` | `float` | `???` | Native lambda |
| `algorithm.norm_adv_by_std_in_grpo` | `bool` | `???` | Native GRPO normalization |
| `trainer.total_epochs` | `int` | `???` | Native epoch count |
| `trainer.total_training_steps` | `int` | `???` | Native total-step budget |
| `trainer.logger` | `list[str]` | `???` | Native logger list |
| `trainer.project_name` | `str` | `???` | Project name |
| `trainer.experiment_name` | `str` | `???` | Experiment name |
| `trainer.test_freq` | `int` | `???` | Validation cadence |
| `trainer.save_freq` | `int` | `???` | Save cadence |
| `trainer.val_before_train` | `bool` | `???` | Validate before training |
| `trainer.val_only` | `bool` | `???` | Validation-only mode |

#### Forwarding to common `rllm.*`

Verl backend forwards the following:

- `rllm.algorithm.{adv_estimator,gamma,lam,norm_adv_by_std_in_grpo}`
- `rllm.rollout.{n,n_val}`
- `rllm.trainer.{total_epochs,total_batches,logger,project_name,experiment_name,test_freq,save_freq,val_before_train,val_only}`

It also extends Hydra search path with Verl configs:

```yaml
rllm:
  hydra.searchpath:
    - pkg://verl.trainer.config
```

---

## Config Forwarding Mechanism

The rLLM configuration system supports a **forwarding mechanism** that allows users familiar with a specific backend (Tinker or Verl) to specify configurations in their native format. These backend-specific configs are then automatically forwarded to the corresponding rLLM configs for backward compatibility.

### How It Works

Backend-specific config files can override rLLM settings using Hydra's `oc.select` resolver. This mechanism:

1. First checks if a backend-specific config value is provided
2. If provided, uses that value to populate the rLLM config
3. If not provided, falls back to the rLLM default value

### Example: Verl Backend Forwarding

In `rllm/experimental/config/rllm/backend/verl.yaml`, you can see how Verl's native trainer configuration is forwarded to rLLM:

```yaml
# In Verl's native config format
trainer:
  total_epochs: 15
  project_name: 'my-verl-project'
  experiment_name: 'verl-experiment-1'

# These are automatically forwarded to rLLM configs
rllm:
  trainer:
    total_epochs: ${oc.select:trainer.total_epochs, 10}  # Uses 15 from above
    project_name: ${oc.select:trainer.project_name, 'rllm-training'}  # Uses 'my-verl-project'
    experiment_name: ${oc.select:trainer.experiment_name, 'default'}  # Uses 'verl-experiment-1'
```

In this example:
- Users can specify `trainer.total_epochs` in Verl's native format
- The value is automatically forwarded to `rllm.trainer.total_epochs`
- If the Verl config is not specified, the rLLM default (10) is used

### Example: Algorithm Configuration Forwarding

Similarly, algorithm configurations can be forwarded:

```yaml
# Backend-specific algorithm config
algorithm:
  adv_estimator: gae
  gamma: 0.99
  lam: 0.95

# Forwarded to rLLM
rllm:
  algorithm:
    adv_estimator: ${oc.select:algorithm.adv_estimator, grpo}  # Uses 'gae'
    gamma: ${oc.select:algorithm.gamma, 1.0}  # Uses 0.99
    lam: ${oc.select:algorithm.lam, 0.95}  # Uses 0.95
```

### Benefits

This forwarding mechanism provides several benefits:

- **Backward Compatibility**: Users can continue using their familiar backend-specific config formats
- **Gradual Migration**: Projects can migrate to rLLM configs incrementally
- **Flexibility**: Supports both backend-specific and rLLM-native configuration styles
- **Consistency**: Ensures backend configs and rLLM configs stay synchronized

---

## Configuration Best Practices

1. **Use rLLM configs for new projects**: If starting from scratch, use the rLLM backend-agnostic configs for better portability across backends.

2. **Leverage forwarding for migration**: If migrating from a specific backend, use the forwarding mechanism to maintain existing configs while gradually adopting rLLM conventions.

3. **Check the unified config**: The `unified.yaml` file shows how all configs are combined and is useful for debugging configuration issues.

4. **Understand defaults hierarchy**: Backend-specific configs override rLLM defaults, which in turn override Hydra's base defaults.

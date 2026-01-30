# rLLM Configuration System

The rLLM framework provides a unified configuration system that separates backend-agnostic settings from backend-specific configurations. This design allows you to switch between different RL backends (Tinker, Verl) while maintaining consistent core training logic.

## Configuration Structure

The configuration system is organized into three main components:

1. **rLLM Backend-Agnostic Configs**: Core training settings shared across all backends
2. **Backend-Specific Configs**: Settings specific to Tinker or Verl backends
3. **Forwarding Mechanism**: Allows backend-specific configs to override rLLM configs for backward compatibility

All configuration files are located in `rllm/experimental/config/`:

- `rllm/base.yaml`: Backend-agnostic rLLM configurations
- `rllm/backend/tinker.yaml`: Tinker-specific configurations
- `rllm/backend/verl.yaml`: Verl-specific configurations
- `unified.yaml`: Main entry point that combines all configs

---

## rLLM Backend-Agnostic Configurations

These configurations are defined in `rllm/base.yaml` and are used across different backends.

### Agent Configuration

Settings for the agent that interacts with the environment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `math_agent` | Name of the agent |
| `max_steps` | `int` | `20` | Maximum number of steps per trajectory |
| `trajectory_timeout` | `int/null` | `null` | Timeout for trajectory execution (seconds) |
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
| `workflow_args.agent_cls` | `str/null` | `null` | Agent class to use in workflow |
| `workflow_args.agent_args` | `dict` | `{}` | Agent arguments in workflow |
| `workflow_args.env_cls` | `str/null` | `null` | Environment class to use in workflow |
| `workflow_args.env_args` | `dict` | `{}` | Environment arguments in workflow |
| `workflow_args.timeout` | `float` | `1e6` | Workflow execution timeout |
| `workflow_args.gamma` | `float` | `0.0` | Discount factor (0.0 = no discounting) |
| `workflow_args.reward_bonus_coeff` | float | `0.0` | Reward shaping coefficient |
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
| `logger` | `list` | `['console']` | Logging backends (options: `console`, `wandb`, `tensorboard`) |
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
| `adv_estimator` | `str` | `grpo` | Advantage estimator (options: `grpo`, `reinforce`, `gae`) |
| `gamma` | `float` | `1.0` | Discount factor for future rewards |
| `lam` | `float` | `0.95` | Lambda for GAE (Generalized Advantage Estimation) |
| `norm_adv_by_std_in_grpo` | `bool` | `True` | Whether to normalize advantages by standard deviation in GRPO |
| `use_rllm` | `bool` | `False` | Whether to use rLLM-specific features |
| `loss_fn` | `str/null` | `null` | Loss function for Tinker backend (options: `importance_sampling`, `ppo`, `cispo`, `dro`, `cross_entropy`) |

### Stepwise Advantage Configuration

Settings for computing advantages at each step in multi-step trajectories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `False` | Whether to enable stepwise advantage computation |
| `mode` | str | `broadcast` | Advantage computation mode (options: `broadcast`, `per_step`) |
| `normalize_by_steps` | bool | `False` | Whether to normalize advantages by number of steps |

### Trajectory Processing Flags

Top-level flags for trajectory processing and filtering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable_thinking` | bool | `False` | Whether to disable thinking tokens in responses |
| `accumulate_reasoning` | bool | `False` | Whether to accumulate reasoning across steps |
| `mask_truncated_samples` | bool | `False` | Whether to mask trajectories that were truncated |
| `filter_token_mismatch` | bool | `True` | Whether to filter out trajectories with token mismatches |

### Compact Filtering Configuration

Fine-grained filtering of trajectories based on various termination conditions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `False` | Whether to enable compact filtering |
| `mask_max_prompt_length_exceeded` | bool | `True` | Mask trajectories that exceed max prompt length |
| `mask_max_response_length_exceeded` | bool | `True` | Mask trajectories that exceed max response length |
| `mask_env_done` | bool | `False` | Mask trajectories where environment signaled done |
| `mask_max_turns_exceeded` | bool | `True` | Mask trajectories that exceed max turns |
| `mask_timeout` | bool | `True` | Mask trajectories that timed out |
| `mask_unknown` | bool | `False` | Mask trajectories with unknown termination reasons |
| `mask_error` | bool | `True` | Mask trajectories that encountered errors |

### Rejection Sampling Configuration

Settings for rejection sampling to improve training data quality.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `False` | Whether to enable rejection sampling |
| `multiplier` | int | `1` | Multiplier for number of rollouts to generate |
| `min_partial_solve_tasks` | int | `1` | Minimum number of tasks that must be partially solved |
| `min_trajs_per_group` | int | `2` | Minimum number of trajectories per group to keep |

### SDK Configuration

Settings for the rLLM SDK, including trace storage and proxy server.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store.path` | str | `~/.rllm/traces.db` | Path to trace database |
| `processing.groupby_key` | str/null | `null` | Key to group trajectories by |
| `processing.traj_name_key` | str/null | `null` | Key to use as trajectory name |
| `proxy.host` | str | `127.0.0.1` | Proxy server host |
| `proxy.port` | int | `4000` | Proxy server port |
| `proxy.mode` | str | `subprocess` | Proxy mode (options: `subprocess`, `external`) |
| `proxy.admin_token` | str | `my-shared-secret` | Admin token for proxy authentication |

### Episode Logging Configuration

Settings for logging full episode trajectories to disk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_episodes` | bool | `false` | Whether to log full episodes to disk |
| `episode_log_dir` | str | `logs/${rllm.trainer.project_name}/${rllm.trainer.experiment_name}` | Directory for episode logs |

---

## Backend-Specific Configurations

### Tinker Backend Configuration

_[To be filled in]_

### Verl Backend Configuration

_[To be filled in]_

---

## Config Forwarding Mechanism

The rLLM configuration system supports a **forwarding mechanism** that allows users familiar with a specific backend (Tinker or Verl) to specify configurations in their native format. These backend-specific configs are then automatically forwarded to the corresponding rLLM configs for backward compatibility.

### How It Works

Backend-specific config files can override rLLM settings using Hydra's `oc.select` resolver. This mechanism:

1. First checks if a backend-specific config value is provided
2. If provided, uses that value to populate the rLLM config
3. If not provided, falls back to the rLLM default value

### Example: Verl Backend Forwarding

In `rllm/backend/verl.yaml`, you can see how Verl's native trainer configuration is forwarded to rLLM:

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

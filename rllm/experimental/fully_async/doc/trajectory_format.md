# Trajectory Save Format

This document describes the format of saved trajectory data from the fully async trainer.

## Overview

When `save_trajectories: true` is set in the `async_training` config, trajectory groups are saved to JSON files after each training step. Files are saved to:

```
{default_local_dir}/trajectories/step_{global_step}.json
```

## File Structure

```json
{
  "global_step": 1,
  "param_version": 0,
  "num_trajectory_groups": 128,
  "trajectory_groups": [...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `global_step` | int | The training step number when this batch was processed |
| `param_version` | int | The parameter version used for training (increments on param sync) |
| `num_trajectory_groups` | int | Number of trajectory groups in this batch |
| `trajectory_groups` | list | List of TrajectoryGroup objects |

## TrajectoryGroup

A `TrajectoryGroup` contains multiple trajectories that are compared for advantage computation (e.g., GRPO).

```json
{
  "trajectories": [...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `trajectories` | list[Trajectory] | List of trajectories in this group |

## Trajectory

A `Trajectory` represents a single rollout with its sequences and reward.

```json
{
  "sequences": [...],
  "reward": 1.0,
  "metadata": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | list[Sequence] | List of sequences (prompt-response pairs) in this trajectory |
| `reward` | float | The reward for this trajectory (default: 0.0) |
| `metadata` | dict \| null | Optional metadata associated with this trajectory |

## Sequence

A `Sequence` represents a single prompt-response pair with token-level information.

```json
{
  "prompt_ids": [1, 2, 3, ...],
  "response_ids": [10, 11, 12, ...],
  "response_logprobs": [-0.5, -0.3, -0.8, ...],
  "response_masks": [1, 1, 1, ...],
  "start_version": 0,
  "end_version": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt_ids` | list[int] | Token IDs of the prompt |
| `response_ids` | list[int] | Token IDs of the model response |
| `response_logprobs` | list[float] | Log probabilities for each response token |
| `response_masks` | list[int] | Mask indicating valid tokens (1) vs padding (0) |
| `start_version` | int \| null | Parameter version when generation started |
| `end_version` | int \| null | Parameter version when generation ended |

### Version Tracking

The `start_version` and `end_version` fields track policy staleness:
- If `start_version == end_version`: The sequence was generated with a single policy version
- If `start_version != end_version`: The sequence spans multiple policy updates (stale data)

This information is used for:
- Staleness tracking metrics
- Off-policy correction algorithms
- Debugging async training dynamics

## Example

```json
{
  "global_step": 42,
  "param_version": 5,
  "num_trajectory_groups": 2,
  "trajectory_groups": [
    {
      "trajectories": [
        {
          "sequences": [
            {
              "prompt_ids": [1, 2, 3, 4, 5],
              "response_ids": [100, 101, 102],
              "response_logprobs": [-0.5, -0.3, -0.2],
              "response_masks": [1, 1, 1],
              "start_version": 4,
              "end_version": 5
            }
          ],
          "reward": 1.0,
          "metadata": {"task_id": "math_001"}
        },
        {
          "sequences": [
            {
              "prompt_ids": [1, 2, 3, 4, 5],
              "response_ids": [200, 201, 202, 203],
              "response_logprobs": [-0.6, -0.4, -0.3, -0.5],
              "response_masks": [1, 1, 1, 1],
              "start_version": 5,
              "end_version": 5
            }
          ],
          "reward": 0.0,
          "metadata": {"task_id": "math_001"}
        }
      ]
    }
  ]
}
```

## Configuration

To enable trajectory saving, add to your config:

```yaml
async_training:
  save_trajectories: true
```

## Loading Saved Trajectories

```python
import json
from rllm.experimental.fully_async.protocol import TrajectoryGroup

# Load from file
with open("trajectories/step_42.json", "r") as f:
    data = json.load(f)

# Reconstruct TrajectoryGroup objects
trajectory_groups = [
    TrajectoryGroup.from_dict(tg_data) 
    for tg_data in data["trajectory_groups"]
]
```

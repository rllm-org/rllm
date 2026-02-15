# Remote Solver-Judge Example

This example demonstrates **online RL training with remote agent episode generation**. The agent (SolverJudgeWorkflow) runs in a separate process (or Docker container), while the trainer handles model weights, policy updates, and exposes an inference API that the remote agent calls.

## Architecture

```
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│           Trainer                │     │    Remote Agent Server           │
│                                  │     │                                  │
│  ┌────────────────────────────┐  │     │  ┌────────────────────────────┐  │
│  │  Inference API Server      │◄─┼─────┼──│  RemoteRolloutEngine       │  │
│  │  (POST /v1/model_response) │  │     │  │  (calls inference API)     │  │
│  │  wraps RolloutEngine       │  │     │  └─────────────┬──────────────┘  │
│  └────────────────────────────┘  │     │                │                 │
│                                  │     │  ┌─────────────▼──────────────┐  │
│  ┌────────────────────────────┐  │     │  │  SolverJudgeWorkflow       │  │
│  │  RemoteEpisodeCollector    │──┼─────┼─►│  (Solver + Judge)          │  │
│  │  (POST /generate_episode)  │  │     │  │  Returns: Episode          │  │
│  └─────────────┬──────────────┘  │     │  └────────────────────────────┘  │
│                │                 │     │                                  │
│  ┌─────────────▼──────────────┐  │     │  POST /generate_episode         │
│  │  Training Pipeline         │  │     └──────────────────────────────────┘
│  │  (advantages, PPO, etc.)   │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

**Flow:**
1. Trainer sends a task to the remote agent via `POST /generate_episode`
2. Remote agent receives the task and the trainer's `inference_api_url`
3. Remote agent runs `SolverJudgeWorkflow`, calling back to the trainer's native `/v1/model_response` endpoint for each model inference (preserving full `ModelOutput` with token IDs and logprobs)
4. Remote agent returns the completed `Episode` (with trajectories, rewards, etc.)
5. Trainer collects all episodes, computes advantages, updates the policy
6. Repeat -- the inference API automatically serves the updated model weights

## Files

| File | Description |
|------|-------------|
| `remote_agent_server.py` | FastAPI server implementing the remote agent endpoint |
| `train_remote_solver_judge.py` | Hydra training script (same as local, but enables `remote_agent` config) |
| `train_remote_solver_judge.sh` | Shell script with all config overrides |

## Quick Start

### 1. Prepare the dataset

```bash
python -m examples.solver_judge.prepare_countdown_data
```

### 2. Start the remote agent server

```bash
# In a separate terminal (or Docker container)
python -m examples.remote_solver_judge.remote_agent_server --port 5100
```

You can start multiple servers for parallelism:
```bash
python -m examples.remote_solver_judge.remote_agent_server --port 5100 &
python -m examples.remote_solver_judge.remote_agent_server --port 5101 &
```

### 3. Launch training

```bash
# Using the shell script (edit variables inside as needed):
bash examples/remote_solver_judge/train_remote_solver_judge.sh

# Or directly with Hydra overrides:
python -m examples.remote_solver_judge.train_remote_solver_judge \
    rllm/backend=tinker \
    rllm.remote_agent.enabled=true \
    'rllm.remote_agent.endpoints=["http://localhost:5100"]' \
    rllm.remote_agent.inference_api.port=8089 \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    data.train_batch_size=32
```

### Multiple remote agents

To distribute across multiple agent servers:

```bash
python -m examples.remote_solver_judge.train_remote_solver_judge \
    rllm.remote_agent.enabled=true \
    'rllm.remote_agent.endpoints=["http://agent1:5100","http://agent2:5100"]' \
    ...
```

Tasks are distributed across endpoints using round-robin load balancing.

## Configuration

The remote agent config section in `rllm.remote_agent`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable remote agent mode |
| `endpoints` | `[]` | List of remote agent server URLs |
| `inference_api.host` | `0.0.0.0` | Host for the trainer's inference API |
| `inference_api.port` | `8089` | Port for the trainer's inference API |
| `timeout` | `600` | Per-episode HTTP timeout (seconds) |
| `max_concurrent` | `128` | Max concurrent requests to remote agents |
| `retry_limit` | `3` | Retries per failed remote call |

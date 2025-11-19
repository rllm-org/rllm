# Eval Protocol FrozenLake Example

This example shows how to use **Eval Protocol**'s FrozenLake environment from within **rLLM** using the generic `EvalProtocolWorkflow`.

For a conceptual overview of how this integration works and how it generalizes to other benchmarks, see the core-concepts page on [Eval Protocol Integration](../../docs/core-concepts/eval-protocol.md).

---

## Quick Start

### Prepare FrozenLake dataset

From the project root:

```bash
cd examples/eval_protocol
python prepare_frozen_lake_data.py
```

This script builds and registers the `frozen_lake_eval_protocol` train/test splits in the rLLM `DatasetRegistry`.

### Run FrozenLake workflow (inference)

Once your Fireworks API credentials are configured, you can run a small batch of FrozenLake tasks through Eval Protocol and rLLM:

```bash
python run_frozen_lake_flow.py
```

This will:

- Load the `frozen_lake_eval_protocol` test split.
- Use `EvalProtocolWorkflow` (with `env_path="eval_protocol.benchmarks.test_frozen_lake"`) to run rollouts via Eval Protocol.
- Print per-task rewards/accuracy and save results to `logs/frozen_lake_results.json`.

### Train an RL agent

To train an agent against the same Eval Protocol FrozenLake environment:

```bash
bash train_frozen_lake_flow.sh
```

This uses `EvalProtocolWorkflow` inside `AgentTrainer` (via Hydra configs) to:

- Generate rollouts using Eval Protocol’s rollout processor and MCP server.
- Compute rewards via the Eval Protocol evaluation function.
- Optimize the underlying model with PPO/GRPO.

You can edit `train_frozen_lake_flow.sh` to customize model path, Fireworks deployment, and training hyperparameters.

---

## Code Reference

### Data preparation

Script that builds and registers the FrozenLake Eval Protocol dataset:

```python title="examples/eval_protocol/prepare_frozen_lake_data.py"
--8<-- "examples/eval_protocol/prepare_frozen_lake_data.py"
```

### Workflow runner

Main script for running the FrozenLake Eval Protocol workflow through rLLM:

```python title="examples/eval_protocol/run_frozen_lake_flow.py"
--8<-- "examples/eval_protocol/run_frozen_lake_flow.py"
```

### Training script

Agent training implementation using `EvalProtocolWorkflow` and `AgentTrainer`:

```python title="examples/eval_protocol/train_frozen_lake_flow.py"
--8<-- "examples/eval_protocol/train_frozen_lake_flow.py"
```

# Eval Protocol Integration

This page explains how **Eval Protocol** environments are integrated into **rLLM** via the generic `EvalProtocolWorkflow`.

[Eval Protocol](https://github.com/eval-protocol/python-sdk) is an is an open-source, language-agnostic framework that makes it easy to do reinforcement fine-tuning on agents, across any framework, environment, or trainer. It ships a large collection of pre-implemented and integrated environments and benchmarks, which is why this integration exists: it lets rLLM plug into a wide variety of Eval Protocol tasks with minimal extra glue code.

At a high level:

- Eval Protocol defines **benchmarks and evaluation tests** using the `@evaluation_test` decorator.
- rLLM’s `EvalProtocolWorkflow` discovers those tests by **module path** (the `env_path` argument).
- rLLM **reads the rollout/evaluation configuration** from the Eval Protocol test (rollout processor, MCP server path, etc.).
- The workflow then exposes the Eval Protocol benchmark as a standard **rLLM Workflow**, returning `Episode` / `Trajectory` objects usable for evaluation and RL training.

We’ll use the FrozenLake benchmark as a concrete example, but the same pattern applies to any Eval Protocol environment.

---

## 1. Wiring Eval Protocol into AgentWorkflowEngine

In rLLM, Eval Protocol benchmarks are surfaced to the workflow engine via `EvalProtocolWorkflow`. For example, in `examples/eval_protocol/run_frozen_lake_flow.py`:

```python
engine = AgentWorkflowEngine(
    workflow_cls=EvalProtocolWorkflow,
    workflow_args={
        "env_path": "eval_protocol.benchmarks.test_frozen_lake",
        "lite_llm_prefix": "fireworks_ai/",
        "steps": 30,
        "temperature": 1.0,
        "max_tokens": 16384,
    },
    rollout_engine=rollout_engine,
    n_parallel_tasks=n_parallel_tasks,
    retry_limit=1,
)
```

### Key pieces

- **`workflow_cls=EvalProtocolWorkflow`**
  - Tells `AgentWorkflowEngine` to instantiate `EvalProtocolWorkflow` for each task.
  - This workflow adapts Eval Protocol’s rollout/eval to rLLM’s `Episode` abstraction.

- **`env_path`**
  - A Python module path pointing to the Eval Protocol test module, e.g. `"eval_protocol.benchmarks.test_frozen_lake"`.
  - `EvalProtocolWorkflow` imports this module and discovers the evaluation test function decorated with Eval Protocol’s `@evaluation_test`.

- **`lite_llm_prefix`**
  - Prefix added to the rollout model name passed to Eval Protocol (e.g. `"fireworks_ai/"`).
  - Combined with `rollout_engine.model` to form the full model id used by Eval Protocol’s rollout processor.

- **`steps`, `temperature`, `max_tokens`**
  - Standard generation/rollout parameters.
  - Forwarded as part of the rollout processor configuration so eval-protocol controls how many steps and tokens are used.

Once `AgentWorkflowEngine` is configured this way, any task you pass to `engine.execute_tasks(...)` is run through the Eval Protocol environment wrapped by `EvalProtocolWorkflow`.

---

## 2. How Eval Protocol defines the environment (FrozenLake example)

On the Eval Protocol side, a benchmark is defined using the `@evaluation_test` decorator. The FrozenLake test looks like:

```python
@evaluation_test(
    input_dataset=["tests/pytest/data/frozen_lake_dataset.jsonl"],
    dataset_adapter=frozen_lake_to_evaluation_row,
    completion_params=[
        {
            "temperature": 0.0,
            "max_tokens": 4096,
            "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct",
        }
    ],
    rollout_processor=MCPGymRolloutProcessor(),
    passed_threshold=0.66,
    num_runs=1,
    max_concurrent_rollouts=3,
    mode="pointwise",
    server_script_path="examples/frozen_lake_mcp/server.py",
)
def test_frozen_lake_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluate how well the model plays FrozenLake by checking if it reaches the
    goal while avoiding holes.
    """
    score = row.get_total_reward()

    if score == 1.0:
        reason = "Agent reached the goal"
    else:
        reason = "Agent did not reach the goal"

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=reason,
    )

    return row
```

The decorator attaches metadata that describes:

- Which **rollout processor** to use (`MCPGymRolloutProcessor`).
- How to start the **MCP environment server** (`server_script_path`).
- Any additional **rollout configuration** (e.g. `mode`, `max_concurrent_rollouts`).

`EvalProtocolWorkflow` reads exactly this metadata to know how to run the environment.

---

## 3. What EvalProtocolWorkflow does

Putting it all together, when `EvalProtocolWorkflow.run(task, uid, **kwargs)` is called it:

1. Builds an `EvaluationRow` from the rLLM task dict (id, system prompt, environment context, user prompt template).
2. Combines Eval Protocol’s metadata from `@evaluation_test` (rollout processor, MCP config, mode, etc.) with the workflow args (model, steps, temperature, max_tokens) to create a rollout config.
3. Runs the rollout via the Eval Protocol `rollout_processor`, calls the Eval Protocol evaluation function, and converts the resulting `EvaluationRow` into an rLLM `Episode` / `Trajectory` / `Step` (attaching the final score and metrics).

If an error occurs at any point, the workflow returns an `Episode` marked incorrect with zero reward and an `"error"` field in metrics instead of raising.

---

## 4. Using Eval Protocol benchmarks out of the box

Because the integration is driven entirely by `env_path` and `@evaluation_test` metadata:

- **Any Eval Protocol test** that uses `@evaluation_test` can be used as an rLLM workflow.
- You do **not** need to write a custom workflow per environment.
- You can plug in:
  - Benchmarks like **tau2-bench**, **AIME25**, **HealthBench**.
  - Any **custom eval-protocol environment** you write yourself.
  - Various **MCP-based environments**, for example Frozen Lake is MCP tool-based.
  - Ongoing integrations such as **OpenEnv**-backed environments as they are wired into eval-protocol.

To use a different benchmark, you typically:

1. Change `env_path` to point to the target test module (e.g. `"eval_protocol.benchmarks.test_tau2_bench"`).
2. Provide a dataset whose task rows can be mapped to `EvaluationRow` by `_task_to_evaluation_row`.
3. Reuse the same `EvalProtocolWorkflow` with `AgentWorkflowEngine` (for evaluation) or `AgentTrainer` (for RL training).

This makes Eval Protocol a **plug-and-play source of environments** for rLLM, without rewriting workflow logic for each benchmark.

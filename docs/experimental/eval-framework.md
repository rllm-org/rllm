# Benchmark Evaluation Framework

Evaluate any model on any benchmark with one command.

```bash
# One-time setup
rllm setup
# > Provider: openai
# > API key: ****
# > Default model: gpt-4o-mini

# Run eval — no --base-url needed
rllm eval gsm8k

# Override model for a single run
rllm eval gsm8k --model gpt-4o

# Or specify --base-url directly (bypasses proxy)
rllm eval gsm8k --agent math --base-url http://localhost:8000/v1 --model qwen3.5-35b

# Use a custom evaluator
rllm eval gsm8k --evaluator my_module:MyEvaluator --base-url ... --model ...
```

## Overview

The eval framework uses a two-stage pipeline with four layers:

1. **Datasets** — what to evaluate on (pulled from HuggingFace, stored in `~/.rllm/datasets/`)
2. **AgentFlows** — run an agent program that produces an `Episode` (trajectories without rewards)
3. **Evaluators** — score the Episode, producing reward + signals
4. **Runner** — orchestrates the pipeline in parallel and produces results

```
Task → AgentFlow.run(task, config) → Episode (trajectories without rewards)
                                        │
                                        ▼
                        Evaluator.evaluate(task, episode)
                                        │
                                        ▼
                        Episode (mutated: trajectory.reward + signals set,
                                episode.is_correct set)
```

This separation means you can swap evaluation logic without modifying agent code, produce multiple evaluation signals (accuracy, format, F1), and run diverse agent programs (multi-agent, ADK, OpenAI SDK) through `rllm eval`.

Everything lives in `rllm/experimental/` since this design will eventually deprecate the old agent classes (`MathAgent`, `CodeAgent`, etc.).

## Quick Start

```bash
# Install
pip install -e .

# Configure provider (one-time)
rllm setup

# List available benchmarks
rllm dataset list --all

# Pull a dataset
rllm dataset pull gsm8k

# List available agents
rllm agent list

# Run evaluation (uses config from 'rllm setup')
rllm eval gsm8k

# Override model
rllm eval gsm8k --model gpt-4o

# Quick test with limited examples
rllm eval gsm8k --max-examples 10

# Use a custom agent flow
rllm eval gsm8k --agent my_module:my_agent

# Use a custom evaluator
rllm eval gsm8k --evaluator my_module:MyEvaluator

# Bypass proxy — connect directly to an endpoint
rllm eval gsm8k --base-url http://localhost:8000/v1 --model qwen3.5-35b
```

---

## Architecture

### Directory Layout

```
rllm/
├── registry/
│   ├── datasets.json          # Dataset catalog (what's available on HF)
│   └── agents.json            # Agent flow catalog (built-in agents)
├── experimental/
│   ├── agents/                # Built-in AgentFlow classes
│   │   ├── __init__.py
│   │   ├── math_agent.py      # MathAgentFlow, CountdownAgentFlow
│   │   ├── code_agent.py      # CodeAgentFlow
│   │   └── qa_agent.py        # QAAgentFlow
│   ├── eval/                  # Evaluation engine
│   │   ├── __init__.py
│   │   ├── types.py           # AgentFlow/Evaluator protocols, built-in evaluators
│   │   ├── agent_loader.py    # Resolves agent by name or import path
│   │   ├── evaluator_loader.py # Resolves evaluator by name or import path
│   │   ├── runner.py          # EvalRunner: two-stage parallel eval orchestrator
│   │   ├── results.py         # EvalResult: metrics + signals + save/report
│   │   ├── config.py          # RllmConfig: persistent provider/model settings
│   │   └── proxy.py           # EvalProxyManager: LiteLLM proxy for eval
│   └── cli/                   # CLI commands
│       ├── __init__.py
│       ├── main.py            # Entry point: rllm [dataset|eval|agent|setup]
│       ├── dataset.py         # rllm dataset [list|pull|info|inspect|remove]
│       ├── eval.py            # rllm eval <benchmark> ...
│       ├── agent.py           # rllm agent [list|info]
│       ├── setup.py           # rllm setup (interactive provider config)
│       ├── _display.py        # Table formatting
│       └── _pull.py           # HF download + catalog loading
├── data/
│   ├── dataset.py             # Dataset + DatasetRegistry (v2 format)
│   └── dataset_types.py       # DatasetMetadata, DatasetConfig, enums
└── rewards/                   # Reward functions (reused by evaluators)
    ├── math_reward.py
    ├── code_reward.py
    ├── countdown_reward.py
    └── search_reward.py

~/.rllm/
├── config.json                # Provider/model config (from 'rllm setup')
├── datasets/
│   ├── registry.json          # Local dataset registry (v2 format)
│   ├── gsm8k/
│   │   ├── train.parquet
│   │   ├── train_verl.parquet
│   │   ├── test.parquet
│   │   └── test_verl.parquet
│   └── ...
└── eval_results/
    └── gsm8k_qwen3.5-35b_20260301_143022.json
```

### The AgentFlow Contract

Every agent implements the `AgentFlow` protocol — a class with a `run` method:

```python
class AgentFlow(Protocol):
    def run(self, task: dict, config: AgentConfig) -> Episode: ...
```

**`task`** — a single dataset entry as a dict. Fields vary by dataset:

| Dataset   | Key fields                                     |
|-----------|------------------------------------------------|
| gsm8k     | `question`, `ground_truth`, `data_source`      |
| math500   | `question`, `ground_truth`, `data_source`      |
| countdown | `target`, `nums`, `data_source`                |
| deepcoder | `question`, `ground_truth` (tests), `data_source` |
| hotpotqa  | `question`, `ground_truth`, `data_source`      |

**`config`** — an `AgentConfig` dataclass with:

| Field         | Description                                  |
|---------------|----------------------------------------------|
| `base_url`    | OpenAI-compatible API endpoint               |
| `model`       | Model name to use                            |
| `session_uid` | Unique ID for trace correlation              |
| `metadata`    | Optional metadata dict                       |

**Returns** — an `Episode` containing one or more `Trajectory` objects and optional `artifacts` (structured outputs for evaluators).

Key design points:
- AgentFlows do **not** compute reward — that's the evaluator's job
- `Episode.artifacts` stores extracted answers (e.g., `{"answer": "42"}`) for evaluators to consume
- An AgentFlow can orchestrate multiple agents, each contributing a trajectory to the Episode

### The Evaluator Contract

Every evaluator implements the `Evaluator` protocol:

```python
class Evaluator(Protocol):
    def evaluate(self, task: dict, episode: Episode) -> EvalOutput: ...
```

`EvalOutput` contains:
- `reward: float` — overall reward (typically 0.0 or 1.0)
- `is_correct: bool` — whether the response was correct
- `signals: list[Signal]` — named evaluation signals (e.g., accuracy, format, f1)
- `metadata: dict` — optional metadata

After evaluation, the runner writes results back onto each trajectory:

```python
eval_output = evaluator.evaluate(task, episode)
for traj in episode.trajectories:
    traj.reward = eval_output.reward
    traj.signals = {s.name: s.value for s in eval_output.signals}
episode.is_correct = eval_output.is_correct
```

### Relationship to Training Workflow

Both `AgentFlow` (eval) and `Workflow` (training) produce `Episode` objects:

```
Workflow (training)                    AgentFlow (eval)
├── requires RolloutEngine             ├── lightweight: just needs base_url + model
├── handles TerminationEvent           ├── no training infra dependencies
├── postprocess_episode() with gamma   ├── produces clean Episode with trajectories
└── run(task, uid) -> Episode          └── run(task, config) -> Episode
```

In the future, an `AgentFlow` could be wrapped into a `Workflow` for training, or vice versa.

### Example AgentFlow

```python
from openai import OpenAI
from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

class MyMathAgent:
    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "Solve the math problem. Put answer in \\boxed{}."},
                {"role": "user", "content": task["question"]},
            ],
        )
        answer = response.choices[0].message.content or ""

        step = Step(input=task["question"], output=answer, done=True)
        traj = Trajectory(name="solver", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": answer})

# Singleton for registry
my_math_agent = MyMathAgent()
```

Use it:
```bash
rllm eval gsm8k --agent my_module:my_math_agent --base-url ... --model ...
```

### Example: Multi-Agent Flow

An AgentFlow can orchestrate multiple agents, each contributing a trajectory:

```python
class PlanAndExecuteFlow:
    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        # Agent 1: Planner
        plan_resp = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": f"Plan steps to solve: {task['question']}"}],
        )
        plan = plan_resp.choices[0].message.content
        planner_traj = Trajectory(
            name="planner",
            steps=[Step(input=task["question"], output=plan, done=True)],
        )

        # Agent 2: Executor
        exec_resp = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": f"Execute this plan:\n{plan}"}],
        )
        result = exec_resp.choices[0].message.content
        executor_traj = Trajectory(
            name="executor",
            steps=[Step(input=plan, output=result, done=True)],
        )

        return Episode(
            task=task,
            trajectories=[planner_traj, executor_traj],
            artifacts={"plan": plan, "answer": result},
        )
```

### Example: Custom Evaluator

```python
from rllm.experimental.eval.types import EvalOutput, Signal, _extract_agent_answer

class StrictMatchEvaluator:
    """Evaluator that checks exact string match."""

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        answer = _extract_agent_answer(episode)
        expected = task.get("ground_truth", "")
        is_correct = answer.strip() == str(expected).strip()
        return EvalOutput(
            reward=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            signals=[Signal(name="exact_match", value=1.0 if is_correct else 0.0)],
        )
```

---

## The Two Registries

### Dataset Registry (`rllm/registry/datasets.json`)

Checked into the repo. Defines what benchmarks are available to pull from HuggingFace:

```json
{
  "version": 1,
  "datasets": {
    "gsm8k": {
      "description": "Grade school math word problems (8.5K train, 1.3K test)",
      "source": "rllm-org/gsm8k",
      "category": "math",
      "splits": ["train", "test"],
      "default_agent": "math",
      "reward_fn": "math_reward_fn",
      "eval_split": "test"
    }
  }
}
```

| Field           | Purpose                                          |
|-----------------|--------------------------------------------------|
| `source`        | HuggingFace repo (used by `rllm dataset pull`)   |
| `category`      | Grouping: `math`, `code`, `qa`                   |
| `splits`        | Available splits on HF                           |
| `default_agent` | Agent used when `--agent` is omitted             |
| `reward_fn`     | Evaluator auto-resolved from this name           |
| `eval_split`    | Default split for `rllm eval`                    |

The `reward_fn` field is used to auto-resolve an evaluator when `--evaluator` is omitted. The mapping is:

| `reward_fn` value      | Evaluator class      |
|------------------------|----------------------|
| `math_reward_fn`       | `MathEvaluator`      |
| `countdown_reward_fn`  | `CountdownEvaluator` |
| `code_reward_fn`       | `CodeEvaluator`      |
| `f1_reward_fn`         | `F1Evaluator`        |

**To add a new dataset:** add an entry here and push the data to a HuggingFace repo.

### Agent Registry (`rllm/registry/agents.json`)

Checked into the repo. Maps agent names to their Python `AgentFlow` instances:

```json
{
  "version": 1,
  "agents": {
    "math": {
      "description": "Single-turn math reasoning with step-by-step and boxed answer",
      "module": "rllm.experimental.agents.math_agent",
      "function": "math_agent"
    }
  }
}
```

The `"function"` field points to an `AgentFlow` instance (an object with a `.run()` method).

**To add a new built-in agent:** create the `AgentFlow` class in `rllm/experimental/agents/`, add a singleton instance, then add an entry here.

---

## Local Dataset Registry (v2 Format)

When datasets are pulled or registered, they're tracked in `~/.rllm/datasets/registry.json` using the v2 format:

```json
{
  "version": 2,
  "datasets": {
    "gsm8k": {
      "metadata": {
        "source": "rllm-org/gsm8k",
        "description": "Grade school math word problems",
        "category": "math"
      },
      "splits": {
        "test": {
          "path": "gsm8k/test.parquet",
          "num_examples": 1319,
          "fields": ["question", "ground_truth", "data_source"]
        }
      }
    }
  }
}
```

- Paths are relative to `~/.rllm/datasets/`
- Old v1 registries (`{name: {split: abs_path}}`) are auto-migrated on first load
- Override the storage location with the `RLLM_HOME` environment variable

---

## CLI Reference

### `rllm setup`

```
rllm setup
```

Interactive configuration for provider, API key, and default model. Saves to `~/.rllm/config.json`. Re-running shows current config and allows overriding values.

### `rllm eval`

```
rllm eval <benchmark> [OPTIONS]

Options:
  --agent TEXT         Agent flow: registry name or module:object path
  --evaluator TEXT     Evaluator: registry name or module:class path
  --base-url TEXT      OpenAI-compatible API endpoint (if omitted, auto-starts LiteLLM proxy)
  --model TEXT         Model name (defaults to config from 'rllm setup'; required with --base-url)
  --split TEXT         Dataset split (default: from catalog eval_split)
  --concurrency INT    Parallel requests (default: 64)
  --max-examples INT   Limit examples (for dev/testing)
  --output TEXT        Output file path for results JSON
```

**Proxy mode (default):** When `--base-url` is omitted, the command loads config from `rllm setup`, auto-starts a LiteLLM proxy, runs the eval, and shuts down the proxy.

**Direct mode:** When `--base-url` is provided, connects directly to the given endpoint (requires `--model`).

**Evaluator resolution:** When `--evaluator` is omitted, the evaluator is auto-resolved from the dataset catalog's `reward_fn` field. You can override with `--evaluator <name>` (registry name) or `--evaluator my_module:MyEvaluator` (import path).

Auto-pulls the dataset if not available locally. Uses the catalog's `default_agent` if `--agent` is omitted.

Output:
```
Benchmark: gsm8k (test, 1319 examples)
Model:     qwen3.5-35b
Agent:     math (Single-turn math reasoning with step-by-step and boxed answer)
Evaluator: math_reward_fn

Evaluating: 100%|████████████████████████| 1319/1319 [02:34<00:00, 8.54 it/s]

Results:
  Accuracy:  82.3% (1086/1319)
  Errors:    0
  Accuracy:  0.823

Saved to ~/.rllm/eval_results/gsm8k_qwen3.5-35b_20260301_143022.json
```

### `rllm dataset`

| Command                          | Description                              |
|----------------------------------|------------------------------------------|
| `rllm dataset list`              | List locally pulled datasets             |
| `rllm dataset list --all`        | Show all available + pull status          |
| `rllm dataset pull <name>`       | Pull from HuggingFace                    |
| `rllm dataset info <name>`       | Metadata, splits, default agent, reward  |
| `rllm dataset inspect <name>`    | Sample data rows                         |
| `rllm dataset remove <name>`     | Remove local dataset                     |

### `rllm agent`

| Command                   | Description                              |
|---------------------------|------------------------------------------|
| `rllm agent list`         | List registered agent scaffolds          |
| `rllm agent info <name>`  | Show description + compatible datasets   |

---

## Built-in AgentFlows

| Name        | Class                | Module                                       | Datasets            |
|-------------|----------------------|----------------------------------------------|---------------------|
| `math`      | `MathAgentFlow`      | `rllm.experimental.agents.math_agent`        | gsm8k, math500      |
| `countdown` | `CountdownAgentFlow` | `rllm.experimental.agents.math_agent`        | countdown           |
| `code`      | `CodeAgentFlow`      | `rllm.experimental.agents.code_agent`        | deepcoder           |
| `qa`        | `QAAgentFlow`        | `rllm.experimental.agents.qa_agent`          | hotpotqa            |

Each AgentFlow:
1. Constructs a prompt from the task fields
2. Calls the model via `openai.OpenAI(base_url=config.base_url)`
3. Returns an `Episode` with trajectories and `artifacts` (e.g., `{"answer": "..."}`)
4. Does **not** compute reward — evaluation is handled separately by the evaluator

## Built-in Evaluators

| Registry Name          | Class                | Source                                   |
|------------------------|----------------------|------------------------------------------|
| `math_reward_fn`       | `MathEvaluator`      | Uses `extract_answer` + `grade_answer_*` from `rllm/rewards/math_utils/` |
| `countdown_reward_fn`  | `CountdownEvaluator` | Uses `compute_score` from `rllm/rewards/countdown_reward.py` |
| `code_reward_fn`       | `CodeEvaluator`      | Uses `RewardCodeFn` from `rllm/rewards/code_reward.py` |
| `f1_reward_fn`         | `F1Evaluator`        | Token-overlap F1 score (self-contained) |

Additional evaluators:
- **`CompoundEvaluator`** — runs multiple evaluators, merges signals, computes weighted reward

---

## Programmatic Usage

### EvalRunner (two-stage pipeline)

```python
import asyncio
from rllm.data import DatasetRegistry
from rllm.experimental.eval import EvalRunner, load_agent, load_evaluator

dataset = DatasetRegistry.load_dataset("gsm8k", "test")
agent = load_agent("math")
evaluator = load_evaluator("math_reward_fn")

runner = EvalRunner(base_url="http://localhost:8000/v1", model="qwen3.5-35b", concurrency=64)
result = asyncio.run(runner.run(dataset, agent, evaluator, agent_name="math"))

print(result.summary_table())
print(result.signal_averages)  # e.g., {"accuracy": 0.823}
result.save()  # saves to ~/.rllm/eval_results/
```

### Agent Loader

```python
from rllm.experimental.eval import load_agent

# By registry name — returns an AgentFlow instance
agent = load_agent("math")

# By import path
agent = load_agent("my_module:my_agent")
```

### Evaluator Loader

```python
from rllm.experimental.eval import load_evaluator, resolve_evaluator_from_catalog

# By registry name
evaluator = load_evaluator("math_reward_fn")

# By import path (class is instantiated automatically)
evaluator = load_evaluator("my_module:MyEvaluator")

# Auto-resolve from datasets.json reward_fn field
evaluator = resolve_evaluator_from_catalog("gsm8k")  # returns MathEvaluator
```

### DatasetRegistry

```python
from rllm.data import DatasetRegistry

# Register manually
data = [{"question": "1+1", "ground_truth": "2"}]
ds = DatasetRegistry.register_dataset("my_ds", data, split="test", category="math")

# Load
ds = DatasetRegistry.load_dataset("gsm8k", "test")

# Query
names = DatasetRegistry.get_dataset_names()
splits = DatasetRegistry.get_dataset_splits("gsm8k")
info = DatasetRegistry.get_dataset_info("gsm8k")
exists = DatasetRegistry.dataset_exists("gsm8k", "test")
```

---

## Adding a New Benchmark

1. **Upload data** to a HuggingFace repo (e.g., `rllm-org/my_benchmark`) with parquet files per split.

2. **Add catalog entry** in `rllm/registry/datasets.json`:
   ```json
   "my_benchmark": {
     "description": "My new benchmark (1K test)",
     "source": "rllm-org/my_benchmark",
     "category": "math",
     "splits": ["test"],
     "default_agent": "math",
     "reward_fn": "math_reward_fn",
     "eval_split": "test"
   }
   ```

3. **Test it:**
   ```bash
   rllm dataset pull my_benchmark
   rllm dataset inspect my_benchmark
   rllm eval my_benchmark --base-url ... --model ...
   ```

## Adding a New AgentFlow

1. **Create the class** in `rllm/experimental/agents/`:
   ```python
   # rllm/experimental/agents/my_agent.py
   from rllm.experimental.eval.types import AgentConfig
   from rllm.types import Episode, Step, Trajectory

   class MyAgentFlow:
       def run(self, task: dict, config: AgentConfig) -> Episode:
           # ... call model, build trajectories, return Episode
           step = Step(input=task["question"], output=answer, done=True)
           traj = Trajectory(name="solver", steps=[step])
           return Episode(task=task, trajectories=[traj], artifacts={"answer": answer})

   my_agent = MyAgentFlow()
   ```

2. **Register it** in `rllm/registry/agents.json`:
   ```json
   "my_agent": {
     "description": "My custom agent flow",
     "module": "rllm.experimental.agents.my_agent",
     "function": "my_agent"
   }
   ```

3. **Export it** in `rllm/experimental/agents/__init__.py`.

4. **Use it:**
   ```bash
   rllm eval gsm8k --agent my_agent --base-url ... --model ...
   ```

Or skip registration and use the import path directly:
```bash
rllm eval gsm8k --agent rllm.experimental.agents.my_agent:my_agent --base-url ... --model ...
```

## Adding a New Evaluator

1. **Create the class:**
   ```python
   # my_evaluators.py
   from rllm.experimental.eval.types import EvalOutput, Signal, _extract_agent_answer
   from rllm.types import Episode

   class MyEvaluator:
       def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
           answer = _extract_agent_answer(episode)
           expected = task.get("ground_truth", "")
           is_correct = answer.strip() == str(expected).strip()
           return EvalOutput(
               reward=1.0 if is_correct else 0.0,
               is_correct=is_correct,
               signals=[Signal(name="exact_match", value=1.0 if is_correct else 0.0)],
           )
   ```

2. **Use it via import path:**
   ```bash
   rllm eval gsm8k --evaluator my_evaluators:MyEvaluator --base-url ... --model ...
   ```

3. **Or register it** by adding to the `_EVALUATOR_REGISTRY` in `rllm/experimental/eval/evaluator_loader.py` and linking it to a `reward_fn` name in `datasets.json`.

---

## Key Reuse Map

| Component | Source | Used by |
|-----------|--------|---------|
| `AgentFlow` protocol | `rllm/experimental/eval/types.py` | All agents, EvalRunner |
| `Evaluator` protocol | `rllm/experimental/eval/types.py` | All evaluators, EvalRunner |
| `Step`, `Trajectory`, `Episode` types | `rllm/types.py` | All agents, evaluators |
| `extract_answer`, `grade_answer_*` | `rllm/rewards/math_utils/utils.py` | `MathEvaluator` |
| `compute_score` | `rllm/rewards/countdown_reward.py` | `CountdownEvaluator` |
| `RewardCodeFn` | `rllm/rewards/code_reward.py` | `CodeEvaluator` |
| `Dataset`, `DatasetRegistry` | `rllm/data/dataset.py` | CLI, EvalRunner |
| `datasets.json` `reward_fn` field | `rllm/registry/datasets.json` | Evaluator auto-resolution |

---

## Tests

```bash
# All eval framework tests
pytest tests/data/ tests/eval/ tests/cli/ -v

# Individual suites
pytest tests/data/test_registry_migration.py -v   # Registry v1→v2 migration
pytest tests/eval/test_eval_types.py -v            # AgentFlow/Evaluator protocols + evaluators
pytest tests/eval/test_evaluator_loader.py -v      # Evaluator loading + catalog resolution
pytest tests/eval/test_eval_runner.py -v           # Two-stage runner pipeline
pytest tests/eval/test_runner.py -v                # EvalRunner basic contract
pytest tests/eval/test_agents.py -v                # Built-in AgentFlow classes
pytest tests/eval/test_eval_config.py -v           # Eval config load/save
pytest tests/eval/test_eval_proxy.py -v            # EvalProxyManager
pytest tests/cli/test_dataset_commands.py -v       # Dataset CLI
pytest tests/cli/test_eval_command.py -v           # Eval CLI (incl. --evaluator flag)
pytest tests/cli/test_setup_command.py -v          # Setup CLI
```

Test fixtures use `monkeypatch` to redirect `DatasetRegistry` paths to temp directories, preventing pollution of `~/.rllm/`.

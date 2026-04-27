# Task Format

A **task** in rLLM is a directory of files that fully describes one RL
environment instance: what the agent sees (instruction), where it works
(environment), and how it is graded (verifier). The framework reads the
files and orchestrates everything else.

This is a Harbor-compatible format with a small `[rllm]` extension —
existing Harbor tasks run without modification.

## Quick reference

```
my-benchmark/
├── dataset.toml          # optional manifest (name, default_agent, verifier)
├── instruction.md.tpl    # optional template for row-based datasets
├── data/                 # for row-based datasets
│   └── test.jsonl
├── tests/                # verifier scripts
│   ├── test.sh             # shell verifier (runs in sandbox)
│   └── evaluate.py         # Python verifier (runs on host or hybrid)
├── environment/          # for sandbox tasks
│   ├── Dockerfile
│   └── files/              # files seeded into the agent's workdir
└── task-001/             # for task-per-directory shape
    ├── task.toml
    ├── instruction.md
    └── tests/test.sh
```

## Two physical shapes

### Shape A — task-per-directory (Harbor-style)

Each task is its own subdirectory with `task.toml`, `instruction.md`,
`environment/`, and `tests/`. Use this for SWE-bench, terminal-bench,
or any benchmark where each instance has its own seed files and tests.

```
my-coding-bench/
├── dataset.toml
├── fix-sort/
│   ├── task.toml
│   ├── instruction.md
│   ├── environment/Dockerfile
│   └── tests/test.sh
└── fix-search/
    └── ...
```

### Shape B — rows-with-shared-verifier (gsm8k-style)

A single `data/<split>.jsonl` provides per-task data; all rows share
one verifier in `tests/evaluate.py`. Use this for math, MCQ, code,
or any benchmark with thousands of small instances.

```
gsm8k/
├── dataset.toml
├── instruction.md.tpl
├── data/test.jsonl
└── tests/evaluate.py
```

## Verifier resolution

The Runner reads `dataset.toml` (or `task.toml`) to find each task's
verifier. Four ways to declare it:

```toml
# A. shell script in the benchmark dir, runs in sandbox
[verifier]
script = "tests/test.sh"

# B. Python module in the benchmark dir
[verifier]
module = "tests.evaluate"        # uses tests/evaluate.py
function = "evaluate"            # default, can omit

# C. registered name (works with @evaluator-decorated functions
#    and built-in score_fns)
[verifier]
name = "math_reward_fn"

# D. import path
[verifier]
import_path = "rllm.eval.score_fns.math:evaluate"
```

If unset, the loader auto-detects: `tests/test.sh` → A, `tests/evaluate.py` → B.

### Verifier function signature

```python
def evaluate(task: Task, episode: Episode) -> EvalOutput:  # canonical
def evaluate(metadata: dict, trajectory: dict) -> dict:    # lightweight
```

The framework picks based on the function's parameter names. Both forms
work; `(metadata, trajectory)` is convenient for ad-hoc verifiers.

Returns are coerced: `float`, `bool`, `dict`, `tuple[float, bool]`,
`EvalOutput` are all accepted.

## Reward contract for shell verifiers

The script writes a reward file in the sandbox; the framework reads it.
Search order (first existing wins):

1. `/tmp/rllm/reward.json`
2. `/logs/verifier/reward.json` (Harbor convention)
3. `/logs/verifier/reward.txt` (Harbor convention; single float)

JSON shape:
```json
{
  "reward": 0.75,
  "is_correct": false,
  "signals": {"tests_passed": 0.75},
  "metadata": {"tests_total": 8, "passed": 6}
}
```

## Agent harnesses

`--agent` picks the **harness** — the agent driver. Built-ins:

| Name | Where it runs | Best for |
|---|---|---|
| `simple` | host | one-shot LLM calls (math, mcq, qa) |
| `react` | sandbox | multi-turn bash tool use (coding, system tasks) |
| `claude-code` | sandbox | spawning the Claude Code CLI inside the container |

Add your own with `register_harness("my-harness", MyHarness)` or the
`@rollout` decorator. Use a `module:Class` import path on the CLI to
load harness classes from any module.

## Running

```bash
# Local benchmark directory
rllm eval ./my-benchmark/ --agent simple
rllm eval ./my-coding-bench/ --agent react --sandbox-backend docker

# Catalog dataset (auto-materialised on first pull)
rllm dataset pull gsm8k
rllm eval gsm8k --agent simple --max-examples 10

# Single task directory (one-off)
rllm eval ./harbor/examples/tasks/hello-world --agent react --sandbox-backend docker
```

## User isolation (optional, for adversarial agents)

To stop an agent from writing the reward file directly, declare a
non-root agent user in `task.toml` and create that user in the
Dockerfile:

```toml
[agent]
user = "agent"

[verifier]
user = "root"
```

```dockerfile
RUN useradd -m -u 1000 agent
```

The framework then chowns the workdir to `agent`, locks
`/logs/verifier`, `/tmp/rllm`, and `/tests` to root-only, and runs
agent commands as `agent` while the verifier runs as `root`. The
kernel enforces the boundary — `echo 1 > /logs/verifier/reward.txt`
returns `Permission denied`.

## How tasks become benchmark dirs

Three ways:

1. **Hand-author** — write the directory structure yourself.
2. **`rllm dataset pull <name>`** — pulls from HuggingFace and
   materialises into `~/.rllm/datasets/<name>/`. The catalog entry's
   `transform` function shapes the rows; `verifier` gets written to
   `dataset.toml` as a registered name.
3. **Harbor packages** — the `harbor:` prefix on the CLI resolves
   tasks from the Harbor registry; they ship as Shape A directories.

## Architecture summary

```
       ┌────────────────────┐
       │       Task         │  pure data: instruction + metadata + path
       │  (rllm.task.Task)  │  knows where its tests/ and environment/ live
       └──────────┬─────────┘
                  │
                  ▼
       ┌────────────────────┐
       │      Runner        │  orchestrator
       │  (rllm.runner)     │  - resolves verifier from task config
       └──────────┬─────────┘  - sets up sandbox if needed
                  │             - runs AgentFlow → Episode
       ┌──────────┴─────────┐   - runs Evaluator → reward
       ▼                    ▼
┌────────────┐      ┌──────────────┐
│ AgentFlow  │      │  Evaluator   │
│  (harness) │      │  (verifier)  │
└────────────┘      └──────────────┘
```

The `AgentFlow` and `Evaluator` protocols are unchanged from the
prior eval framework — the new `Task` is the unifying input shape.

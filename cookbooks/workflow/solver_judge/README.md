# Solver-Judge (legacy Workflow API)

Solver-judge multi-agent flow for the countdown task, built on the **legacy `Workflow` abstraction** (`rllm.workflows.workflow.Workflow` + `RolloutEngine` + `AgentWorkflowEngine`). The training side runs through the **unified trainer** (`rllm.experimental.unified_trainer.AgentTrainer`), which keeps the `workflow_class=...` interface while routing to the maintained backend launchers. The old `rllm.trainer.AgentTrainer(backend="tinker")` path has been removed.

For the equivalent example written against the newer `@rllm.rollout` AgentFlow protocol, see [`cookbooks/solver_judge_flow/`](../../solver_judge_flow/).

## Pattern

- **Solver** generates `n_solutions` candidate answers in parallel.
- **Judge** picks the index of the best candidate; its answer is the workflow's final answer.
- The workflow emits one episode with `n_solutions + 1` trajectories (named `solver` / `judge`), each carrying its own reward so GRPO can group advantages per role.

## Layout

| File | Description |
|------|-------------|
| `solver_judge_flow.py` | `SolverJudgeWorkflow` — subclass of `rllm.workflows.workflow.Workflow` |
| `run.py` | Inference / eval driver via `AgentWorkflowEngine` + `OpenAIEngine` |
| `train.py` | Hydra entrypoint — `unified_trainer.AgentTrainer(workflow_class=...)` |
| `train_tinker.sh` | Tinker-backend launch (`rllm/backend=tinker`) |
| `train_verl.sh` | Verl-backend launch (`rllm/backend=verl`) |

## Data

```bash
rllm dataset pull countdown
```

This registers `countdown` train/test splits in the `DatasetRegistry`. The workflow expects each task to carry `question`, `target`, `nums`, and `ground_truth = {"target", "numbers"}` (see `process_countdown_fn` in `run.py`).

## Eval / inference

Start an OpenAI-compatible server (sglang, vllm, ...) on `http://localhost:30000/v1`, then run from this cookbook directory:

```bash
cd cookbooks/workflow/solver_judge
python run.py
```

Reports pass@1 / pass@k over the countdown test split and dumps the full episode log to `logs/solver_judge_countdown.json`.

## Train

### Tinker (single-machine)

```bash
cd cookbooks/workflow/solver_judge
bash train_tinker.sh
```

### Verl (distributed GPU)

```bash
cd cookbooks/workflow/solver_judge
bash train_verl.sh
```

Both scripts call the same `train.py` with `rllm/backend=tinker|verl` and any additional Hydra overrides. Append more overrides on the command line — they will be forwarded.

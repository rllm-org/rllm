# Geo3K (legacy Workflow API)

Single-turn VLM geometry solver for the [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) dataset, built on the **legacy `Workflow` abstraction** (`rllm.workflows.workflow.Workflow` + `RolloutEngine` + `AgentWorkflowEngine`). The training side runs through the **unified trainer** (`rllm.experimental.unified_trainer.AgentTrainer`), which keeps the `workflow_class=...` interface while routing to the maintained backend launchers. The old `rllm.trainer.AgentTrainer(backend="tinker")` path has been removed.

For the equivalent example written against the newer `@rllm.rollout` AgentFlow protocol, see [`cookbooks/geo3k/`](../../geo3k/).

## Pattern

- **Solver** receives a geometry problem with a diagram image. The workflow assembles a `{"role": "user", "content": <text>, "images": [PIL.Image]}` message and calls `rollout_engine.get_model_response` directly. Backend-specific image rendering (OpenAI multimodal blocks for verl/vLLM, the tinker-renderer-aware path for tinker) happens inside the `RolloutEngine`.
- One trajectory per episode, named `solver`, scored by `math_reward_fn` (boxed-answer extraction + symbolic math grading).

## Layout

| File | Description |
|------|-------------|
| `geo3k_flow.py` | `Geo3KWorkflow` — subclass of `rllm.workflows.workflow.Workflow` |
| `run.py` | Inference / eval driver via `AgentWorkflowEngine` + `OpenAIEngine` |
| `train.py` | Hydra entrypoint — `unified_trainer.AgentTrainer(workflow_class=...)` |
| `train_tinker.sh` | Tinker-backend launch (`rllm/backend=tinker`) |
| `train_verl.sh` | Verl-backend launch (`rllm/backend=verl`) |

## VLM dependency note

`Qwen/Qwen3-VL-30B-A3B-Instruct` (and the wider Qwen3-VL family) ships a `Qwen3VLVideoProcessor` that imports torchvision at load time. Even though the tinker rollout only uses the image side, `AutoProcessor.from_pretrained` will fail without it:

```bash
uv pip install torchvision
```

The tinker backend tries to load the processor lazily and now fails fast with an actionable message if torchvision is missing — install it once before kicking off training.

## Data

```bash
rllm dataset pull geo3k
```

## Eval / inference

Start an OpenAI-compatible VLM server (sglang, vllm, ...) on `http://localhost:30000/v1`, then run from this cookbook directory:

```bash
cd cookbooks/workflow/geo3k
python run.py
```

Reports pass@1 / pass@k over the geo3k test split and dumps the full episode log to `logs/geo3k.json`.

## Train

### Tinker (single-machine)

```bash
cd cookbooks/workflow/geo3k
bash train_tinker.sh
```

### Verl (distributed GPU)

```bash
cd cookbooks/workflow/geo3k
bash train_verl.sh
```

Both scripts call the same `train.py` with `rllm/backend=tinker|verl` and any additional Hydra overrides. Append more overrides on the command line — they will be forwarded.

## Config parity with `cookbooks/geo3k/`

This cookbook intentionally matches the AgentFlow geo3k cookbook's hyperparameters so the two are apples-to-apples comparable:

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- LoRA: rank 32 (alpha 32, merge true) on verl
- `training.group_size`: 8
- Tinker: `total_epochs=3`, `test_freq=10`, train/val temperature 0.6
- Verl: `actor_rollout_ref.rollout.n=4`, `data.train_batch_size=32`, `lr=2e-5`, `loss_agg_mode=token-mean`, vLLM async rollout, `tensor_model_parallel_size=2`

The only intentional differences are `project_name` (`geo3k_workflow` vs `geo3k`) and the underlying agent shape (legacy `Workflow` vs `@rllm.rollout` AgentFlow).

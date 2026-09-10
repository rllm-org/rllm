# rLLM Recipes

Ready-to-run training recipes built on top of `examples/` and `cookbooks/`, with opinionated defaults for specific model + task combinations.

| Recipe | Backend | Task | Model |
|--------|---------|------|-------|
| [qwen3_5_swe_grpo](./qwen3_5_swe_grpo/) | verl (GRPO) | Native SWE (`mini-swe-agent`) | `Qwen/Qwen3.5-4B` |

Each recipe folder contains a `README.md`, `train.py`, launch script(s), and config helpers. Recipes are prepared for launch but are not executed automatically.

## Evaluation

[`eval/`](./eval/) holds the `rllm eval` guide for SWE-bench, with both the native rLLM harness and the Harbor harness (`--agent harbor:*`).

| Guide | Dataset | Contents |
|-------|---------|----------|
| [eval/README.md](./eval/README.md) | — | Setup, native vs Harbor harness, model connection, troubleshooting |
| [eval/swebench-pro](./eval/swebench-pro/) | `swebenchpro_100` (100 of 731) | Subset script, oracle test, evaluation commands |
| [eval/swebench-verified](./eval/swebench-verified/) | `swebench_verified_100` (100 of 500) | Same layout as swebench-pro |

Shared sampling params live in [`eval/config/`](./eval/config/) (`qwen3_5.yaml`).

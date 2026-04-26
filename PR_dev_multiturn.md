# Multi-turn AgentFlow stabilization for verl, calculator/dataset upgrades, dual-backend cookbooks

Merges `dev-multiturn` → `main`. Five new commits on top of three already-merged PRs (#503, #508, #509).

## Summary

- **Unblocks AgentFlow training under the verl backend** with three correctness fixes: per-step ID collisions in advantage scatter, trailing-malformed traces failing whole rollouts, and contaminated traces leaking from prior retry attempts.
- **Adds a `calculate_debug_metrics_compat` shim** in `rllm/experimental/verl/metrics.py` that backfills the legacy empty-mask behavior verl removed, and routes the four trainers (`agent_ppo_trainer`, `agent_sdk_trainer`, `agent_workflow_trainer`, `fully_async_trainer`) plus `verl_backend` through it. Also covered by a new `tests/test_verl_metrics.py`.
- **Hardens the gateway** so `/admin/flush` drains in-flight `_safe_store` tasks, `adelete_session` flushes before deletion, and noisy admin/health paths stop spamming `uvicorn.access`.
- **Registers `deepscaler_math`** (`agentica-org/DeepScaleR-Preview-Dataset`, ~40K AIME/AMC/Omni-MATH problems) in the dataset catalog and switches `math_tool_agent` to it to reduce GRPO uniform-group rate.
- **Rewrites the math_tool_agent calculator** with an AST whitelist (sqrt, log, trig, factorial, comb/binom, etc.) so models can write `sqrt(64 + 225)` or `binom(10, 3)` without being rejected as "invalid characters".
- **Restructures cookbook scripts** for `geo3k`, `solver_judge_flow`, and `math_tool_agent` into matched `train_tinker.sh` / `train_verl.sh` pairs, and ports the verl-stable hyperparameters (`lr 1e-6`, `loss_agg_mode=seq-mean-token-mean`, asymmetric clip 0.2/0.28) into the verl scripts.

## Why

Multi-turn AgentFlow training under verl was crashing within 3–5 steps with `EnrichMismatchError` and silently producing wrong advantages even when it didn't crash. Three issues compounded:

1. `_process_trajectory` keyed `step_id` as `f"{task_id}_{trajectory.name}_step{step_idx}"`. With multiple rollouts per task, or multiple same-named trajectories per episode (solver_judge with N solver trajectories), the dict in `update_dataproto_with_advantages` saw colliding writes — only the last advantage survived, then got scattered to **all** colliding rows. Silent GRPO destruction.
2. The strict `EnrichMismatchError` rejected any rollout where traces and agent steps disagreed. In practice, multi-turn agents commonly produce one trailing malformed trace (vLLM returns an empty body when prompt+max_tokens hits `max_model_len`, or disconnects mid-stream during weight sync; the OpenAI client raises; the agent breaks without recording its Nth Step). With `MAX_TURNS=8`, the failure rate compounded enough to exhaust retries.
3. On retry, the gateway still held the failed attempt's traces, and the positional match in `_enrich_episode` mixed stale traces with new steps.

End-to-end verification on math_tool_agent with the new config: training reward climbs from 0.27 baseline to ~0.40–0.50 by step 20; math500 val pass@1 = 0.671 @ step 10, 0.686 @ step 20.

## Files of note

| Area | Files |
|---|---|
| AgentFlow/verl correctness | `rllm/experimental/verl/transform.py`, `rllm/experimental/engine/agent_flow_engine.py`, `rllm/experimental/engine/gateway_manager.py` |
| Debug-metrics shim | `rllm/experimental/verl/metrics.py` (new), `tests/test_verl_metrics.py` (new), `rllm/experimental/verl/verl_backend.py`, `rllm/experimental/fully_async/fully_async_trainer.py`, `rllm/trainer/verl/agent_{ppo,sdk,workflow}_trainer.py` |
| Gateway robustness | `rllm-model-gateway/src/rllm_model_gateway/server.py` |
| Dataset / agent | `rllm/registry/datasets.json`, `rllm/data/transforms.py`, `cookbooks/math_tool_agent/math_tool_agent.py`, `cookbooks/math_tool_agent/test.py`, `cookbooks/math_tool_agent/train.py` |
| Cookbook scripts | `cookbooks/{geo3k,math_tool_agent,solver_judge_flow}/train_tinker.sh`, `cookbooks/{geo3k,math_tool_agent,solver_judge_flow}/train_verl.sh`, READMEs |

## Behavior changes worth flagging to reviewers

- **`step_id` format change** (`{trajectory.uid}_step{idx}` instead of `{task_id}_{name}_step{idx}`). External code that re-keyed off the old format will break. `trajectory_ids` is *intentionally unchanged* — verl's `compute_advantage` and the broadcast helper rely on those collisions for grouping.
- **Trailing malformed traces are now silently dropped** when `traces > agent_steps` and the extras have empty `prompt_ids`/`completion_ids`. Mid-sequence empties and `traces < agent_steps` still raise `EnrichMismatchError`. A warning log is emitted on every drop.
- **`adelete_session` now flushes the gateway client before deletion**; the sync `delete_session` was removed (no callers).
- **Padded rows now zero `response_mask`** (in `_pad_to_dp_divisor`), keeping pad tokens out of `batch_num_tokens` so loss magnitude is invariant to `pad_size`.
- **`_get_aggregate_dp_size` lost branches for `rm_wg`, `actor_wg`, `rollout_wg`, and the `hybrid_engine=False` split.** This matches the new EngineWorker path (`use_legacy_worker_impl='disable'` in `verl_launcher`); legacy worker layouts will no longer compute their DP size correctly here.
- **`math_tool_agent` default training set is now `deepscaler_math`**, not `hendrycks_math`. Run `rllm dataset pull deepscaler_math` before training.

## Test plan

- [ ] `pytest tests/test_verl_metrics.py`
- [ ] Pull and prep dataset: `rllm dataset pull deepscaler_math`
- [ ] math_tool_agent verl smoke: `bash cookbooks/math_tool_agent/train_verl.sh` — confirm reward climbs past 0.40 by step ~20 and math500 val pass@1 lands ≥0.65
- [ ] solver_judge_flow verl smoke: `bash cookbooks/solver_judge_flow/train_verl.sh` — confirm solver does not collapse from 0.6 → <0.1 in one update (the failure mode the old config showed)
- [ ] geo3k tinker smoke: `bash cookbooks/geo3k/train_tinker.sh`
- [ ] Calculator tool: `python cookbooks/math_tool_agent/test.py` exercises `sqrt`, `binom`, `pi`; rejects attribute access / unknown names
- [ ] Gateway: trigger `/admin/flush` while a trace is mid-persist and confirm `aget_traces` returns the full trace count

## Out of scope / follow-ups

- The new `metrics.py` shim swallows verl's empty-mask `RuntimeError` and returns NaN-filled defaults. If verl pins move past that path entirely, this can be deleted.
- Cookbook `train_verl.sh` scripts hardcode the hyperparameters that worked for our 30B-scale runs; smaller models may want different `lr` / clip ranges.

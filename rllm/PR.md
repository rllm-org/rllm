# Unified Task / Runner / Harness — eval & training stack refactor

This PR reorganizes the eval and training stack around three abstractions: **`Task`** (data), **`Runner`** (orchestrator), and **`AgentFlow`/harness** (agent). It folds catalog datasets, locally-authored benchmarks, and sandbox-style task directories onto a single code path; consolidates specialized evaluator classes into `reward_fns/`; promotes everything under `rllm/experimental/` into proper homes; and adds a per-run episode store with a `rllm view` browser viewer.

> **Scope note.** Harbor integration changes (under `rllm/integrations/harbor/`, `examples/harbor_swe/`, `harbor:` dataset prefix, ATIF trajectory bridge, Harbor-compatible `task.toml`, etc.) shipped in a prior PR and are intentionally excluded from the description below.

---

## On-disk reorganization

| Move | From | To |
|---|---|---|
| CLI | `rllm/experimental/cli/` | `rllm/cli/` |
| Sandbox | `rllm/sdk/sandbox/` | `rllm/sandbox/` |
| Eval | `rllm/experimental/eval/` | `rllm/eval/` |
| Reward fns (was specialized evaluators) | `rllm/eval/evaluator/` (deleted) | `rllm/eval/reward_fns/` |
| Harnesses | `rllm/experimental/agents/` (mixed) | `rllm/harnesses/` |
| Core protocols + `Task` | new top-level | `rllm/types.py` |

`rllm/experimental/agents/` is gone; its contents were redistributed:
- `sandboxed_agent.py` → `rllm/sandbox/sandboxed_flow.py`
- `vlm_utils.py` → `rllm/eval/`
- `tool_calling.py` + `tools/` → `rllm/harnesses/`
- `react_agent.py` and `search/` deleted as legacy
- `agenthub/` external scaffolds (`frozenlake_agent`, `langgraph_agent`, `react_agent`, `smolagents_agent`, `strands_agent`, `swe_agent`, `terminal_agent`) deleted entirely.

---

## New top-level concepts

| File | Purpose |
|---|---|
| `rllm/types.py` | Canonical `Task`, `Step`, `Trajectory`, `Episode`, plus producer/consumer protocols (`AgentFlow`, `Evaluator`, `AgentConfig`, `run_agent_flow`) |
| `rllm/runner.py` | `Runner` — verifier dispatch + sandbox lifecycle + AgentFlow invocation. One code path for all task kinds |
| `rllm/harnesses/react.py` | `ReActHarness` — one-shot LLM (default for data tasks) |
| `rllm/harnesses/bash.py` | `BashHarness` — multi-turn ReAct loop with bash tool calls in sandbox |
| `rllm/harnesses/claude_code.py` | `ClaudeCodeHarness` — Claude Code CLI in sandbox |
| `rllm/cli/view.py` | `rllm view` — boots a local web viewer over the per-run episode store |

`Task` is pure data: `id`, `instruction`, `metadata`, `dataset_dir`, `sub_dir`. No methods, no callbacks. Two physical shapes both produce `Task` instances:

1. **Task-per-directory** (`task-NNN/` per task, verifier in `tests/`).
2. **Rows-with-shared-verifier** (one JSONL row per task, verifier shared).

---

## Eval pipeline plumbing

| File | Role |
|---|---|
| `rllm/eval/script_evaluator.py` | `ShellScriptEvaluator` — runs `tests/test.sh` in sandbox, reads `/logs/verifier/reward.{txt,json}` |
| `rllm/eval/module_evaluator.py` | `PythonModuleEvaluator` — imports `tests/evaluate.py`, supports `(task, episode)` and `(metadata, trajectory)` signatures |
| `rllm/eval/materialize.py` | `materialize_benchmark()` — writes catalog rows into `~/.rllm/datasets/<name>/` (`data/`, `instruction.md.tpl`, `dataset.toml`) |
| `rllm/eval/reward_fns/_resolver.py` | Reads `[verifier]` from task config, returns the `SYSTEM_PROMPT` exported by the matching reward_fn |
| `rllm/eval/reward_fns/{math,mcq,code,f1,countdown,iou,point_in_mask,depth,bfcl,ifeval,llm_equality,llm_judge,translation,widesearch}.py` | 14 inlined reward functions; each exports `evaluate()` and `SYSTEM_PROMPT` |
| `rllm/eval/runner.py` | Sole entry point `run_dataset()` — async-gather over `Runner.run` with concurrency, error handling, per-episode callback. The legacy `EvalRunner` class was deleted in the final consolidation pass |
| `rllm/eval/episode_store.py` | Per-run JSONL store of completed `Episode`s under `~/.rllm/runs/<run-id>/` |
| `rllm/eval/visualizer.py` | HTML/JS viewer used by `rllm view` |

---

## Modified

| File | Change |
|---|---|
| `rllm/cli/eval.py` | Single dispatch path through `Runner`. Local benchmarks load via `BenchmarkLoader`; catalog datasets either materialise to a local benchmark dir or wrap dict-rows as `Task`s on the fly (`_dict_rows_to_tasks` — Harbor rows get `dataset_dir=Path(row["task_path"])` so HarborRuntime reads `task.metadata["task_path"]` natively). `--evaluator` and catalog `reward_fn` flow through `evaluator_override`. Persists episodes to the run store as they complete |
| `rllm/cli/train.py` | Mirrors `eval.py`'s single-path dispatch. `--agent` resolves through `load_agent` for everything (built-in harnesses, user-registered, plugin entry points, `module:Class` paths, `harbor:<scaffold>`) |
| `rllm/cli/_pull.py` | `pull_dataset()` calls `materialize_benchmark()` after registering parquet; materializes on-the-fly for already-pulled catalog datasets |
| `rllm/tasks/loader.py` | `BenchmarkLoader.load()` returns `list[Task]` (handles per-directory and row shapes); supports VLM content blocks and renders `{{choices}}` for MCQ catalog datasets |
| `rllm/eval/agent_loader.py` | Single registry: catalog (`registry/agents.json`) + user-registered (`~/.rllm/agents.json`) + plugin entry points (`rllm.agents`). Built-in harnesses (`react`, `bash`, `claude-code`) live in the catalog as `module:Class` pointers; classes are auto-instantiated. `rllm agent list` enumerates all sources |
| `rllm/eval/evaluator_loader.py` | `_EVALUATOR_REGISTRY` maps reward-fn names → reward_fn import paths; `_FunctionEvaluator` adapts dict-style legacy callers automatically |
| `rllm/eval/types.py` | Trimmed to `EvalOutput` and `Signal` only — protocols moved to `rllm/types.py` |
| `rllm/sandbox/sandboxed_flow.py` | Moved out of `experimental/`; `set_sandbox()` for Runner-managed sandbox lifecycle; `teardown_sandbox` is a no-op when externally managed |
| `rllm/sandbox/protocol.py` + 3 backends | `exec(... user=...)` for the agent/verifier user split (tampering protection) |
| `rllm/registry/agents.json` | Single source of truth for built-in agents — lists `react`, `bash`, `claude-code` with `module:Class` pointers (replaces the parallel `_HARNESS_REGISTRY` that lived in `tasks/harness.py`) |
| `rllm/runner.py` | When `evaluator_override` is set, skip `_detect_verifier` and per-task sandbox creation. Why: harbor task dirs ship `environment/` + `tests/test.sh` that would otherwise trigger an extra sandbox per task on top of the one HarborRuntime spins up internally |
| `Task.dataset_dir` | Renamed from `Task.benchmark_dir` (the term "dataset" is used everywhere else) |
| Reward-fn output | `score_fns` declared expected output format; harness consumes it (renders system prompt, parses model output) |

---

## Deleted

- `rllm/experimental/cli/`, `rllm/sdk/sandbox/`, `rllm/experimental/eval/`, `rllm/experimental/agents/` (moved or redistributed).
- `rllm/eval/evaluator/` — 6 specialized evaluator modules, logic moved into `reward_fns/`.
- `rllm/tasks/runner.py:TaskRunner` — replaced by `rllm/runner.py:Runner`.
- `rllm/tasks/evaluator.py:TaskEvaluator` — replaced by `ShellScriptEvaluator`.
- `rllm/tasks/simple_evaluator.py:SimpleEvaluator` — replaced by `PythonModuleEvaluator`.
- `rllm/tasks/task.py` — old sandbox-task class; `rllm.types.Task` is canonical.
- `rllm/tasks/harness.py` — separate harness registry collapsed into `rllm/registry/agents.json` + `agent_loader.load_agent`. `register_harness`, `load_harness`, `list_harnesses`, `is_harness_name` are gone.
- `rllm/eval/runner.py:EvalRunner` — legacy two-stage dict-row runner; `run_dataset()` is the sole entry point now.
- `rllm/eval/task_spec.py` — `TaskSpec` + `build_task_spec` were exported but not called anywhere; pure dead code.
- 9 legacy evaluator classes from `rllm/eval/types.py` (`MathEvaluator`, `MCQEvaluator`, `CodeEvaluator`, `F1Evaluator`, `CountdownEvaluator`, `IoUEvaluator`, `PointInMaskEvaluator`, `DepthEvaluator`, `BfclEvaluator`).
- `rllm/trajectory_visualizer.py` — old viewer (replaced by `rllm/eval/visualizer.py` + `rllm view`).
- `agenthub/` — external integrations dropped.

---

## Per-run episode store + `rllm view`

Every eval run writes completed `Episode`s to a JSONL stream under `~/.rllm/runs/<run-id>/episodes.jsonl` as they finish. `rllm view` boots a local browser viewer over those runs:

- Lists runs (newest first), shows score / agent / dataset.
- Per-episode trajectory viewer with messages, tool calls, rewards.
- Works without re-running anything; useful for after-the-fact inspection and for sharing a single run with a teammate.

---

## VLM, MCQ, `--evaluator`, and harness train

The PR closes the previously-open TODO list:

- **VLM datasets** — `task.instruction` may be a `list[dict]` of content blocks; materialize and the loader preserve them end-to-end.
- **MCQ rendering** — `_render_instruction` now substitutes `{{choices}}` in addition to `{{question}}`; MMLU/MMLU-Pro etc. work through the unified path.
- **`--evaluator` override** — honored on the new path; previously silently ignored for local benchmarks and harness-routed catalog datasets.
- **Harness train** — `rllm train` mirrors `eval`'s dispatch and runs through Runner.

---

## Tests

- `tests/eval/test_agents.py` deleted (legacy class identity tests).
- `tests/eval/test_eval_runner.py` and `tests/eval/test_runner.py` deleted along with `EvalRunner`.
- `tests/eval/test_evaluator_loader.py` rewritten to assert protocol conformance and exercise the dict-style backward-compat adapter.
- New: `tests/eval/test_materialize.py`, `tests/eval/test_resolver.py`, `tests/eval/test_script_module_evaluators.py`, `tests/eval/test_unified_runner.py`.
- Catalog/CLI/agentic/IFEval/MCQ/VLM tests updated to the new shapes.
- ~+1.2k LoC test, ~–950 LoC removed.

---

## Verified working

- `rllm eval gsm8k --max-examples 3` → **100%** (catalog → materialize → Runner → reward_fn).
- `rllm eval ./path/to/local-bench --sandbox-backend docker --agent bash` → **100%** (sandbox → Runner → ShellScriptEvaluator).
- Adversarial agent run with verifier-user split → **0% with `Permission denied`** (tampering blocked).
- All 14 reward-fn names resolve via `load_evaluator`.
- Legacy dict-style `evaluator.evaluate({...}, episode)` still works via the auto-adapter.
- `rllm view` lists runs and renders trajectories from the per-run episode store.
- `rllm agent list` enumerates the built-in harnesses (previously they were hidden behind a separate registry).

---

## Net diff (excluding Harbor integration)

~6.2k insertions / ~5.0k deletions across the eval, CLI, sandbox, harness, and tests trees. Most of the deletions are the old `experimental/agents/` and `agenthub/` trees and the per-evaluator class files.

## Commits in this PR (excluding Harbor)

```
cd7dff14 refactor(eval): collapse agent/harness registries and EvalRunner into one path
716434fb fix: update stale rllm.eval.types imports after rllm.types lift
5c4fff16 fix: post-refactor cleanup — Task shape + CLI templates + numpy advantage
02f2ca23 feat(eval): persist Episodes per-run and add `rllm view` browser viewer
f64719c1 refactor: rename Task.benchmark_dir → Task.dataset_dir
f96d72bf refactor: remove agenthub/ external integrations
7bee3e58 refactor: redistribute experimental/agents/ into proper homes
0f8dd1b7 refactor: unify harnesses under rllm/harnesses/
df62f950 test: migrate tests/eval/ + add unit tests for new modules
73fffda4 feat(eval+train): wire up TODOs — VLM, MCQ choices, --evaluator override, harness train
1634e540 refactor: lift core protocols + Task into rllm.types
db7a7d65 refactor: rename rllm/eval/score_fns/ → rllm/eval/reward_fns/
bfc56864 refactor: delete legacy evaluator classes; reward_fns are the source of truth
fd68e64b fix(eval): reward_fns declare expected output format; harness uses it
5d7a9f0e fix(cli): materialise on-the-fly for already-pulled catalog datasets
14ed9730 refactor: rename harnesses, drop legacy "react" catalog agent
e75b2fd7 fix(cli): first-run UX for catalog datasets with harness --agent
49ef3b60 feat(eval): catalog auto-dispatch through Runner + docs (PR 4)
c2bfd08c fix(cli): clearer error when a path-like benchmark doesn't resolve
43e24681 feat(eval): reward_fns + materialize + SimpleHarness (PR 3)
4a0c88a3 refactor(eval): migrate sandbox tasks to unified Runner (PR 2)
e8c44050 feat(eval): foundation for unified Task + Runner abstraction (PR 1)
84f6b4d6 refactor: group specialized evaluators under rllm/eval/evaluator/
a39e1860 refactor: move eval out of experimental to rllm/eval
99998ec9 refactor: move sandbox out of sdk to rllm/sandbox
94ff54b0 feat: agent/verifier user split for tampering protection
e925d7e0 refactor: split TaskExecutor into Task + AgentHarness + TaskRunner
1d46d274 fix: sanitize container names and clean up reward file probing
094b8589 fix: sanitize container names and clean up reward file probing
240316ed fix: sanitize dataset name in eval results save path
374bc97e fix: update relative paths after cli move out of experimental
65c11c9f refactor: move cli out of experimental to rllm/cli
```

## Remaining follow-ups

- `pyproject.toml` has `[tool.uv.extra-build-dependencies]` which uv doesn't recognize; blocks `uv run pytest`. Pre-existing.
- `rllm dataset list` should surface materialization status (pulled? new format?).

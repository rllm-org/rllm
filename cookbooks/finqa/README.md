# FinQA Agent

A multi-turn ReAct-style financial-QA agent for rLLM, ported to the **AgentFlow protocol**. Replaces `projects/finqa/` (which depended on `BaseAgent` + `ToolAgent` + `ToolEnvironment` + `AgentExecutionEngine`).

## Overview

The agent answers questions about SEC 10-K financial statements by querying structured tables extracted from filings. Four tools are exposed via native OpenAI function calling:

| Tool | Description |
|------|-------------|
| `get_table_names` | List queryable tables for a given company |
| `get_table_info` | Column metadata, dtypes, sample values for a table |
| `sql_query` | Run a filtered SQLite query against the in-memory store |
| `calculator` | Safely evaluate a numeric expression (asteval) |

All ~6,900 tables across 207 companies are pre-loaded into a process-wide `:memory:` SQLite at module import; the tools just look up cached metadata or run a read-only query under a thread lock.

The agent's final reply ends with a `FINAL ANSWER: …` block. The evaluator extracts it and grades against ground truth via a judge LLM (gpt-5-nano for single-table, gpt-5-mini with a structured rubric for multi-table), routed through Portkey for caching/retry. A small bonus is awarded when the agent inspects the *expected* tables via `get_table_info` (the `right_table_access_reward`).

[Model Weights](https://huggingface.co/rLLM/rLLM-FinQA-4B) | [Dataset](https://huggingface.co/datasets/rLLM/finqa)

## Architecture

```
AgentFlow.run(task, config)
  │
  └── Multi-turn loop (up to 20 turns, native OpenAI tool calls)
        │
        ├── client.chat.completions.create(messages, tools=TOOL_SPECS)
        │
        ├── If msg.tool_calls is empty → that's the final answer, break.
        │
        └── Else: dispatch each tool call → append a `tool` message → repeat.
              (track table_name in `accessed_tables` whenever
               `get_table_info` is invoked, for the table-access bonus)
  │
  └── episode.artifacts = {"answer": full_response, "accessed_tables": [...], "turns": N}

Evaluator.evaluate(task, episode)
  │
  └── extract FINAL ANSWER from artifacts["answer"], grade via judge LLM,
      add table-access bonus, return EvalOutput.
```

## Installation

```bash
uv pip install -e ".[tinker]"                      # rllm + tinker backend
uv pip install --no-deps -e cookbooks/finqa        # this cookbook
```

After installation:

```bash
rllm agent list      # should show "finqa"
```

## Dataset

```bash
python cookbooks/finqa/prepare_data.py
```

This will:
- Download the dataset tarball from [rLLM/finqa](https://huggingface.co/datasets/rLLM/finqa)
- Extract company tables to `cookbooks/finqa/data/company_tables/` (207 companies, 6,923 tables)
- Register `finqa/{train, val, test}` with `DatasetRegistry` (4,030 / 522 / 558 examples)

## Eval (rllm CLI)

Set the judge keys (used by the evaluator), then run:

```bash
export OPENAI_API_KEY=sk-…
export PORTKEY_API_KEY=pk-…

rllm eval finqa \
    --agent finqa \
    --evaluator finqa \
    --model rLLM/rLLM-FinQA-4B \
    --base-url http://localhost:30000/v1 \
    --split test \
    --max-examples 20
```

Episode JSONs land under `~/.rllm/eval_results/`. If `OPENAI_API_KEY` or `PORTKEY_API_KEY` is missing, the evaluator silently returns `reward=0` rather than crashing.

## Training (rllm CLI)

```bash
export OPENAI_API_KEY=sk-…
export PORTKEY_API_KEY=pk-…

rllm train finqa \
    --agent finqa \
    --evaluator finqa \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --group-size 8 \
    --batch-size 16 \
    --lora-rank 32 \
    --epochs 1 \
    --val-freq 20
```

## Training (shell scripts)

### Tinker (single-machine, LoRA on 30B)

```bash
bash cookbooks/finqa/train_tinker.sh
```

### Verl (distributed GPU, 4B base)

```bash
uv pip install -e ".[verl]"
bash scripts/install_megatron.sh <cu128|cu129|...>
bash cookbooks/finqa/train_verl.sh
```

## Tests

```bash
pytest cookbooks/finqa/test.py -v
```

## Files

| File | Description |
|------|-------------|
| `finqa_flow.py` | `finqa_flow` — multi-turn AgentFlow with native tool calling |
| `finqa_tools.py` | The four tools as plain functions + OpenAI tool specs |
| `finqa_eval.py` | `finqa_evaluator` — judge-LLM correctness + table-access bonus |
| `finqa_constants.py` | Path constants (data, prompts) |
| `prepare_data.py` | HF download + register train / val / test splits |
| `train.py` | Python API training script (Hydra config) |
| `train_tinker.sh` | Tinker backend — single-machine LoRA training |
| `train_verl.sh` | Verl backend — distributed multi-GPU training |
| `pyproject.toml` | Plugin metadata and entry points |
| `test.py` | Unit tests (calculator, FINAL ANSWER parsing, table-access scoring) |
| `prompts/` | System prompt + correctness rubric prompts |

## Migration notes

This cookbook replaces `projects/finqa/`, which used the legacy stack:

| Legacy | Replacement |
|---|---|
| `FinQAAgent(ToolAgent)` (qwen tool-call parsing) | `finqa_flow` AgentFlow + native OpenAI function calling |
| `FinQAEnvironment(ToolEnvironment)` (in-loop reward) | Evaluator outside the loop, called once on the final episode |
| `MultiTurnWorkflow` + `AgentExecutionEngine` | Plain async function — `client.chat.completions.create(tools=…)` in a `for` loop |
| `Tool` base class wrappers | Plain Python callables (`get_table_names`, `get_table_info`, `sql_query`, `calculator`) |

The cookbook has zero imports from `rllm.agents`, `rllm.environments`, `rllm.engine.agent_execution_engine`, or `rllm.workflows`. After this migration, `ToolAgent` / `ToolEnvironment` / `MCPToolAgent` are no longer reachable from any active code path and can be removed.

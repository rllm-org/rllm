# LangGraph Math

A multi-turn math agent authored with [LangGraph](https://github.com/langchain-ai/langgraph)'s `create_react_agent`, trained end-to-end with rLLM. Demonstrates that any LangGraph agent integrates with rLLM training **without writing a callback handler** — pointing LangChain's `ChatOpenAI` at `config.base_url` is enough, because the rLLM model gateway captures every LLM call.

It is intentionally a near-clone of [`cookbooks/math_tool_agent/`](../math_tool_agent/) so you can compare a hand-rolled tool loop against a LangGraph one on the same dataset.

## Why so little code?

```python
@rllm.rollout(name="langgraph-math")
async def langgraph_math(task: Task, config: AgentConfig) -> None:
    llm = ChatOpenAI(model=config.model, base_url=config.base_url, api_key="EMPTY", temperature=1.0)
    agent = create_react_agent(llm, tools=[calculate], prompt=SYSTEM_PROMPT)
    await agent.ainvoke({"messages": [("user", task.instruction)]})
    return None
```

That's the whole agent. No callback handler, no message format conversion, no manual `Step`/`Trajectory` construction. The mechanism:

- LangChain's `ChatOpenAI(base_url=…)` issues OpenAI Chat Completions requests against the gateway session URL the trainer provides.
- The gateway middleware extracts the session id from the URL path (`/sessions/{sid}/v1/...`) and persists every request/response as a `TraceRecord` keyed by that session.
- The flow returns `None`. The framework's coercion (`rllm.types._coerce_to_episode`) builds an empty single-trajectory `Episode`. During enrichment the gateway's traces become the trajectory's `Step`s, populated with prompt/response token IDs and per-token logprobs ready for training.
- The evaluator reads the agent's final assistant message from `episode.trajectories[-1].steps[-1].model_response` and grades it against ground truth.

Because the trajectory's `name` is `"langgraph-math"` (set on `@rllm.rollout`), all rollouts of the same task hash to the same `f"{task_id}:langgraph-math"` key when the trainer builds `TrajectoryGroup`s for GRPO advantage computation.

## Architecture

```
AgentFlow.run(task, config) → None
  │
  └── LangGraph create_react_agent (recursion limit 25)
        │
        ├── ChatOpenAI(base_url=config.base_url)
        │     → POST /sessions/{sid}/v1/chat/completions
        │     → gateway captures TraceRecord (model, messages, token_ids, logprobs)
        │
        ├── @tool calculate(expression) → safe AST eval
        │
        └── repeat until LLM emits no more tool calls
```

## Installation

```bash
# From the rllm repo root
uv pip install -e ".[tinker]"                          # rllm + tinker backend
uv pip install --no-deps -e cookbooks/langgraph_math   # this cookbook + LangGraph deps
```

After installation:

```bash
rllm agent list   # should show "langgraph_math" as a plugin
```

## Dataset

Same datasets as `cookbooks/math_tool_agent/` so you can compare learning curves:

```bash
rllm dataset pull deepscaler_math   # ~40K competition math problems
rllm dataset pull math500           # 500-problem test benchmark
```

## Training

### Tinker (single-machine)

```bash
bash cookbooks/langgraph_math/train_tinker.sh
```

Or directly via the Python API:

```bash
python cookbooks/langgraph_math/train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=8
```

### Verl (distributed GPU)

```bash
uv pip install -e ".[verl]"
bash scripts/install_megatron.sh <cu128|cu129|...>
bash cookbooks/langgraph_math/train_verl.sh
```

## Tests

```bash
pytest cookbooks/langgraph_math/test.py -v
```

The tests cover the calculator's safe-eval whitelist, answer-extraction patterns, and the evaluator's behavior over synthetic Episodes — including the case where the assistant turn is several Steps back behind tool messages.

## Files

| File | Description |
|------|-------------|
| `langgraph_math.py` | `langgraph_math` — LangGraph `create_react_agent` AgentFlow |
| `evaluator.py` | `langgraph_math_evaluator` — reads answer from gateway-captured trajectory |
| `train.py` | Python API training script (Hydra config) |
| `train_tinker.sh` | Tinker backend — single-machine training |
| `train_verl.sh` | Verl backend — distributed multi-GPU training |
| `pyproject.toml` | Plugin metadata and entry points |
| `test.py` | Unit tests for calculator, parsing, and evaluation |

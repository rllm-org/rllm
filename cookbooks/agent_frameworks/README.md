# Agent Frameworks

One math task — solve a competition problem with a calculator tool — built four ways, one file per framework. Use this cookbook to compare LangGraph, OpenAI Agents SDK, smolagents, and Strands on the same dataset, or as a template for plugging your own framework into rLLM.

The point of this cookbook is to make the AgentFlow + model-gateway architecture concrete: every framework integration collapses to ~6 lines of agent body that points the framework's LLM client at `config.base_url` and returns `None`. The gateway captures every LLM call by URL-routed session, the framework auto-builds an `Episode` from those captured traces, and the evaluator parses the answer out of the resulting trajectory. No callback handler, no traced chat client, no manual `Step`/`Trajectory` construction.

## Layout

```
cookbooks/agent_frameworks/
├── README.md              # this file
├── pyproject.toml         # one package, four entry-point agents
├── calculator.py          # safe_eval — shared by every flow
├── system_prompt.py       # shared system prompt
├── evaluator.py           # math_evaluator — reads last assistant message
├── agentflow/
│   ├── __init__.py
│   ├── langgraph.py       # langgraph_math
│   ├── openai_agents.py   # openai_agents_math
│   ├── smolagents.py      # smolagents_math
│   └── strands.py         # strands_math
├── train.py               # `python train.py +rllm.agent_name=<agent>`
├── train_tinker.sh        # `bash train_tinker.sh <agent>`
├── train_verl.sh          # `bash train_verl.sh <agent>`
└── test.py                # calculator + evaluator + protocol smoke tests
```

## What each flow does

```python
# agentflow/langgraph.py
@rllm.rollout(name="langgraph-math")
async def langgraph_math(task, config):
    llm = ChatOpenAI(model=config.model, base_url=config.base_url, api_key="EMPTY", temperature=1.0)
    agent = create_react_agent(llm, tools=[calculate], prompt=SYSTEM_PROMPT)
    await agent.ainvoke({"messages": [("user", task.instruction)]})
    return None

# agentflow/openai_agents.py
@rllm.rollout(name="openai-agents-math")
async def openai_agents_math(task, config):
    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    model = OpenAIChatCompletionsModel(model=config.model, openai_client=client)
    agent = Agent(name="solver", instructions=SYSTEM_PROMPT, tools=[calculate], model=model)
    await Runner.run(agent, input=task.instruction)
    return None

# agentflow/smolagents.py
@rllm.rollout(name="smolagents-math")
def smolagents_math(task, config):
    model = OpenAIServerModel(model_id=config.model, api_base=config.base_url, api_key="EMPTY")
    agent = ToolCallingAgent(tools=[calculate], model=model)
    agent.run(SYSTEM_PROMPT + "\n\n" + str(task.instruction))
    return None

# agentflow/strands.py
@rllm.rollout(name="strands-math")
async def strands_math(task, config):
    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    model = OpenAIModel(client=client, model_id=config.model)
    agent = Agent(model=model, tools=[calculate], system_prompt=SYSTEM_PROMPT)
    await agent.invoke_async(task.instruction)
    return None
```

Each flow's trajectory `name` is set on the `@rllm.rollout` decorator. That's what the trainer uses to group rollouts of the same task into a `TrajectoryGroup` for advantage computation, so the four agents stay in their own GRPO groups even when run side-by-side on the same dataset.

## Install

```bash
# rllm + the backend you want to train on
uv pip install -e ".[tinker]"

# Pick one framework — or `[all]` for everything:
uv pip install --no-deps -e "cookbooks/agent_frameworks[langgraph]"
uv pip install --no-deps -e "cookbooks/agent_frameworks[openai-agents]"
uv pip install --no-deps -e "cookbooks/agent_frameworks[smolagents]"
uv pip install --no-deps -e "cookbooks/agent_frameworks[strands]"
uv pip install --no-deps -e "cookbooks/agent_frameworks[all]"

# Verify discovery — lists every entry-point agent whose framework is installed.
rllm agent list
```

## Datasets

```bash
rllm dataset pull deepscaler_math   # ~40K AIME/AMC/Omni-MATH/STILL competition math (train)
rllm dataset pull math500           # 500-problem test benchmark
```

## Eval

```bash
rllm eval math500 \
    --agent strands_math \
    --evaluator math_evaluator \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --base-url http://localhost:8000/v1 \
    --max-examples 20
```

Substitute `--agent` with `langgraph_math`, `openai_agents_math`, `smolagents_math`, or `strands_math`. Same evaluator name (`math_evaluator`) for every flow.

## Training

```bash
# Tinker (single-machine LoRA) — pick the agent as the first arg
bash cookbooks/agent_frameworks/train_tinker.sh langgraph_math
bash cookbooks/agent_frameworks/train_tinker.sh strands_math

# Verl (distributed GPU)
bash cookbooks/agent_frameworks/train_verl.sh openai_agents_math
```

Or directly via `train.py`:

```bash
python cookbooks/agent_frameworks/train.py \
    +rllm.agent_name=smolagents_math \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507
```

## Tests

```bash
pytest cookbooks/agent_frameworks/test.py -v
```

Tests cover the shared `calculator.py` and `evaluator.py`, plus a parameterized smoke test that confirms each flow satisfies the `AgentFlow` protocol (skipped automatically if the framework SDK isn't installed).

## Adding a new framework

1. Create `agentflow/<framework>.py` with one `@rllm.rollout(name="<framework>-math")` function that wires the framework's LLM client to `config.base_url`, runs the agent on `task.instruction`, and `return None`s.
2. Add it to `pyproject.toml`'s `[project.entry-points."rllm.agents"]` and `[tool.setuptools] py-modules` lists; declare the framework's package in `[project.optional-dependencies].<framework>`.
3. Reinstall the cookbook with `uv pip install --no-deps -e "cookbooks/agent_frameworks[<framework>]"` and your agent shows up under `rllm agent list`.

That's the entire integration surface.

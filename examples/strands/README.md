# Train Strands RAG Agent with rLLM SDK

In this tutorial, you'll train a retrieval-augmented generation (RAG) agent built with [Strands Agents SDK](https://github.com/strands-agents/sdk-python). This demonstrates that rLLM SDK works seamlessly with popular agent frameworks — your Strands code runs unchanged.

Based on the [LangGraph RAG tutorial](https://rllm-project.readthedocs.io/en/latest/examples/sdk_langgraph_rag/) (`examples/sdk/langgraph/`).

## Overview

By the end of this tutorial, you will have:

- Built a Strands agent with a `@tool` retrieval tool
- Injected rLLM SDK tracing into Strands' OpenAI model
- Trained the agent to search effectively using RL

## Concepts

We will cover:

- **Client injection**: Swap Strands' OpenAI client with traced SDK client
- **NonStreamingOpenAIModel**: Subclass that forces `stream=False` for LiteLLM proxy compatibility
- **Multi-turn tracing**: All LLM calls in an agentic loop are captured
- **Tool-turn budget**: Enforce step limits via tool call counting

## Setup

Move to rllm folder, install dependencies:

```bash
pip install strands-agents strands-agents-tools httpx
```

Download HotpotQA dataset, Wikipedia corpus and pre-built FAISS indices (data scripts are shared with the LangGraph example):

```bash
cd examples/sdk/langgraph
python data/prepare_hotpotqa_data.py
python data/download_search_data.py --data_dir ./search_data
cat search_data/prebuilt_indices/part_aa search_data/prebuilt_indices/part_ab > search_data/prebuilt_indices/e5_Flat.index
mv search_data/wikipedia/wiki-18.jsonl search_data/prebuilt_indices/corpus.json
cd ../../..
```

The data is stored in `examples/sdk/langgraph/search_data/`. Pass this path when launching the RAG server below.

Install env for retrieval server (recommend starting a fresh env):

```bash
conda create -n rag-server python=3.10 pip -y
pip install faiss-gpu==1.7.2 fastapi uvicorn numpy==1.26.4 sentence-transformers torch
```

Start the RAG server on port 9002:

```bash
cd examples/strands/rag
bash launch_rag.sh ../../sdk/langgraph/search_data/prebuilt_indices 9002
```

## 1. Client Injection

Strands' `OpenAIModel` accepts a custom `client` parameter. By injecting our traced async client, all LLM calls flow through our proxy automatically.

### 1.1 Normal Strands (no tracing)

```python
from strands import Agent
from strands.models.openai import OpenAIModel

# Standard usage - no tracing
model = OpenAIModel(
    model_id="Qwen/Qwen3-4B",
)
agent = Agent(model=model, tools=[...])
```

### 1.2 With rLLM SDK tracing

```python
from strands import Agent
from strands.models.openai import OpenAIModel
from rllm.sdk import get_chat_client_async

# Create traced async client
traced_client = get_chat_client_async(
    base_url="http://localhost:4000/v1",
    api_key="token-abc123",
    use_proxy=True,
)

# Inject into OpenAIModel
model = OpenAIModel(
    model_id="Qwen/Qwen3-4B",
    client=traced_client,
)
agent = Agent(model=model, tools=[...])
```

That's it! Your Strands agent now has full tracing with zero code changes to the workflow logic.

## 2. NonStreamingOpenAIModel

Strands SDK hardcodes `stream=True` in all requests, but LiteLLM proxy's `async_post_call_success_hook` only fires for non-streaming requests. Streaming responses skip the hook entirely, so zero traces are written and training crashes.

We subclass `OpenAIModel` to force `stream=False` and convert `ChatCompletion` responses back to the `StreamEvent` dicts that Strands' event loop expects:

```python
from strands.models.openai import OpenAIModel

class NonStreamingOpenAIModel(OpenAIModel):
    def format_request(self, messages, tool_specs=None, system_prompt=None,
                       tool_choice=None, **kwargs):
        request = super().format_request(messages, tool_specs, system_prompt,
                                         tool_choice, **kwargs)
        request["stream"] = False
        request.pop("stream_options", None)
        return request

    async def stream(self, messages, tool_specs=None, system_prompt=None,
                     *, tool_choice=None, **kwargs):
        """Non-streaming: make one API call, yield StreamEvent dicts."""
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)

        async with self._get_client() as client:
            response = await client.chat.completions.create(**request)

            yield {"messageStart": {"role": "assistant"}}

            choice = response.choices[0]
            message = choice.message

            if message.content:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": message.content}}}
                yield {"contentBlockStop": {}}

            if message.tool_calls:
                for tc in message.tool_calls:
                    yield {"contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": tc.id, "name": tc.function.name}}
                    }}
                    yield {"contentBlockDelta": {
                        "delta": {"toolUse": {"input": tc.function.arguments}}
                    }}
                    yield {"contentBlockStop": {}}

            stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
            stop_reason = stop_map.get(choice.finish_reason, "end_turn")
            yield {"messageStop": {"stopReason": stop_reason}}
```

## 3. Define the Retrieval Tool

Strands uses the `@tool` decorator — simpler than LangGraph's `to_langchain_tool()`:

```python
import httpx
from strands import tool

RETRIEVAL_SERVER_URL = "http://127.0.0.1:9002"

_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0, connect=5.0),
    limits=httpx.Limits(max_connections=2000, max_keepalive_connections=200),
)

@tool
async def local_search(query: str, top_k: int = 5) -> str:
    """Search for information using a dense retrieval server with Wikipedia corpus."""
    top_k = min(max(1, top_k), 50)
    response = await _client.post(
        f"{RETRIEVAL_SERVER_URL}/retrieve",
        json={"query": query, "top_k": top_k},
    )
    response.raise_for_status()
    results = response.json().get("results", [])

    if not results:
        return "No relevant documents found for the query."

    formatted = []
    for i, result in enumerate(results[:top_k], 1):
        doc_id = result.get("id", f"doc_{i}")
        content = result.get("content", "")
        if isinstance(content, dict):
            content = content.get("contents", "")
        score = result.get("score", 0.0)
        if len(content) > 300:
            content = content[:300] + "..."
        formatted.append(f"[Document {i}] (ID: {doc_id}, Score: {score:.3f})\n{content}\n")

    return "\n".join(formatted)
```

## 4. Build the Agent and Run Function

### 4.1 Create the agent

```python
from strands import Agent
from strands.handlers.callback_handler import null_callback_handler

SEARCH_SYSTEM_PROMPT = """You are a helpful AI assistant that can search for information.

When answering questions:
1. Use the search tool to find relevant information
2. Synthesize information from multiple sources
3. Put your final answer in \\boxed{} format

Example: \\boxed{Paris}"""

model = NonStreamingOpenAIModel(
    model_id="Qwen/Qwen3-4B",
    client=traced_client,
    params={"temperature": 0.7, "max_tokens": 2048},
)

agent = Agent(
    model=model,
    tools=[local_search],
    system_prompt=SEARCH_SYSTEM_PROMPT,
    callback_handler=null_callback_handler,
)
```

### 4.2 Run with tool-turn budget

Unlike LangGraph (which streams full node steps), Strands streams token-by-token. Budget is enforced by counting `current_tool_use` events:

```python
import re
from strands.event_loop.event_loop import MaxTokensReachedException

MAX_TURNS = 5

async def run_search_agent(question: str, ground_truth: str, max_turns: int = MAX_TURNS) -> dict:
    agent = Agent(model=model, tools=[local_search],
                  system_prompt=SEARCH_SYSTEM_PROMPT,
                  callback_handler=null_callback_handler)

    content = ""
    final_answer = None
    num_tool_turns = 0
    timed_out = False

    stream = agent.stream_async(question)
    try:
        async for event in stream:
            # Count tool calls
            is_tool_use = (isinstance(event, dict)
                          and "current_tool_use" in event
                          and event.get("current_tool_use", {}).get("name"))
            if is_tool_use:
                num_tool_turns += 1
                if num_tool_turns > max_turns:
                    timed_out = True
                    break

            # Accumulate text content
            if isinstance(event, dict) and "contentBlockDelta" in event:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                content += delta.get("text", "")

            # Extract \boxed{} answer
            match = re.search(r"\\boxed\{([^}]+)\}", content)
            if match:
                final_answer = match.group(1)

    except MaxTokensReachedException:
        pass
    finally:
        await stream.aclose()

    # Compute reward
    reward = 0.0
    if final_answer and not timed_out:
        from rllm.rewards.search_reward import RewardConfig, RewardSearchFn
        from rllm.rewards.reward_fn import RewardInput
        reward_fn = RewardSearchFn(RewardConfig())
        reward = reward_fn(RewardInput(
            task_info={"ground_truth": ground_truth}, action=final_answer
        )).reward

    return {"final_answer": final_answer, "reward": reward,
            "num_tool_turns": num_tool_turns, "timed_out": timed_out}
```

### 4.3 Test the agent

```python
result = await run_search_agent(
    question="What is the capital of France?",
    ground_truth="Paris"
)
print(f"Answer: {result['final_answer']}")
print(f"Reward: {result['reward']}")
print(f"Tool turns: {result['num_tool_turns']}")
```

Expected output:

```
Answer: Paris
Reward: 1.0
Tool turns: 2
```

## 5. Set Up Training

### 5.1 Training wrapper

```python
import hydra
from rllm.data import DatasetRegistry
from rllm.trainer import AgentTrainer
from search_agent_strands import run_search_agent

async def run_agent(question, ground_truth, max_steps=5, **kwargs):
    """Training wrapper - returns reward only."""
    try:
        result = await run_search_agent(question, ground_truth, max_turns=max_steps)
    except Exception:
        return 0.0
    return result["reward"]

@hydra.main(config_path="pkg://rllm.trainer.config",
            config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa-small", "test")

    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_run_func=run_agent,
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

### 5.2 Launch script

```bash
#!/bin/bash
# train_strands_agent.sh
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

python train_strands_agent.py \
    algorithm.adv_estimator=rloo \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=100 \
    rllm.sdk.store.path="./rllm-traces-strands.db"
```

## 6. Run Training

```bash
cd ~/rllm/examples/strands
bash train_strands_agent.sh
```

## 7. Training Results

Trained for ~700 steps on HotpotQA with RLOO, Qwen3-4B, 8x H100:

| Metric | Step 10 | Step 300 | Step 500 | Step 700+ |
|--------|---------|----------|----------|-----------|
| pass@1 | 0.2656 | 0.3672 | 0.3984 | 0.405-0.417 |
| reward | 0.20 | 0.33 | 0.36 | 0.37-0.39 |

**+15 percentage points pass@1 gain** (0.2656 → 0.417 peak), averaging ~0.405 at convergence.

## Files

```
examples/strands/
├── search_agent_strands.py   # Agent: NonStreamingOpenAIModel + budget enforcement
├── retrieve_tool.py          # @tool RAG client (httpx async, 30s timeout)
├── train_strands_agent.py    # Training entry point (Hydra)
├── train_strands_agent.sh    # Training launch script (8x H100 config)
├── rag/
│   ├── rag_server.py         # FastAPI auto-batching RAG server
│   └── launch_rag.sh         # RAG server launch script
└── README.md                 # This file
```

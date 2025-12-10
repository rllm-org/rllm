# Train LangGraph RAG Agent with rLLM SDK

In this tutorial, you'll train a retrieval-augmented generation (RAG) agent built with LangGraph. This demonstrates that rLLM SDK works seamlessly with popular agent frameworks—your LangGraph code runs unchanged.

## Overview

By the end of this tutorial, you will have:

1. Built a LangGraph agent with retrieval tool calling
2. Injected rLLM SDK tracing into LangChain's ChatOpenAI
3. Trained the agent to search effectively using RL

### Concepts

We will cover:

- **Client injection**: Swap ChatOpenAI's internal client with traced SDK client
- **LangGraph workflow**: StateGraph, nodes, edges, and `tools_condition`
- **Multi-turn tracing**: All LLM calls in an agentic loop are captured

---

## Setup

Install dependencies:

```bash
pip install langchain-openai langgraph
```

Start the retrieval server (HotPotQA):

```bash
python examples/sdk/langgraph/rag_server.py --port 9002 &
```

---

## 1. The Key Insight: Client Injection

LangChain's `ChatOpenAI` accepts custom `client` and `async_client` parameters. By injecting our traced clients, all LLM calls flow through our proxy automatically.

### 1.1 Normal LangChain (no tracing)

```python
from langchain_openai import ChatOpenAI

# Standard usage - no tracing
llm = ChatOpenAI(model="gpt-4o")
```

### 1.2 With rLLM SDK tracing

```python
from langchain_openai import ChatOpenAI
from rllm.sdk import get_chat_client, get_chat_client_async

# Create traced clients
sync_client = get_chat_client(
    base_url="http://localhost:4000/v1",
    api_key="EMPTY"
)
async_client = get_chat_client_async(
    base_url="http://localhost:4000/v1",
    api_key="EMPTY"
)

# Inject into ChatOpenAI
llm = ChatOpenAI(
    model="Qwen/Qwen3-4B",
    client=sync_client,        # ← Traced!
    async_client=async_client, # ← Traced!
)
```

**That's it!** Your LangGraph agent now has full tracing with zero code changes to the workflow logic.

---

## 2. Build the LangGraph Agent

### 2.1 Import dependencies

```python
import os
import re
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rllm.sdk import get_chat_client, get_chat_client_async
```

### 2.2 Configure the model with traced clients

```python
MODEL = "Qwen/Qwen3-4B"
MAX_RESPONSE_TOKENS = 2048

# Create traced clients
sync_client = get_chat_client(
    base_url="http://localhost:4000/v1",
    api_key="EMPTY"
)
async_client = get_chat_client_async(
    base_url="http://localhost:4000/v1",
    api_key="EMPTY"
)

# Inject into ChatOpenAI
response_model = ChatOpenAI(
    model=MODEL,
    temperature=0.7,
    max_tokens=MAX_RESPONSE_TOKENS,
    client=sync_client,
    async_client=async_client,
)
```

### 2.3 Define the retrieval tool

```python
from examples.sdk.langgraph.local_retrieval_tool import to_langchain_tool

retriever_tool = to_langchain_tool(
    server_url="http://127.0.0.1:9002",
    max_results=5,
    timeout=30.0,
)
```

### 2.4 Create the agent node

```python
SYSTEM_PROMPT = """You are a helpful AI assistant that can search for information.

When answering questions:
1. Use the search tool to find relevant information
2. Synthesize information from multiple sources
3. Put your final answer in \\boxed{} format

Example: \\boxed{Paris}"""

async def agent_step(state: MessagesState):
    """Agent decides: call tools or provide final answer."""
    response = await response_model.bind_tools([retriever_tool]).ainvoke(
        state["messages"]
    )
    return {"messages": [response]}
```

### 2.5 Assemble the graph

```python
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("agent", agent_step)
workflow.add_node("tools", ToolNode([retriever_tool]))

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Routes to "tools" or END based on tool calls
    {
        "tools": "tools",
        END: END,
    },
)
workflow.add_edge("tools", "agent")

# Compile
graph = workflow.compile()
```

### 2.6 Test the graph

```python
async def test_agent():
    async for chunk in graph.astream(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        },
        {"recursion_limit": 10},
    ):
        for node_name, update in chunk.items():
            print(f"Node: {node_name}")
            if "messages" in update:
                print(f"  → {update['messages'][-1].content[:100]}...")

# Run test
import asyncio
asyncio.run(test_agent())
```

**Expected output:**
```
Node: agent
  → Tool call: retrieve_documents(query="capital of France")
Node: tools
  → Paris is the capital and largest city of France...
Node: agent
  → The capital of France is \boxed{Paris}...
```

---

## 3. Create the Run Function

Wrap the graph execution with reward computation.

### 3.1 Define the run function

```python
from rllm.rewards.search_reward import RewardConfig, RewardSearchFn

async def run_search_agent(question: str, ground_truth: str, max_turns: int = 5) -> dict:
    """Run agent and compute reward."""
    
    final_answer = None
    num_turns = 0
    timed_out = False

    async for chunk in graph.astream(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
        },
        {"recursion_limit": max_turns * 2 + 5},
    ):
        for node_name, update in chunk.items():
            if node_name == "agent":
                num_turns += 1
                if num_turns > max_turns:
                    timed_out = True
                    break

            # Extract answer from \boxed{}
            if "messages" in update and update["messages"]:
                content = update["messages"][-1].content
                match = re.search(r"\\boxed\{([^}]+)\}", content)
                if match:
                    final_answer = match.group(1)

        if timed_out:
            break

    # Compute reward
    reward = 0.0
    if final_answer and not timed_out:
        reward_fn = RewardSearchFn(RewardConfig())
        reward = reward_fn({"ground_truth": ground_truth}, final_answer).reward

    return {
        "final_answer": final_answer,
        "reward": reward,
        "num_turns": num_turns,
        "timed_out": timed_out,
    }
```

### 3.2 Test the run function

```python
result = await run_search_agent(
    question="What is the capital of France?",
    ground_truth="Paris"
)
print(f"Answer: {result['final_answer']}")
print(f"Reward: {result['reward']}")
print(f"Turns: {result['num_turns']}")
```

**Expected output:**
```
Answer: Paris
Reward: 1.0
Turns: 2
```

---

## 4. Set Up Training

### 4.1 Training wrapper

```python
import hydra
from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

async def run_agent(question, ground_truth, **kwargs):
    """Training wrapper - returns reward only."""
    try:
        result = await run_search_agent(question, ground_truth)
        return result["reward"]
    except Exception:
        return 0.0

@hydra.main(
    config_path="pkg://rllm.trainer.config", 
    config_name="agent_ppo_trainer", 
    version_base=None
)
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

### 4.2 Launch script

```bash
#!/bin/bash
# train_rag_agent.sh
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

# Start retrieval server
python examples/sdk/langgraph/rag_server.py --port 9002 &
sleep 5

MODEL_PATH=Qwen/Qwen3-4B

python3 -m examples.sdk.langgraph.train_rag_agent \
    algorithm.adv_estimator=rloo \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.total_epochs=3 \
    trainer.project_name=langgraph-rag \
    trainer.experiment_name=hotpotqa-rloo
```

---

## 5. Run Training

```bash
chmod +x train_rag_agent.sh
./train_rag_agent.sh
```

---

## 6. What Gets Traced

Every LLM call in the agentic loop is captured:

```
Turn 1: agent → "I need to search for..."  [TRACED]
Turn 2: tools → (retrieval results)
Turn 3: agent → "Based on the search..."   [TRACED]
Turn 4: agent → "\boxed{Paris}"            [TRACED]
```

The SDK captures:

- **Prompt token IDs** for each turn
- **Response token IDs** for each turn
- **Tool call metadata** (not trained on, but logged)

---

## Next Steps

- **[Tutorial 1](sdk_math.md)**: Review basics with a single-step agent
- **[Tutorial 2](sdk_solver_judge.md)**: Multi-agent patterns with `@trajectory`
- **[SDK Documentation](../core-concepts/sdk.md)**: Full API reference

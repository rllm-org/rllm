#!/usr/bin/env python3
"""
LangGraph-based search agent for HotPotQA using local retrieval server.

This is a LangGraph implementation similar to examples/search/run_search_agent.py
but using LangGraph's agent framework instead of RLLM's AgentExecutionEngine.

Prerequisites:
1. Start the retrieval server:
   python examples/sdk/langgraph/retrieve_server.py --data_dir ./search_data/prebuilt_indices --port 9002

2. Set environment variables:
   export OPENAI_API_KEY="your-api-key"
   export RETRIEVAL_SERVER_URL="http://127.0.0.1:9002"

3. Run this script:
   python examples/sdk/langgraph/search_agent_langgraph.py
"""

import asyncio
import os
import re

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rllm.data import DatasetRegistry
from rllm.rewards.reward_fn import RewardInput
from rllm.rewards.search_reward import RewardConfig, RewardSearchFn
from rllm.sdk import get_chat_client, get_chat_client_async
from rllm.sdk.session.base import _ensure_tracer_initialized

# Import the conversion function
from .local_retrieval_tool import to_langchain_tool

MODEL = "Qwen/Qwen3-4B"
# MODEL = "gpt-4.1"
MAX_TURNS = 5
MAX_RESPONSE_TOKENS = 2048

TRAIN = True

if TRAIN:
    base_url = "http://localhost:4000/v1"
    api_key = ""
    use_proxy = True
else:
    base_url = "http://localhost:8000/v1"
    api_key = "token-abc123"
    use_proxy = False

SEARCH_SYSTEM_PROMPT = """You are a helpful AI assistant that can search for information to answer questions accurately.

When answering questions:
1. Use the available search tools to find relevant and reliable information
2. Synthesize information from multiple sources when needed
3. Provide accurate and comprehensive answers based on your search results
4. Always put your final answer in \\boxed{} format

For example:
- If the answer is "American", write: \\boxed{American}
- If the answer is "yes", write: \\boxed{yes}
- If the answer is a year like "1985", write: \\boxed{1985}

Remember to search thoroughly and provide your final answer clearly within the \\boxed{} format."""


# Convert LocalRetrievalTool to LangChain StructuredTool
retriever_tool = to_langchain_tool(
    server_url=os.getenv("RETRIEVAL_SERVER_URL", "http://127.0.0.1:9002"),
    max_results=5,  # Match retrieval server default
    timeout=30.0,
)

# Initialize the chat model with RLLM SDK
# Use both sync client (for LangChain internals) and async client (for async operations)
sync_client = get_chat_client(
    api_key=api_key,
    base_url=base_url,
    use_proxy=use_proxy,
)
async_client = get_chat_client_async(
    api_key=api_key,
    base_url=base_url,
    use_proxy=use_proxy,
)

# Pass both clients to ChatOpenAI - it will use async_client for ainvoke()
response_model = ChatOpenAI(
    model=MODEL,
    temperature=0.7,
    max_tokens=MAX_RESPONSE_TOKENS,
    client=sync_client,
    async_client=async_client,
)
_ensure_tracer_initialized("search_agent")


async def agent_step(state: MessagesState):
    """
    Agent decides whether to call tools or provide final answer.
    Uses async ainvoke for better concurrency.
    """
    response = await response_model.bind_tools([retriever_tool]).ainvoke(state["messages"])
    return {"messages": [response]}


# Build the workflow - simple multi-step agent
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("agent", agent_step)
workflow.add_node("tools", ToolNode([retriever_tool]))

# Add edges
workflow.add_edge(START, "agent")

# After agent: if tool calls, execute tools; otherwise end
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Built-in function that checks for tool calls
    {
        "tools": "tools",  # If tool calls exist, go to tools node
        END: END,  # If no tool calls, we're done
    },
)

# After tools execute, go back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
graph = workflow.compile()

# Maximum number of turns (agent steps) before timeout


async def run_search_agent(question: str, ground_truth: str = None, max_turns: int = MAX_TURNS) -> dict:
    """
    Run the search agent on a single question asynchronously.

    Args:
        question: The question to answer
        ground_truth: Optional ground truth answer for evaluation
        max_turns: Maximum number of turns before timeout (default: 20)

    Returns:
        dict: Results including the final answer and trajectory
    """
    messages = []
    final_answer = None
    num_turns = 0
    timed_out = False

    # Use recursion_limit to enforce max turns
    # Each turn = agent step + tool step, so recursion_limit = max_turns * 2
    # Use astream for async iteration
    async for chunk in graph.astream(
        {"messages": [{"role": "system", "content": SEARCH_SYSTEM_PROMPT}, {"role": "user", "content": question}]},
        {"recursion_limit": max_turns * 2 + 5},  # +5 for safety margin
    ):
        for node_name, update in chunk.items():
            # Count agent steps as turns (not tool executions)
            if node_name == "agent":
                num_turns += 1

                # Check if we've exceeded max turns
                if num_turns > max_turns:
                    timed_out = True
                    break

            if "messages" in update and update["messages"]:
                last_msg = update["messages"][-1]
                messages.append(last_msg)

                # Extract final answer if present
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                match = re.search(r"\\boxed\{([^}]+)\}", content)
                if match:
                    final_answer = match.group(1)

        if timed_out:
            break

    result = {
        "question": question,
        "final_answer": final_answer,
        "ground_truth": ground_truth,
        "messages": messages,
        "num_turns": num_turns,
        "timed_out": timed_out,
    }

    # Evaluate if ground truth is provided
    if ground_truth:
        reward_fn = RewardSearchFn(RewardConfig())

        # If timed out or no answer, score is 0
        if timed_out or not final_answer:
            result["is_correct"] = False
            result["reward"] = 0.0
            result["evaluation_metadata"] = {"reason": "timeout" if timed_out else "no_answer"}
        else:
            # Normal evaluation
            reward_input = RewardInput(task_info={"ground_truth": ground_truth}, action=final_answer)
            reward_output = reward_fn(reward_input)

            result["is_correct"] = reward_output.is_correct
            result["reward"] = reward_output.reward
            result["evaluation_metadata"] = reward_output.metadata

    return result


async def main():
    """Main async entry point."""
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "RETRIEVAL_SERVER_URL" not in os.environ:
        os.environ["RETRIEVAL_SERVER_URL"] = "http://127.0.0.1:9002"

    test_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
    results = []
    for task in test_dataset.get_data()[:5]:  # Run on first 5 for demo
        result = await run_search_agent(question=task["question"], ground_truth=task.get("ground_truth"))
        results.append(result)

        print(f"\nQuestion: {result['question'][:100]}...")
        print(f"Answer: {result['final_answer']}")
        print(f"Correct: {result.get('is_correct', 'N/A')}")

    # Summary statistics
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {correct_count}/{len(results)} correct")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())

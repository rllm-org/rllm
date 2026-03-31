"""LangGraph framework ReAct agent plugin for rLLM.

Uses LangGraph's ``StateGraph`` with a ``ChatOpenAI`` model routed through
the eval proxy.  Adapts to any benchmark via TaskSpec.

Supports both text and multimodal (VLM) benchmarks — LangChain's
``HumanMessage`` natively accepts OpenAI-style multimodal content blocks.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph

from rllm.experimental.eval.types import AgentConfig
from rllm.sdk.integrations.langgraph import RLLMTrajectoryCallbackHandler
from rllm.types import Episode


def _get_spec_and_data(task: Any) -> tuple[Any, dict]:
    """Extract TaskSpec and raw data from a task (Task object or dict)."""
    spec = None
    data = task
    if hasattr(task, "spec"):
        spec = task.spec
    if hasattr(task, "data"):
        data = task.data
    if isinstance(data, dict):
        return spec, data
    return spec, {"question": str(data)}


def _bridge_tool(tool_def: dict) -> StructuredTool:
    """Bridge an OpenAI tool dict (with ``_execute``) to a LangChain ``StructuredTool``."""
    func_spec = tool_def.get("function", {})
    name = func_spec.get("name", "tool")
    description = func_spec.get("description", "A tool.")
    executor = tool_def.get("_execute")

    def tool_func(**kwargs):
        if executor is not None:
            return str(executor(**kwargs))
        return ""

    return StructuredTool.from_function(
        func=tool_func,
        name=name,
        description=description,
    )


def _extract_final_answer(messages: list) -> str:
    """Extract the final AI message content from graph output."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                if text_parts:
                    return "\n".join(text_parts)
    return ""


_MAX_TURNS = 16


class LangGraphAgentFlow:
    """LangGraph ReAct agent that works with any benchmark via TaskSpec."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        spec, data = _get_spec_and_data(task)

        system_prompt = "You are a capable AI assistant.\n\n"
        system_prompt += spec.instruction if spec else "Solve the given task."

        if spec:
            user_content = spec.render_input(data)
        else:
            user_content = data.get("question", str(data))

        # user_content may be a string (text tasks) or a list of content
        # blocks (multimodal tasks).  HumanMessage accepts both natively.
        # Extract a text summary for the tracer input field.
        if isinstance(user_content, list):
            user_text = ""
            for block in user_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    user_text = block.get("text", "")
                    break
            if not user_text:
                user_text = str(user_content)
        elif isinstance(user_content, str):
            user_text = user_content
        else:
            user_content = str(user_content)
            user_text = user_content

        # Create model routed through the eval proxy
        llm = ChatOpenAI(
            model=config.model,
            base_url=config.base_url,
            api_key="EMPTY",
        )

        # Set up tracing
        callback_handler = RLLMTrajectoryCallbackHandler()
        callback_handler._user_input = {"message": user_text[:500]}

        # Bridge tools from config metadata
        tools_meta: list[dict] = config.metadata.get("tools", [])
        bridged_tools = [_bridge_tool(t) for t in tools_meta if "function" in t]

        has_tools = len(bridged_tools) > 0

        if has_tools:
            llm_with_tools = llm.bind_tools(bridged_tools)
        else:
            llm_with_tools = llm

        # Build the graph
        def agent_node(state: MessagesState) -> dict:
            response = llm_with_tools.invoke(
                state["messages"],
                config={"callbacks": [callback_handler]},
            )
            return {"messages": [response]}

        def should_continue(state: MessagesState) -> str:
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tools"
            return END

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")

        if has_tools:
            tool_map = {t.name: t for t in bridged_tools}

            def tools_node(state: MessagesState) -> dict:
                last_msg = state["messages"][-1]
                tool_messages = []
                for tc in last_msg.tool_calls:
                    tool = tool_map.get(tc["name"])
                    if tool is not None:
                        result = tool.invoke(tc["args"])
                    else:
                        result = f"Tool '{tc['name']}' not found."
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                return {"messages": tool_messages}

            graph.add_node("tools", tools_node)
            graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
            graph.add_edge("tools", "agent")
        else:
            graph.add_edge("agent", END)

        app = graph.compile()

        # Run the graph
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        result = app.invoke(
            {"messages": initial_messages},
            config={"recursion_limit": _MAX_TURNS * 2 + 5},
        )

        answer = _extract_final_answer(result.get("messages", []))

        # Update tracer output
        callback_handler._last_output = answer
        callback_handler._trajectory_built = False
        traj = callback_handler.get_trajectory()

        return Episode(
            task=data,
            trajectories=[traj],
            artifacts={"answer": answer},
        )


langgraph_agent = LangGraphAgentFlow()

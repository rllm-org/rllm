"""{{project_name}}: A LangChain/LangGraph agent for rllm eval.

Prerequisites:
    pip install langchain-openai langgraph

Usage:
    rllm model setup
    rllm eval gsm8k --agent {{module_name}}.agent:{{agent_instance}}
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory


SYSTEM_PROMPT = """\
You are a helpful assistant. Solve the problem step by step.
Put your final answer inside \\boxed{}."""


# Define your tools here
# from langchain_core.tools import tool
# @tool
# def search(query: str) -> str:
#     """Search for information."""
#     return "search result"


class {{agent_class}}:
    """LangGraph-based agent wrapped as an rllm AgentFlow."""

    def _build_graph(self, model: ChatOpenAI):
        """Build the LangGraph workflow."""
        tools = []  # Add your tools here: [search, ...]

        def agent_step(state: MessagesState):
            if tools:
                response = model.bind_tools(tools).invoke(state["messages"])
            else:
                response = model.invoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_step)

        if tools:
            graph.add_node("tools", ToolNode(tools))
            graph.add_edge(START, "agent")
            graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
            graph.add_edge("tools", "agent")
        else:
            graph.add_edge(START, "agent")
            graph.add_edge("agent", END)

        return graph.compile()

    def run(self, task: dict, config: AgentConfig) -> Episode:
        model = ChatOpenAI(
            model=config.model,
            base_url=config.base_url,
            api_key="EMPTY",
        )
        graph = self._build_graph(model)
        question = task.get("question", "")

        result = graph.invoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        })

        answer = result["messages"][-1].content or ""
        step = Step(input=question, output=answer, done=True)
        traj = Trajectory(name="solver", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": answer})


{{agent_instance}} = {{agent_class}}()

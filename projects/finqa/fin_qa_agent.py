# Third Party Imports
from rllm.agents.tool_agent import ToolAgent

from .constants import REACT_SYSTEM_PROMPT_PATH

# Local Imports
from .fin_qa_tools import Calculator, GetTableInfo, GetTableNames, SQLQuery

# Load ReAct system prompt
with open(REACT_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
    FINQA_REACT_SYSTEM_PROMPT = f.read().strip()


class FinQAAgent(ToolAgent):
    """
    An agent that can answer questions about financial statements.
    """

    def __init__(
        self,
        system_prompt=FINQA_REACT_SYSTEM_PROMPT,
        parser_name="qwen",
    ):
        """
        Initialize the FinQAAgent.

        Args:
            system_prompt: System prompt for the agent.
                Default: FINQA_REACT_SYSTEM_PROMPT
            parser_name: Name of the parser to use for tool calls.
                Default: "qwen", same default as ToolAgent.
        """

        # Initialize the tool map.
        fin_qa_tool_map = {
            "calculator": Calculator,
            "get_table_info": GetTableInfo,
            "get_table_names": GetTableNames,
            "sql_query": SQLQuery,
        }

        # Initialize the ToolAgent.
        super().__init__(
            system_prompt=system_prompt,
            parser_name=parser_name,
            tool_map=fin_qa_tool_map,
        )

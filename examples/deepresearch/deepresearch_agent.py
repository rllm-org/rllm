"""
DeepResearch Agent - Adapted from Tongyi DeepResearch for rLLM

This is the core ReAct agent that implements DeepResearch's reasoning and tool-calling logic,
adapted to work with rLLM's OpenAI engine instead of the original server-based approach.

Original: https://github.com/Alibaba-NLP/DeepResearch/blob/main/inference/react_agent.py
"""

import asyncio
import time
from datetime import datetime

import json5

# rLLM imports
from rllm.engine.rollout import RolloutEngine

# Constants from original DeepResearch
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = 100

# System prompt adapted from DeepResearch
DEEPRESEARCH_SYSTEM_PROMPT = """You are an autonomous intelligent agent tasked with answering questions and performing research tasks.

You have access to the following tools:
- Search: for web searches to find current information
- FileParser: for reading and analyzing files
- Scholar: for academic research and paper searches
- Visit: for visiting and analyzing web pages
- PythonInterpreter: for running Python code and calculations

Use the following format for your reasoning and actions:

<think>
Your thoughts about what to do next, analyzing the question and planning your approach.
</think>

When you need to use a tool, format it as:
<tool_call>
{"name": "ToolName", "arguments": {"arg1": "value1", "arg2": "value2"}}
</tool_call>

For Python code execution, use:
<tool_call>
python
<code>
# Your Python code here
print("Hello World")
</code>
</tool_call>

When you have gathered enough information and can provide a final answer, format it as:
<answer>
Your final answer based on your research and analysis
</answer>

Current date: """


def today_date():
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().date().strftime("%Y-%m-%d")


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent adapted from Tongyi DeepResearch.

    This agent implements the core reasoning loop with tool calling capabilities,
    using rLLM's OpenAI engine for model inference.
    """

    def __init__(self, rollout_engine: RolloutEngine, tools: dict = None, **kwargs):
        """
        Initialize the ReAct agent.

        Args:
            rollout_engine: rLLM OpenAI engine for model inference
            tools: Dictionary of available tools {tool_name: tool_instance}
        """
        self.rollout_engine = rollout_engine
        self.tools = tools or {}

        # Configuration from original DeepResearch
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        self.max_tokens = 108 * 1024  # Context length limit
        self.max_time = 150 * 60  # 150 minutes timeout

    def sanity_check_output(self, content: str) -> bool:
        """Check if the model output contains the expected thinking structure."""
        return "<think>" in content and "</think>" in content

    async def call_server(self, messages: list[dict], max_tries: int = 10) -> str:
        """
        Call rLLM OpenAI engine (replacement for original call_server method).

        Args:
            messages: List of chat completion messages
            max_tries: Maximum number of retry attempts

        Returns:
            Model response text
        """
        for attempt in range(max_tries):
            try:
                print(
                    f"--- Attempting to call rLLM engine, try {attempt + 1}/{max_tries} ---"
                )

                # Call rLLM OpenAI Engine with DeepResearch parameters
                response = await self.rollout_engine.get_model_response(
                    messages=messages,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=4096,  # Reasonable for GPT-4o 128k context
                    presence_penalty=1.1,
                )

                # Extract text from ModelOutput
                content = response.text if hasattr(response, "text") else str(response)

                if content and content.strip():
                    print("--- rLLM engine call successful ---")
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received empty response")

            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed: {e}")
                if attempt < max_tries - 1:
                    # Exponential backoff
                    sleep_time = 2**attempt
                    print(f"Waiting {sleep_time} seconds before retry...")
                    await asyncio.sleep(sleep_time)

        raise Exception(f"Failed to get response after {max_tries} attempts")

    def count_tokens(self, messages: list[dict], model: str = "gpt-4o") -> int:
        """
        Estimate token count for messages (simplified version).

        Args:
            messages: List of chat completion messages
            model: Model name (for compatibility)

        Returns:
            Estimated token count
        """
        total_text = ""
        for msg in messages:
            total_text += msg.get("content", "")

        # Rough estimate: 4 characters per token
        return len(total_text) // 4

    async def _run(self, question: str, answer: str = None, **kwargs) -> dict:
        """
        Main reasoning loop adapted from original DeepResearch.

        This is the core ReAct implementation that handles:
        - Multi-turn conversation
        - Tool calling and execution
        - Context length management
        - Termination conditions

        Args:
            question: The research question to answer
            answer: Ground truth answer (for evaluation)

        Returns:
            Dictionary with results including messages, prediction, and termination reason
        """
        start_time = time.time()

        # Setup system prompt with current date
        system_prompt = DEEPRESEARCH_SYSTEM_PROMPT + today_date()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        num_llm_calls_available = self.max_llm_calls
        round_num = 0
        termination = None
        prediction = ""

        print(f"🔍 Starting DeepResearch for question: {question}")

        while num_llm_calls_available > 0:
            # Check time limit (150 minutes)
            if time.time() - start_time > self.max_time:
                prediction = "No answer found after 2h30mins"
                termination = "timeout"
                break

            round_num += 1
            num_llm_calls_available -= 1

            print(
                f"\n--- Round {round_num} ({num_llm_calls_available} calls remaining) ---"
            )

            # Get model response
            content = await self.call_server(messages)
            print(
                f"Model response: {content[:200]}..."
                if len(content) > 200
                else f"Model response: {content}"
            )

            # Clean up content if it contains tool_response
            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]

            messages.append({"role": "assistant", "content": content.strip()})

            # Check for final answer
            if "<answer>" in content and "</answer>" in content:
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
                termination = "answer"
                print(f"✅ Final answer found: {prediction}")
                break

            # Handle tool calls
            if "<tool_call>" in content and "</tool_call>" in content:
                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[
                    0
                ]
                try:
                    # Special handling for Python code
                    if "python" in tool_call_text.lower() and "<code>" in content:
                        code = content.split("<code>")[1].split("</code>")[0].strip()
                        result = await self.execute_python(code)
                        print(f"🐍 Python execution result: {result[:100]}...")
                    else:
                        # Parse JSON tool call
                        tool_call = json5.loads(tool_call_text)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})
                        result = await self.custom_call_tool(tool_name, tool_args)
                        print(f"🔧 Tool {tool_name} result: {result[:100]}...")

                except Exception as e:
                    result = f"Error: Tool call parsing failed: {e}"
                    print(f"❌ Tool call error: {e}")

                # Add tool response
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})

            # Check if we've exceeded call limit
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                messages[-1]["content"] = (
                    "Sorry, the number of llm calls exceeds the limit."
                )

            # Handle context length limit
            token_count = self.count_tokens(messages)
            print(f"Token count: {token_count}")

            if token_count > self.max_tokens:
                print(f"⚠️ Token limit exceeded: {token_count} > {self.max_tokens}")
                final_msg = "You have reached the maximum context length. Please provide your best answer based on the information above in the format: <think>your final thinking</think>\n<answer>your answer</answer>"
                messages[-1]["content"] = final_msg

                content = await self.call_server(messages)
                messages.append({"role": "assistant", "content": content.strip()})

                if "<answer>" in content and "</answer>" in content:
                    prediction = (
                        content.split("<answer>")[1].split("</answer>")[0].strip()
                    )
                    termination = "answer_token_limit"
                else:
                    prediction = content
                    termination = "token_limit_no_answer"
                break

        # Final result
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination or "max_rounds_reached",
            "rounds": round_num,
            "time_taken": time.time() - start_time,
        }

        print("\n🏁 DeepResearch completed:")
        print(f"   Rounds: {round_num}")
        print(f"   Time: {result['time_taken']:.1f}s")
        print(f"   Termination: {termination}")
        print(f"   Prediction: {prediction}")

        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> str:
        """
        Execute tool calls with the available tools.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        if tool_name in self.tools:
            try:
                # Call the tool
                if hasattr(self.tools[tool_name], "call"):
                    # Async tool
                    if asyncio.iscoroutinefunction(self.tools[tool_name].call):
                        result = await self.tools[tool_name].call(**tool_args)
                    else:
                        result = self.tools[tool_name].call(**tool_args)
                elif callable(self.tools[tool_name]):
                    # Direct callable
                    result = self.tools[tool_name](**tool_args)
                else:
                    result = f"Tool {tool_name} is not callable"

                return str(result)

            except Exception as e:
                return f"Error calling tool {tool_name}: {e}"
        else:
            available_tools = list(self.tools.keys())
            return f"Tool {tool_name} not found. Available tools: {available_tools}"

    async def execute_python(self, code: str) -> str:
        """
        Execute Python code (placeholder for now).

        Args:
            code: Python code to execute

        Returns:
            Execution result as string
        """
        try:
            # For now, just return the code - will be replaced with actual execution
            return f"[Python code executed]\nCode: {code}\n[Placeholder - actual execution not implemented yet]"
        except Exception as e:
            return f"[Python execution error]: {e}"

    def reset(self):
        """Reset the agent state (for compatibility with rLLM workflow)."""
        # The agent is stateless - each run() creates fresh state
        # No need to reset anything
        pass

    async def run(self, question: str, answer: str = None, **kwargs) -> dict:
        """
        Public interface for running the agent.

        Args:
            question: Research question to answer
            answer: Ground truth answer (optional, for evaluation)

        Returns:
            Result dictionary
        """
        return await self._run(question, answer, **kwargs)

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
DEEPRESEARCH_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with the following tools:
- Search: for web searches to find current information
- Scholar: for academic research and paper searches
- Visit: for visiting and analyzing web pages
- PythonInterpreter: for running Python code and calculations
- FileParser: for reading and analyzing files

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

For Python code execution, use:
<tool_call>
python
<code>
# Your Python code here
print("Hello World")
</code>
</tool_call>

Current date: """


def today_date():
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().date().strftime("%Y-%m-%d")


def build_text_completion_prompt(
    messages: list[dict], allow_special: bool = True
) -> str:
    """
    Build text completion prompt from messages list.
    Adapted from qwen_agent.utils.utils.build_text_completion_prompt

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        allow_special: Whether to allow special tokens (for compatibility)

    Returns:
        Formatted prompt string
    """
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt_parts = []

    # Handle system message
    if messages and messages[0]["role"] == "system":
        sys_content = messages[0]["content"]
        prompt_parts.append(f"{im_start}system\n{sys_content}{im_end}")
        messages = messages[1:]

    # Ensure chat completes with assistant
    if messages and messages[-1]["role"] != "assistant":
        messages = messages + [{"role": "assistant", "content": ""}]

    # Format each message
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"{im_start}{role}\n{content}{im_end}")

    return "\n".join(prompt_parts)


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent adapted from Tongyi DeepResearch.

    This agent implements the core reasoning loop with tool calling capabilities,
    using rLLM's OpenAI engine for model inference.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Args:
            rollout_engine: rLLM OpenAI engine for model inference
            tools: Dictionary of available tools {tool_name: tool_instance}
        """
        self.rollout_engine = rollout_engine
        self.tools = tools or {}
        self.system_prompt = system_prompt

        # Configuration from original DeepResearch
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        self.max_time = 150 * 60  # 150 minutes timeout

        # Smart context management using actual API consumption
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Use the same conservative limit as original DeepResearch
        # This works for most modern models (GPT-4o 128k, Qwen 128k, etc.)
        self.max_context_tokens = 108 * 1024  # 110,592 tokens, same as original

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
                # Call rLLM OpenAI Engine with DeepResearch parameters
                response = await self.rollout_engine.get_model_response(
                    messages=messages,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=4096,  # Reasonable for GPT-4o 128k context
                    presence_penalty=1.1,
                )

                # Track actual token consumption from API
                if hasattr(response, "prompt_tokens") and hasattr(
                    response, "completion_tokens"
                ):
                    self.total_prompt_tokens += response.prompt_tokens
                    self.total_completion_tokens += response.completion_tokens

                # Extract text from ModelOutput
                content = response.text if hasattr(response, "text") else str(response)

                if content and content.strip():
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

    def get_total_tokens_used(self) -> int:
        """
        Get total tokens consumed so far from actual API usage.
        This is much more accurate than any tokenizer estimation.

        Returns:
            Total tokens used (prompt + completion)
        """
        return self.total_prompt_tokens + self.total_completion_tokens

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
        system_prompt = (
            self.system_prompt or DEEPRESEARCH_SYSTEM_PROMPT
        ) + today_date()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        num_llm_calls_available = self.max_llm_calls
        round = 0
        termination = None
        prediction = ""

        print(f"🔍 Starting DeepResearch for question: {question}")

        while num_llm_calls_available > 0:
            # Check time limit (150 minutes)
            if time.time() - start_time > self.max_time:
                prediction = "No answer found after 2h30mins"
                termination = "No answer found after 2h30mins"
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                }
                return result

            round += 1
            num_llm_calls_available -= 1

            # Get model response
            content = await self.call_server(messages)

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
                    # Special handling for Python code (match original logic)
                    if "python" in tool_call_text.lower():
                        try:
                            # Extract code from the original content (not just tool_call_text)
                            code_raw = (
                                content.split("<tool_call>")[1]
                                .split("</tool_call>")[0]
                                .split("<code>")[1]
                                .split("</code>")[0]
                                .strip()
                            )
                            result = await self.execute_python(code_raw)
                            print(f"🐍 Python execution result: {result[:100]}...")
                        except Exception:
                            result = "[Python Interpreter Error]: Formatting error."
                            print("❌ Python code formatting error")
                    else:
                        # Parse JSON tool call
                        tool_call = json5.loads(tool_call_text)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})
                        result = await self.custom_call_tool(tool_name, tool_args)
                        print(f"🔧 Tool {tool_name} result: {result[:100]}...")

                except Exception as e:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                    print(f"❌ Tool call error: {e}")

                # Add tool response
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})

            # Check if we've exceeded call limit
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                messages[-1]["content"] = (
                    "Sorry, the number of llm calls exceeds the limit."
                )

            # Handle context length limit using actual API consumption
            total_tokens_used = self.get_total_tokens_used()

            if total_tokens_used > self.max_context_tokens:
                print(
                    f"⚠️ Token limit exceeded: {total_tokens_used} > {self.max_context_tokens}"
                )

                # Instead of replacing the last message, add a clear instruction
                final_instruction = {
                    "role": "user",
                    "content": "You have reached the maximum context length. Based on all the information above, please provide your best answer now in the format: <think>your final thinking</think>\n<answer>your answer</answer>",
                }

                # Truncate conversation history to make room for final answer
                # Keep system prompt, original question, and recent context
                if len(messages) > 4:  # system + user + at least 2 exchanges
                    # Keep first 2 messages (system + original question) and last 2 meaningful exchanges
                    truncated_messages = messages[:2]  # system + original question
                    recent_messages = messages[-4:]  # last 4 messages for context
                    truncated_messages.extend(recent_messages)
                    messages = truncated_messages

                messages.append(final_instruction)

                # Note: After truncation, we'll let the next API call handle any remaining limits
                print("Context truncated, proceeding with final answer request")

                content = await self.call_server(messages)
                messages.append({"role": "assistant", "content": content.strip()})

                if "<answer>" in content and "</answer>" in content:
                    prediction = (
                        content.split("<answer>")[1].split("</answer>")[0].strip()
                    )
                    termination = "answer generated due to token limit"
                else:
                    prediction = content.strip()
                    termination = (
                        "response generated due to token limit (no answer format)"
                    )

                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                }
                return result

        # Final validation logic from original Tongyi implementation
        if "<answer>" in messages[-1]["content"]:
            prediction = (
                messages[-1]["content"].split("<answer>")[1].split("</answer>")[0]
            )
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"

        # Final result
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": round,
            "time_taken": time.time() - start_time,
        }

        print("\n🏁 DeepResearch completed:")
        print(f"   Rounds: {round}")
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
        Execute Python code using the PythonInterpreter tool.

        Args:
            code: Python code to execute

        Returns:
            Execution result as string
        """
        if "PythonInterpreter" in self.tools:
            try:
                # Use the PythonInterpreter tool
                tool = self.tools["PythonInterpreter"]
                if hasattr(tool, "call"):
                    if asyncio.iscoroutinefunction(tool.call):
                        result = await tool.call(code=code)
                    else:
                        result = tool.call(code=code)
                    return str(result)
                else:
                    return "PythonInterpreter tool is not callable"
            except Exception as e:
                return f"Python execution error: {e}"
        else:
            return "PythonInterpreter tool not available"

    def reset(self):
        """Reset the agent state (for compatibility with rLLM workflow)."""
        # Reset token counters for each new task
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

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

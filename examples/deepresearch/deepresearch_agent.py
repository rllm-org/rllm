"""
DeepResearch Agent - Adapted from Tongyi DeepResearch for rLLM

This is the core ReAct agent that implements DeepResearch's reasoning and tool-calling logic,
adapted to work with rLLM's OpenAI engine instead of the original server-based approach.

Original: https://github.com/Alibaba-NLP/DeepResearch/blob/main/inference/react_agent.py
"""

import asyncio
import time
from datetime import datetime


# rLLM imports
from rllm.engine.rollout import RolloutEngine

# Constants from original DeepResearch
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = 100

# System prompt adapted from DeepResearch
DEEPRESEARCH_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You MUST use the provided tools to research and verify information before answering. Do NOT answer directly from memory - always use tools to gather current, accurate information.

IMPORTANT: You are REQUIRED to use at least one tool before providing any answer. Even if you think you know the answer, you must verify it using the appropriate tools. Direct answers without tool use are not acceptable.

When you have gathered sufficient information through tool use and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You MUST use one or more of the following tools to research the query:

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

    async def _run(
        self, question: str, answer: str = None, images: list = None, **kwargs
    ) -> dict:
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
            images: List of image data URLs (base64 encoded)

        Returns:
            Dictionary with results including messages, prediction, and termination reason
        """
        start_time = time.time()

        # Setup system prompt with current date
        system_prompt = (
            self.system_prompt or DEEPRESEARCH_SYSTEM_PROMPT
        ) + today_date()

        # Construct initial user message (multimodal if images present)
        if images:
            # Build multimodal message with images
            user_content = [{"type": "text", "text": question}]
            for image_data in images:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": image_data}}
                )
            user_message = {"role": "user", "content": user_content}
        else:
            # Plain text message
            user_message = {"role": "user", "content": question}

        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]

        num_llm_calls_available = self.max_llm_calls
        round = 0
        termination = None
        prediction = ""

        # Truncate question for display
        q_display = str(question).replace("\n", " ").strip()
        if len(q_display) > 200:
            q_display = q_display[:200] + "..."
        print(f"🔍 Starting DeepResearch for question: {q_display}")

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

            # Debug: Print raw model response to see format
            if round == 1:
                print(f"[DEBUG] Raw model response (first 500 chars): {content[:500]}")

            # Print concise round info with truncation
            MAX_PRINT_LENGTH = 200

            # Simple truncation for all prints
            def truncate(text, max_len=MAX_PRINT_LENGTH):
                text = str(text).replace("\n", " ").strip()
                # Special handling for base64 images
                if "data:image" in text or ";base64," in text:
                    # Find the base64 part and truncate it
                    if "base64," in text:
                        parts = text.split("base64,", 1)
                        return parts[0] + "base64,[truncated]"
                    return "[base64 image data]"
                if len(text) > max_len:
                    return text[:max_len] + "..."
                return text

            if "<tool_call>" in content:
                # Extract tool name for display
                if "python" in content.lower() and "<code>" in content:
                    print(f"Round {round}: 🐍 Executing Python code")
                elif '"name":' in content:
                    try:
                        import json5

                        tool_text = content.split("<tool_call>")[1].split(
                            "</tool_call>"
                        )[0]
                        tool_text = tool_text[:1000]  # Limit for parsing
                        tool_data = json5.loads(tool_text)
                        tool_name = tool_data.get("name", "Unknown")
                        if "arguments" in tool_data:
                            args_str = truncate(str(tool_data["arguments"]), 100)
                            print(
                                f"Round {round}: 🔧 Calling {tool_name} with args: {args_str}"
                            )
                        else:
                            print(f"Round {round}: 🔧 Calling {tool_name}")
                    except Exception:
                        print(f"Round {round}: 🔧 Tool call")
                else:
                    print(f"Round {round}: 🔧 Tool call")
            elif "<answer>" in content:
                # Final answer
                answer_preview = content.split("<answer>")[1].split("</answer>")[0]
                print(
                    f"Round {round}: ✅ Final answer: {truncate(answer_preview, 100)}"
                )
            else:
                # Model reasoning
                print(f"Round {round}: 💭 Reasoning: {truncate(content)}")

            # Clean up content if it contains tool_response
            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]

            messages.append({"role": "assistant", "content": content.strip()})

            # Handle tool calls FIRST (before checking for answer)
            # This allows o3 to include both tool_call and answer in same message
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
                        except Exception:
                            result = "[Python Interpreter Error]: Formatting error."
                    else:
                        # Parse JSON tool call
                        tool_call = json5.loads(tool_call_text)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})
                        result = await self.custom_call_tool(tool_name, tool_args)

                except Exception:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'

                # Add tool response
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})

            # Check for final answer AFTER processing tools
            # This allows o3 to execute tools even when it includes answer in same message
            if "<answer>" in content and "</answer>" in content:
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
                termination = "answer"
                break

            # Check if we've exceeded call limit
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                messages[-1]["content"] = (
                    "Sorry, the number of llm calls exceeds the limit."
                )

            # Handle context length limit using actual API consumption
            total_tokens_used = self.get_total_tokens_used()

            if total_tokens_used > self.max_context_tokens:
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
                print(
                    f"Round {round + 1}: ⚠️ Context limit reached, requesting final answer"
                )

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
        # Truncate prediction for display
        pred_display = str(prediction).replace("\n", " ").strip()
        if len(pred_display) > 200:
            pred_display = pred_display[:200] + "..."
        print(f"   Prediction: {pred_display}")

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
        # Reset token counters for each new run
        self.reset()
        return await self._run(question, answer, **kwargs)

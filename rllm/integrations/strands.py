import json
import re
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, TypeVar

from pydantic import BaseModel
from strands import Agent
from strands.models.model import Model
from strands.types.content import ContentBlock, Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout import ModelOutput, RolloutEngine

T = TypeVar("T", bound=BaseModel)


class RLLMModel(Model):
    """Model class that uses rLLM's RolloutEngine for inference."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        """Initialize the RLLMModel.

        Args:
            rollout_engine: The rLLM RolloutEngine instance to use for inference
        """
        self.rollout_engine = rollout_engine
        # Minimal config to satisfy strands Model interface
        model_id = kwargs.pop("model_id", None)
        self._config: dict[str, Any] = {"model_id": model_id, "params": dict(kwargs)}

    def update_config(self, **model_config: Any) -> None:
        if "model_id" in model_config:
            self._config["model_id"] = model_config.pop("model_id")
        params = self._config.get("params") or {}
        params.update(model_config)
        self._config["params"] = params

    def get_config(self) -> dict[str, Any]:
        return {"model_id": self._config.get("model_id"), "params": dict(self._config.get("params") or {})}

    async def structured_output(self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.

        Note: This is a basic implementation that converts the text response to the output model.
        For more advanced structured output, consider using the native OpenAI structured output features.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments.

        Yields:
            Model events with the last being the structured output.
        """
        # Convert Strands messages to chat completion format
        messages = self._convert_messages_to_chat_format(prompt, system_prompt)

        # Add instruction for structured output
        if messages and messages[-1]["role"] == "user":
            original_content = messages[-1]["content"]
            messages[-1]["content"] = f"{original_content}\n\nPlease respond with a JSON object that matches this schema: {output_model.model_json_schema()}"

        # Get response from rollout engine
        response_text = (await self.rollout_engine.get_model_response(messages, **kwargs)).text

        print(f"response_text: {response_text}")
        try:
            # Try to parse the response as JSON and convert to the output model
            import json

            # Extract JSON from response if it's wrapped in other text
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                structured_output = output_model(**parsed_data)
                yield {"output": structured_output}
            else:
                raise ValueError("No valid JSON found in response")

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse structured output: {e}") from e

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream conversation with the model using RolloutEngine.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments.

        Yields:
            Standard StreamEvents that will be processed by the Agent's event loop.
        """
        # Convert Strands messages to chat completion format
        chat_messages = self._convert_messages_to_chat_format(messages, system_prompt)

        # Convert tool specs to OpenAI format for the model
        if tool_specs:
            tools_param = self._convert_tool_specs_to_openai_format(tool_specs)
            kwargs["tools"] = tools_param

            # Force tool use when tools available
            kwargs.setdefault("tool_choice", "required")

        # Disable Qwen3 thinking mode for faster rollouts
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}
        if "chat_template_kwargs" not in kwargs["extra_body"]:
            kwargs["extra_body"]["chat_template_kwargs"] = {}
        kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] = False

        # Filter out Strands-specific kwargs
        strands_only_params = ["system_prompt_content", "tool_execution_handler", "event_handler"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in strands_only_params}

        # Call rollout engine
        model_output: ModelOutput = await self.rollout_engine.get_model_response(chat_messages, **filtered_kwargs)

        # Generate standard StreamEvents
        yield {"messageStart": {"role": "assistant"}}

        # Check if we have tool calls
        if getattr(model_output, "tool_calls", None):
            # Generate content blocks for each tool use
            for tc in model_output.tool_calls:
                tool_call_info = self._extract_tool_call_info(tc)

                # Notify the Agent to record tool call information (for trajectory tracking)
                if hasattr(self, "agent") and hasattr(self.agent, "_record_tool_call_info"):
                    self.agent._record_tool_call_info(tool_call_info)

                # Generate toolUse content block (Strands streaming format)
                yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_call_info["id"], "name": tool_call_info["name"]}}}}

                # Stream the tool input as JSON string
                input_str = json.dumps(tool_call_info["input"])
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": input_str}}}}

                yield {"contentBlockStop": {}}

            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            # Try fallback text parsing for Qwen3 plain text tool calls
            response_text = model_output.text or ""
            fallback_tool_calls = self._parse_tool_calls_from_text(response_text)

            if fallback_tool_calls:
                # Process tool calls found in plain text
                for tc in fallback_tool_calls:
                    tool_call_info = {
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"]
                    }

                    if hasattr(self, "agent") and hasattr(self.agent, "_record_tool_call_info"):
                        self.agent._record_tool_call_info(tool_call_info)

                    yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_call_info["id"], "name": tool_call_info["name"]}}}}
                    input_str = json.dumps(tool_call_info["input"])
                    yield {"contentBlockDelta": {"delta": {"toolUse": {"input": input_str}}}}
                    yield {"contentBlockStop": {}}

                yield {"messageStop": {"stopReason": "tool_use"}}
            else:
                # No tool calls found, return text response
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": response_text}}}
                yield {"contentBlockStop": {}}

                stop_reason = getattr(model_output, "finish_reason", "end_turn")
                yield {"messageStop": {"stopReason": stop_reason}}

    def _convert_tool_specs_to_openai_format(self, tool_specs: list[ToolSpec]) -> list[dict]:
        """Convert tool specs to OpenAI tools format for model use."""
        tools_param = []

        for spec in tool_specs:
            try:
                # Handle decorated function tools (like calculator.calculator)
                if hasattr(spec, "tool_spec"):
                    tool_spec = spec.tool_spec
                    if isinstance(tool_spec, dict):
                        name = tool_spec.get("name")
                        input_schema = tool_spec.get("inputSchema", {})
                        if isinstance(input_schema, dict) and "json" in input_schema:
                            params = input_schema["json"]
                            if name and isinstance(params, dict):
                                tools_param.append(
                                    {
                                        "type": "function",
                                        "function": {"name": name, "parameters": params},
                                    }
                                )
                                continue

                # Handle strands_tools modules with TOOL_SPEC (uppercase)
                if hasattr(spec, "TOOL_SPEC"):
                    tool_spec = spec.TOOL_SPEC
                    if isinstance(tool_spec, dict):
                        name = tool_spec.get("name")
                        input_schema = tool_spec.get("inputSchema", {})
                        if isinstance(input_schema, dict) and "json" in input_schema:
                            params = input_schema["json"]
                            if name and isinstance(params, dict):
                                tools_param.append(
                                    {
                                        "type": "function",
                                        "function": {"name": name, "parameters": params},
                                    }
                                )
                                continue

                # Prefer SDK helper if available
                if hasattr(spec, "to_openai_tool"):
                    maybe = spec.to_openai_tool()
                    if isinstance(maybe, dict):
                        tools_param.append(maybe)
                        continue

                # Dict-shaped specs
                if isinstance(spec, dict):
                    if spec.get("type") == "function" and isinstance(spec.get("function"), dict):
                        tools_param.append(spec)
                        continue

                    # Extract name and parameters from common fields
                    name = spec.get("name")
                    params = None

                    # Try multiple parameter extraction patterns
                    if "parameters" in spec:
                        params = spec["parameters"]
                    elif "input_schema" in spec:
                        params = spec["input_schema"]
                    elif "inputSchema" in spec and isinstance(spec["inputSchema"], dict):
                        # Handle nested inputSchema.json pattern
                        input_schema = spec["inputSchema"]
                        if "json" in input_schema:
                            params = input_schema["json"]

                    if name and isinstance(params, dict):
                        tools_param.append(
                            {
                                "type": "function",
                                "function": {"name": name, "parameters": params},
                            }
                        )
                    continue

                # Object specs: attempt attribute-based extraction
                name = getattr(spec, "name", None)
                params = getattr(spec, "input_schema", None) or getattr(spec, "parameters", None)
                if name and isinstance(params, dict):
                    tools_param.append(
                        {
                            "type": "function",
                            "function": {"name": name, "parameters": params},
                        }
                    )

            except Exception as e:
                print(f"[RLLMModel] Warning: Failed to convert tool spec {spec}: {e}")
                continue
        return tools_param

    def _extract_tool_call_info(self, tool_call) -> dict:
        """Extract tool call information from ModelOutput tool call."""
        try:
            try:
                func = tool_call.get("function", {}) if isinstance(tool_call, dict) else getattr(tool_call, "function", {})
            except Exception:
                func = tool_call
            fname = func.get("name") if isinstance(func, dict) else getattr(func, "name", None)
            fargs_raw = func.get("arguments") if isinstance(func, dict) else getattr(func, "arguments", None)

            # Parse arguments if they're JSON strings
            fargs = {}
            if isinstance(fargs_raw, str):
                import json as _json

                try:
                    fargs = _json.loads(fargs_raw) if fargs_raw.strip() else {}
                except Exception:
                    fargs = {"_raw": fargs_raw}
            elif isinstance(fargs_raw, dict):
                fargs = fargs_raw

            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)

            return {"id": tool_id, "name": fname or "tool", "input": fargs}
        except Exception:
            return {"id": "unknown", "name": "unknown", "input": {}}

    def _parse_tool_calls_from_text(self, text: str) -> list[dict]:
        """
        Fallback parser for Qwen3 plain text tool calls.

        Detects JSON tool patterns like: {"name": "retrieve", "arguments": {...}}
        """
        tool_calls = []

        # Match {"name": ..., "arguments": ...} pattern
        pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}'

        for match in re.finditer(pattern, text):
            try:
                data = json.loads(match.group(0))
                if "name" in data and "arguments" in data:
                    tool_calls.append({
                        "id": f"fallback_call_{len(tool_calls)}",
                        "name": data["name"],
                        "arguments": data["arguments"]
                    })
            except json.JSONDecodeError:
                continue

        # Try alternate pattern with arguments before name
        if not tool_calls:
            pattern_alt = r'\{\s*"arguments"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}'
            for match in re.finditer(pattern_alt, text):
                try:
                    data = json.loads(match.group(0))
                    if "name" in data and "arguments" in data:
                        tool_calls.append({
                            "id": f"fallback_call_{len(tool_calls)}",
                            "name": data["name"],
                            "arguments": data["arguments"]
                        })
                except json.JSONDecodeError:
                    continue

        return tool_calls

    def _convert_messages_to_chat_format(self, messages: Messages, system_prompt: str | None = None) -> list[dict[str, str]]:
        """Convert Strands messages to chat completion format.

        This reuses logic similar to OpenAIModel but outputs the simpler format expected by RolloutEngine.

        Args:
            messages: Strands messages to convert
            system_prompt: Optional system prompt to prepend

        Returns:
            List of chat completion messages
        """
        chat_messages = []

        # Add system prompt if provided
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            role = message["role"]
            content = message["content"]

            # Extract text content from Strands format
            text_content = ""
            for content_block in content:
                if "text" in content_block:
                    text_content += content_block["text"]
                elif "toolUse" in content_block:
                    # For now, represent tool use as text
                    tool_use = content_block["toolUse"]
                    text_content += f"[Tool: {tool_use['name']} with input: {tool_use.get('input', {})}]"
                elif "toolResult" in content_block:
                    # For now, represent tool result as text
                    tool_result = content_block["toolResult"]
                    text_content += f"[Tool Result: {tool_result.get('content', [])}]"
                # TODO: Handle other content types like images, documents if needed

            if text_content.strip():  # Only add if there's actual content
                chat_messages.append({"role": role, "content": text_content})

        return chat_messages


class StrandsAgent(Agent):
    def __init__(self, model: RLLMModel, **kwargs):
        """Initialize StrandsAgent with trajectory tracking.

        Args:
            model: RLLMModel instance to use
            **kwargs: Additional arguments to pass to the base Agent class
        """
        super().__init__(model=model, **kwargs)

        # Set agent reference in the model for tool info passing
        model.agent = self

        self._trajectory = Trajectory()
        self._current_step = None
        self._pending_tool_calls = []  # Track tool calls for current step

    @property
    def trajectory(self) -> Trajectory:
        """Get the current trajectory object."""
        return self._trajectory

    @property
    def chat_completions(self):
        """Convert agent's messages into chat completions format."""
        completions = []
        for message in self.messages:
            # Convert Strands message format to chat completion format
            if isinstance(message.get("content"), list):
                # Handle multi-content messages
                text_content = ""

                for content_block in message["content"]:
                    if isinstance(content_block, dict):
                        if "text" in content_block:
                            text_content += content_block["text"]
                        elif "toolUse" in content_block:
                            # Represent tool use in chat completion format
                            tool_use = content_block["toolUse"]
                            text_content += f"[Using tool: {tool_use.get('name', 'unknown')}]"
                        elif "toolResult" in content_block:
                            # Represent tool result in chat completion format
                            tool_result = content_block["toolResult"]
                            if tool_result.get("content"):
                                for result_item in tool_result["content"]:
                                    if isinstance(result_item, dict) and "text" in result_item:
                                        text_content += result_item["text"]

                # Only add message if it has meaningful content
                if text_content.strip():
                    completions.append({"role": message["role"], "content": text_content})
            else:
                # Handle simple string content
                content = str(message.get("content", ""))
                if content.strip():  # Only add non-empty content
                    completions.append({"role": message["role"], "content": content})
        return completions

    def reset(self):
        """Reset the agent for workflow pooling support."""
        # Clear trajectory and internal state
        self._trajectory = Trajectory()
        self._current_step = None
        self._pending_tool_calls = []

        # Clear message history to prevent contamination
        if hasattr(self, "messages"):
            self.messages.clear()
        if hasattr(self, "_messages"):
            self._messages.clear()

    def reset_trajectory(self, task: Any = None):
        """Reset the trajectory for a new episode."""
        self._trajectory = Trajectory(task=task)
        self._current_step = None
        self._pending_tool_calls = []

    def _record_tool_call_info(self, tool_call_info: dict):
        """Record tool call information for trajectory tracking.

        This method is called by RLLMModel to pass tool call information
        without executing the tools (execution is handled by Strands event loop).
        """
        self._pending_tool_calls.append(tool_call_info)

    def _start_new_step(self, observation: Any = None):
        """Start a new step in the trajectory."""
        self._current_step = Step(chat_completions=self.chat_completions.copy(), observation=observation)
        # print(f"\n🔄 Starting new step...")
        # print(f"   📥 Observation: {str(observation)[:100]}{'...' if len(str(observation)) > 100 else ''}")

    def _finish_current_step(self, model_response: str = "", action: Any = None, reward: float = 0.0, done: bool = False):
        """Finish the current step and add it to the trajectory."""
        if self._current_step is not None:
            self._current_step.model_response = model_response
            self._current_step.reward = reward
            self._current_step.done = done
            self._current_step.chat_completions = self.chat_completions.copy()

            # 🔧 Check if there are tool calls, if so set as tool_calls type
            if self._pending_tool_calls:
                # Set action as tool_calls format
                self._current_step.action = {
                    "type": "tool_calls",
                    "tool_calls": self._pending_tool_calls.copy(),
                    "final_response": action,  # Preserve original action info
                }
                # Clear tool calls cache
                self._pending_tool_calls.clear()
                # print(f"✅ Step {len(self._trajectory.steps) + 1}: Tool calls completed")
            else:
                self._current_step.action = action
                # print(f"✅ Step {len(self._trajectory.steps) + 1}: Completed")

            self._trajectory.steps.append(self._current_step)
            self._trajectory.reward += reward

            self._current_step = None

    def _finish_current_step_from_result(self, result: Any):
        """Finish the current step by extracting info from AgentResult."""
        model_response = ""
        action = None

        if hasattr(result, "message") and result.message:
            # Extract text content from the final message
            if hasattr(result.message, "content") and isinstance(result.message.content, list):
                for content_block in result.message.content:
                    if isinstance(content_block, dict) and "text" in content_block:
                        model_response += content_block["text"]
            elif hasattr(result.message, "content"):
                model_response = str(result.message.content)

            action = result.message if hasattr(result, "message") else result

        # Determine if this step is done
        done = hasattr(result, "stop_reason") and result.stop_reason in ["end_turn", "stop_sequence"]

        self._finish_current_step(model_response=model_response, action=action, done=done)

    def __call__(self, prompt: str | list[ContentBlock], **kwargs) -> Any:
        """Enhanced call method that tracks trajectory."""
        # Start a new step with the user prompt as observation
        self._start_new_step(observation=prompt)

        # Let the original Strands Agent handle everything (including tool execution)
        result = super().__call__(prompt, **kwargs)

        # Finish the current step based on the result
        self._finish_current_step_from_result(result)

        return result

    async def invoke_async(self, prompt: str | list[ContentBlock], **kwargs) -> Any:
        """Async invoke method with trajectory tracking."""
        # Start a new step with the user prompt as observation
        self._start_new_step(observation=prompt)

        # Let the original Strands Agent handle everything
        result = None
        async for event in super().stream_async(prompt, **kwargs):
            if "result" in event:
                result = event["result"]
                break

        # Finish the current step based on the result
        if result:
            self._finish_current_step_from_result(result)

        return result

    def get_current_state(self) -> Step | None:
        """Get the current step state."""
        if self._trajectory.steps:
            return self._trajectory.steps[-1]
        return self._current_step

    def update_step_reward(self, reward: float):
        """Update the reward for the current or last step."""
        if self._current_step is not None:
            self._current_step.reward = reward
        elif self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            # Update trajectory total reward
            self._trajectory.reward = sum(step.reward for step in self._trajectory.steps)

    def update_step_info(self, info: dict):
        """Update the info for the current or last step."""
        if self._current_step is not None:
            self._current_step.info.update(info)
        elif self._trajectory.steps:
            self._trajectory.steps[-1].info.update(info)

    def get_tool_call_summary(self) -> list[dict]:
        """Get a summary of all tool calls made during the trajectory."""
        # Tool calls will now be tracked by the standard Strands event loop
        # This method can be enhanced to extract tool information from messages
        return []

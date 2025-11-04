import json
import logging
import re

from r2egym.agenthub.action import Action as SWEAction

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_SYSTEM_PROMPT_FN_CALL, SWE_USER_PROMPT, SWE_USER_PROMPT_FN_CALL, SWEAGENT_SYSTEM_PROMPT, SWEAGENT_USER_PROMPT
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.parser.chat_template_parser import ChatTemplateParser

TOKEN_WARNING_THRESHOLD = 28000


# Mapping of scaffold types to their tool schema definitions
# These are imported directly from R2E-Gym


def get_tools_for_scaffold(scaffold: str = "sweagile"):
    """
    Get the OpenAI function calling tools schema for a given scaffold.

    Args:
        scaffold: The scaffold type ("r2egym", "sweagent", or "sweagile")

    Returns:
        List of tool schemas in OpenAI function calling format
    """
    from r2egym.agenthub.tools import (
        execute_bash_tool,
        file_editor,
        finish_tool,
        r2egym_bash_execute_tool,
        search_tool,
        str_replace_editor_tool,
        submit_tool,
    )

    if scaffold == "r2egym":
        return [
            file_editor,
            search_tool,
            r2egym_bash_execute_tool,
            finish_tool,
        ]
    elif scaffold == "sweagent":
        return [
            str_replace_editor_tool,
            execute_bash_tool,
            submit_tool,
        ]
    raise ValueError(f"Invalid scaffold: {scaffold}")


def parse_oai_response(response: ModelOutput) -> tuple[str, SWEAction]:
    if isinstance(response, ModelOutput):
        content = response.content
        if len(response.tool_calls) == 0:
            logger.warning(f"No tool calls found in the ModelOutput. Last 500 chars of the response: ...{response.text[-500:]} Returning empty action.")
            return content, SWEAction(function_name="", parameters={})
        if not isinstance(response.tool_calls[0].arguments, dict):
            logger.warning(f"Arguments is not a dict, got {type(response.tool_calls[0].arguments)}: {response.tool_calls[0].arguments}")
            response.tool_calls[0].arguments = {}
        action = SWEAction(function_name=response.tool_calls[0].name, parameters=response.tool_calls[0].arguments)
        return content, action
    else:
        raise ValueError(f"Invalid response type: {type(response)}. Expected ChatCompletion or ModelOutput object.")


def parse_xml_response(response_text: str) -> tuple[str, SWEAction]:
    """
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the entire first <function=...></function> block
    Returns (thought, action).
    """
    # Regex to match (non-greedily) from `<function=` up to the first `</function>`
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action = match.group(1)  # The entire <function=...></function> block
        thought = response_text[: match.start()]  # Everything before the block
    else:
        # If no match, treat entire text as "thought"
        thought = response_text
        action = ""

    # Strip leading/trailing whitespace
    thought = thought.strip()
    action = action.strip()

    # convert action to Action object
    action = SWEAction.from_string(action)

    return thought, action


logger = logging.getLogger(__name__)


class SWEAgent(BaseAgent):
    def __init__(self, use_tool_calling: bool = True, scaffold: str = "r2egym", chat_template_parser: ChatTemplateParser = None, accumulate_reasoning: bool = False, **kwargs):
        self.use_tool_calling = use_tool_calling
        self.scaffold = scaffold
        self.accumulate_reasoning = accumulate_reasoning
        assert scaffold in ["r2egym", "sweagent"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        self.system_prompt = SWE_SYSTEM_PROMPT_FN_CALL if use_tool_calling else SWE_SYSTEM_PROMPT
        if scaffold == "sweagent":
            self.system_prompt = SWEAGENT_SYSTEM_PROMPT
        self.user_prompt_template = SWE_USER_PROMPT_FN_CALL if use_tool_calling else SWE_USER_PROMPT
        if scaffold == "sweagent":
            self.user_prompt_template = SWEAGENT_USER_PROMPT

        self.chat_template_parser = chat_template_parser
        if self.use_tool_calling:
            tools_schema = json.dumps(get_tools_for_scaffold(scaffold))
            self.tools_prompt = self.chat_template_parser.tool_parser.get_tool_prompt(tools_schema)

        self._trajectory = Trajectory()
        self.reset()

    def process_model_response(self, response: str) -> tuple[str, str]:
        """
        Processes the model's response to extract thought and action components.

        Parses the response using either function calling or XML parsing based on agent configuration.

        Args:
            response (str): The raw text response from the model.

        Returns:
            Tuple[str, str]: A tuple containing:
                - The action string in XML format
                - The processed response (may be reformatted if self.format_model_response is True)
        """
        if self.use_tool_calling:
            thought, action = parse_oai_response(response)
        else:
            thought, action = parse_xml_response(response)

        action_str = action.to_xml_string()
        if self.format_model_response:
            response = f"{thought}\n\n{action_str}"
        return action.to_xml_string(), {
            "thought": thought,
        }

    def update_from_env(self, observation, reward, done, info):
        # If the first step in environment, we need to update the state from the environment
        if self._trajectory.steps:
            observation = str(observation)
        else:
            observation = str(observation)
            observation = self.user_prompt_template.format(problem_statement=observation)

        max_steps = info.get("max_steps", None)
        if max_steps:
            remaining_steps = max_steps - self.step - 1
            if remaining_steps > 0:
                observation += f"\nSteps Remaining: {remaining_steps}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."

        cur_tokens = info.get("cur_tokens", None)
        if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
            observation += "\nYou are running out of tokens. Please submit your answer NOW."

        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        self.messages.append({"role": "user", "content": observation})
        self.cur_step = Step(observation=observation)

    def update_from_model(self, model_output: ModelOutput, **kwargs) -> Action:
        """
        Updates the agent's internal state after an environment step.

        This function is called during environment interaction to incorporate the latest action's
        outcome into the agent's learning process.

        Args:
            model_output ModelOutput: The response from the model.
        Returns:
            Action: The action to take.
        """
        response = model_output.text
        self._trajectory.steps.append(self.cur_step)

        if self.use_tool_calling:
            content, action = parse_oai_response(model_output)
        else:
            content, action = parse_xml_response(response)
        if len(model_output.tool_calls) > 0:
            action_str = self.chat_template_parser.tool_parser.tool_call_to_str(model_output.tool_calls[0])
        else:
            action_str = ""
        logger.debug(f"update_from_model: action_str: {action_str}")
        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."

        # Update Trajectory
        cur_step = self._trajectory.steps[-1]
        cur_step.reasoning = model_output.reasoning
        cur_step.content = model_output.content
        cur_step.text = model_output.text
        cur_step.action = action
        cur_step.model_response = response

        # Update Chat Completions
        self.messages.append({"role": "assistant", "content": response})
        self.step += 1
        return Action(action=cur_step.action)

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]

    def reset(self):
        self._trajectory = Trajectory()
        if self.use_tool_calling:
            prompt = self.system_prompt + self.tools_prompt
        else:
            prompt = self.system_prompt
        self.messages = [{"role": "system", "content": prompt}]
        self.step = 0

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages

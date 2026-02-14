import json
import logging
import re

from rllm.environments.swe.action import Action as SWEAction

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_SYSTEM_PROMPT_FN_CALL, SWE_USER_PROMPT, SWE_USER_PROMPT_FN_CALL, SWEAGENT_SYSTEM_PROMPT, SWEAGENT_USER_PROMPT

TOKEN_WARNING_THRESHOLD = 28000


def parse_oai_response(response):
    thought = response.choices[0].message.content
    if not thought:
        thought = ""
    try:
        function_name = response.choices[0].message.tool_calls[0].function.name
        parameters = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        action = SWEAction(function_name, parameters)
    except Exception:
        action = SWEAction(function_name="", parameters={})
    return thought, action


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
    def __init__(self, use_fn_calling: bool = False, format_model_response: bool = False, scaffold: str = "r2egym"):
        self.use_fn_calling = use_fn_calling
        self.format_model_response = format_model_response
        self.scaffold = scaffold
        assert scaffold in ["r2egym", "sweagent"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        self.system_prompt = SWE_SYSTEM_PROMPT_FN_CALL if use_fn_calling else SWE_SYSTEM_PROMPT
        if scaffold == "sweagent":
            self.system_prompt = SWEAGENT_SYSTEM_PROMPT
        self.user_prompt_template = SWE_USER_PROMPT_FN_CALL if use_fn_calling else SWE_USER_PROMPT
        if scaffold == "sweagent":
            self.user_prompt_template = SWEAGENT_USER_PROMPT

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
        if self.use_fn_calling:
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

    def update_from_model(self, response: str, **kwargs):
        """
        Updates the agent's internal state after an environment step.

        This function is called during environment interaction to incorporate the latest action's
        outcome into the agent's learning process.

        Args:
            response (str): The response from the model.

        Returns:
            None
        """
        self._trajectory.steps.append(self.cur_step)
        if self.use_fn_calling:
            thought, action = parse_oai_response(response)
        else:
            thought, action = parse_xml_response(response)
        action_str = action.to_xml_string()
        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."

        # Update Trajectory
        cur_step = self._trajectory.steps[-1]
        cur_step.thought = thought
        cur_step.action = action_str
        cur_step.model_response = response

        # Update Chat Completions
        if self.format_model_response:
            self.messages.append({"role": "assistant", "content": f"{thought}\n\n{action_str}"})
        else:
            self.messages.append({"role": "assistant", "content": response})
        self.step += 1
        return Action(action=cur_step.action)

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.step = 0

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages

"""
This workflow bridges eval-protocol's MCPGymRolloutProcessor with rllm-fw's Workflow pattern
for the FrozenLake environment.
"""

import asyncio
import signal
import traceback
from pathlib import Path

import eval_protocol
from eval_protocol.benchmarks.test_frozen_lake import test_frozen_lake_evaluation
from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.default_mcp_gym_rollout_processor import (
    MCPGymRolloutProcessor,
)
from eval_protocol.pytest.types import RolloutProcessorConfig

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.workflows.workflow import Workflow

_orig_signal = signal.signal

def tracing_signal(sig, handler):
    print(f"[DEBUG] signal.signal({sig}, {handler}) called")
    traceback.print_stack()
    return _orig_signal(sig, handler)

signal.signal = tracing_signal

class FrozenLakeWorkflow(Workflow):
    """
    Workflow that executes frozen lake tasks using MCPGymRolloutProcessor.

    Task format expected:
    {
        "id": "frozen_lake_task_0",
        "system_prompt": "...",
        "environment_context": {...},
        "user_prompt_template": "{observation}"
    }
    """

    # Class variables (shared across all workflow instances)
    _shared_server_started = False
    _server_lock = asyncio.Lock()
    _shared_rollout_processor = MCPGymRolloutProcessor()

    def __init__(
        self,
        rollout_engine: OpenAIEngine,
        lite_llm_prefix: str = "fireworks_ai/",
        steps: int = 30,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(rollout_engine, **kwargs)
        self.steps = steps
        self.model = lite_llm_prefix + rollout_engine.model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use shared rollout processor across all instances
        self.rollout_processor = FrozenLakeWorkflow._shared_rollout_processor

        eval_protocol_path = Path(eval_protocol.__file__).parent
        server_script_path = eval_protocol_path / "mcp_servers" / "frozen_lake" / "server.py"

        self.config = RolloutProcessorConfig(
            completion_params={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            mcp_config_path="",
            server_script_path=str(server_script_path),
            steps=self.steps,
            semaphore=asyncio.Semaphore(1),
            kwargs={"start_server": False},
        )

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute the frozen lake workflow.

        Args:
            task: Dict containing frozen lake task data
            uid: Unique identifier for this episode
            **kwargs: Additional arguments

        Returns:
            Episode with trajectory and computed rewards
        """
        # Thread-safe server startup (double-checked locking pattern)
        if not FrozenLakeWorkflow._shared_server_started:
            # Only acquire lock if server not started yet
            async with FrozenLakeWorkflow._server_lock:
                # Check again inside lock (another workflow might have started it)
                if not FrozenLakeWorkflow._shared_server_started:
                    # First workflow to reach here starts the server
                    self.config.kwargs["start_server"] = True
                    FrozenLakeWorkflow._shared_server_started = True
                else:
                    self.config.kwargs["start_server"] = False
        else:
            self.config.kwargs["start_server"] = False

        self.reset(task=task, uid=uid)

        try:
            eval_row = self._task_to_evaluation_row(task)

            tasks = self.rollout_processor([eval_row], self.config)

            if not tasks:
                raise ValueError("MCPGymRolloutProcessor returned no tasks")

            result_row: EvaluationRow = await tasks[0]

            episode = await self._evaluate_and_create_episode(result_row, task, uid)

            return episode

        except Exception as e:
            # Gracefully handle failures - return a failed Episode instead of crashing
            print(f"⚠️  Task {uid} failed: {e}")

            failed_episode = Episode(
                id=uid,
                task=task,
                is_correct=False,
                trajectories=[],
                metrics={"frozen_lake_reward": 0.0, "error": str(e)},
            )
            return failed_episode

    def _task_to_evaluation_row(self, task: dict) -> EvaluationRow:
        """Convert rllm task dict to eval protocol EvaluationRow."""
        return EvaluationRow(
            messages=[Message(role="system", content=task["system_prompt"])],
            input_metadata=InputMetadata(
                row_id=task["id"],
                dataset_info={
                    "environment_context": task["environment_context"],
                    "user_prompt_template": task["user_prompt_template"],
                },
            ),
        )

    async def _evaluate_and_create_episode(
        self,
        row: EvaluationRow,
        task: dict,
        uid: str,
    ) -> Episode:
        """
        Evaluate the rollout and convert to rllm Episode.
        """
        # Call the evaluation function
        evaluated_row: EvaluationRow = await test_frozen_lake_evaluation(row)

        # Extract reward and metrics from evaluation_result
        if evaluated_row.evaluation_result is None:
            raise ValueError("Evaluation function did not return a result")

        reward = evaluated_row.evaluation_result.score
        reward_info = evaluated_row.evaluation_result.metrics or {}

        def msg_to_dict(msg: Message) -> dict:
            """Convert eval_protocol Message to chat completion dict."""
            d = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            if msg.name:
                d["name"] = msg.name
            return d

        trajectory = Trajectory()
        all_messages = []

        for msg in row.messages:
            msg_dict = msg_to_dict(msg)
            all_messages.append(msg_dict)

            # Create Step with only observation and chat_completions for user or tool message
            if msg.role in ["user", "tool"]:
                new_step = Step(observation=str(msg.content or ""), chat_completions=all_messages.copy())
                trajectory.steps.append(new_step)

            # Create new Step with action/response for assistant message
            elif msg.role == "assistant":
                # Extract action: tool calls if present, otherwise message content
                action_data = msg_dict.get("tool_calls") if msg.tool_calls else str(msg.content or "")

                new_step = Step(
                    model_response=str(msg.content) if msg.content else "",
                    action=action_data,
                    chat_completions=all_messages.copy(),
                )
                trajectory.steps.append(new_step)

        # Assign final reward to the last step (sparse reward)
        if trajectory.steps:
            trajectory.steps[-1].reward = reward
            trajectory.steps[-1].info = reward_info

        trajectory.reward = reward
        trajectory.task = task

        # Create episode
        episode = Episode(
            id=uid,
            task=task,
            is_correct=(reward == 1.0),
            trajectories=[trajectory],
            metrics={"frozen_lake_reward": reward, **reward_info},
        )

        return episode

    def cleanup(self):
        """Cleanup MCP server resources."""
        if self.rollout_processor:
            self.rollout_processor.cleanup()
            self.rollout_processor = None
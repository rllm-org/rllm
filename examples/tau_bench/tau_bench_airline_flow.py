"""
Tau-Bench Airline Workflow for rllm-fw

This workflow bridges eval-protocol's MCPGymRolloutProcessor with rllm-fw's Workflow pattern.
It executes tau-bench airline tasks and returns Episodes with computed rewards.
"""

import asyncio
from pathlib import Path

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.workflows.workflow import Workflow

import eval_protocol
from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.benchmarks.test_tau_bench_airline import test_tau_bench_airline_evaluation

class TauBenchAirlineWorkflow(Workflow):
    """
    Workflow that executes tau-bench airline tasks using MCPGymRolloutProcessor.
    
    Task format expected:
    {
        "id": "airline_task_0",
        "environment_context": {...},
        "user_simulation": {...},
        "evaluation_criteria": {...},
        "user_prompt_template": {...},
    }
    """
    
    _shared_server_started = False
    _server_lock = asyncio.Lock()
    _shared_rollout_processor = MCPGymRolloutProcessor()
    
    def __init__(
        self,
        steps: int = 30,
        model: str = "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        temperature: float = 0.8,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.steps = steps
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Use shared rollout processor across all instances
        self.rollout_processor = TauBenchAirlineWorkflow._shared_rollout_processor

        eval_protocol_path = Path(eval_protocol.__file__).parent
        server_script_path = eval_protocol_path / "mcp_servers" / "tau2" / "server.py"

        self.config = RolloutProcessorConfig(
            completion_params={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            mcp_config_path="",  # Not used for tau-bench (uses server_script_path instead)
            server_script_path=str(server_script_path),
            steps=self.steps,
            semaphore=asyncio.Semaphore(1),  # This semaphore is redundant because AgentWorkflowEngine already handles concurrency
            kwargs={"domain": "airline", "start_server": False},  # Default False, will be set to True on first run only
        )
        
    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute the tau-bench airline workflow.
        
        Args:
            task: Dict containing tau-bench airline task data
            uid: Unique identifier for this episode
            **kwargs: Additional arguments
            
        Returns:
            Episode with trajectory and computed rewards
        """
        # Thread-safe server startup (double-checked locking pattern)
        if not TauBenchAirlineWorkflow._shared_server_started:
            # Only acquire lock if server not started yet
            async with TauBenchAirlineWorkflow._server_lock:
                # Check again inside lock (another workflow might have started it)
                if not TauBenchAirlineWorkflow._shared_server_started:
                    # First workflow to reach here starts the server
                    self.config.kwargs["start_server"] = True
                    TauBenchAirlineWorkflow._shared_server_started = True
                    # Note: kwargs["start_server"] stays False for subsequent workflows since it's already False by default
        
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
                metrics={"tau_bench_reward": 0.0, "error": str(e)},
            )
            return failed_episode
    
    def _task_to_evaluation_row(self, task: dict) -> EvaluationRow:
        """Convert rllm task dict to eval protocol EvaluationRow."""
        # Load system prompt from eval_protocol package
        domain = task["environment_context"]["domain"]
        eval_protocol_path = Path(eval_protocol.__file__).parent
        prompt_file = eval_protocol_path / "mcp_servers" / "tau2" / "tests" / "system_prompts" / f"{domain}_agent_system_prompt.md"
        
        with open(prompt_file, "r") as f:
            system_prompt = f.read().strip()
        
        return EvaluationRow(
            messages=[Message(role="system", content=system_prompt)],
            input_metadata=InputMetadata(
                row_id=task["id"],
                dataset_info={
                    "environment_context": task["environment_context"],
                    "user_simulation": task["user_simulation"],
                    "evaluation_criteria": task["evaluation_criteria"],
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
        Evaluate the rollout using tau2 evaluators and convert to rllm Episode.
        
        This combines reward computation and Episode creation into one step.
        """
        # Call the evaluation function which computes tau2 rewards
        evaluated_row: EvaluationRow = await test_tau_bench_airline_evaluation(row)
        
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
        
        # Assign final reward to the last step only because tau2-bench is sparse reward
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
            trajectories=[("tau_bench_agent", trajectory)],
            metrics={"tau_bench_reward": reward, **reward_info},
        )
        
        return episode
    
    def cleanup(self):
        """Cleanup MCP server resources."""
        if self.rollout_processor:
            self.rollout_processor.cleanup()
            self.rollout_processor = None


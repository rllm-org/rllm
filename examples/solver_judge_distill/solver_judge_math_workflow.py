import asyncio
import re

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.math_reward import rllm_reward_fn_math
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow


def _get_full_response(output: ModelOutput) -> str:
    """Get the full response from ModelOutput, handling thinking models."""
    content = output.content or ""
    reasoning = output.reasoning or ""

    # If content is empty but reasoning exists, the model might have put everything in reasoning
    if not content and reasoning:
        return reasoning

    # If both exist, content is the main response
    if content:
        return content

    return ""


class MathSolver:
    """Generates solutions to math problems."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def generate_solution(self, problem: str) -> tuple[Trajectory, bool]:
        """Generate a single solution to a math problem.
        
        Returns:
            Tuple of (trajectory, is_length_exceeded)
        """
        messages = [
            {
                "role": "user",
                "content": f"{problem}\n\nThink step by step and output your final answer within \\boxed{{}}.",
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)

        # Check for length exceeded
        is_length_exceeded = output.finish_reason == "length"
        
        # Get full response for reward evaluation
        full_response = _get_full_response(output)

        trajectory = Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=full_response,  # Store full response for reward evaluation
                    model_output=output,
                )
            ],
        )
        return trajectory, is_length_exceeded

    async def generate_solutions(self, problem: str, n_solutions: int = 2) -> list[tuple[Trajectory, bool]]:
        """Generate multiple solutions in parallel.
        
        Returns:
            List of tuples (trajectory, is_length_exceeded)
        """
        tasks = [asyncio.create_task(self.generate_solution(problem)) for _ in range(n_solutions)]
        return await asyncio.gather(*tasks)

    def _extract_boxed_answer(self, response: str) -> str:
        """Extract the boxed answer from solver response."""
        if not response:
            return ""
        boxed_match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response)
        if boxed_match:
            return boxed_match.group(1).strip()
        return response  # Return full response if no boxed answer found


class MathJudge:
    """Evaluates and selects the best solution from multiple candidates."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def judge_solutions(self, problem: str, solutions: list[str]) -> Trajectory:
        """Judge multiple solutions and select the best one."""
        messages = [{"role": "user", "content": self._create_judge_prompt(problem, solutions)}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)

        full_response = _get_full_response(output)
        selected_solution = self._parse_judge_response(full_response, solutions)

        return Trajectory(
            name="judge",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=selected_solution,
                    model_output=output,
                )
            ],
        )

    def _parse_judge_response(self, response: str, solutions: list[str]) -> str:
        """Extract selected solution index from judge response."""
        if not response:
            return solutions[0] if solutions else ""

        boxed_match = re.search(r"\\boxed\{(\d+)\}", response)
        if boxed_match:
            try:
                solution_index = int(boxed_match.group(1))
                if 1 <= solution_index <= len(solutions):
                    return solutions[solution_index - 1]
            except (ValueError, IndexError):
                pass

        # Fallback: look for "Solution X" pattern
        solution_match = re.search(r"[Ss]olution\s*(\d+)", response)
        if solution_match:
            try:
                solution_index = int(solution_match.group(1))
                if 1 <= solution_index <= len(solutions):
                    return solutions[solution_index - 1]
            except (ValueError, IndexError):
                pass

        return solutions[0] if solutions else ""

    def _create_judge_prompt(self, problem: str, solutions: list[str]) -> str:
        """Create a prompt for the judge to evaluate solutions."""
        prompt = f"""You are an expert math verifier. Given a math problem and multiple solution attempts, select the correct solution.

Problem:
{problem}

Solutions to evaluate:
"""
        for i, solution in enumerate(solutions, 1):
            boxed_match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
            display_solution = boxed_match.group(1) if boxed_match else solution[:500]
            prompt += f"\nSolution {i}:\n{display_solution}\n"

        prompt += """
Evaluate each solution for correctness.
Output the index of your selected solution within \\boxed{{}}, e.g., \\boxed{{1}} for the first solution.
If multiple solutions are correct, select the first correct one.
If no solution is correct, select the one closest to being correct."""
        return prompt


def math_reward_fn(task: dict, action: str):
    """Wrapper for math reward function compatible with workflow interface."""
    ground_truth = task.get("ground_truth", "")
    data_source = task.get("data_source", "math")
    return rllm_reward_fn_math(data_source, action, ground_truth)


class SolverJudgeMathWorkflow(Workflow):
    """Solver-Judge workflow for math problems.

    This workflow:
    1. Has the solver generate multiple solution attempts
    2. Assigns rewards to each solver trajectory
    3. Has the judge select the best solution
    4. Returns an episode with all trajectories for training
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        n_solutions: int = 2,
        reward_function: RewardFunction = None,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.n_solutions = n_solutions
        self.reward_function = reward_function or math_reward_fn
        self.solver = MathSolver(rollout_engine)
        self.judge = MathJudge(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the solver-judge workflow."""
        from rllm.workflows.workflow import TerminationReason
        
        self.reset(task, uid)
        problem = task.get("question") or task.get("problem", "")

        # Step 1: Solver generates multiple solutions in parallel
        solver_results = await self.solver.generate_solutions(problem, self.n_solutions)

        # Collect full solutions and assign rewards
        # Handle length-exceeded responses with reward=0
        solutions = []
        solver_trajectories = []
        any_length_exceeded = False
        
        for traj, is_length_exceeded in solver_results:
            solver_trajectories.append(traj)
            solution = traj.steps[0].action 
            solutions.append(solution)
            
            if is_length_exceeded:
                # Length exceeded: assign reward=0
                traj.steps[0].reward = 0.0
                traj.reward = 0.0
                any_length_exceeded = True
            else:
                reward_result = self.reward_function(task, solution)
                traj.steps[0].reward = reward_result.reward
                traj.reward = reward_result.reward 

        # Step 2: Judge selects the best solution
        judge_trajectory = await self.judge.judge_solutions(problem, solutions)
        selected_solution = judge_trajectory.steps[0].action

        # Check if judge response exceeded length
        judge_length_exceeded = judge_trajectory.steps[0].model_output.finish_reason == "length"

        # Evaluate the selected solution
        if judge_length_exceeded:
            judge_trajectory.steps[0].reward = 0.0
            judge_trajectory.reward = 0.0
            is_correct = False
            any_length_exceeded = True
        else:
            reward_result = self.reward_function(task, selected_solution)
            judge_trajectory.steps[0].reward = reward_result.reward
            judge_trajectory.reward = reward_result.reward
            is_correct = reward_result.is_correct

        # Compute metrics
        solver_acc = sum(traj.steps[0].reward for traj in solver_trajectories) / len(solver_trajectories)
        judge_acc = int(is_correct)

        # Return episode with all trajectories
        episode = Episode(
            id=uid,
            task=task,
            trajectories=[*solver_trajectories, judge_trajectory],
            is_correct=is_correct,
            metrics={"solver_acc": solver_acc, "judge_acc": judge_acc},
        )
        
        # Set termination reason if any response exceeded length
        if any_length_exceeded:
            episode.termination_reason = TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
        
        return episode

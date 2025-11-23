"""
Solver-Judge workflow using @step and @trajectory decorators.

This is a cleaner version of simple_solver_judge_flow.py that demonstrates
how the new decorators streamline the workflow implementation.

Key improvements:
- @step decorator eliminates manual session management
- Automatic StepView creation with result field
- Cleaner code with less boilerplate
- Same functionality, better ergonomics
"""

import asyncio
import re

from rllm.engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.sdk import TrajectoryView, get_chat_client_async, trajectory
from rllm.workflows.workflow import Workflow


class Solver:
    def __init__(self, **kwargs):
        self.client = get_chat_client_async(base_url="http://localhost:4000/v1", api_key="EMPTY")
        self.model = "Qwen/Qwen3-4B-Instruct-2507"

    @trajectory(name="solver")
    async def generate_solution(self, problem: str):
        """
        Generate a solution using @step decorator.

        The decorator:
        - Creates a session internally
        - Tracks LLM calls automatically
        - Returns StepView with result field set to the return value
        """
        messages = [{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
            max_tokens=1000,
        )
        response_text = response.choices[0].message.content

        # Parse and return the answer (matches original behavior)
        return self._parse_solver_response(response_text)

    async def generate_solutions(self, problem: str, n_solutions: int = 2):
        """Generate multiple solutions in parallel."""
        tasks = [asyncio.create_task(self.generate_solution(problem)) for _ in range(n_solutions)]
        # Returns list of StepView objects
        return await asyncio.gather(*tasks)

    def _parse_solver_response(self, response: str) -> str:
        """Extract answer from response."""
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return f"<answer>{answer_match.group(1).strip()}</answer>"
        else:
            return "No solution found"


class Judge:
    def __init__(self, **kwargs):
        self.client = get_chat_client_async(base_url="http://localhost:4000/v1", api_key="EMPTY")
        self.model = "Qwen/Qwen3-4B-Instruct-2507"

    @trajectory(name="judge")
    async def judge_solutions(self, problem: str, solutions: list[str]):
        """
        Judge solutions using @step decorator.

        Returns StepView with the selected solution in the result field.
        """
        messages = [{"role": "user", "content": self._create_judge_prompt(problem, solutions)}]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
            max_tokens=1000,
        )
        response_text = response.choices[0].message.content

        # Parse and return the selected solution (matches original behavior)
        return self._parse_judge_response(response_text, solutions)

    def _parse_judge_response(self, response: str, solutions: list[str]) -> str:
        """Parse judge response to get selected solution."""
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            try:
                solution_index = int(answer_text)
                return solutions[solution_index - 1]
            except (ValueError, IndexError):
                return ""
        return ""

    def _create_judge_prompt(self, problem: str, solutions: list[str]) -> str:
        """Create a prompt for the judge to evaluate solutions."""
        prompt = f"""You are an expert verifier. Given a countdown problem and multiple solution attempts, select a correct solution.
Problem:
{problem}
Solutions to evaluate:
"""
        for i, solution in enumerate(solutions, 1):
            prompt += f"\nSolution {i}:\n{solution}\n"

        prompt += """
A correct solution must satisfy the following criteria:
1. The solution uses only the given numbers.
2. Each number is used exactly once.
3. Only basic arithmetic operations (+, -, *, /) are used.
4. The calculation results in the target number.
5. The final answer is clearly marked within <answer>...</answer> tags.
Output the index of your selected solution within <answer>...</answer> tags, e.g., <answer>1</answer> for the first solution, <answer>2</answer> for the second solution, etc. If multiple solutions are correct, output the index of the first correct solution."""
        return prompt


class SolverJudgeWorkflowDecorated(Workflow):
    """
    Decorated version using @trajectory.
    """

    def __init__(self, rollout_engine: RolloutEngine, n_solutions: int = 2, reward_function: RewardFunction = None, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.n_solutions = n_solutions
        self.reward_function = reward_function
        self.solver = Solver()
        self.judge = Judge()

    async def run(self, task: dict, uid: str, **kwargs) -> list[TrajectoryView]:
        """
        Run workflow - manually construct trajectories for now.

        Note: We could use @trajectory decorator on helper methods,
        but the workflow engine expects a specific return format,
        so we construct TrajectoryViews manually here.
        """
        self.reset(task, uid)
        problem = task["question"]

        # Generate solutions
        solver_trajs = await self.solver.generate_solutions(problem, self.n_solutions)

        # Process solutions
        solutions = []
        for traj in solver_trajs:
            parsed_answer = traj.result
            traj.steps[0].reward = self.reward_function(task, parsed_answer).reward
            solutions.append(parsed_answer)

        # Judge solutions
        judge_traj = await self.judge.judge_solutions(problem, solutions)
        selected_solution = judge_traj.result
        reward = self.reward_function(task, selected_solution).reward
        judge_traj.steps[0].reward = reward
        judge_traj.reward = reward

        return solver_trajs + [judge_traj]

import asyncio
import re
import uuid

from rllm.agents.agent import Trajectory
from rllm.engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.sdk import StepView, get_chat_client_async, session
from rllm.sdk.protocol import TrajectoryView
from rllm.workflows.workflow import Workflow


class Solver:
    def __init__(self, **kwargs):
        self.client = get_chat_client_async(base_url="http://localhost:4000/v1", api_key="EMPTY")
        self.model = "Qwen/Qwen3-4B-Instruct-2507"

    async def generate_solution(self, problem: str) -> StepView:
        with session(agent="solver", groupby_key=str(uuid.uuid4())) as sess:
            messages = [{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1,
                max_tokens=1000,
            )
            response_text = response.choices[0].message.content

        step = sess.steps[0]
        step.action = self._parse_solver_response(response_text)
        return step

    async def generate_solutions(self, problem: str, n_solutions: int = 2) -> list[Trajectory]:
        tasks = [asyncio.create_task(self.generate_solution(problem)) for _ in range(n_solutions)]
        return await asyncio.gather(*tasks)

    def _parse_solver_response(self, response: str) -> str:
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return f"<answer>{answer_match.group(1).strip()}</answer>"
        else:
            return "No solution found"


class Judge:
    def __init__(self, **kwargs):
        self.client = get_chat_client_async(base_url="http://localhost:4000/v1", api_key="EMPTY")
        self.model = "Qwen/Qwen3-4B-Instruct-2507"

    async def judge_solutions(self, problem: str, solutions: list[str]) -> Trajectory:
        with session(agent="judge") as sess:
            messages = [{"role": "user", "content": self._create_judge_prompt(problem, solutions)}]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1,
                max_tokens=1000,
            )
            response_text = response.choices[0].message.content

        step = sess.steps[0]
        step.action = self._parse_judge_response(response_text, solutions)

        return step

    def _parse_judge_response(self, response: str, solutions: list[str]) -> str:
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


class SolverJudgeWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, n_solutions: int = 2, reward_function: RewardFunction = None, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.n_solutions = n_solutions
        self.reward_function = reward_function
        self.solver = Solver()
        self.judge = Judge()

    async def run(self, task: dict, uid: str, **kwargs) -> list[TrajectoryView]:
        self.reset(task, uid)
        problem = task["question"]

        # Step 1: Solver generates multiple solutions in parallel
        solver_steps = await self.solver.generate_solutions(problem, self.n_solutions)

        # Assign rewards to solver trajectories
        solutions = []
        for step in solver_steps:
            solutions.append(step.action)
            step.reward = self.reward_function(task, step.action).reward

        # Step 2: Judge selects the best solution
        judge_step = await self.judge.judge_solutions(problem, solutions)

        # Evaluate the selected solution
        reward_result = self.reward_function(task, judge_step.action)
        judge_step.reward = reward_result.reward

        return [TrajectoryView(name="solver", steps=[solver_step]) for solver_step in solver_steps] + [TrajectoryView(name="judge", steps=[judge_step])]

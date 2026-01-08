import asyncio
import re

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.workflow import Workflow

class MathSolver:
    """Generates solutions to math problems."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def generate_solution(self, problem: str) -> Trajectory:
        """Generate a single solution to a math problem."""
        messages = [
            {
                "role": "user",
                "content": f"{problem}\nThink step by step and output your final answer within \\boxed{{}}.",
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        action = output.content if output.finish_reason != "length" else "No Solution Found"

        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions = messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    action=action,
                    model_output=output,
                )
            ],
        )

    async def generate_solutions(self, problem: str, n_solutions: int = 2) -> list[Trajectory]:
        """Generate multiple solutions in parallel."""
        tasks = [asyncio.create_task(self.generate_solution(problem)) for _ in range(n_solutions)]
        return await asyncio.gather(*tasks)

class MathJudge:
    """Evaluates and selects the best solution from multiple candidates."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def judge_solutions(self, problem: str, solutions: list[str]) -> Trajectory:
        """Judge multiple solutions and select the best one."""
        messages = [{"role": "user", "content": self._create_judge_prompt(problem, solutions)}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)

        return Trajectory(
            name="judge",
            steps=[
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    action=self._parse_judge_response(output.content, solutions),
                    model_output=output,
                )
            ],
        )

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
        prompt = f"""You are an expert verifier. Given a math problem and multiple solution attempts, select a correct solution.
Problem:
{problem}
Solutions to evaluate:
"""
        for i, solution in enumerate(solutions, 1):
            prompt += f"\nSolution {i}:\n{solution}\n"

        prompt += """
Evaluate each solution for correctness and output the index of your selected solution within <answer>...</answer> tags, e.g., <answer>1</answer> for the first solution, <answer>2</answer> for the second solution, etc. If multiple solutions are correct, output the index of the first correct solution. If no solution is correct, output the index of the solution closest to being correct."""
        return prompt


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
        reward_function = None,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.n_solutions = n_solutions
        self.reward_function = reward_function or math_reward_fn
        self.solver = MathSolver(rollout_engine)
        self.judge = MathJudge(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the solver-judge workflow."""
        self.reset(task, uid)
        problem = task.get("question")

        # Step 1: Solver generates multiple solutions in parallel
        solver_trajectories = await self.solver.generate_solutions(problem, self.n_solutions)

        # Collect full solutions and assign rewards
        solutions = []
        for traj in solver_trajectories:
            solution = traj.steps[0].action 
            solutions.append(solution)
            reward_result = self.reward_function(task, solution)
            traj.steps[0].reward = reward_result.reward
            traj.reward = reward_result.reward

        # Step 2: Judge selects the best solution
        judge_trajectory = await self.judge.judge_solutions(problem, solutions)
        selected_solution = judge_trajectory.steps[0].action

        # Evaluate the selected solution
        reward_result = self.reward_function(task, selected_solution)
        judge_trajectory.steps[0].reward = reward_result.reward
        judge_trajectory.reward = reward_result.reward
        is_correct = reward_result.is_correct

        # Compute metrics
        solver_acc = sum(traj.steps[0].reward for traj in solver_trajectories) / len(solver_trajectories)
        judge_acc = int(is_correct)

        # Return episode with all trajectories
        return Episode(
            id=uid,
            task=task,
            trajectories=[*solver_trajectories, judge_trajectory],
            is_correct=is_correct,
            metrics={"solver_acc": solver_acc, "judge_acc": judge_acc},
        )

from typing import Any

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow


class Solver:
    def __init__(self, rollout_engine, **kwargs):
        self._trajectory = Trajectory()
        self.rollout_engine = rollout_engine

    async def generate_multiple_solutions(self, problem: str, n_solutions: int = 2) -> list[str]:
        """Generate multiple solutions to the problem and log each as a step."""
        responses = []
        solutions = []

        for i in range(n_solutions):
            messages = [{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}]

            response = (await self.rollout_engine.get_model_response(messages)).text

            if "</think>" in response:
                action = response[response.find("</think>") + len("</think>") :].strip()
            else:
                action = "No solution found"

            responses.append(response)
            solutions.append(action)

        return responses, solutions

    def reset(self):
        """Reset the solver's trajectory."""
        self._trajectory = Trajectory()

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        return self.messages


class Judge:
    def __init__(self, rollout_engine, **kwargs):
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": ""}]
        self.rollout_engine = rollout_engine

    async def select_best_solution(self, problem: str, solutions: list[str]) -> dict[str, Any]:
        """Evaluate all solutions and select the best one, logging the verification process."""

        # Create initial step for verification task
        verification_prompt = self._create_verification_prompt(problem, solutions)

        # Get model response for verification
        messages = [{"role": "user", "content": verification_prompt}]
        response = (await self.rollout_engine.get_model_response(messages)).text

        # Parse the verification response to extract the selected solution
        selected_solution, confidence_score = self._parse_verification_response(response, solutions)

        # Create step for the verification result
        verification_step = Step(model_response=response, action=Action(selected_solution), chat_completions=messages + [{"role": "assistant", "content": response}], info={"selected_solution": selected_solution, "confidence_score": confidence_score, "total_solutions_evaluated": len(solutions)})
        self._trajectory.steps.append(verification_step)

        return {"selected_solution": selected_solution, "confidence_score": confidence_score, "verification_response": response}

    def _create_verification_prompt(self, problem: str, solutions: list[str]) -> str:
        """Create a prompt for the verifier to evaluate solutions."""
        prompt = f"""You are an expert verifier. Given a countdown problem and multiple solution attempts, select the best solution.

Problem:
{problem}

Solutions to evaluate:
"""
        for i, solution in enumerate(solutions, 1):
            prompt += f"\nSolution {i}:\n{solution}\n"

        prompt += """
Please evaluate each solution for correctness, clarity, and completeness. For countdown problems, check:
1. Does the solution use only the given numbers?
2. Is each number used exactly once?
3. Are only basic arithmetic operations (+, -, *, /) used?
4. Does the calculation result in the target number?
5. Is the final answer clearly marked within <answer>...</answer> tags?

Respond in the following format:
Selected Solution: [number of the best solution, e.g., 1, 2, 3, or 4]
Confidence Score: [0.0-1.0]
Reasoning: [Your detailed reasoning for why this solution is best]
"""
        return prompt

    def _parse_verification_response(self, response: str, solutions: list[str]) -> tuple[str, float]:
        """Parse the verifier's response to extract selected solution and confidence."""
        lines = response.strip().split("\n")
        selected_solution = ""  # default to empty string
        confidence_score = 0.5  # default confidence

        for line in lines:
            if line.startswith("Selected Solution:"):
                try:
                    solution_num = int(line.split(":")[1].strip())
                    if 1 <= solution_num <= len(solutions):
                        selected_solution = solutions[solution_num - 1]
                except (ValueError, IndexError):
                    pass
            elif line.startswith("Confidence Score:"):
                try:
                    confidence_score = float(line.split(":")[1].strip())
                    confidence_score = max(0.0, min(1.0, confidence_score))  # clamp to [0,1]
                except (ValueError, IndexError):
                    pass

        return selected_solution, confidence_score

    def reset(self):
        """Reset the verifier's trajectory."""
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        return self.messages


class SolverJudgeWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, n_solutions: int = 2, reward_function: RewardFunction = None, **kwargs):
        super().__init__(rollout_engine, **kwargs)

        self.n_solutions = n_solutions
        self.reward_function = reward_function
        self.solver = Solver(rollout_engine)
        self.judge = Judge(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the solver-judge workflow."""
        # Reset components for new task
        self.reset(task, uid)

        problem = task["question"]  # Changed from "query" to "question" for countdown format

        solver_trajectories = [Trajectory() for _ in range(self.n_solutions)]

        # Step 1: Solver generates multiple solutions
        responses, solutions = await self.solver.generate_multiple_solutions(problem, self.n_solutions)

        user_message = {"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}

        for i in range(self.n_solutions):
            reward_result = self.reward_function(task, solutions[i])

            solver_trajectories[i].steps.append(
                Step(
                    model_response=responses[i],
                    action=Action(solutions[i]),
                    chat_completions=[user_message, {"role": "assistant", "content": responses[i]}],
                    reward=reward_result.reward,
                )
            )
            solver_trajectories[i].reward = reward_result.reward

        # Step 2: Judge selects the best solution
        verification_result = await self.judge.select_best_solution(problem, solutions)

        # Apply reward function if provided
        if self.reward_function is not None and self.judge.trajectory.steps:
            final_step = self.judge.trajectory.steps[-1]
            selected_solution = verification_result["selected_solution"]

            # Apply reward function
            reward_result = self.reward_function(task, selected_solution)
            final_step.reward = reward_result.reward
            final_step.info = {**(final_step.info or {}), **reward_result.metadata}

            self.judge.trajectory.reward = reward_result.reward

            # Set correctness on the episode
            is_correct = reward_result.is_correct

        else:
            is_correct = False

        solver_acc = sum(traj.steps[-1].reward for traj in solver_trajectories) / len(solver_trajectories)
        judge_acc = int(is_correct)

        # Create episode with trajectories as list of tuples
        episode = Episode(
            id=uid,
            task=task,
            is_correct=is_correct,
            trajectories=[("solver", traj) for i, traj in enumerate(solver_trajectories)] + [("judge", self.judge.trajectory)],
            # all solver trajectories will get grouped together in GRPO
            metrics={"solver_acc": solver_acc, "judge_acc": judge_acc},
        )

        return episode

    def reset(self, task: dict, uid: str):
        self.solver.reset()
        self.judge.reset()
        self.trajectory = Trajectory()
        self.messages = []
        self.task = task
        self.uid = uid

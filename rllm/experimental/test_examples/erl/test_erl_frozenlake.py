"""ERL FrozenLake training example for the rLLM unified trainer.

Reproduces the FrozenLake experiment from:
  Shi et al., "Experiential Reinforcement Learning", 2026.

Uses the generic ErlWorkflow with FrozenLake-specific callables.
Launch via the companion shell script ``tmp/test_erl_frozenlake.sh``.
"""

from __future__ import annotations

from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from rllm.agents.agent import Trajectory
from rllm.data.dataset import DatasetRegistry
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.experimental.erl import DEFAULT_ERL_ADV_ESTIMATOR_MAP, ErlConfig, ErlWorkflow
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.workflows.store import InMemoryStore

# ---------------------------------------------------------------------------
# Constants — ported from the original ERL codebase
# ---------------------------------------------------------------------------

ACTION_CODE_TO_TEXT = {1: "Left", 2: "Down", 3: "Right", 4: "Up"}

# Obfuscated symbols (the core of the ERL experiment — agent must discover
# what each symbol means through interaction, not from the prompt).
ERL_FROZENLAKE_SYMBOLS = {
    "player": "A",
    "goal": "B",
    "hole": "C",
    "frozen": "D",
    "player_hole": "X",
    "player_goal": "\u221a",
}

# Game-agnostic system prompt — does NOT reveal game mechanics or symbol meanings.
# Uses a structured <reason> block for chain-of-thought reasoning.
ERL_FROZENLAKE_SYSTEM_PROMPT = """
You are an agent playing a game on a grid, acting as a reasoning engine.
Your decisions are based on your current game rules (your best guess of how the game works)
and your strategic playbook (your learned strategies). These may be incomplete or incorrect.
Your only way to interact with the environment is by choosing your NEXT ACTION.

Instructions:
1. Analyze State: Summarize the current state.
2. Predict Long-term Value of Outcomes (Value Function Evaluation): Evaluate the strategic value
   and potential of the current state for the future.
3. Predict Immediate Consequences (World Model Simulation): For the top two candidate actions,
   predict their consequences using a "result-because" structure.
4. Select the Best Action: Choose the action leading to the most advantageous future state.

Your response MUST strictly follow this structure:
<reason>
**1. Analysis of the Current State:**
[Summary of the board state.]

**2. Prediction of the Value of Current States:**
[Assessment of the state's strategic value.]
- **Value:** High / Medium / Low value with justification.

**3. Prediction of Immediate Consequences:**
[Analyze ONLY the top 2 candidate actions using the "result-because" structure.]
- **Action A:** result-because structure.
- **Action B:** result-because structure.
</reason>

Then output the NEXT ACTION inside triple backticks, like this:
```Up```

Always remember:
- Valid actions: Up, Down, Left, Right.
- Think step by step, but make the final line only the next action wrapped in triple backticks, e.g., ```Up```.
""".strip()

# Updater system prompt — instructs the model to produce a structured
# <prompt> block with game_rules and strategy sections.
FROZENLAKE_UPDATER_PROMPT = (
    "You are a chief scientific strategist and master tactician. "
    "Your mission is to analyze extensive field data from numerous operations to distill and refine the "
    "Master Rulebook of a complex game. "
    "You will be presented with a large collection of highly successful trajectories and critical failure "
    "trajectories, collected over a long period. "
    "Your primary task is to perform a deep, comparative analysis to understand the fundamental principles "
    "of victory and defeat. Act as a grand strategist, identifying universal patterns and high-level causal "
    "relationships. Your goal is to synthesize these insights to produce the next generation's Master Rulebook, "
    "making it more robust, accurate, and effective. "
    "Core Principles: Think Long-Term\u2014focus on universal, strategic truths that hold across diverse scenarios; "
    "Learn from Contrast\u2014extract insights by comparing winners and losers; Synthesize and Consolidate\u2014produce a "
    "single unified theory; Be Authoritative and Concise\u2014state rules as definitive principles. "
    "Your output MUST be a single consolidated <prompt> block representing the new Master Rulebook:\n"
    "<prompt>\n"
    "<game_rules>\n"
    "**1. Symbol Meanings:** [Clarify what each key symbol represents within the game world "
    "(entities, terrain types, goals, risks, etc.), and explicitly identify which symbol "
    "corresponds to the player/agent.]\n"
    "**2. Information & Interpretation:** [Define how elements reliably inform about the game state.]\n"
    "**3. Gameplay & Actions:** [Define the core mechanics and interactions.]\n"
    "**4. Action Effects:** [Describe the predictable outcomes of actions.]\n"
    "**5. Game Objective & Termination:** [State the ultimate win/loss conditions.]\n"
    "</game_rules>\n"
    "<strategy>\n"
    "**1. Core Strategies:** [Describe foundational, high-level strategic priorities that lead to victory.]\n"
    "**2. Tactical Tips:** [List widely applicable, advantageous situational plays.]\n"
    "</strategy>\n"
    "</prompt>"
)

MAX_AGENT_STEPS = 8
MAX_ENV_STEPS = 8


# ---------------------------------------------------------------------------
# Symbol map application helper
# ---------------------------------------------------------------------------


def _apply_symbol_map(env, symbol_map: dict[str, str]) -> None:
    """Override the environment's GRID_LOOKUP to use obfuscated symbols."""
    env.GRID_LOOKUP = {
        0: f" {symbol_map['player']} \t",
        1: f" {symbol_map['frozen']} \t",
        2: f" {symbol_map['hole']} \t",
        3: f" {symbol_map['goal']} \t",
        4: f" {symbol_map['player_hole']} \t",
        5: f" {symbol_map['player_goal']} \t",
    }


# ---------------------------------------------------------------------------
# Solver function factory
# ---------------------------------------------------------------------------


def make_solver_fn(
    max_steps: int = MAX_AGENT_STEPS,
    env_max_steps: int = MAX_ENV_STEPS,
    is_slippery: bool = False,
    symbol_map: dict[str, str] | None = None,
):
    """Return an async ``solver_fn`` compatible with :class:`ErlWorkflow`."""
    symbols = symbol_map or ERL_FROZENLAKE_SYMBOLS

    async def solver_fn(prompt: str, task: dict[str, Any], engine: RolloutEngine) -> Trajectory:
        from rllm.agents.frozenlake_agent import FrozenLakeAgent
        from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv

        agent = FrozenLakeAgent(max_steps=max_steps, use_accumulate_history=True)
        agent.reset()
        agent.messages[0]["content"] = prompt

        env = FrozenLakeEnv(
            size=task.get("size", 4),
            seed=task.get("seed", 42),
            p=task.get("p", 0.8),
            max_steps=env_max_steps,
            is_slippery=is_slippery,
        )
        _apply_symbol_map(env, symbols)
        observation, info = env.reset(task)
        _apply_symbol_map(env, symbols)  # re-apply after reset re-inits
        # Re-render with obfuscated symbols
        observation = env.render(mode="tiny_rgb_array")
        info["max_steps"] = env_max_steps
        agent.update_from_env(observation, 0, False, info)

        for _ in range(max_steps):
            output = await engine.get_model_response(agent.chat_completions)
            response = output.text or output.content or ""
            agent.update_from_model(response)

            last_step = agent.trajectory.steps[-1]
            last_step.model_output = output
            if output.prompt_ids is not None:
                last_step.model_post_init(None)

            obs, reward, done, info = env.step(int(last_step.action))
            info["max_steps"] = env_max_steps
            agent.update_from_env(obs, reward, done, info)

            if done:
                break

        traj = agent.trajectory
        traj.reward = 1.0 if env.success() else 0.0
        return traj

    return solver_fn


# ---------------------------------------------------------------------------
# Feedback function
# ---------------------------------------------------------------------------


def frozenlake_feedback_fn(task: dict[str, Any], trajectory: Trajectory) -> str:
    """Generate detailed textual feedback from a FrozenLake trajectory."""
    action_steps = [s for s in trajectory.steps if s.action is not None]
    steps_taken = len(action_steps)
    succeeded = (trajectory.reward or 0.0) >= 1.0
    reward = trajectory.reward if trajectory.reward is not None else 0.0

    if not action_steps:
        return f"No valid actions recorded. Reward={reward:.2f}, steps=0."

    last_step = action_steps[-1]
    last_info = dict(last_step.info or {})
    max_steps_limit = last_info.get("max_steps")

    # Determine outcome
    if succeeded:
        outcome = "The agent reached the goal."
    elif max_steps_limit and steps_taken >= max_steps_limit:
        outcome = f"Hit the max step limit ({max_steps_limit})."
    elif last_step.done:
        outcome = "Fell into a hole."
    else:
        outcome = "Episode ended unexpectedly."

    # Ineffective moves
    ineffective = sum(1 for s in action_steps if s.info and s.info.get("action_is_effective") is False)
    if not succeeded and ineffective:
        outcome += f" {ineffective} action(s) had no effect (likely attempted to move out of bounds or into a wall)."
        # Trailing ineffective
        trailing = 0
        for s in reversed(action_steps):
            if s.info and s.info.get("action_is_effective") is False:
                trailing += 1
            else:
                break
        if trailing >= 2:
            outcome += f" Last {trailing} action(s) in a row did nothing."

    # Last action
    action_code = last_step.action
    action_text = ACTION_CODE_TO_TEXT.get(int(action_code) if action_code is not None else 0, str(action_code))
    outcome += f" Last action taken: {action_text}."

    # Final state hint
    if last_step.done and not succeeded:
        outcome += " Final state: in a hole."
    elif max_steps_limit and steps_taken >= max_steps_limit:
        outcome += " Final state: still on the board when max steps were reached."

    return f"{outcome} Reward={reward:.2f}, steps={steps_taken}."


# ---------------------------------------------------------------------------
# State builder function
# ---------------------------------------------------------------------------


def frozenlake_state_builder_fn(
    base_prompt: str,
    task: dict[str, Any],
    trajectory: Trajectory,
    feedback: str,
) -> str:
    """Build context for the updater model from a single FrozenLake attempt."""
    lines: list[str] = [
        "## Inferred information from past attempts (may be inaccurate)",
        base_prompt.strip(),
        "",
        "## Recent Attempt",
        f"Seed: {task.get('seed')} | Size: {task.get('size')} | p: {task.get('p')}",
    ]

    lines.append("Trace:")
    for turn, step in enumerate(trajectory.steps, 1):
        obs = step.observation or "<missing observation>"
        action_code = step.action
        action_text = ACTION_CODE_TO_TEXT.get(int(action_code) if action_code is not None else 0, str(action_code))
        info = step.info or {}
        effective = info.get("action_is_effective")
        status = f"reward={step.reward:.2f}"
        if effective is not None:
            status += f", effective={bool(effective)}"
        if step.done:
            status += ", done=True"

        lines.append(f"- Observation {turn} (seen before choosing an action):\n{obs}")
        lines.append(f"- Action {turn} (taken after Observation {turn}): {action_text} | {status}")

        # Action comments — explain what happened
        action_comment = ""
        if effective is False:
            action_comment = "The agent did not move (likely hit a wall or tried to go out of bounds)."
        elif effective is True:
            if step.done and step.reward <= 0:
                action_comment = "The agent moved into a hole; episode ended."
            elif step.reward > 0 or step.done:
                action_comment = "The agent reached the goal."
            else:
                action_comment = "The agent moved to a new safe tile; goal not reached yet."
        if action_comment:
            lines.append(f"  -> {action_comment}")

    reward = trajectory.reward if trajectory.reward is not None else 0.0
    lines.append(f"Reward: {reward:.4f} | Correct: {reward >= 1.0}")
    lines.append(f"Feedback: {feedback}")
    lines.append("")
    lines.append("Think step by step and provide an improved FrozenLake instruction enclosed in <prompt>...</prompt> tags.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset preparation (inlined to avoid PYTHONPATH issues)
# ---------------------------------------------------------------------------


def _prepare_frozenlake_datasets(train_size: int = 10000, test_size: int = 100):
    rng = np.random.RandomState(42)
    train_seeds = rng.randint(0, 100000, size=train_size)
    test_seeds = rng.randint(0, 100000, size=test_size)
    train_sizes = rng.randint(2, 10, size=train_size)
    test_sizes = rng.randint(2, 10, size=test_size)
    train_ps = rng.uniform(0.6, 0.85, size=train_size)
    test_ps = rng.uniform(0.6, 0.85, size=test_size)

    def _make(seeds, sizes, ps):
        return [{"seed": int(s), "size": int(sz), "p": float(p), "index": i, "uid": f"{s}_{sz}_{p}"} for i, (s, sz, p) in enumerate(zip(seeds, sizes, ps, strict=False))]

    train_data = _make(train_seeds, train_sizes, train_ps)
    test_data = _make(test_seeds, test_sizes, test_ps)

    train_ds = DatasetRegistry.register_dataset("frozenlake", train_data, "train")
    test_ds = DatasetRegistry.register_dataset("frozenlake", test_data, "test")
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="pkg://rllm.experimental.config",
    config_name="unified",
    version_base=None,
)
def main(config: DictConfig):
    train_dataset, test_dataset = _prepare_frozenlake_datasets()

    erl_config = ErlConfig(
        initial_system_prompt=ERL_FROZENLAKE_SYSTEM_PROMPT,
        updater_system_prompt=FROZENLAKE_UPDATER_PROMPT,
        updater_sampling_params={"temperature": 0.7, "top_p": 0.9},
        train_first_attempt=True,
        train_second_attempt=True,
        train_distilled=True,
        train_updater=True,
        success_reward_threshold=1.0,
    )

    store = InMemoryStore()

    trainer = AgentTrainer(
        workflow_class=ErlWorkflow,
        workflow_args={
            "solver_fn": make_solver_fn(
                max_steps=MAX_AGENT_STEPS,
                env_max_steps=MAX_ENV_STEPS,
                symbol_map=ERL_FROZENLAKE_SYMBOLS,
            ),
            "feedback_fn": frozenlake_feedback_fn,
            "state_builder_fn": frozenlake_state_builder_fn,
            "erl_config": erl_config,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="verl",
        traj_group_adv_estimator_map=DEFAULT_ERL_ADV_ESTIMATOR_MAP,
        store=store,
    )
    trainer.train()


if __name__ == "__main__":
    main()

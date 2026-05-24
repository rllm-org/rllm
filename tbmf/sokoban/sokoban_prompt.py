##### SOKOBAN PROMPT TEMPLATES FOR METARL #####
SOKOBAN_PLAY_PROMPT = """
You are an expert agent operating in the Sokoban environment.

# Symbols and Their Meaning
- Walls (`#`): These block movement. You can't move through or push anything into walls.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you! You'll move around the grid to push boxes.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Goal
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!

# Rules
Your admissible actions are ["up", "down", "left", "right"].
You can only push one box at a time. You can't pull boxes, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls (`#`) or other boxes.
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Observations
The initial state of the game is:
{init_observation}{past_trajectories_reflections}{current_trajectory}
Now it's your turn to make moves (choose the next {num_actions_per_turn} actions).

- Your response first be step-by-step reasoning about the current situation — observe the positions of boxes and targets, plan a path to push a box toward a target, and avoid traps like corners or walls.
- Then choose {num_actions_per_turn} admissible actions and present them within <action> </action> tags (separated by comma).
"""


SOKOBAN_REFLECT_PROMPT = '''
You are an expert agent operating in the Sokoban environment.

# Symbols and Their Meaning
- Walls (`#`): These block movement. You can't move through or push anything into walls.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you! You'll move around the grid to push boxes.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Your Goal
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!

# Rules
Your admissible actions are ["up", "down", "left", "right"].
You can only push one box at a time. You can't pull boxes, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls (`#`) or other boxes.
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Your Task
You will be given the history of a past experience.
Your job is to **reflect on the past sequence**, identify any **mistakes or inefficiencies**, and then devise a **concise, improved plan** starting from the original initial state.

# Past Experience
The initial state of the game is:
{init_observation}
{current_trajectory}
The task is NOT successfully completed.

Now it's your turn to reflect on the past experience and come up with a new plan of action.

- Your response should first be step-by-step reasoning about the strategy and path you took to attempt to complete the task. Identify where things went wrong or could be better.
- Then devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags, to guide the next trial.
'''

# Prompt templates for parsing past trajectories and reflections
PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = '''

On trial #{traj_idx}, you take the following actions: 
{past_trajectory}
The task is NOT successfully completed. Your reflection and improved plan is: 
{reflection}'''

HISTORY_ONLY_TEMPLATE = '''

On trial #{traj_idx}, you take the following actions: 
{past_trajectory}
The task is NOT successfully completed.
'''

REFLECTION_ONLY_TEMPLATE = '''

On trial #{traj_idx}, the task is NOT successfully completed. Your reflection is: 
{reflection}'''

def parse_reflection(traj_idx, past_traj, reflection, reflection_type):
    if traj_idx == 0 or len(reflection) == 0:
        return '\n'
    else:
        memories = []
        for _idx in range(traj_idx):
            if reflection_type == 'history_and_reflection':
                memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    reflection=reflection[_idx]
                )
            elif reflection_type == 'history_only':
                memory = HISTORY_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                )
            elif reflection_type == 'reflection_only':
                memory = REFLECTION_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    reflection=reflection[_idx]
                )
            else:
                raise ValueError(f"Unknown reflection_type: {reflection_type}")

            memories.append(memory)
        return ''.join(memories)


# Prompt templates for parsing current trajectory
CURR_TRAJ_AT_TRAJ1 = '''
You have already taken the following actions:
{current_trajectory}
'''

CURR_TRAJ_AT_TRAJ2toN = '''

Currently you're on trial #{traj_idx}. You have already taken the following actions: 
{current_trajectory}
'''

TRAJ_2toN_INIT = '''

Currently you're on trial #{traj_idx}, starting from the initial state.'''


def parse_current_trajectory(turn_idx, traj_idx, curr_traj):
    if traj_idx == 0:
        if turn_idx == 0:
            return ""
        else:
            return CURR_TRAJ_AT_TRAJ1.format(
                current_trajectory=curr_traj
            )
    else:
        if turn_idx == 0:
            return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
        else:
            return CURR_TRAJ_AT_TRAJ2toN.format(
                traj_idx=traj_idx + 1,
                current_trajectory=curr_traj
            )

def get_sokoban_prompt(phase: str = 'play', 
                       turn_idx: int = 0,
                       traj_idx: int = 0,
                       init_observation: str = '',
                       curr_traj: str = '',
                       past_traj: dict = {},
                       reflection: str='',
                       num_actions_per_turn=3,
                       reflection_type: str = 'reflection_only',
                       ):
    assert phase in ['play', 'reflect']    

    num_actions_per_turn = {
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE"
    }[num_actions_per_turn]

    if phase == 'play':
        past_trajectories_reflections = parse_reflection(traj_idx, past_traj, reflection, reflection_type)
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = SOKOBAN_PLAY_PROMPT.format(
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
            num_actions_per_turn=num_actions_per_turn,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = SOKOBAN_REFLECT_PROMPT.format(
            init_observation=init_observation,
            current_trajectory=current_trajectory
        )
    return prompt.strip()


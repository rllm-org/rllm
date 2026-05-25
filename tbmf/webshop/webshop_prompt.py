"""LaMer WebShop prompt module.

Provides prompt construction for the WebShop environment following the
LaMer agent framework conventions (play and reflect phases).
"""

WEBSHOP_PLAY_PROMPT = """
You are an expert autonomous agent operating in the WebShop e\u2011commerce environment.
Your task is to: {task_description}.{past_trajectories_reflections}{current_trajectory}

Your admissible actions of the current situation are:
[
{admissible_actions}
].

Now it's your turn to take one action for the current step.
Your response should briefly reason about the current situation, then choose the one admissible action that best advances the shopping goal.
End your response with exactly one action in <action> </action> tags, and write nothing after the closing action tag.
"""

WEBSHOP_REFLECT_PROMPT = '''
You are an expert autonomous agent operating in the WebShop e\u2011commerce environment.
Your task is to: {task_description}.

You will be given the history of a past experience.
Your job is to **reflect on the past sequence**, identify any **mistakes or inefficiencies**, and then devise a **concise, improved plan** starting from the original initial state.

Below are the last few actions and corresponding observations you have:
{current_trajectory}
The task is NOT successfully completed.

Now it's your turn to reflect on the past experience and come up with a new plan of action.

- Your response should first be step-by-step reasoning about the strategy and path you took to attempt to complete the task. Identify where things went wrong or could be better.
- Then devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags, to guide the next trial.
'''


# Prompt templates for parsing past trajectories and reflections
PAST_EXPERIENCE_REFLECTION_TEMPLATE = '''

On trial #{traj_idx}, the last few actions and observations you have are:
{past_trajectory}
The task is NOT successfully completed. Your reflection is:
{reflection}'''

HISTORY_ONLY_TEMPLATE = '''

On trial #{traj_idx}, the last few actions and observations you have are:
{past_trajectory}
The task is NOT successfully completed. '''

REFLECTION_ONLY_TEMPLATE = '''

On trial #{traj_idx}, the task is NOT successfully completed. Your reflection is:
{reflection}'''

def parse_reflection(traj_idx, past_traj, reflection, reflection_type='reflection_only'):
    if traj_idx == 0:
        return ''
    else:
        memories = []
        for _idx in range(traj_idx):
            has_reflection = _idx in reflection
            has_history = _idx in past_traj

            if reflection_type == 'reflection_and_history' and has_reflection and has_history:
                memory = PAST_EXPERIENCE_REFLECTION_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    reflection=reflection[_idx]
                )
            elif reflection_type == 'history_only' and has_history:
                memory = HISTORY_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                )
            elif reflection_type == 'reflection_only' and has_reflection:
                memory = REFLECTION_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    reflection=reflection[_idx]
                )
            elif has_history:
                memory = HISTORY_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                )
            else:
                continue

            memories.append(memory)
        return ''.join(memories)


CURR_TRAJ_AT_TRAJ1 = '''

Below are the last few actions and corresponding observations you have:
{current_trajectory}'''

CURR_TRAJ_AT_TRAJ2toN = '''

Currently you're on trial #{traj_idx}, below are the last few actions and observations:
{current_trajectory}'''

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

def get_webshop_prompt(phase: str = 'play',
                        turn_idx: int = 0,
                        traj_idx: int = 0,
                        task_description: str = '',
                        curr_traj: str='',
                        past_traj: dict={},
                        admissible_actions: str='',
                        reflection: str='',
                        reflection_type : str='reflection_only'
                        ):
    assert phase in ['play', 'reflect']
    if phase == 'play':
        past_trajectories_reflections = parse_reflection(traj_idx, past_traj, reflection, reflection_type)
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)

        prompt = WEBSHOP_PLAY_PROMPT.format(
            task_description=task_description,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
            admissible_actions=admissible_actions,
        )

    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = WEBSHOP_REFLECT_PROMPT.format(
            task_description=task_description,
            current_trajectory=current_trajectory
        )
    return prompt

def get_webshop_prompt_short(phase: str = 'play',
                        turn_idx: int = 0,
                        traj_idx: int = 0,
                        task_description: str = '',
                        curr_traj: str='',
                        past_traj: dict={},
                        admissible_actions: str='',
                        reflection: str='',
                        reflection_type : str='reflection_only'
                        ):

    assert phase in ['play', 'reflect']
    if phase == 'play':
        reflection_type = 'reflection_only'
        curr_traj = curr_traj.split('\n')[-1]

        past_trajectories_reflections = parse_reflection(traj_idx, past_traj, reflection, reflection_type)
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)

        prompt = WEBSHOP_PLAY_PROMPT.format(
            task_description=task_description,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
            admissible_actions=admissible_actions,
        )

    else:
        curr_traj = curr_traj.split('\n')[-1]
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = WEBSHOP_REFLECT_PROMPT.format(
            task_description=task_description,
            current_trajectory=current_trajectory
        )
    return prompt

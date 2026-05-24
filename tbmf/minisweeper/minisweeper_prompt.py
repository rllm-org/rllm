MINESWEEPER_PLAY_PROMPT = """
You are an expert agent operating in the Minesweeper game.
You will be given a two dimensional {board_size} by {board_size} board, with {n_mines} hidden mines. 
The rows and columns are indexed from 1 to {board_size}.

# Cell States
- Unopened cells (?): cells that are yet to be revealed and may contain a mine.
- Blank cells (.): opened and non-mine cells, and they have no neighboring mines
- Numbered cells (1-8): opened and non-mine cells, and the number indicates how many mines are in the eight neighboring cells, including those diagonally adjacent. For example, a cell with a `8' means all its neighboring cells contain mines.
- Mine cells (*): opened cells that contain a mine.

# Your Goal
Your goal is to clear the board by revealing all the cells that don't contain mines, without detonating any of the hidden mines scattered throughout the board.
Use clues about the number of neighboring mines in each field to reason about the position of mines and non-mine cells.

# Reveal Rules
Your admissible action is to choose ONE unopened cell (?) to reveal per turn. The outcome depends on the content of that cell:
- Blank cell (.): That cell is revealed, and all contiguous blank cells plus their bordering numbered cells are automatically revealed (auto-cascade).
- Numbered cell (1–8): Only that single cell is revealed, showing the count of neighboring mines.
- Mine (*): The game ends immediately in a loss.

# Observation
The initial state of the game is:
{init_observation}{past_trajectories_reflections}{current_trajectory}
Now it's your turn to make a move.
- Your should first reason step-by-step about the current situation — observe the status of the board, inferring the states of unopened cells (?).
- Then choose ONE unopened cell (?) to reveal. Put the index of cell in the format of "(row, col)" within the <action> </action> tag.
"""


MINESWEEPER_REFLECT_PROMPT = '''
You are an expert agent operating in the Minesweeper game.
You will be given a two dimensional {board_size} by {board_size} board, with {n_mines} hidden mines. 
The rows and columns are indexed from 1 to {board_size}

# Cell States
- Unopened cells (?): cells that are yet to be revealed and may contain a mine.
- Blank cells (.): opened and non-mine cells, and they have no neighboring mines
- Numbered cells (1-8): opened and non-mine cells, and the number indicates how many mines are in the eight neighboring cells, including those diagonally adjacent. For example, a cell with a `8' means all its neighboring cells contain mines.
- Mine cells (*): opened cells that contain a mine.

# Your Goal
Your goal is to clear the board by revealing all the cells that don't contain mines, without detonating any of the hidden mines scattered throughout the board.
Use clues about the number of neighboring mines in each field to reason about the position of mines and non-mine cells.

# Reveal Rules
Your admissible action is to choose ONE unopened cell (?) to reveal per turn. The outcome depends on the content of that cell:
- Blank cell (.): That cell is revealed, and all contiguous blank cells plus their bordering numbered cells are automatically revealed (auto-cascade).
- Numbered cell (1–8): Only that single cell is revealed, showing the count of neighboring mines.
- Mine (*): The game ends immediately in a loss.

# Your Task
You will be given the history of a past experience.
Your job now is to **reflect on the past experience**, identify any **mistakes or inefficiencies**, and then devise a **concise, improved plan** for your next try starting from the original initial state.

# Past Experience
The initial state of the game is:
{init_observation}{current_trajectory}
The task is NOT successfully completed.

Now it's your turn to reflect on the past experience and come up with a new plan of action.

- Your response should first be step-by-step reasoning about the strategy and path you took to attempt to complete the task. Identify where things went wrong or could be better.
- Then devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags, to guide the next trial.
'''

# Prompt templates for parsing past trajectories and reflections
PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = '''

On trial #{traj_idx}, you have taken the following actions: 
{past_trajectory}
The task is NOT successfully completed. Your reflection is: 
{reflection}'''

HISTORY_ONLY_TEMPLATE = '''

On trial #{traj_idx}, you have taken the following actions: 
{past_trajectory}
The task is NOT successfully completed.'''

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

def get_minesweeper_prompt(n_mines: int, 
                           board_size: int,
                           phase: str = 'play',
                           turn_idx: int = 0,
                           traj_idx: int = 0,
                           init_observation: str = '', 
                           curr_traj: str = '',
                           past_traj: str = '',
                           reflection: str='',
                           reflection_type: str = 'reflection_only',
                           ):
    assert phase in ['play', 'reflect']    

    if phase == 'play':
        past_trajectories_reflections = parse_reflection(traj_idx, past_traj, reflection, reflection_type)
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = MINESWEEPER_PLAY_PROMPT.format(
            board_size=board_size,
            n_mines=n_mines,
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = MINESWEEPER_REFLECT_PROMPT.format(
            board_size=board_size,
            n_mines=n_mines,
            init_observation=init_observation,
            current_trajectory=current_trajectory
        )
    return prompt.strip()


import copy
import numpy as np
from .core import MineField, ActionFeedback

class MineSweeper(MineField):
    def __init__(self, **env_kwargs):
        env_kwargs["n_rows"] = env_kwargs.get("board_size", 5)
        env_kwargs["n_cols"] = env_kwargs.get("board_size", 5)
        self.board_type = env_kwargs.get("board_type", "board")
        self.board_size = env_kwargs.get("board_size", 5)
        super().__init__(**env_kwargs)

    def step(self, act, x, y):
        # convert to int if not already
        x = int(x)
        y = int(y)
        if act == "L":
            status = self.on_left_click(x, y)
        elif act == "M":
            status = self.on_middle_click(x, y) 
        elif act == "R":
            status = self.on_right_click(x, y)
        else:
            raise ValueError(f"Invalid action: {act}. Must be one of 'L', 'M', or 'R'.")
        
        if self.board_type == "table":
            obs = self.to_str_repr()
        elif self.board_type == "coord":
            obs = self.to_coord_repr()
        elif self.board_type == "board":
            obs = self.to_board_str_repr()
        else:
            raise ValueError(f"Invalid board type: {self.board_type}. Must be one of 'table' or 'coord'.")
        
        # check if board_disp and board_disp_prev are the same
        action_is_effective = np.any(self.board_disp != self.board_disp_prev)
        info = {"action_is_effective": action_is_effective, "won": False}

        if not action_is_effective:
            reward = -1.0
            done = False
        elif status == ActionFeedback.SUCCESS:
            # reward = -0.1 # 1.0 # 0.2 * num_revealed_cells or normalize by length # n_max = 7
            if np.sum(self.board_disp != self.board_disp_prev) == 1:
                reward = 0.5
            else:
                reward = 2.0
            done = False
        elif status == ActionFeedback.GAME_WIN:
            reward = 10.0
            done = True
            info["won"] = True
        elif status == ActionFeedback.GAME_OVER:
            reward = -0.1
            done = True
        else:
            reward = -1.0
            done = False
        
        return obs, reward, done, info

    def reset(self, seed):
        self.board_true = None
        self.board_disp = None
        self.board_mine = None
        self.board_disp_with_index = None
        self.board_disp_prev = None
        
        self.first_move = True
        self.game_over = False
        self.action_history = list()
        
        self.seed = seed
        self.init_disp_board()
        
        rng = np.random.default_rng(seed)
        for _ in range(1000): 
            x, y = int(rng.integers(0, self.board_size)), int(rng.integers(0, self.board_size))

            self.on_first_move(x, y)
        
            if not self.check_game_win():
                break

        if self.board_type == "table":
            obs = self.to_str_repr()
        elif self.board_type == "coord":
            obs = self.to_coord_repr()
        elif self.board_type == "board":
            obs = self.to_board_str_repr()
        else:
            raise ValueError(f"Invalid board type: {self.board_type}. Must be one of 'table' or 'coord'.")
        
        info = {"action_is_effective": True, "won": False}
        
        return obs, info
    
    def copy(self):
        """
        Create a deep copy of the MineSweeper environment.
        Returns a new MineSweeper instance with identical state.
        """
        # Create new instance with same constructor parameters
        new_self = MineSweeper(
            board_size=self.board_size,
            board_type=self.board_type,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            n_mines=self.n_mines,
            seed=self.seed,
            display_on_action=self.display_on_action,
            empty_cell=self.emt,
            mine_cell=self.mine,
            flag_cell=self.flg,
            unchecked_cell=self.unc,
            number_cells=self.nums_disp,
            strict_winning_condition=self.strict_winning_condition
        )
        
        # Copy all board states (numpy arrays)
        new_self.board_true = self.board_true.copy() if self.board_true is not None else None
        new_self.board_disp = self.board_disp.copy() if self.board_disp is not None else None
        new_self.board_mine = self.board_mine.copy() if self.board_mine is not None else None
        new_self.board_disp_with_index = self.board_disp_with_index.copy() if self.board_disp_with_index is not None else None
        new_self.board_disp_prev = self.board_disp_prev.copy() if self.board_disp_prev is not None else None
        
        # Copy game state variables
        new_self.first_move = self.first_move
        new_self.game_over = self.game_over
        new_self.action_history = copy.deepcopy(self.action_history)
        
        # Copy display characters and mappings
        new_self.emt = self.emt
        new_self.mine = self.mine
        new_self.flg = self.flg
        new_self.unc = self.unc
        new_self.nums_disp = self.nums_disp.copy() if isinstance(self.nums_disp, list) else self.nums_disp
        new_self.num2disp = copy.deepcopy(self.num2disp)
        
        # Copy dimensions and settings
        new_self.n_rows = self.n_rows
        new_self.n_cols = self.n_cols
        new_self.n_mines = self.n_mines
        new_self.display_on_action = self.display_on_action
        new_self.strict_winning_condition = self.strict_winning_condition
        new_self.seed = self.seed
        
        # Copy MineSweeper-specific attributes
        new_self.board_type = self.board_type
        new_self.board_size = self.board_size
        
        return new_self
        

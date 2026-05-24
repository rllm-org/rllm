"""
# Author: Yinghao Li
# Modified: February 19th, 2024
# ---------------------------------------
# Description: Implement the minesweeper game.
# Reference: https://github.com/pyGuru123/Python-Games/tree/master/MineSweeper
"""

import re
import json
import copy
import os
import os.path as op
import numpy as np
import logging
from itertools import product
from enum import Enum
from typing import Union
from typing import Optional


logger = logging.getLogger(__name__)

whitespace_start_end = re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE)
tailing_comma = re.compile(r",$", re.MULTILINE)


def prettify_json(text, indent=2, collapse_level=4):
    pattern = r"[\r\n]+ {%d,}" % (indent * collapse_level)
    text = re.sub(pattern, " ", text)
    text = re.sub(r"([\[({])+ +", r"\g<1>", text)
    text = re.sub(
        r"[\r\n]+ {%d}([])}])" % (indent * (collapse_level - 1)),
        r"\g<1>",
        text,
    )
    text = re.sub(r"(\S) +([])}])", r"\g<1>\g<2>", text)
    return text


def replace_idx_quotes(s):
    lines = s.split("\n")
    lines[0] = re.sub(r"[`']", r'"', lines[0])
    for idx, line in enumerate(lines[1:], start=1):
        lines[idx] = re.sub(r"[`'](\d+)'", r'"\g<1>"', line, count=1)
    return "\n".join(lines)


class ActionFeedback(Enum):
    SUCCESS = 0
    UNEXIST_CELL = 1
    MIDDLE_CLICK_UNCHECKED_CELL = 2
    MIDDLE_CLICK_FLAG_CELL = 3
    MIDDLE_CLICK_EMPTY_CELL = 4
    RIGHT_CLICK_NUMBER_CELL = 5
    RIGHT_CLICK_EMPTY_CELL = 6
    LEFT_CLICK_NUMBER_CELL = 7
    LEFT_CLICK_EMPTY_CELL = 8
    LEFT_CLICK_FLAG_CELL = 9
    GAME_OVER = 10
    GAME_WIN = 11
    MIDDLE_CLICK_NUMBER_CELL_NO_FLAG = 12
    MIDDLE_CLICK_NUMBER_CELL_NUMBER_MISMATCH = 13
    START_BY_RIGHT_CLICK = 14
    START_BY_MIDDLE_CLICK = 15


class MineField:
    """
    Class to implement the minesweeper game.
    """

    def __init__(
        self,
        n_rows=9,
        n_cols=9,
        n_mines=10,
        seed=42,
        display_on_action=False,
        empty_cell=".",
        mine_cell="*",
        flag_cell="F",
        unchecked_cell="?",
        number_cells=None,
        strict_winning_condition=False,
        *args, 
        **kwargs
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_mines = n_mines
        self.display_on_action = display_on_action
        self.strict_winning_condition = strict_winning_condition

        self.board_true = None
        self.board_disp = None
        self.board_mine = None
        self.board_disp_with_index = None
        self.board_disp_prev = None

        self.emt = empty_cell
        self.mine = mine_cell
        self.flg = flag_cell
        self.unc = unchecked_cell
        self.nums_disp = number_cells if number_cells is not None else ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.num2disp = {str(i): self.nums_disp[i - 1] for i in range(1, 9)}

        self.first_move = True
        self.game_over = False
        self.action_history = list()

        self.seed = seed
        self.init_disp_board()

    def place_mines(self, num: int = None, exclude: tuple[int, int] = None):
        if num is None:
            num = self.n_mines

        assert num <= self.n_rows * self.n_cols

        np.random.seed(self.seed)
        board_rand = np.random.rand(self.n_rows, self.n_cols)

        board_rand_flat_sorted = np.sort(board_rand.flatten())
        threshold = board_rand_flat_sorted[num - 1]

        board_bool = board_rand <= threshold

        if exclude and board_bool[exclude]:
            threshold = board_rand_flat_sorted[num]
            board_bool = board_rand <= threshold
            board_bool[exclude] = False

        self.board_mine = board_bool

        return self

    def init_disp_board(self):
        self.board_disp = np.empty((self.n_rows, self.n_cols), dtype=str)
        self.board_disp.fill(self.unc)
        return self

    def infer_board(self):
        self.board_true = np.empty((self.n_rows, self.n_cols), dtype=str)
        for i, j in product(range(self.n_rows), range(self.n_cols)):
            self.board_true[i, j] = str(self.count_mines(i, j))
        self.board_true[self.board_mine] = self.mine
        self.board_true[self.board_true == "0"] = self.emt

    def on_first_move(self, x, y):
        if self.board_mine is None:
            self.place_mines(exclude=(x, y))

        self.infer_board()
        self.init_disp_board()
        self.add_index()

        if self.board_true[x, y] == self.mine:
            return self.on_game_over()

        # self.board_disp_prev = copy.deepcopy(self.board_disp)
        # self.board_disp[x, y] = self.board_true[x, y]
        self.update_adjacent_cells(x, y)
        self.first_move = False

        if self.display_on_action:
            self.display()

        self.board_disp_prev = copy.deepcopy(self.board_disp)
        if self.check_game_win():
            return ActionFeedback.GAME_WIN

        return ActionFeedback.SUCCESS

    def on_left_click(self, x: int, y: int):
        """
        Left click event handler.
        """
        self.action_history.append(f"L({x},{y})")
        x -= 1
        y -= 1

        if not self.is_valid_cell(x, y):
            return ActionFeedback.UNEXIST_CELL

        if self.game_over:
            return self.on_game_over()

        if self.first_move:
            return self.on_first_move(x, y)

        # invalid operations
        if self.board_disp[x, y] == self.flg:
            return ActionFeedback.LEFT_CLICK_FLAG_CELL
        elif self.board_disp[x, y] == self.emt:
            return ActionFeedback.LEFT_CLICK_EMPTY_CELL
        elif self.board_disp[x, y] in "12345678":
            return ActionFeedback.LEFT_CLICK_NUMBER_CELL

        if self.board_true[x, y] == self.mine:
            self.board_disp[x, y] = self.board_true[x, y]
            return self.on_game_over()

        self.board_disp_prev = copy.deepcopy(self.board_disp)
        if self.board_true[x, y] == self.emt:
            self.update_adjacent_cells(x, y)
        # number cell
        else:
            self.board_disp[x, y] = self.board_true[x, y]

        if self.display_on_action:
            self.display()

        if self.check_game_win():
            return ActionFeedback.GAME_WIN

        return ActionFeedback.SUCCESS

    def on_right_click(self, x: int, y: int):
        """
        Right click event handler.
        """
        self.action_history.append(f"R({x},{y})")
        if self.first_move:
            return ActionFeedback.START_BY_RIGHT_CLICK

        x -= 1
        y -= 1

        if not self.is_valid_cell(x, y):
            return ActionFeedback.UNEXIST_CELL

        if self.game_over:
            return self.on_game_over()

        if self.board_disp[x, y] in "12345678":
            return ActionFeedback.RIGHT_CLICK_NUMBER_CELL
        elif self.board_disp[x, y] == self.emt:
            return ActionFeedback.RIGHT_CLICK_EMPTY_CELL

        self.board_disp_prev = copy.deepcopy(self.board_disp)
        if self.board_disp[x, y] == self.unc:
            self.board_disp[x, y] = self.flg
        elif self.board_disp[x, y] == self.flg:
            self.board_disp[x, y] = self.unc

        if self.display_on_action:
            self.display()

        if self.check_game_win():
            return ActionFeedback.GAME_WIN

        return ActionFeedback.SUCCESS

    def on_middle_click(self, x: int, y: int):
        """
        Middle click event handler.
        """
        self.action_history.append(f"M({x},{y})")
        if self.first_move:
            return ActionFeedback.START_BY_MIDDLE_CLICK

        x -= 1
        y -= 1

        if not self.is_valid_cell(x, y):
            return ActionFeedback.UNEXIST_CELL

        if self.game_over:
            return self.on_game_over()

        if self.board_disp[x, y] == self.emt:
            return ActionFeedback.MIDDLE_CLICK_EMPTY_CELL
        elif self.board_disp[x, y] == self.flg:
            return ActionFeedback.MIDDLE_CLICK_FLAG_CELL
        elif self.board_disp[x, y] == self.unc:
            return ActionFeedback.MIDDLE_CLICK_UNCHECKED_CELL

        r_start = max(x - 1, 0)
        r_end = min(x + 2, self.n_rows)
        c_start = max(y - 1, 0)
        c_end = min(y + 2, self.n_cols)

        if not np.sum(self.board_disp[r_start:r_end, c_start:c_end] == self.flg):
            return ActionFeedback.MIDDLE_CLICK_NUMBER_CELL_NO_FLAG

        if np.any(
            self.board_true[r_start:r_end, c_start:c_end][self.board_disp[r_start:r_end, c_start:c_end] == self.flg]
            != self.mine
        ):
            self.board_disp_prev = copy.deepcopy(self.board_disp)
            return self.on_game_over()

        if np.sum(self.board_disp[r_start:r_end, c_start:c_end] == self.flg) != int(self.board_true[x, y]):
            return ActionFeedback.MIDDLE_CLICK_NUMBER_CELL_NUMBER_MISMATCH

        self.board_disp_prev = copy.deepcopy(self.board_disp)
        for i, j in product(range(r_start, r_end), range(c_start, c_end)):
            if self.board_disp[i, j] == self.flg:
                continue
            self.update_adjacent_cells(i, j)

        if self.display_on_action:
            self.display()

        if self.check_game_win():
            return ActionFeedback.GAME_WIN

        return ActionFeedback.SUCCESS

    def on_game_over(self):
        self.game_over = True
        if self.display_on_action:
            logger.info("Game Over! Please Restart!")
            self.display()
        return ActionFeedback.GAME_OVER

    def check_game_win(self):
        if np.all((self.board_disp == self.flg) == self.board_mine):
            return True

        if not self.strict_winning_condition:
            if np.all((self.board_disp[~self.board_mine] == self.board_true[~self.board_mine])):
                return True

        return False

    @property
    def n_correctly_flagged_mines(self):
        number = int(np.sum(self.board_mine[self.board_disp == self.flg]))
        return number

    def update_adjacent_cells(self, x, y):
        if not self.is_valid_cell(x, y) or not self.board_disp[x, y] == self.unc:
            return None

        if self.board_true[x, y] == self.emt:
            self.board_disp[x, y] = self.emt

            for i, j in product(range(x - 1, x + 2), range(y - 1, y + 2)):
                self.update_adjacent_cells(i, j)
        else:
            self.board_disp[x, y] = self.board_true[x, y]

        return None

    def is_mine(self, row, col):
        return self.board_mine[row, col]

    def is_valid_cell(self, row, col):
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols

    def count_mines(self, row, col):
        r_start = max(row - 1, 0)
        r_end = min(row + 2, self.n_rows)
        c_start = max(col - 1, 0)
        c_end = min(col + 2, self.n_cols)
        n_mines = np.sum(self.board_mine[r_start:r_end, c_start:c_end])
        return n_mines

    def add_index(self):
        if self.board_disp_with_index is None:
            self.board_disp_with_index = np.empty((self.n_rows + 1, self.n_cols + 1), dtype=str)
            self.board_disp_with_index.fill(self.emt)
            self.board_disp_with_index[0, :] = np.arange(self.n_cols + 1).astype(str)
            self.board_disp_with_index[1:, 0] = np.arange(1, self.n_rows + 1).astype(str)
        self.board_disp_with_index[1:, 1:] = self.board_disp
        return None

    @property
    def n_revealed_cells(self):
        return int(np.sum(self.board_disp != self.unc))

    def to_str_repr(self, with_row_column_ids=False) -> str:
        if with_row_column_ids:
            self.add_index()
            str_arr = np.array2string(self.board_disp_with_index, separator=",")
        else:
            str_arr = np.array2string(self.board_disp, separator=",")
        str_arr = re.sub(r"([\[\]])", "", str_arr)
        str_arr = re.sub(r"'([1-8%s%s%s])'" % (self.emt, self.flg, self.unc), r"`\g<1>'", str_arr)
        if with_row_column_ids:
            str_arr = replace_idx_quotes(str_arr)
        str_arr = tailing_comma.sub("", str_arr)
        str_arr = whitespace_start_end.sub("", str_arr)
        for k, v in self.num2disp.items():
            str_arr = str_arr.replace(f"`{k}'", f"`{v}'")
        return str_arr
    
    def to_board_str_repr(self) -> str:
        # str_arr = np.array2string(self.board_disp, separator=",")
        # str_arr = re.sub(r"([\[\]])", "", str_arr)
        # str_arr = re.sub(r"'([1-8%s%s%s])'" % (self.emt, self.flg, self.unc), r"`\g<1>'", str_arr)
        # str_arr = tailing_comma.sub("", str_arr)
        # str_arr = whitespace_start_end.sub("", str_arr)
        # for k, v in self.num2disp.items():
        #     str_arr = str_arr.replace(f"`{k}'", f"`{v}'")
        rows = []
        for idx, row in enumerate(self.board_disp):
            rows.append(f'Row {idx+1}: ' + " ".join(row))

        return '\n'.join(rows)

    def to_coord_repr(self) -> dict:
        coord_value_list = [
            f"({i+1},{j+1}): {self.board_disp[i, j]}" for i, j in product(range(self.n_cols), range(self.n_rows))
        ]
        coord_str = "\n".join(coord_value_list)
        for k, v in self.num2disp.items():
            coord_str = coord_str.replace(f": {k}", f": {v}")
        return coord_str

    def save_board(self, path: str, additional_addribute: Union[list[str], str] = None, **kwargs) -> None:
        if additional_addribute is None:
            additional_addribute = list()
        elif isinstance(additional_addribute, str):
            additional_addribute = [additional_addribute]

        board_dict = {
            "seed": self.seed,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_mines": self.n_mines,
            "board_mine": self.board_mine.astype(int).tolist(),
        }
        for addr in additional_addribute:
            board_dict[addr] = getattr(self, addr)
        for k, v in kwargs.items():
            board_dict[k] = v
        save_json(board_dict, path, collapse_level=3)
        return None

    def load_board(self, path: str, load_action_history: bool = False) -> "MineField":
        with open(path, "r", encoding="utf-8") as f:
            board_dict = json.load(f)

        self.seed = board_dict["seed"]
        self.n_rows = board_dict["n_rows"]
        self.n_cols = board_dict["n_cols"]
        self.n_mines = board_dict["n_mines"]

        self.board_true = None
        self.board_disp = None
        self.board_disp_with_index = None

        self.first_move = True
        self.game_over = False

        self.board_mine = np.array(board_dict["board_mine"]).astype(bool)
        self.init_disp_board()

        if load_action_history:
            self.action_history = board_dict.get("action_history", list())

        return self
    

def save_json(obj, path: str, collapse_level: Optional[int] = None):
    """
    Save objective to a json file.
    Create this function so that we don't need to worry about creating parent folders every time

    Parameters
    ----------
    obj: the objective to save
    path: the path to save
    collapse_level: set to any collapse value to prettify output json accordingly

    Returns
    -------
    None
    """
    file_dir = op.dirname(op.normpath(path))
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    json_obj = json.dumps(obj, indent=2, ensure_ascii=False)
    if collapse_level:
        json_obj = prettify_json(json_obj, collapse_level=collapse_level)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_obj)

    return None

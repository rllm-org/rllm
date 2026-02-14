#!/root/.venv/bin/python

# Control what commands are visible to agents
ALLOWED_STR_REPLACE_EDITOR_COMMANDS = ["view", "create", "str_replace", "insert"]
# To add undo functionality for agents: ["view", "create", "str_replace", "insert", "undo_edit"]
# To make read-only: ["view"]

"""
Description: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`

Parameters:
  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`.
Allowed values: [`view`, `create`, `str_replace`, `insert`]
  (2) path (string, required): Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.
  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
  (8) enable_linting (boolean, optional): Optional parameter to enable Python linting checks before saving changes. Default is `false`.
"""

import argparse
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings

import sys
import chardet

# sys.stdout.reconfigure(encoding='utf-8')

STATE_FILE = "/var/tmp/editor_state.json"
SNIPPET_LINES = 4

# We ignore certain warnings from tree_sitter (optional).
warnings.simplefilter("ignore", category=FutureWarning)

_LINT_ERROR_TEMPLATE = """Your proposed edit has introduced new syntax error(s).
Please read this error message carefully and then retry editing the file.
ERRORS:
"""

TRUNCATED_MESSAGE = (
    "<response clipped><NOTE>To save on context only part of this file has been "
    "shown to you. You should retry this tool after you have searched inside the file "
    "with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
)
MAX_RESPONSE_LEN = 10000  # 4000 #12000 # 16000


import sys
import io

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
else:
    # Fallback
    sys.stderr.write("sys.stdout does not have a 'buffer' attribute.\n")


def safe_print(x):
    try:
        print(x)
    except UnicodeEncodeError:
        print(x.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def maybe_truncate(content: str, truncate_after: Optional[int] = MAX_RESPONSE_LEN):
    if not truncate_after or len(content) <= truncate_after:
        return content
    return content[:truncate_after] + TRUNCATED_MESSAGE


class EditorError(Exception):
    """Raised for usage or file system errors within the editor tool."""


class EditorResult:
    """
    Simple container for output and optional error messages.
    """

    def __init__(self, output: str, error: str = ""):
        self.output = output
        self.error = error

    def __str__(self):
        if self.error:
            return f"ERROR: {self.error}\n\n{self.output}"
        return self.output


def load_history() -> Dict[str, List[str]]:
    """
    Load the file edit history from STATE_FILE if it exists.
    """
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k: v for k, v in data.items()}
    except FileNotFoundError:
        return {}
    except Exception as e:
        safe_print(f"Warning: Could not load editor history from {STATE_FILE}: {e}")
        return {}


def save_history(history: Dict[str, List[str]]):
    """
    Save the file edit history to STATE_FILE as JSON.
    """
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f)
    except Exception as e:
        safe_print(f"Warning: Could not write editor history to {STATE_FILE}: {e}")


class StrReplaceEditor:
    """
    A file editor that supports the following commands:
        - view
        - create
        - str_replace
        - insert
        - undo_edit

    The edit history is kept in memory (self.file_history) and also persisted to disk.
    """

    def __init__(
        self, file_history: Dict[str, List[str]], enable_linting: bool = False
    ):
        self.file_history = defaultdict(list, file_history)
        self.enable_linting = enable_linting

    def run(
        self,
        command: str,
        path_str: str,
        file_text: str = None,
        view_range: List[int] = None,
        old_str: str = None,
        new_str: str = None,
        insert_line: int = None,
        python_only: bool = True,
    ) -> EditorResult:
        path = Path(path_str)
        self.validate_path(command, path)

        if command == "view":
            return self.view(path, view_range, python_only=python_only)
        elif command == "create":
            return self.create(path, file_text)
        elif command == "str_replace":
            return self.str_replace(path, old_str, new_str)
        elif command == "insert":
            return self.insert(path, insert_line, new_str)
        elif command == "undo_edit":
            return self.undo_edit(path)
        else:
            raise EditorError(
                f"Unrecognized command '{command}'. "
                "Allowed commands: view, create, str_replace, insert, undo_edit."
            )

    def validate_path(self, command: str, path: Path):
        if command == "create":
            if path.exists():
                raise EditorError(
                    f"File already exists at: {path}. Cannot overwrite with 'create'."
                )
        else:
            if not path.exists():
                raise EditorError(f"The path '{path}' does not exist.")

        if path.is_dir() and command != "view":
            raise EditorError(
                f"The path '{path}' is a directory. Only 'view' can be used on directories."
            )

    @staticmethod
    def read_path(path: Path) -> str:
        encoding = chardet.detect(path.read_bytes())["encoding"]
        if encoding is None:
            encoding = "utf-8"
        return path.read_text(encoding=encoding)

    def view(
        self,
        path: Path,
        view_range: Optional[List[int]] = None,
        python_only: bool = True,
    ) -> EditorResult:
        """
        If path is a directory, list contents (2 levels deep, excluding hidden).
        If path is a file, display the file contents with line numbers.
        Then apply [start_line, end_line] slicing if provided.
        """
        if path.is_dir():
            if not python_only:
                cmd = ["find", str(path), "-maxdepth", "2", "-not", "-path", "*/.*"]
            else:
                # Use `-type d -o -name '*.py'` to only list directories or *.py files
                cmd = [
                    "find",
                    str(path),
                    "-maxdepth",
                    "2",
                    "-not",
                    "-path",
                    "*/.*",
                    "(",
                    "-type",
                    "d",
                    "-o",
                    "-name",
                    "*.py",
                    ")",
                ]
            try:
                # Try using the newer parameters first (Python 3.7+)
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, check=False
                    )
                except TypeError:
                    # Fallback for Python 3.5 and 3.6 where capture_output and text are not supported
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        check=False,
                    )

                stderr = proc.stderr.strip()
                stdout = proc.stdout
                if stderr:
                    return EditorResult(output="", error=stderr)

                msg = (
                    f"Here's the files and directories up to 2 levels deep in {path}, "
                    "excluding hidden:\n" + stdout
                )
                msg = maybe_truncate(msg)
                return EditorResult(output=msg)
            except Exception as e:
                return EditorResult(
                    output="",
                    error=f"Ran into {e} while trying to list directory contents of {path}.",
                )

        # ====================
        # NEW RESTRICTION: only .py files are allowed for viewing
        if path.suffix != ".py" and python_only:
            error_msg = (
                f"ERROR: Viewing non-Python files is disallowed for saving context. "
                f"File '{path.name}' is not a .py file."
            )
            return EditorResult(output="", error=error_msg)
        # ====================

        # For a file - always use normal reading
        file_text = self.read_path(path)
        file_text = file_text.expandtabs()
        lines_with_original_numbers = [
            (i, line) for i, line in enumerate(file_text.splitlines())
        ]

        # Optionally slice by [start_line, end_line]
        total_lines = len(lines_with_original_numbers)
        if view_range and len(view_range) == 2:
            start, end = view_range
            if not (1 <= start <= total_lines):
                return EditorResult(
                    output="",
                    error=(
                        f"Invalid view_range {view_range}: start line must be in [1, {total_lines}]"
                    ),
                )
            if end != -1 and (end < start or end > total_lines):
                return EditorResult(
                    output="",
                    error=(
                        f"Invalid view_range {view_range}: end must be >= start "
                        f"and <= {total_lines}, or -1 to view until end."
                    ),
                )

            # Filter lines by 1-based index
            sliced_lines = []
            for i, text in lines_with_original_numbers:
                one_based = i + 1
                if one_based < start:
                    continue
                if end != -1 and one_based > end:
                    continue
                sliced_lines.append((i, text))
        else:
            # No slicing
            sliced_lines = lines_with_original_numbers

        # Now produce a cat-like output (line numbering = i+1)
        final_output = (
            f"Here's the result of running `cat -n` on the file: {path}:\n"
        )
        # Then maybe truncate
        output_str_list = []
        for i, text in sliced_lines:
            # i is 0-based
            output_str_list.append(f"{i+1:6d} {text}")

        final_output += "\n".join(output_str_list)
        final_output = maybe_truncate(final_output)
        return EditorResult(output=final_output)

    def create(self, path: Path, file_text: str) -> EditorResult:
        if file_text is None:
            raise EditorError("Cannot create file without 'file_text' parameter.")

        if self.enable_linting and path.suffix == ".py":
            lint_error = self._lint_check(file_text, str(path))
            if lint_error:
                return EditorResult(output="", error=_LINT_ERROR_TEMPLATE + lint_error)

        try:
            path.write_text(file_text, encoding="utf-8")
            self.file_history[str(path)].append("")
        except Exception as e:
            raise EditorError(f"Error creating file at {path}: {e}")

        success_msg = f"File created at {path}. "
        success_msg += self._make_output(file_text, str(path))
        success_msg += "Review the file and make sure that it is as expected. Edit the file if necessary."

        return EditorResult(output=f"{success_msg}")

    def str_replace(self, path: Path, old_str: str, new_str: str) -> EditorResult:
        if old_str is None:
            raise EditorError("Missing required parameter 'old_str' for 'str_replace'.")

        file_content = self.read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str else ""
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise EditorError(
                f"No occurrences of '{old_str}' found in {path} for replacement."
            )
        if occurrences > 1:
            raise EditorError(
                f"Multiple occurrences of '{old_str}' found in {path}. "
                "Please ensure it is unique before using str_replace."
            )

        old_text = file_content
        updated_text = file_content.replace(old_str, new_str if new_str else "")

        if self.enable_linting and path.suffix == ".py":
            lint_error = self._lint_check(updated_text, str(path))
            if lint_error:
                return EditorResult(output="", error=_LINT_ERROR_TEMPLATE + lint_error)

        self.file_history[str(path)].append(old_text)
        self.write_file(path, updated_text)

        # Original snippet logic
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + (new_str or "").count("\n")
        snippet = "\n".join(updated_text.split("\n")[start_line : end_line + 1])

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return EditorResult(output=success_msg)

    def insert(self, path: Path, insert_line: int, new_str: str) -> EditorResult:
        if new_str is None:
            raise EditorError("Missing required parameter 'new_str' for 'insert'.")

        old_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = old_text.split("\n")

        if insert_line < 0 or insert_line > len(file_text_lines):
            raise EditorError(
                f"Invalid insert_line {insert_line}. Must be in [0, {len(file_text_lines)}]."
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        updated_text = "\n".join(new_file_text_lines)

        if self.enable_linting and path.suffix == ".py":
            lint_error = self._lint_check(updated_text, str(path))
            if lint_error:
                return EditorResult(output="", error=_LINT_ERROR_TEMPLATE + lint_error)

        self.file_history[str(path)].append(old_text)
        self.write_file(path, updated_text)

        # Original snippet logic
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )
        snippet = "\n".join(snippet_lines)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += (
            "Review the changes and make sure they are as expected "
            "(correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        )

        return EditorResult(output=success_msg)

    def undo_edit(self, path: Path) -> EditorResult:
        path_str = str(path)
        if not self.file_history[path_str]:
            raise EditorError(f"No previous edits found for {path} to undo.")

        old_text = self.file_history[path_str].pop()
        self.write_file(path, old_text)

        return EditorResult(
            output=(
                f"Last edit to {path} undone successfully. "
                + self._make_output(old_text, str(path))
            )
        )

    def read_file(self, path: Path) -> str:
        try:
            return self.read_path(path)
        except Exception as e:
            raise EditorError(f"Failed to read file {path}: {e}")

    def write_file(self, path: Path, content: str):
        try:
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            raise EditorError(f"Failed to write file {path}: {e}")

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ) -> str:
        """
        Mimics cat -n style numbering, plus maybe_truncate to avoid huge outputs.
        """
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()

        lines = file_content.split("\n")
        numbered = "\n".join(
            f"{i + init_line:6}\t{line}" for i, line in enumerate(lines)
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + numbered
            + "\n"
        )

    def _lint_check(self, new_content: str, file_path: str) -> str:
        import ast

        try:
            ast.parse(new_content, filename=file_path)
            return ""
        except SyntaxError as e:
            return str(e)


def main():
    def parse_view_range(range_str: str):
        # Remove surrounding brackets if present
        range_str = range_str.strip().strip("[]()")

        # Split on commas or whitespace
        parts = range_str.replace(",", " ").split()

        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Expected two numbers, got: {range_str}")
        try:
            start_line = int(parts[0])
            end_line = int(parts[1])
        except ValueError:
            raise argparse.ArgumentTypeError(f"Could not convert {parts} to integers.")
        return [start_line, end_line]

    parser = argparse.ArgumentParser(
        description=(
            "A disk-backed file editing tool (view, create, str_replace, insert, undo_edit) "
            "with optional linting."
        )
    )
    parser.add_argument(
        "command",
        type=str,
        help="One of: view, create, str_replace, insert, undo_edit",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the target file or directory (absolute path recommended)",
    )
    parser.add_argument(
        "--file_text",
        type=str,
        default=None,
        help="File content (for 'create')",
    )
    parser.add_argument(
        "--view_range",
        type=parse_view_range,
        default=None,
        help="Line range to view [start_line, end_line], use -1 for end.",
    )
    parser.add_argument(
        "--old_str",
        type=str,
        default=None,
        help="Old string (for 'str_replace')",
    )
    parser.add_argument(
        "--new_str",
        type=str,
        default=None,
        help="New string (for 'str_replace' or 'insert')",
    )
    parser.add_argument(
        "--insert_line",
        type=int,
        default=None,
        help="Line number to insert text at (for 'insert')",
    )
    parser.add_argument(
        "--enable_linting",
        type=bool,
        default=False,
        help="Enable linting checks for Python files before saving changes.",
    )
    parser.add_argument(
        "--python_only",
        type=bool,
        default=True,
        help="If True, attempts to limit view (for both dir and file level) to Python files only.",
    )

    args = parser.parse_args()

    file_history = load_history()
    editor = StrReplaceEditor(file_history, enable_linting=args.enable_linting)

    try:
        result = editor.run(
            command=args.command,
            path_str=args.path,
            file_text=args.file_text,
            view_range=args.view_range,
            old_str=args.old_str,
            new_str=args.new_str,
            insert_line=args.insert_line,
            python_only=args.python_only,
        )
        safe_print(result.output)
        if result.error:
            safe_print(f"ERROR: {result.error}")

    except EditorError as e:
        safe_print(f"ERROR: {e}")
    except Exception as e:
        safe_print(f"ERROR: Unhandled exception: {e}")
        import traceback

        traceback.print_exc()

    save_history(dict(editor.file_history))


if __name__ == "__main__":
    main()

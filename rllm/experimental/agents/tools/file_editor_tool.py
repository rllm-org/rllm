"""FileEditorTool: view, create, and edit files inside a sandbox."""

from __future__ import annotations

import shlex

from rllm.experimental.agents.sandboxed_agent import _safe_exec
from rllm.sdk.sandbox.protocol import Sandbox


class FileEditorTool:
    """File operations: view, create, str_replace, insert."""

    schema = {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "View, create, or edit files. Commands: view, create, str_replace, insert.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert"],
                        "description": "The operation to perform.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "File content (for 'create' command).",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "String to replace (for 'str_replace').",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement string (for 'str_replace').",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Line number to insert after (for 'insert').",
                    },
                    "new_str_insert": {
                        "type": "string",
                        "description": "Text to insert (for 'insert').",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [start, end] line range for 'view'.",
                    },
                },
                "required": ["command", "path"],
            },
        },
    }

    def execute(self, sandbox: Sandbox, **kwargs) -> str:
        command = kwargs.get("command", "")
        path = kwargs.get("path", "")

        if command == "view":
            view_range = kwargs.get("view_range")
            if view_range and len(view_range) == 2:
                start, end = view_range
                return _safe_exec(sandbox, f"sed -n '{start},{end}p' {shlex.quote(path)} | cat -n")
            return _safe_exec(sandbox, f"cat -n {shlex.quote(path)}")

        elif command == "create":
            content = kwargs.get("file_text", "")
            # Ensure parent directory exists
            _safe_exec(sandbox, f"mkdir -p $(dirname {shlex.quote(path)})")
            # Use heredoc for safe content writing
            return _safe_exec(
                sandbox,
                f"cat > {shlex.quote(path)} << 'RLLM_EOF'\n{content}\nRLLM_EOF",
            )

        elif command == "str_replace":
            old_str = kwargs.get("old_str", "")
            new_str = kwargs.get("new_str", "")
            if not old_str:
                return "Error: old_str is required for str_replace"
            # Use python for exact string replacement (avoids sed escaping issues)
            py_script = "import sys; content = open(sys.argv[1]).read(); old = sys.argv[2]; new = sys.argv[3]; count = content.count(old); print(f'Replaced {count} occurrence(s)') if count > 0 else print('Error: old_str not found in file'); open(sys.argv[1], 'w').write(content.replace(old, new, 1)) if count > 0 else None"
            return _safe_exec(
                sandbox,
                f"python3 -c {shlex.quote(py_script)} {shlex.quote(path)} {shlex.quote(old_str)} {shlex.quote(new_str)}",
            )

        elif command == "insert":
            insert_line = kwargs.get("insert_line", 0)
            new_str_insert = kwargs.get("new_str_insert", kwargs.get("new_str", ""))
            if not new_str_insert:
                return "Error: new_str or new_str_insert is required for insert"
            sed_arg = shlex.quote(str(insert_line) + "a\\\\" + new_str_insert)
            quoted_path = shlex.quote(path)
            return _safe_exec(
                sandbox,
                f"sed -i {sed_arg} {quoted_path}",
            )

        else:
            return f"Error: unknown command '{command}'. Use: view, create, str_replace, insert"

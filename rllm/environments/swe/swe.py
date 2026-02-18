import base64
import json
import logging
import os
import re

import numpy as np
from datasets import Dataset, load_dataset

from arl import SandboxSession

from rllm.environments.swe.action import Action
from rllm.environments.swe.reward import calculate_reward
from rllm.environments.base.base_env import BaseEnv

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "tools")
CONTINUE_MSG = """
You forgot to use a function call in your response.
YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
"""

CMD_TIMEOUT = 120  # seconds

SKIP_FILES_NEW = ["run_tests.sh", "r2e_tests"]

# Only tools that do real work inside the sandbox.
# execute_bash/finish/submit are handled directly in step().
R2EGYM_TOOL_FILES = [
    os.path.join(TOOLS_DIR, "r2egym/file_editor.py"),
    os.path.join(TOOLS_DIR, "r2egym/search.py"),
]

SWEAGENT_TOOL_FILES = [
    os.path.join(TOOLS_DIR, "sweagent/str_replace_editor.py"),
]

BLOCKED_COMMANDS = frozenset(["git", "ipython", "jupyter", "nohup"])

R2E_ENV_IDS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "R2E-Gym/SWE-Bench-Lite",
]
DEFAULT_R2E_ENV_ID = "R2E-Gym/R2E-Gym-Lite"
_TRUNCATION_LINES = 40


def format_observation(output: str, error_code: str, action_name: str) -> str:
    """Format command output into an observation string.

    For bash commands: includes exit code and truncates long output
    (keeps first/last 40 lines) to save LLM context.
    For tool commands (file_editor, search, etc.): light header only,
    since tool scripts handle their own truncation internally.
    """
    if action_name in ("execute_bash", "bash"):
        lines = output.splitlines() if output else []
        if len(lines) > 2 * _TRUNCATION_LINES:
            top = "\n".join(lines[:_TRUNCATION_LINES])
            bottom = "\n".join(lines[-_TRUNCATION_LINES:])
            divider = "-" * 50
            output = (
                f"{top}\n"
                f"{divider}\n"
                f"<Observation truncated in middle for saving context>\n"
                f"{divider}\n"
                f"{bottom}"
            )
        return (
            f"Exit code: {error_code}\nExecution output of [{action_name}]:\n{output}"
        )

    return f"Execution output of [{action_name}]:\n{output}"


def _derive_pool_ref(ds: dict) -> str:
    """Derive WarmPool name from dataset entry (matches batch_prefetch.py convention)."""
    repo_name = ds.get("repo_name", ds.get("repo", ""))
    commit_hash = ds.get("commit_hash", ds.get("base_commit", ""))
    safe_repo = re.sub(r"[^a-z0-9]", "-", repo_name.lower()).strip("-")
    hash_prefix = commit_hash[:12].lower()
    name = f"{safe_repo}-{hash_prefix}"
    return name[:63].rstrip("-")


class SWEEnv(BaseEnv):
    """Software Engineering Environment backed by ARL sandbox sessions."""

    def __init__(
        self,
        entry: dict | None = None,
        dataset: Dataset | None = None,
        idx: int | None = None,
        step_timeout: int = 90,
        reward_timeout: int = 300,
        gateway_url: str | None = None,
        namespace: str = "default",
        pool_ref: str | None = None,
        verbose: bool = False,
        scaffold: str = "r2egym",
    ):
        if entry is not None:
            self.entry = entry
            self.dataset = None
            self.idx = None
        else:
            if dataset is None:
                dataset = load_dataset(DEFAULT_R2E_ENV_ID, split="test")
            self.dataset = dataset
            if idx is None:
                idx = np.random.randint(0, len(self.dataset))
            assert 0 <= idx < len(self.dataset), "Selected index out of range"
            self.idx = idx
            self.entry = self.dataset[idx]

        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.gateway_url = gateway_url or os.environ.get(
            "ARL_GATEWAY_URL", "http://localhost:8080"
        )
        self.namespace = namespace
        self.pool_ref = pool_ref or _derive_pool_ref(self.entry)
        self.total_steps = 0
        self.verbose = verbose
        self.scaffold = scaffold
        self._cmd_counter = 0
        assert scaffold in ["r2egym", "sweagent"], (
            f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        )

        # ARL session (created in reset)
        self.session: SandboxSession | None = None

        # Detect dataset type
        image = self.entry.get("docker_image", self.entry.get("image_name", ""))
        self.swebench_verified = "swebench" in image
        self.repo_path = "/testbed"
        self.alt_path = "/" if self.swebench_verified else "/root"

        # Logger
        self.logger = logging.getLogger(f"SWEEnv.{self.pool_ref}")
        if not verbose:
            self.logger.setLevel(logging.CRITICAL)

    def _execute_raw(
        self, cmd: str, timeout: int = CMD_TIMEOUT, workdir: str | None = None
    ) -> tuple[str, str, int]:
        """Execute a command and return raw (stdout, stderr, exit_code).

        Low-level method that returns stdout/stderr separately so callers
        can format them as needed (e.g. with [STDOUT]/[STDERR] headers).
        """
        self._cmd_counter += 1
        workdir = workdir or self.repo_path
        assert self.session is not None, "Session not initialized"
        response = self.session.execute(
            steps=[
                {
                    "name": f"cmd_{self._cmd_counter}",
                    "command": ["sh", "-c", f"timeout {timeout} {cmd}"],
                    "workDir": workdir,
                    "timeout": timeout + 10,
                }
            ]
        )
        result = response.results[0]
        stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", result.output.stdout or "")
        stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", result.output.stderr or "")
        return stdout, stderr, result.output.exit_code

    def _run(
        self, cmd: str, timeout: int = CMD_TIMEOUT, workdir: str | None = None
    ) -> tuple[str, str]:
        """Execute a shell command in the sandbox session.

        Returns (output, error_code_str) matching the previous DockerRuntime interface.
        """
        stdout, stderr, exit_code = self._execute_raw(cmd, timeout, workdir)
        output = stdout
        if stderr:
            output = output + "\n" + stderr if output else stderr

        if exit_code == 124:  # timeout exit code
            return f"The command took too long to execute (>{timeout}s)", "-1"
        if exit_code != 0:
            return output, f"Error: Exit code {exit_code}"
        return output, str(exit_code)

    def _copy_to_sandbox(self, src_path: str, dest_path: str):
        """Copy a local file into the sandbox via base64 encoding."""
        with open(src_path, "rb") as f:
            content = f.read()
        b64 = base64.b64encode(content).decode()
        dir_path = os.path.dirname(dest_path)
        # Split into chunks to avoid shell argument length limits
        chunk_size = 65536
        if len(b64) <= chunk_size:
            self._run(
                f"mkdir -p {dir_path} && printf '%s' '{b64}' | base64 -d > {dest_path}"
            )
        else:
            self._run(f"mkdir -p {dir_path} && : > {dest_path}")
            for i in range(0, len(b64), chunk_size):
                chunk = b64[i : i + chunk_size]
                self._run(f"printf '%s' '{chunk}' | base64 -d >> {dest_path}")

    def _setup_env(self):
        """Initialize the sandbox environment (same steps as DockerRuntime.setup_env)."""
        if self.swebench_verified:
            self._run("chmod +x /run_tests.sh")
            self._run("ln -sf /opt/miniconda3/envs/testbed /root/.venv")
            self._run("python -m pip install chardet")
        else:
            self._run(f"ln -sf {self.repo_path}/.venv {self.alt_path}/.venv")
            self._run(
                f"ln -sf {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python"
            )
            self._run(
                f"ln -sf {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3"
            )
            self._run(
                f"find {self.repo_path}/.venv/bin -type f -executable "
                f"-exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;"
            )
            self._run("uv pip install chardet")
            self._run("find . -name '*.pyc' -delete")
            self._run("find . -name '__pycache__' -exec rm -rf {} +")
            self._run("find /r2e_tests -name '*.pyc' -delete")
            self._run("find /r2e_tests -name '__pycache__' -exec rm -rf {} +")
            for skip_file in SKIP_FILES_NEW:
                if skip_file == "r2e_tests":
                    continue
                self._run(
                    f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}"
                )
            self._run(f"mv /r2e_tests {self.alt_path}/r2e_tests")
            self._run(f"ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests")

    def _provision_tools(self, tool_files: list[str]):
        """Copy tool scripts into sandbox and make them executable."""
        for tool_file in tool_files:
            _, ext = os.path.splitext(tool_file)
            cmd_name = os.path.basename(tool_file)
            if ext == ".py":
                container_cmd_name = cmd_name[:-3]  # strip .py
            else:
                container_cmd_name = cmd_name
            container_path = f"/usr/local/bin/{container_cmd_name}"
            self._copy_to_sandbox(tool_file, container_path)
            self._run(f"chmod +x {container_path}")

    def _get_task_instruction(self) -> str:
        """Extract problem statement from dataset entry."""
        try:
            content = self.entry["problem_statement"]
            match = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL)
            return match.group(1) if match else content
        except Exception:
            return self.entry.get("problem_statement", "")

    def _create_session(self):
        """Create a new ARL sandbox session."""
        self.session = SandboxSession(
            pool_ref=self.pool_ref,
            namespace=self.namespace,
            gateway_url=self.gateway_url,
            keep_alive=True,
            timeout=max(self.reward_timeout, self.step_timeout) + 60,
        )
        self.session.create_sandbox()

    # =====================================================
    # BaseEnv interface
    # =====================================================

    def reset(self) -> tuple[str, dict]:
        if self.session:
            self.close()
        self._create_session()
        self._setup_env()

        tool_files = (
            R2EGYM_TOOL_FILES if self.scaffold == "r2egym" else SWEAGENT_TOOL_FILES
        )
        self._provision_tools(tool_files)
        self.total_steps = 0
        self._cmd_counter = 0
        return self._get_task_instruction(), {}

    def step(self, action: str | Action) -> tuple[str, float, bool, dict]:
        if isinstance(action, str):
            action_obj = Action.from_string(action)
        else:
            action_obj = action

        fn = action_obj.function_name.lower() if action_obj.function_name else ""

        # No function call — remind the agent.
        if not fn:
            return CONTINUE_MSG, 0, False, {}

        done = fn in ("finish", "submit")
        if done:
            self.total_steps += 1
            return "<<< Finished >>>", 0, True, {}

        if fn in ("execute_bash", "bash"):
            # Run the command directly in the sandbox.
            # Output is formatted with [STDOUT]/[STDERR] headers to match
            # the old execute_bash tool script's output format.
            cmd = action_obj.parameters.get("command") or action_obj.parameters.get(
                "cmd", ""
            )
            first_token = cmd.strip().split()[0] if cmd.strip() else ""
            if first_token in BLOCKED_COMMANDS:
                output = (
                    f"Bash command '{first_token}' is not allowed. "
                    "Please use a different command or tool."
                )
                error_code = "Error: Exit code 1"
            else:
                stdout, stderr, exit_code = self._execute_raw(
                    cmd, timeout=self.step_timeout
                )
                if exit_code == 124:
                    output = (
                        f"The command took too long to execute (>{self.step_timeout}s)"
                    )
                    error_code = "-1"
                elif exit_code != 0:
                    output = (
                        f"Error executing command:\n\n"
                        f"[STDOUT]\n\n{stdout.strip()}\n\n"
                        f"[STDERR]\n\n{stderr.strip()}"
                    )
                    error_code = f"Error: Exit code {exit_code}"
                else:
                    output = (
                        f"[STDOUT]\n\n{stdout.strip()}\n\n[STDERR]\n\n{stderr.strip()}"
                    )
                    error_code = str(exit_code)
        else:
            # file_editor, search, str_replace_editor — run the tool binary.
            bash_cmd = action_obj.to_bashcmd()
            output, error_code = self._run(bash_cmd, timeout=self.step_timeout)

        self.total_steps += 1
        return format_observation(output, error_code, fn), 0, False, {}

    def compute_final_reward(self) -> float:
        return calculate_reward(
            session=self.session,
            ds=self.entry,
            repo_path=self.repo_path,
            alt_path=self.alt_path,
            timeout=self.reward_timeout,
        )

    def close(self):
        if self.session:
            try:
                self.session.delete_sandbox()
            except Exception as e:
                self.logger.error(f"Error deleting sandbox: {e}")
            self.session = None

    @staticmethod
    def from_dict(extra_info: dict | str) -> "SWEEnv":
        import inspect

        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        sig = inspect.signature(SWEEnv.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        init_params["entry"] = extra_info
        return SWEEnv(**init_params)

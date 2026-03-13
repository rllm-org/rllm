"""SWE-bench agent: bash-only tool-calling agent in a sandboxed environment.

Extends SandboxedAgentFlow so the EvalRunner manages the sandbox lifecycle.
Uses a custom agent loop inspired by mini-swe-agent for better performance:
- Format error recovery (continues when LLM skips tool calls)
- Smart head/tail output truncation
- Environment variable setup for non-interactive execution
- Submission detection via magic string

Supports both Docker (default) and Modal sandbox backends:
- Docker: builds per-instance images via swebench harness
- Modal: builds modal.Image chains from swebench test spec scripts
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import uuid
from pathlib import Path

import openai
from swebench.harness.test_spec.test_spec import make_test_spec

from rllm.experimental.agents.sandboxed_agent import (
    SandboxedAgentFlow,
    _safe_exec,
    create_sandbox,
)
from rllm.sdk.sandbox.protocol import Sandbox
from rllm.types import Episode, Step, Trajectory

from .prompts import BASH_TOOL, FORMAT_ERROR_MSG, INSTANCE_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Serialize Docker image builds to avoid race conditions
_image_build_lock = threading.Lock()

# Output truncation: keep head + tail when output is too long
OUTPUT_LIMIT = 10_000
OUTPUT_HEAD = 5_000
OUTPUT_TAIL = 5_000


def _truncate_output(output: str) -> str:
    """Smart truncation: keep first and last N chars for long outputs."""
    if len(output) <= OUTPUT_LIMIT:
        return output
    total = len(output)
    head = output[:OUTPUT_HEAD]
    tail = output[-OUTPUT_TAIL:]
    return f"{head}\n\n... [{total - OUTPUT_HEAD - OUTPUT_TAIL} characters truncated] ...\n\n{tail}"


def _execute_tool(sandbox: Sandbox, command: str) -> str:
    """Execute a bash command in the sandbox and return truncated output.

    Wraps the command to:
    1. Source conda so the testbed environment is active
    2. Set non-interactive env vars (PAGER=cat, etc.)
    3. Run from /testbed
    """
    wrapped = f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && export PAGER=cat GIT_PAGER=cat TERM=dumb NO_COLOR=1 PIP_PROGRESS_BAR=off && cd /testbed && {command}"
    output = _safe_exec(sandbox, wrapped, timeout=120)
    return _truncate_output(output)


def _run_agent_loop(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    sandbox: Sandbox,
    max_turns: int = 40,
    temperature: float = 0.0,
) -> tuple[list[Step], list[dict], str | None]:
    """Custom agent loop with format error recovery and submission detection.

    Improvements over basic ToolCallingMixin:
    - If the LLM responds without tool calls, send a format error message
      and continue (up to 3 retries) instead of stopping.
    - Detects SUBMIT_PATCH in command output to end the loop early.
    - Smart head/tail truncation for long outputs.
    - All commands run from /testbed.
    """
    tool_schemas = [BASH_TOOL]
    steps: list[Step] = []
    format_errors = 0
    max_format_errors = 3

    for turn in range(max_turns):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas,
                temperature=temperature,
            )
        except Exception:
            logger.exception("LLM API call failed on turn %d", turn)
            break

        choice = response.choices[0]
        msg = choice.message

        # Append assistant message to conversation
        messages.append(msg.model_dump(exclude_none=True))

        # No tool calls — send format error and retry
        if not msg.tool_calls:
            content = msg.content or ""

            # Check if the model is trying to submit via text
            if "SUBMIT_PATCH" in content:
                steps.append(Step(input=f"turn_{turn}", output=content, done=True))
                return steps, messages, content

            format_errors += 1
            if format_errors >= max_format_errors:
                logger.warning("Max format errors reached, ending loop")
                steps.append(Step(input=f"turn_{turn}", output=content, done=True))
                return steps, messages, content

            # Send format error recovery message
            messages.append({"role": "user", "content": FORMAT_ERROR_MSG})
            continue

        # Reset format error counter on successful tool use
        format_errors = 0

        # Process each tool call
        submitted = False
        for tc in msg.tool_calls:
            fn_name = tc.function.name

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"command": tc.function.arguments}

            command = args.get("command", "")

            if fn_name != "bash":
                output = f"Unknown tool: {fn_name}. Only 'bash' is available."
            else:
                output = _execute_tool(sandbox, command)

            # Check for submission signal
            if "SUBMIT_PATCH" in command or "SUBMIT_PATCH" in output:
                submitted = True

            steps.append(Step(input=command, output=output))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                }
            )

        if submitted:
            # Agent signaled it's done — capture final state
            return steps, messages, "SUBMITTED"

    return steps, messages, None


def _ensure_docker_image(task: dict) -> str:
    """Build the swebench Docker instance image if it doesn't exist."""
    import docker as docker_lib
    from swebench.harness.docker_build import build_env_images, build_instance_image

    client = docker_lib.from_env()
    test_spec = make_test_spec(task)
    image_name = test_spec.instance_image_key

    # Lock-free fast path
    try:
        client.images.get(image_name)
        logger.info("Image %s already exists", image_name)
        return image_name
    except docker_lib.errors.ImageNotFound:
        pass

    # Serialize builds — swebench writes to shared build directories
    with _image_build_lock:
        try:
            client.images.get(image_name)
            logger.info("Image %s already exists", image_name)
            return image_name
        except docker_lib.errors.ImageNotFound:
            pass

        logger.info("Building environment images for %s...", task.get("instance_id"))
        build_env_images(client, [task], instance_image_tag="latest", env_image_tag="latest")

        logger.info("Building instance image %s...", image_name)
        build_instance_image(test_spec, client, logger=None, nocache=False)

    return image_name


def _build_modal_image(task: dict):
    """Build a modal.Image from swebench test spec scripts.

    Replicates the three-layer Docker build (base → env → instance) as a
    single modal.Image chain. Modal caches image layers automatically so
    identical conda environments are not rebuilt across instances.

    Returns a ``modal.Image`` object.
    """
    import modal

    test_spec = make_test_spec(task)
    instance_id = task.get("instance_id", "unknown")

    env_script = test_spec.setup_env_script
    repo_script = test_spec.install_repo_script

    # Write scripts to temp files for modal.Image.add_local_file()
    tmpdir = Path(tempfile.mkdtemp(prefix=f"rllm-swe-modal-{instance_id}-"))
    env_script_path = tmpdir / "setup_env.sh"
    repo_script_path = tmpdir / "setup_repo.sh"
    env_script_path.write_text(env_script)
    repo_script_path.write_text(repo_script)

    logger.info("Building Modal image for %s...", instance_id)

    # Build the image chain: base → env → instance
    # Mirrors swebench's ModalSandboxRuntime.get_instance_image()
    image = (
        modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
        .run_commands("apt update")
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(
            "wget",
            "git",
            "build-essential",
            "libffi-dev",
            "libtiff-dev",
            "jq",
            "curl",
            "locales",
            "locales-all",
            "tzdata",
        )
        .run_commands(
            # Install Miniconda3
            "wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh' -O miniconda.sh",
            "bash miniconda.sh -b -p /opt/miniconda3",
            "echo 'export PATH=/opt/miniconda3/bin:$PATH' >> ~/.bashrc",
            "/opt/miniconda3/bin/conda init --all",
            "/opt/miniconda3/bin/conda config --append channels conda-forge",
            "adduser --disabled-password --gecos 'dog' nonroot",
        )
        # Layer 2: environment setup (conda env + dependencies)
        .add_local_file(str(env_script_path), "/root/setup_env.sh", copy=True)
        # Layer 3: repo clone + install
        .add_local_file(str(repo_script_path), "/root/setup_repo.sh", copy=True)
        .run_commands(
            "chmod +x /root/setup_env.sh",
            "/bin/bash -c 'source ~/.bashrc && /root/setup_env.sh'",
            "echo 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed' >> /root/.bashrc",
            "/bin/bash /root/setup_repo.sh",
        )
        .workdir("/testbed/")
    )

    logger.info("Modal image built for %s", instance_id)
    return image


class SWEAgentFlow(SandboxedAgentFlow):
    """AgentFlow for SWE-bench: runs a bash-tool agent in a sandboxed container.

    Sandbox lifecycle is managed by EvalRunner via SandboxedAgentFlow:
    1. setup_sandbox() creates sandbox with the swebench instance image
    2. on_sandbox_ready() resets the repo and configures the environment
    3. run() executes the custom agent loop
    4. teardown_sandbox() destroys the container

    Supports ``sandbox_backend="docker"`` (default) and ``"modal"``.
    """

    max_concurrent: int = 4
    sandbox_backend: str = "docker"

    def setup_sandbox(self, task: dict, config) -> None:
        """Create sandbox with backend-specific image building."""
        task_id = task.get("instance_id", task.get("task_id", "unknown"))
        name = f"rllm-{task_id}-{uuid.uuid4().hex[:6]}"

        if self.sandbox_backend == "modal":
            # Build modal.Image from swebench test spec scripts
            modal_image = _build_modal_image(task)
            from rllm.sdk.sandbox.backends.modal_backend import ModalSandbox

            self._sandbox = ModalSandbox(
                name=name,
                image=modal_image,
                timeout=30 * 60,
            )
        else:
            # Docker: build local swebench instance image
            image = _ensure_docker_image(task)
            self._sandbox = create_sandbox(self.sandbox_backend, name=name, image=image)

        self.on_sandbox_ready(task, config)

    def get_image(self, task: dict) -> str:
        """Build and return the swebench per-instance Docker image."""
        return _ensure_docker_image(task)

    def on_sandbox_ready(self, task: dict, config) -> None:
        """Reset repo and configure environment for non-interactive use."""
        if self.sandbox is None:
            return

        base_commit = task.get("base_commit", "")
        if base_commit:
            _safe_exec(self.sandbox, f"cd /testbed && git reset --hard {base_commit}")
            # Use -fd (not -fdx) to preserve compiled extensions (.so files)
            # that were built during image construction
            _safe_exec(self.sandbox, "cd /testbed && git clean -fd")

    def run(self, task: dict, config) -> Episode:
        instance_id = task.get("instance_id", "unknown")
        repo = task.get("repo", "")
        base_commit = task.get("base_commit", "")
        problem_statement = task.get("question", "")

        # Build messages
        instance_prompt = INSTANCE_PROMPT.format(
            problem_statement=problem_statement,
            repo=repo,
            base_commit=base_commit,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instance_prompt},
        ]

        client = openai.OpenAI(base_url=config.base_url, api_key="not-needed")

        steps, messages, final = _run_agent_loop(
            client=client,
            model=config.model,
            messages=messages,
            sandbox=self.sandbox,
            max_turns=40,
        )

        # Capture the patch
        patch = _safe_exec(self.sandbox, "cd /testbed && git diff") if self.sandbox else ""

        trajectory = Trajectory(
            name="swe_agent",
            task=task,
            steps=steps,
            output=patch,
        )
        return Episode(
            id=f"{instance_id}:0",
            task=task,
            trajectories=[trajectory],
            artifacts={"answer": patch, "patch": patch},
        )


# Module-level singleton for plugin entry point
swe_agent = SWEAgentFlow()

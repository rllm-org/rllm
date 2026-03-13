# Sandboxed Benchmark Support

This document describes the sandboxed execution infrastructure for running coding benchmarks (SWE-bench, Terminal-Bench, etc.) via the `rllm eval` and `rllm train` CLI.

## Overview

Sandboxed benchmarks run agent code inside isolated containers (Docker or Modal). The system provides:

1. **SandboxedAgentFlow** — base class for agents that need sandboxed environments, with lifecycle hooks managed by EvalRunner
2. **ToolCallingMixin** — reusable multi-turn tool-calling loop extracted from SearchAgentFlow
3. **Tool system** — BashTool, FileEditorTool, SubmitTool for agent-sandbox interaction
4. **EvalRunner integration** — automatic setup/teardown of sandboxes around agent execution
5. **Plugin architecture** — SWE-bench and Terminal-Bench agents as installable plugins

## Architecture

```
EvalRunner (manages sandbox lifecycle)
│
├── setup_sandbox(task, config)     ← creates Sandbox via backend factory
├── agent.run(task, config)          ← agent uses self.sandbox
├── evaluator.evaluate(task, ep)     ← evaluator can access sandbox via artifacts
└── teardown_sandbox()               ← guaranteed cleanup (finally block)
```

### Sandbox Backends

The sandbox backend is pluggable via `sandbox_backend` attribute or `--sandbox-backend` CLI flag:

| Backend | Class | Use Case |
|---------|-------|----------|
| `docker` | `DockerSandbox` | Local development, CI/CD |
| `local` | `LocalSandbox` | Testing without Docker |
| `modal` | `ModalSandbox` | Cloud execution, scaling |

### Class Hierarchy

```
SandboxedAgentFlow (ABC)
│   setup_sandbox() / teardown_sandbox() / create_instance()
│   sandbox_backend, image, max_concurrent, setup_commands
│
├── SWEAgentFlow (agenthub/swe_agent/)
│       Uses ToolCallingMixin + BashTool
│       Per-instance images from swebench harness
│       Supports both Docker and Modal backends
│
└── TerminalAgentFlow (agenthub/terminal_agent/)
        Uses ToolCallingMixin + BashTool
        Docker Compose environments (TODO)
```

## Usage

### Running SWE-bench with Docker (default)

```bash
# Install the plugin
cd agenthub/swe_agent && pip install -e .

# Run evaluation
rllm eval swebench_verified --agent swe --model gpt-4o --max-examples 3
```

### Running SWE-bench with Modal

```bash
# Authenticate with Modal
modal setup

# Run with Modal backend
rllm eval swebench_verified --agent swe --model gpt-4o --max-examples 3 --sandbox-backend modal
```

### CLI Options

```
--sandbox-backend    docker | local | modal (auto-detected from agent if omitted)
--sandbox-concurrency    Override max concurrent sandboxes (default: agent's max_concurrent)
```

## Writing a Sandboxed Agent

```python
from rllm.experimental.agents.sandboxed_agent import SandboxedAgentFlow, _safe_exec
from rllm.experimental.agents.tool_calling import ToolCallingMixin
from rllm.experimental.agents.tools.bash_tool import BashTool

class MyAgentFlow(SandboxedAgentFlow, ToolCallingMixin):
    sandbox_backend = "docker"
    image = "python:3.11-slim"
    max_concurrent = 4

    def on_sandbox_ready(self, task, config):
        """Optional: run setup after sandbox creation."""
        _safe_exec(self.sandbox, "pip install some-package")

    def run(self, task, config):
        from openai import OpenAI
        client = OpenAI(base_url=config.base_url, api_key="not-needed")

        messages = [
            {"role": "system", "content": "You are a helpful assistant..."},
            {"role": "user", "content": task["question"]},
        ]

        steps, messages, final = self.run_tool_loop(
            client, config.model, messages,
            tools=[BashTool()],
            sandbox=self.sandbox,
            max_turns=30,
        )

        return Episode(task=task, trajectories=[Trajectory(steps=steps)])
```

### Key Concepts

- **`create_instance()`** — EvalRunner calls this to create a per-task shallow copy, so each parallel task gets its own sandbox
- **`get_image(task)`** — Override for per-task images (e.g., SWE-bench builds a unique Docker image per instance)
- **`on_sandbox_ready(task, config)`** — Hook for post-creation setup (e.g., git reset)
- **`episode.artifacts["_sandbox"]`** — EvalRunner stores the sandbox reference here so evaluators can access it

### Available Tools

| Tool | Function Name | Description |
|------|--------------|-------------|
| `BashTool` | `bash` | Execute shell commands (output truncated at 16K chars) |
| `FileEditorTool` | `str_replace_editor` | View, create, str_replace, insert file operations |
| `SubmitTool` | `submit` | Signal completion and capture git diff |

## Plugin Structure

Plugins are discovered via Python entry points:

```toml
# pyproject.toml
[project.entry-points."rllm.agents"]
my_agent = "my_plugin.agent:my_agent_instance"

[project.entry-points."rllm.evaluators"]
my_reward_fn = "my_plugin.evaluator:MyEvaluator"
```

## File Layout

```
rllm/experimental/agents/
├── sandboxed_agent.py          # SandboxedAgentFlow base + create_sandbox() factory
├── tool_calling.py             # ToolCallingMixin with run_tool_loop()
└── tools/
    ├── __init__.py
    ├── bash_tool.py            # BashTool
    ├── file_editor_tool.py     # FileEditorTool
    └── submit_tool.py          # SubmitTool

rllm/sdk/sandbox/backends/
├── docker.py                   # DockerSandbox
├── local.py                    # LocalSandbox
└── modal_backend.py            # ModalSandbox (fully implemented)

agenthub/
├── swe_agent/                  # SWE-bench agent (Docker + Modal)
└── terminal_agent/             # Terminal-Bench agent (stub)
```

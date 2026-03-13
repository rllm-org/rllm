"""``rllm init`` — scaffold a new agent project.

Creates a ready-to-use project directory with:
- An agent module (AgentFlow implementation)
- An optional custom evaluator
- A pyproject.toml with rllm plugin entry points
- A README with usage instructions
"""

from __future__ import annotations

import os
import re

import click
from rich.console import Console
from rich.panel import Panel

console = Console()

# Template directory (sibling to this module)
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Available agent templates
TEMPLATES = {
    "react": {
        "file": "react_agent.py.tpl",
        "label": "Multi-turn ReAct agent with tool use",
        "extra_deps": "",
    },
}


def _slugify(name: str) -> str:
    """Convert a project name to a valid Python module name."""
    slug = re.sub(r"[^a-zA-Z0-9]", "_", name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if slug and slug[0].isdigit():
        slug = "_" + slug
    return slug or "my_agent"


def _to_class_name(slug: str) -> str:
    """Convert a slug to PascalCase class name ending in 'Agent'."""
    pascal = "".join(part.capitalize() for part in slug.split("_"))
    if not pascal.endswith("Agent"):
        pascal += "Agent"
    return pascal


def _render_template(template_path: str, variables: dict[str, str]) -> str:
    """Simple mustache-style template rendering using {{var}} placeholders."""
    with open(template_path, encoding="utf-8") as f:
        content = f.read()
    for key, value in variables.items():
        content = content.replace("{{" + key + "}}", value)
    return content


def _write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@click.command("init")
@click.argument("project_name", required=False)
@click.option(
    "--template",
    "-t",
    type=click.Choice(list(TEMPLATES.keys())),
    help="Agent template to use.",
)
@click.option("--evaluator", is_flag=True, default=False, help="Include a custom evaluator.")
@click.option("--output-dir", "-o", type=click.Path(), help="Parent directory (default: current dir).")
def init_cmd(project_name: str | None, template: str | None, evaluator: bool, output_dir: str | None):
    """Scaffold a new rllm agent project.

    Creates a project directory with an agent module, pyproject.toml, and
    optional custom evaluator. The generated project can be installed with
    ``pip install -e .`` and immediately used with ``rllm eval``.

    \b
    Examples:
        rllm init my-math-agent
        rllm init my-agent --template langchain --evaluator
        rllm init my-agent -t openai-agents
    """
    console.print()

    # Track whether we're in interactive mode (args not fully provided)
    was_interactive = not project_name or not template

    # Interactive prompts if not provided via flags
    if not project_name:
        project_name = click.prompt("Project name", default="my-agent")

    if not template:
        # Default to the only available template
        template = list(TEMPLATES.keys())[0]

    # Only prompt if running interactively (template was also prompted)
    if not evaluator and was_interactive:
        evaluator = click.confirm("Include a custom evaluator?", default=False)

    # Derive names
    module_name = _slugify(project_name)
    agent_class = _to_class_name(module_name)
    # Derive instance name: "test_math_agent" -> "test_math_agent"
    # If module already ends with "_agent", use as-is; otherwise append
    if module_name.endswith("_agent"):
        agent_instance = module_name
    else:
        agent_instance = module_name + "_agent"
    # TestMathAgent -> TestMathEvaluator (not TestMathEvaluatorEvaluator)
    if agent_class.endswith("Agent"):
        evaluator_class = agent_class[: -len("Agent")] + "Evaluator"
    else:
        evaluator_class = agent_class + "Evaluator"

    tpl_info = TEMPLATES[template]
    variables = {
        "project_name": project_name,
        "module_name": module_name,
        "agent_name": module_name.replace("_", "-"),
        "agent_class": agent_class,
        "agent_instance": agent_instance,
        "agent_class_name": agent_instance,
        "evaluator_class": evaluator_class,
        "description": f"Custom rllm agent: {project_name}",
        "extra_deps": tpl_info["extra_deps"],
    }

    # Output paths
    base_dir = output_dir or os.getcwd()
    project_dir = os.path.join(base_dir, project_name)

    if os.path.exists(project_dir):
        if not click.confirm(f"Directory '{project_name}' already exists. Overwrite?", default=False):
            raise SystemExit(1)

    # Render and write files
    pkg_dir = os.path.join(project_dir, module_name)

    # __init__.py
    init_content = _render_template(os.path.join(_TEMPLATE_DIR, "init_py.tpl"), variables)
    _write_file(os.path.join(pkg_dir, "__init__.py"), init_content)

    # agent.py
    agent_content = _render_template(os.path.join(_TEMPLATE_DIR, tpl_info["file"]), variables)
    _write_file(os.path.join(pkg_dir, "agent.py"), agent_content)

    # evaluator.py (optional)
    if evaluator:
        eval_content = _render_template(os.path.join(_TEMPLATE_DIR, "custom_evaluator.py.tpl"), variables)
        _write_file(os.path.join(pkg_dir, "evaluator.py"), eval_content)

    # pyproject.toml
    pyproject_content = _render_template(os.path.join(_TEMPLATE_DIR, "pyproject.toml.tpl"), variables)
    if evaluator:
        # Uncomment the evaluator entry point
        pyproject_content = pyproject_content.replace(
            '# [project.entry-points."rllm.evaluators"]',
            '[project.entry-points."rllm.evaluators"]',
        )
        pyproject_content = pyproject_content.replace(
            f'# {variables["agent_name"]} = "{module_name}.evaluator:{evaluator_class}"',
            f'{variables["agent_name"]} = "{module_name}.evaluator:{evaluator_class}"',
        )
    _write_file(os.path.join(project_dir, "pyproject.toml"), pyproject_content)

    # Print summary
    console.print()
    console.print(
        Panel(
            f"[bold green]Project created:[/bold green] {project_dir}\n\n[bold]Template:[/bold] {tpl_info['label']}\n[bold]Agent:[/bold] {module_name}.agent:{agent_instance}\n" + (f"[bold]Evaluator:[/bold] {module_name}.evaluator:{evaluator_class}\n" if evaluator else ""),
            title="[bold cyan]rllm init[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print("[bold]Next steps:[/bold]\n")
    console.print(f"  [cyan]cd {project_name}[/cyan]")
    console.print("  [cyan]pip install -e .[/cyan]")
    console.print()
    console.print("  [dim]# Edit your agent logic:[/dim]")
    console.print(f"  [cyan]$EDITOR {module_name}/agent.py[/cyan]")
    console.print()
    console.print("  [dim]# One-time setup (configure provider, API key, default model):[/dim]")
    console.print("  [cyan]rllm model setup[/cyan]")
    console.print()
    console.print("  [dim]# Run eval (agent is auto-discovered via entry point):[/dim]")
    console.print(f"  [cyan]rllm eval gsm8k --agent {variables['agent_name']}[/cyan]")
    console.print()
    console.print("  [dim]# Or use import path directly (no install needed):[/dim]")
    console.print(f"  [cyan]rllm eval gsm8k --agent {module_name}.agent:{agent_instance}[/cyan]")
    if evaluator:
        console.print()
        console.print("  [dim]# Use your custom evaluator:[/dim]")
        console.print(f"  [cyan]rllm eval gsm8k --evaluator {module_name}.evaluator:{evaluator_class}[/cyan]")
    console.print()

"""Eval launcher panel — placeholder for kicking off eval runs from the UI."""

from rllm.console.panels import Panel, register_panel

register_panel(
    Panel(
        id="eval_launcher",
        title="Eval Launcher",
        icon="play",
        nav_order=40,
        placeholder=True,
    )
)

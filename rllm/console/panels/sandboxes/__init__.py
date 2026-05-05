"""Sandboxes panel — placeholder for live sandbox orchestration view."""

from rllm.console.panels import Panel, register_panel

register_panel(
    Panel(
        id="sandboxes",
        title="Sandboxes",
        icon="box",
        nav_order=30,
        placeholder=True,
    )
)

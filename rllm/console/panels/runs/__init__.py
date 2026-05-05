"""Runs panel — eval-results filesystem browser."""

from rllm.console.panels import Panel, register_panel
from rllm.console.panels.runs.api import router

register_panel(
    Panel(
        id="runs",
        title="Runs",
        icon="list",
        nav_order=20,
        router=router,
    )
)

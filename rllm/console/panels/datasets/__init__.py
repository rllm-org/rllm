"""Datasets panel — registry browser + entry inspector."""

from rllm.console.panels import Panel, register_panel
from rllm.console.panels.datasets.api import router

register_panel(
    Panel(
        id="datasets",
        title="Datasets",
        icon="database",
        nav_order=5,
        router=router,
    )
)

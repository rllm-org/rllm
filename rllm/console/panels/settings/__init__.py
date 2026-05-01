"""Settings panel — config viewer + env-var manager."""

from rllm.console.panels import Panel, register_panel
from rllm.console.panels.settings.api import router

register_panel(
    Panel(
        id="settings",
        title="Settings",
        icon="settings",
        nav_order=900,
        router=router,
    )
)

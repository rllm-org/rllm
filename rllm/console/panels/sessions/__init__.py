"""Sessions panel — cross-run gateway tracing browser."""

from rllm.console.panels import Panel, register_panel
from rllm.console.panels.sessions.api import router

register_panel(
    Panel(
        id="sessions",
        title="Sessions",
        icon="activity",
        nav_order=10,
        router=router,
    )
)

"""Training panel — placeholder for future training-run integration."""

from rllm.console.panels import Panel, register_panel

register_panel(
    Panel(
        id="training",
        title="Training",
        icon="graduation-cap",
        nav_order=50,
        placeholder=True,
    )
)

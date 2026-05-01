"""rLLM Console — operator UI for the gateway.

The console is mounted onto an existing FastAPI app (typically the model
gateway) by calling :func:`mount_console`. It contributes:

* ``GET  {prefix}/api/shell/info``         — list of registered panels
* ``GET  {prefix}/api/panels/<id>/...``    — per-panel data routes
* ``GET  {prefix}/``                       — built SPA shell (when frontend is built)
* ``GET  {prefix}/assets/...``             — built JS/CSS/font assets
* ``GET  {prefix}/{path:path}``            — SPA catch-all for client-side routing

Panels self-register at module import time via :func:`register_panel`.
"""

from rllm.console.app import mount_console
from rllm.console.panels import Panel, list_panels, register_panel

__all__ = ["mount_console", "Panel", "register_panel", "list_panels"]

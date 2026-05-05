"""Panel protocol + global registry.

Panels are declared as :class:`Panel` instances and added to the registry
via :func:`register_panel`. The default panels self-register at import
time (see ``rllm.console.app._import_default_panels``); third-party
extensions can add more by importing the module and calling
``register_panel`` before :func:`mount_console` runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter


@dataclass
class Panel:
    """Declarative description of a console panel.

    Args:
        id: Stable URL slug (``/console/api/panels/<id>/...``).
        title: Human-readable label shown in the sidebar.
        icon: ``lucide-react`` icon name. Falls back to "circle" client-side.
        nav_order: Sort key for sidebar ordering (lower first).
        router: APIRouter mounted under ``/console/api/panels/<id>``. May
            be ``None`` for placeholder panels with no backend yet.
        placeholder: When True, the frontend renders a "coming soon"
            stub regardless of any router.
    """

    id: str
    title: str
    icon: str = "circle"
    nav_order: int = 100
    router: APIRouter | None = None
    placeholder: bool = False


_PANELS: list[Panel] = []


def register_panel(panel: Panel) -> None:
    """Add ``panel`` to the global registry.

    Re-registering the same ``id`` replaces the previous entry. This makes
    repeated imports (test runners, hot reload) safe.
    """
    for i, existing in enumerate(_PANELS):
        if existing.id == panel.id:
            _PANELS[i] = panel
            return
    _PANELS.append(panel)


def list_panels() -> list[Panel]:
    """Return registered panels sorted by ``(nav_order, id)``."""
    return sorted(_PANELS, key=lambda p: (p.nav_order, p.id))


def _reset_panels() -> None:
    """Clear the registry. Test-only escape hatch."""
    _PANELS.clear()


__all__ = ["Panel", "register_panel", "list_panels"]

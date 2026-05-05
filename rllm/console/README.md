# rllm.console

Backend for the rLLM Console — the operator UI mounted onto the model gateway.

This package contributes:

- A `mount_console(app, ...)` helper that attaches the console to any FastAPI app (typically `rllm-model-gateway`).
- A panel-registry contract — each panel is a `Panel` dataclass with an `APIRouter` and a slot in the sidebar.
- The built-in panels: Sessions, Runs, Datasets, Settings (functional) plus Sandboxes, Eval Launcher, Training (placeholders).

The frontend lives in [`rllm-console/`](../../rllm-console/) and is bundled into `rllm/console/static/` at build time. End users get the full UI by running `rllm view`.

## Mounting

```python
from fastapi import FastAPI
from rllm.console import mount_console

app = FastAPI()
mount_console(
    app,
    eval_results_root="~/.rllm/eval_results",
    url_prefix="/console",   # default
    version="0.1.0",
)
```

After mounting, the app exposes:

| Path | Purpose |
|------|---------|
| `GET  {prefix}/api/shell/info` | Sidebar metadata — list of registered panels |
| `GET  {prefix}/api/panels/<id>/...` | Per-panel data routes |
| `GET  {prefix}/` | Built SPA shell (when `static/` is present) |
| `GET  {prefix}/assets/...` | Built JS/CSS/font bundles |
| `GET  {prefix}/{path:path}` | SPA catch-all for client-side routing |

`/console/api/...` always wins over the SPA catch-all, so a panel-route 404 surfaces as JSON, not HTML.

The standard launcher is `rllm/cli/view.py` (`rllm view`), which boots a view-only gateway (TraceStore + console; no proxy workers).

## Adding a panel

A panel is two pieces: an `APIRouter` for backend data and a React component for the frontend. The panel's `id` glues them together.

### Backend (this package)

```python
# rllm/console/panels/myfeat/__init__.py
from rllm.console.panels import Panel, register_panel
from rllm.console.panels.myfeat.api import router

register_panel(
    Panel(
        id="myfeat",
        title="My Feature",
        icon="star",         # any lucide-react icon name
        nav_order=50,        # sidebar sort key
        router=router,       # mounted at /console/api/panels/myfeat
    )
)
```

```python
# rllm/console/panels/myfeat/api.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/things")
def list_things() -> list[dict]:
    return [{"id": "a"}, {"id": "b"}]
```

Register the module in `rllm/console/app.py:_DEFAULT_PANEL_MODULES` so it auto-loads on `mount_console`. Third-party extensions can skip the default-list edit and instead `import` their module + call `register_panel` before `mount_console` runs.

### Frontend (rllm-console/)

See [`rllm-console/README.md`](../../rllm-console/README.md). The panel `id` must match exactly between the backend `Panel(id=...)` and the frontend `panels/registry.tsx` entry.

### Placeholder panels

Set `placeholder=True` and omit `router` to render a "coming soon" stub in the sidebar without a backend yet — useful for advertising upcoming features. See `panels/eval_launcher/__init__.py` for the canonical example.

## Reading data sources

Panels typically read from one of:

| Source | Used by | Notes |
|--------|---------|-------|
| Gateway `TraceStore` | Sessions | Cross-run trace feed; reads via the gateway's app state |
| `~/.rllm/eval_results/<run>/` | Runs | One subdir per eval run; episodes + scores |
| User dataset registry | Datasets | Browses + pulls from `~/.rllm/datasets/registry.json` |
| Process env / config | Settings | Env-var manager, gateway config viewer |

`app.state.console_eval_results_root` is set at mount time so panel routes can read it without closing over args:

```python
from fastapi import Request

@router.get("/runs")
def list_runs(request: Request):
    root = request.app.state.console_eval_results_root
    if root is None:
        return []
    ...
```

## Testing

Each panel has a co-located test under `tests/console/test_panels_<id>.py` that mounts a minimal FastAPI app + the panel router and asserts response shapes. Run:

```bash
uv run pytest tests/console -x -q
```

The panel registry is module-global; tests should call `rllm.console.panels._reset_panels()` between tests if they manipulate registration.

## Layout

```
rllm/console/
├── __init__.py            # Public API (mount_console, Panel, register_panel)
├── app.py                 # mount_console — shell-info route, panel router mounting, SPA static
├── static/                # Built SPA bundle (gitignored; populated by `rllm view --build`)
└── panels/
    ├── __init__.py        # Panel dataclass + global registry
    ├── sessions/          # Cross-run trace feed (gateway TraceStore)
    ├── runs/              # Eval-run grid + episode viewer
    ├── datasets/          # Registry browser + pull-from-UI
    ├── settings/          # Config viewer + env-var manager
    ├── eval_launcher/     # Placeholder
    ├── sandboxes/         # Placeholder
    └── training/          # Placeholder
```

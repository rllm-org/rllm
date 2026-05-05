# rllm-console

Frontend source for the rLLM Console — the operator UI mounted onto the
model gateway by [`rllm.console.mount_console`](../rllm/console/app.py).

End users never need to install anything from this directory: the
release wheel ships pre-built static assets in `rllm/console/static/`.
This source tree is for hacking on the UI itself.

## Stack

- bun (package manager)
- Vite 6 + React 19 + TypeScript strict
- React Router 7 (SPA mode, no SSR)
- Tailwind CSS v4 (`@tailwindcss/vite`)
- Radix UI primitives + lucide-react icons
- TanStack Query 5 for server state

## Dev

```bash
# Terminal 1 — backend (any rllm command that boots the gateway, e.g.):
rllm view ~/.rllm/eval_results --port 7860 --no-build

# Terminal 2 — frontend with hot reload, proxying /console/api/* to :7860:
cd rllm-console
bun install
bun run dev   # opens http://localhost:5173
```

Override the proxy target with `VITE_API_URL=http://host:port`.

## Build

```bash
bun run build
# Output: build/client/   →  copy to rllm/console/static/
```

`rllm view --build` does that copy automatically.

## Layout

```
app/
  root.tsx             # provider tree + error boundary
  routes.ts            # route table (RR7 framework mode)
  routes/
    home.tsx           # redirects to first panel
    panel.tsx          # shell + panel mount point
  components/
    Layout.tsx         # sidebar + main outlet
    Sidebar.tsx
  lib/
    api.ts             # typed fetch helpers (mirrors backend Pydantic)
    format.ts          # timeAgo, compactNumber, percent
    icons.ts           # backend icon-name → lucide component
    utils.ts           # cn()
  panels/
    registry.tsx       # id → React component map
    Placeholder.tsx    # fallback for unimplemented panels
    sessions/index.tsx
    runs/index.tsx
```

## Adding a panel

1. Create `app/panels/<id>/index.tsx` exporting a default React component.
2. Wire it in `panels/registry.tsx`.
3. Register the matching backend `Panel(id="<id>", ...)` in
   `rllm/console/panels/<id>/__init__.py` so it shows up in the sidebar.

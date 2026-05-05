"""HTTP-based dashboard for rLLM eval runs.

``rllm view`` boots a stdlib :mod:`http.server` rooted at
``~/.rllm/eval_results/`` and serves a single Tailwind-styled SPA.

Two views, connected by hash routing:

* **Run list** (``#/``) — table of every run in the directory, with
  benchmark, model, agent, score (color-coded), correct/total, errors,
  status, and created-at. Mirrors the rllm-ui flat-table aesthetic.

* **Run detail** (``#/run/<id>``) — the original per-run drill-down:
  hierarchical Task → Trajectory → Step explorer with collapsible field
  boxes for input/output/action/metadata.

Episode bodies are lazy-fetched from the static file server, so very
large eval folders stay responsive.
"""

from __future__ import annotations

import http.server
import json
import re
import socketserver
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Filesystem layer                                                            #
# --------------------------------------------------------------------------- #


_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


def _resolve_episodes_dir(run_dir: Path) -> Path:
    candidate = run_dir / "episodes"
    if candidate.is_dir():
        return candidate
    return run_dir


def _load_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _parse_run_timestamp(run_id: str) -> str | None:
    """Pull a ``YYYYMMDD_HHMMSS`` suffix off ``run_id`` and ISO-format it."""
    m = _TIMESTAMP_RE.search(run_id)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def _scan_runs(root: Path) -> list[dict[str, Any]]:
    """Discover all eval runs under ``root``.

    A run is a subdirectory with an ``episodes/`` folder. The aggregate
    :class:`EvalResult` JSON sits next to it as ``<run_id>.json``.
    """
    if not root.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        episodes_dir = run_dir / "episodes"
        if not episodes_dir.is_dir():
            continue

        run_id = run_dir.name
        meta = _load_meta(run_dir)

        # Aggregate result JSON: prefer ``<run_dir>/results.json`` (the
        # current layout), fall back to the legacy sibling
        # ``<root>/<run_id>.json`` so older runs still scan correctly.
        agg_path = run_dir / "results.json"
        if not agg_path.is_file():
            legacy = root / f"{run_id}.json"
            agg_path = legacy if legacy.is_file() else agg_path
        agg: dict[str, Any] = {}
        if agg_path.is_file():
            try:
                with open(agg_path, encoding="utf-8") as f:
                    agg = json.load(f)
            except Exception:
                agg = {}

        n_episodes = sum(1 for _ in episodes_dir.glob("episode_*.json"))

        # Status: completed if aggregate exists, else "incomplete".
        status = "completed" if agg else "incomplete"

        created_at = _parse_run_timestamp(run_id)
        if created_at is None:
            try:
                created_at = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                created_at = None

        ts_match = _TIMESTAMP_RE.search(run_id)
        out.append(
            {
                "id": run_id,
                "benchmark": agg.get("dataset_name") or meta.get("benchmark") or "—",
                "model": agg.get("model") or meta.get("model") or "—",
                "agent": agg.get("agent") or meta.get("agent") or "—",
                "split": meta.get("split") or "",
                "timestamp": meta.get("timestamp") or (ts_match.group(1) if ts_match else ""),
                "created_at": created_at,
                "score": agg.get("score"),
                "correct": agg.get("correct"),
                "total": agg.get("total"),
                "errors": agg.get("errors"),
                "n_episodes": n_episodes,
                "status": status,
            }
        )

    # Most-recent first.
    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return out


def _build_episode_index(episodes_dir: Path) -> list[dict[str, Any]]:
    """Read just the headline fields from every episode file in a run."""
    index = []
    for path in sorted(episodes_dir.glob("episode_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        task = data.get("task") if isinstance(data.get("task"), dict) else {}
        n_steps = sum(len(t.get("steps") or []) for t in (data.get("trajectories") or []))
        rewards = [t.get("reward") for t in (data.get("trajectories") or []) if t.get("reward") is not None]
        avg_reward = sum(rewards) / len(rewards) if rewards else None
        index.append(
            {
                "filename": path.name,
                "eval_idx": data.get("eval_idx"),
                "task_id": task.get("id") if isinstance(task, dict) else None,
                "is_correct": data.get("is_correct"),
                "termination_reason": data.get("termination_reason"),
                "n_trajectories": len(data.get("trajectories") or []),
                "n_steps": n_steps,
                "reward": avg_reward,
                "instruction_preview": _preview(task.get("instruction") if isinstance(task, dict) else None),
            }
        )
    return index


def _preview(value: Any, limit: int = 140) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s[: limit - 1] + "…" if len(s) > limit else s


# --------------------------------------------------------------------------- #
# HTML / JS                                                                   #
# --------------------------------------------------------------------------- #


_PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    theme: {
      extend: {
        colors: {
          layer:  { 1:'#fafbfc', 2:'#f6f7f9', 3:'#f1f3f5' },
          accent: { 50:'#eef4fb',100:'#d6e4f3',200:'#b5cde5',300:'#8bb2d4',400:'#6594c0',500:'#3f72af',600:'#345f94',700:'#2a4e7a' },
        },
        fontFamily: {
          sans: ['"DM Sans"','-apple-system','BlinkMacSystemFont','"Segoe UI"','sans-serif'],
          mono: ['"IBM Plex Mono"','"SF Mono"','Monaco','monospace'],
        },
        boxShadow: {
          subtle: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
          card:   '0 1px 3px 0 rgb(0 0 0 / 0.08), 0 1px 2px -1px rgb(0 0 0 / 0.06)',
        },
      },
    },
  };
</script>
<style>
  :root { --color-bg:#fafbfc; }
  html, body { background:var(--color-bg); }
  body {
    font-family:'DM Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    font-size:14px;
    color:#111827;
    -webkit-font-smoothing:antialiased;
  }
  pre, code, .mono { font-family:'IBM Plex Mono','SF Mono',Monaco,monospace; }
  pre { white-space:pre-wrap; word-break:break-word; }
  .field-box { max-height:12rem; overflow:auto; transition:max-height .15s ease; }
  .field-box.expanded { max-height:60vh; }
  ::-webkit-scrollbar { width:8px; height:8px; }
  ::-webkit-scrollbar-thumb { background:#d1d5db; border-radius:4px; }
  ::-webkit-scrollbar-thumb:hover { background:#9ca3af; }
  details > summary { list-style:none; cursor:pointer; }
  details > summary::-webkit-details-marker { display:none; }
  details > summary .chev { transition:transform .15s ease; display:inline-block; }
  details[open] > summary .chev { transform:rotate(90deg); }

  /* Subtle staggered entrance for table rows (run list) */
  .row-in { animation: rowIn .25s cubic-bezier(0.2,0.7,0.2,1) backwards; }
  @keyframes rowIn { from { opacity:0; transform: translateY(2px); } to { opacity:1; transform:none; } }

  .pulse-dot { position:relative; }
  .pulse-dot::after {
    content:''; position:absolute; inset:0; border-radius:9999px; background:inherit;
    animation: ping 1.6s cubic-bezier(0,0,.2,1) infinite;
  }
  @keyframes ping { 75%,100% { transform: scale(2.4); opacity:0; } }
</style>
</head>
<body>

<!-- ─────────────────────────────  Header  ───────────────────────────── -->
<header class="bg-white border-b border-gray-200 sticky top-0 z-30">
  <div class="max-w-7xl mx-auto px-6 h-14 flex items-center gap-4">
    <a href="#/" class="flex items-center gap-2 group">
      <span class="inline-flex items-center justify-center w-7 h-7 rounded-md bg-accent-500 text-white font-semibold text-[12px] tracking-tight group-hover:bg-accent-600 transition-colors">rL</span>
      <span class="font-semibold text-gray-900 tracking-tight">rLLM<span class="text-gray-400 font-normal"> · viewer</span></span>
    </a>
    <nav id="breadcrumb" class="flex items-center gap-2 text-sm text-gray-500 min-w-0"></nav>
    <div class="ml-auto flex items-center gap-3 text-xs text-gray-400">
      <span class="mono truncate max-w-[28rem]" title="__ROOT_PATH__">__ROOT_PATH__</span>
    </div>
  </div>
</header>

<main id="root" class="max-w-7xl mx-auto px-6 py-6"></main>

<!-- ─────────────────────────────  Templates  ──────────────────────────── -->

<template id="tpl-list">
  <section>
    <div class="flex items-center justify-between mb-5">
      <div>
        <h1 class="text-xl font-semibold text-gray-900 tracking-tight">Evaluation runs</h1>
        <p class="text-xs text-gray-500 mt-0.5"><span id="run-count" class="font-medium text-gray-700">0</span> runs · sorted by most recent</p>
      </div>
      <div class="flex items-center gap-2">
        <div class="relative">
          <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-4.35-4.35M11 19a8 8 0 100-16 8 8 0 000 16z"/></svg>
          <input id="run-search" type="text" placeholder="Search runs…"
                 class="pl-9 pr-3 py-1.5 w-64 border border-gray-200 rounded-md text-xs bg-white focus:outline-none focus:border-accent-400 focus:ring-2 focus:ring-accent-100">
        </div>
      </div>
    </div>

    <div id="run-empty" class="hidden bg-white border border-dashed border-gray-300 rounded-xl p-16 text-center">
      <div class="mx-auto w-12 h-12 rounded-full bg-layer-2 flex items-center justify-center mb-4">
        <svg class="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
      </div>
      <h3 class="text-sm font-semibold text-gray-900 mb-1">No evaluation runs yet</h3>
      <p class="text-xs text-gray-500">Run <code class="mono px-1.5 py-0.5 bg-layer-2 border border-gray-200 rounded text-[11px]">rllm eval &lt;benchmark&gt;</code> to populate this directory.</p>
    </div>

    <div id="run-table-wrap" class="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-subtle">
      <table class="w-full">
        <thead class="bg-layer-2 border-b border-gray-200">
          <tr class="text-left text-[11px] uppercase tracking-wider text-gray-500 select-none">
            <th class="px-4 py-3 font-medium cursor-pointer hover:text-gray-700" data-sort="benchmark">Benchmark <span class="sort-arrow"></span></th>
            <th class="px-4 py-3 font-medium cursor-pointer hover:text-gray-700" data-sort="model">Model <span class="sort-arrow"></span></th>
            <th class="px-4 py-3 font-medium cursor-pointer hover:text-gray-700" data-sort="agent">Agent <span class="sort-arrow"></span></th>
            <th class="px-4 py-3 font-medium cursor-pointer hover:text-gray-700 text-right" data-sort="score">Score <span class="sort-arrow"></span></th>
            <th class="px-4 py-3 font-medium text-right">Correct&thinsp;/&thinsp;Total</th>
            <th class="px-4 py-3 font-medium text-right cursor-pointer hover:text-gray-700" data-sort="errors">Errors <span class="sort-arrow"></span></th>
            <th class="px-4 py-3 font-medium" data-sort="status">Status <span class="sort-arrow"></span></th>
            <th class="px-4 py-3 font-medium cursor-pointer hover:text-gray-700" data-sort="created_at">Created <span class="sort-arrow"></span></th>
          </tr>
        </thead>
        <tbody id="run-rows" class="divide-y divide-gray-100"></tbody>
      </table>
    </div>
  </section>
</template>

<template id="tpl-run">
  <section>
    <!-- Run-level summary card -->
    <div class="bg-white border border-gray-200 rounded-xl p-5 mb-5 shadow-subtle">
      <div class="flex items-start justify-between gap-6">
        <div class="min-w-0 flex-1">
          <div class="flex items-center gap-2 mb-1 text-xs text-gray-500">
            <a href="#/" class="hover:text-accent-600 transition-colors flex items-center gap-1">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M15 19l-7-7 7-7"/></svg>
              All runs
            </a>
            <span class="text-gray-300">/</span>
            <span class="mono text-gray-600 truncate" id="run-id">—</span>
          </div>
          <h1 class="text-lg font-semibold text-gray-900 tracking-tight" id="run-title">—</h1>
          <div class="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-xs text-gray-500" id="run-meta-row"></div>
        </div>
        <div class="flex items-center gap-4 shrink-0">
          <div class="text-right">
            <div class="text-[11px] uppercase tracking-wider text-gray-400 font-medium">Score</div>
            <div id="run-score" class="text-3xl font-semibold tabular-nums leading-tight">—</div>
            <div id="run-score-sub" class="text-[11px] text-gray-500 tabular-nums"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Episode toolbar -->
    <div class="flex items-center gap-3 mb-3">
      <div class="inline-flex rounded-md border border-gray-200 bg-white overflow-hidden text-xs shadow-subtle">
        <button data-filter="all"       class="filter-btn px-3 py-1.5 font-medium bg-accent-50 text-accent-700">All</button>
        <button data-filter="correct"   class="filter-btn px-3 py-1.5 font-medium border-l border-gray-200 text-gray-600 hover:bg-gray-50">✓ Correct</button>
        <button data-filter="incorrect" class="filter-btn px-3 py-1.5 font-medium border-l border-gray-200 text-gray-600 hover:bg-gray-50">✗ Incorrect</button>
      </div>
      <input id="ep-search" placeholder="Filter by task id or instruction…"
             class="flex-1 px-3 py-1.5 border border-gray-200 rounded-md text-xs focus:outline-none focus:border-accent-400 focus:ring-2 focus:ring-accent-100">
      <span id="ep-count" class="text-xs text-gray-500 tabular-nums shrink-0"></span>
      <button id="expand-all"   class="text-xs px-2 py-1 rounded hover:bg-gray-100 text-gray-600 transition-colors">Expand all</button>
      <button id="collapse-all" class="text-xs px-2 py-1 rounded hover:bg-gray-100 text-gray-600 transition-colors">Collapse all</button>
    </div>

    <div id="ep-list" class="space-y-2"></div>
  </section>
</template>

<!-- ─────────────────────────────  Script  ─────────────────────────────── -->
<script>
const RUNS_DATA = __RUNS_JSON__;        // server-rendered list
const PRESELECT_RUN = __PRESELECT__;    // optional initial #/run/<id>
const ROOT_PATH = __ROOT_PATH_JSON__;

const root = document.getElementById("root");
const breadcrumb = document.getElementById("breadcrumb");

// ─── Helpers ────────────────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, ch =>
    ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[ch]));
}
function formatValue(v) {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try { return JSON.stringify(v, null, 2); } catch (e) { return String(v); }
}
function scoreColor(s) {
  if (s == null) return "text-gray-400";
  if (s >= 0.5)  return "text-green-600";
  if (s >= 0.2)  return "text-amber-600";
  return "text-red-600";
}
function fmtPct(s) { return s == null ? "—" : (s * 100).toFixed(1) + "%"; }
function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  const date = d.toLocaleDateString("en-US", { month:"short", day:"numeric", year:"numeric" });
  const time = d.toLocaleTimeString("en-US", { hour:"2-digit", minute:"2-digit" });
  return `${date} · ${time}`;
}
function timeAgo(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  const diffMs = Date.now() - d.getTime();
  const m = Math.floor(diffMs / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const days = Math.floor(h / 24);
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString("en-US", { month:"short", day:"numeric" });
}
function setBreadcrumb(parts) {
  if (!parts.length) { breadcrumb.innerHTML = ""; return; }
  breadcrumb.innerHTML = parts.map((p, i) => {
    const sep = i ? `<span class="text-gray-300">/</span>` : "";
    if (p.href) return `${sep}<a href="${p.href}" class="hover:text-accent-600 transition-colors truncate">${escapeHtml(p.label)}</a>`;
    return `${sep}<span class="text-gray-700 truncate">${escapeHtml(p.label)}</span>`;
  }).join(" ");
}
function statusBadge(status) {
  const cfg = ({
    completed:  { label:"Completed",  bg:"bg-gray-100",   text:"text-gray-600",   dot:"bg-gray-400",   pulse:false },
    incomplete: { label:"Incomplete", bg:"bg-amber-100",  text:"text-amber-700",  dot:"bg-amber-500",  pulse:true  },
    running:    { label:"Running",    bg:"bg-green-100",  text:"text-green-700",  dot:"bg-green-500",  pulse:true  },
    failed:     { label:"Failed",     bg:"bg-red-100",    text:"text-red-700",    dot:"bg-red-500",    pulse:false },
  })[status] || { label:status||"—", bg:"bg-gray-100", text:"text-gray-600", dot:"bg-gray-400", pulse:false };
  return `<span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium ${cfg.bg} ${cfg.text}">
    <span class="${cfg.pulse ? 'pulse-dot' : ''} inline-block w-1.5 h-1.5 rounded-full ${cfg.dot}"></span>${cfg.label}</span>`;
}

// ─── Router ─────────────────────────────────────────────────────────────
function parseHash() {
  const h = location.hash.replace(/^#/, "") || "/";
  const m = h.match(/^\/run\/(.+)$/);
  if (m) return { view:"run", id: decodeURIComponent(m[1]) };
  return { view:"list" };
}
function navigate(hash) { location.hash = hash; }
window.addEventListener("hashchange", route);

// ─── List view ──────────────────────────────────────────────────────────
let listSort = { field: "created_at", dir: "desc" };
let listSearch = "";

function renderListView() {
  setBreadcrumb([{ label: "Evaluation runs" }]);
  root.innerHTML = "";
  root.appendChild(document.getElementById("tpl-list").content.cloneNode(true));

  document.getElementById("run-count").textContent = RUNS_DATA.length;

  const search = document.getElementById("run-search");
  search.addEventListener("input", e => { listSearch = e.target.value.toLowerCase(); paintRunRows(); });

  for (const th of document.querySelectorAll("[data-sort]")) {
    th.addEventListener("click", () => {
      const f = th.dataset.sort;
      if (listSort.field === f) listSort.dir = listSort.dir === "asc" ? "desc" : "asc";
      else listSort = { field: f, dir: f === "score" || f === "created_at" ? "desc" : "asc" };
      paintRunRows();
    });
  }
  paintRunRows();
}

function paintRunRows() {
  const tbody = document.getElementById("run-rows");
  const empty = document.getElementById("run-empty");
  const wrap = document.getElementById("run-table-wrap");
  if (!tbody) return;

  const q = listSearch.trim();
  let rows = RUNS_DATA.slice();
  if (q) rows = rows.filter(r =>
    [r.benchmark, r.model, r.agent, r.id].some(v => String(v || "").toLowerCase().includes(q))
  );

  rows.sort((a, b) => {
    const f = listSort.field;
    let av = a[f], bv = b[f];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "number") return listSort.dir === "asc" ? av - bv : bv - av;
    av = String(av); bv = String(bv);
    return listSort.dir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
  });

  // Update sort arrows
  for (const th of document.querySelectorAll("[data-sort]")) {
    const arrow = th.querySelector(".sort-arrow");
    if (!arrow) continue;
    arrow.textContent = th.dataset.sort === listSort.field ? (listSort.dir === "asc" ? "↑" : "↓") : "";
    arrow.className = "sort-arrow text-accent-600";
  }

  if (RUNS_DATA.length === 0) { wrap.classList.add("hidden"); empty.classList.remove("hidden"); return; }
  wrap.classList.remove("hidden"); empty.classList.add("hidden");

  tbody.innerHTML = rows.map((r, i) => `
    <tr class="row-in hover:bg-layer-1 cursor-pointer transition-colors" style="animation-delay:${Math.min(i*15, 200)}ms"
        data-run-id="${escapeHtml(r.id)}">
      <td class="px-4 py-3 align-top">
        <div class="text-sm font-medium text-gray-900">${escapeHtml(r.benchmark)}</div>
        <div class="text-[11px] text-gray-400 mono truncate max-w-[24rem]">${escapeHtml(r.id)}</div>
      </td>
      <td class="px-4 py-3 align-top text-sm text-gray-700 mono">${escapeHtml(r.model)}</td>
      <td class="px-4 py-3 align-top text-sm text-gray-700">${escapeHtml(r.agent)}</td>
      <td class="px-4 py-3 align-top text-right">
        <span class="text-base font-semibold tabular-nums ${scoreColor(r.score)}">${fmtPct(r.score)}</span>
      </td>
      <td class="px-4 py-3 align-top text-right text-sm text-gray-500 tabular-nums">${
        r.total != null ? `${r.correct ?? 0}/${r.total}` : `<span class="text-gray-400">— / ${r.n_episodes}</span>`
      }</td>
      <td class="px-4 py-3 align-top text-right text-sm tabular-nums">${
        r.errors != null ? `<span class="${r.errors > 0 ? 'text-red-600 font-medium' : 'text-gray-400'}">${r.errors}</span>` : '<span class="text-gray-400">—</span>'
      }</td>
      <td class="px-4 py-3 align-top">${statusBadge(r.status)}</td>
      <td class="px-4 py-3 align-top text-xs text-gray-500">
        <div>${escapeHtml(fmtDate(r.created_at))}</div>
        <div class="text-gray-400">${escapeHtml(timeAgo(r.created_at))}</div>
      </td>
    </tr>
  `).join("");

  // Row click → drill
  tbody.querySelectorAll("tr[data-run-id]").forEach(tr => {
    tr.addEventListener("click", () => navigate(`#/run/${encodeURIComponent(tr.dataset.runId)}`));
  });
}

// ─── Run-detail view ────────────────────────────────────────────────────
let runState = { id:null, meta:null, index:[], filter:"all", search:"" };
const episodeCache = new Map();

async function renderRunView(runId) {
  const summary = RUNS_DATA.find(r => r.id === runId);
  setBreadcrumb([
    { label: "Evaluation runs", href: "#/" },
    { label: runId },
  ]);

  root.innerHTML = "";
  root.appendChild(document.getElementById("tpl-run").content.cloneNode(true));

  document.getElementById("run-id").textContent = runId;
  document.getElementById("run-title").textContent = summary
    ? `${summary.benchmark} · ${summary.model}`
    : runId;

  const metaRow = document.getElementById("run-meta-row");
  if (summary) {
    metaRow.innerHTML = [
      `<span><span class="text-gray-400">benchmark</span> <span class="text-gray-700 font-medium">${escapeHtml(summary.benchmark)}</span></span>`,
      `<span><span class="text-gray-400">model</span> <span class="text-gray-700 mono">${escapeHtml(summary.model)}</span></span>`,
      `<span><span class="text-gray-400">agent</span> <span class="text-gray-700">${escapeHtml(summary.agent)}</span></span>`,
      summary.split ? `<span><span class="text-gray-400">split</span> <span class="text-gray-700">${escapeHtml(summary.split)}</span></span>` : "",
      `<span>${statusBadge(summary.status)}</span>`,
      `<span class="text-gray-400">${escapeHtml(fmtDate(summary.created_at))}</span>`,
    ].filter(Boolean).join("");

    const scoreEl = document.getElementById("run-score");
    scoreEl.textContent = fmtPct(summary.score);
    scoreEl.className = `text-3xl font-semibold tabular-nums leading-tight ${scoreColor(summary.score)}`;
    if (summary.total != null) {
      document.getElementById("run-score-sub").textContent = `${summary.correct ?? 0} / ${summary.total} correct${summary.errors ? ` · ${summary.errors} errors` : ""}`;
    } else {
      document.getElementById("run-score-sub").textContent = `${summary.n_episodes} episodes`;
    }
  }

  // Wire toolbar
  for (const btn of document.querySelectorAll(".filter-btn")) {
    btn.addEventListener("click", () => {
      runState.filter = btn.dataset.filter;
      for (const b of document.querySelectorAll(".filter-btn")) {
        b.classList.remove("bg-accent-50", "text-accent-700");
        b.classList.add("text-gray-600");
      }
      btn.classList.remove("text-gray-600");
      btn.classList.add("bg-accent-50", "text-accent-700");
      paintEpisodeList();
    });
  }
  document.getElementById("ep-search").addEventListener("input", e => {
    runState.search = e.target.value;
    paintEpisodeList();
  });
  document.getElementById("expand-all").addEventListener("click", () => {
    document.querySelectorAll("#ep-list details[data-filename]").forEach(d => d.open = true);
  });
  document.getElementById("collapse-all").addEventListener("click", () => {
    document.querySelectorAll("#ep-list details[data-filename]").forEach(d => d.open = false);
  });

  // Field-box expand/collapse delegation
  document.getElementById("root").addEventListener("click", e => {
    const btn = e.target.closest(".toggle-expand");
    if (!btn) return;
    const wrapper = btn.closest("[data-field-wrap]");
    if (!wrapper) return;
    const box = wrapper.querySelector(".field-box");
    if (!box) return;
    box.classList.toggle("expanded");
    btn.textContent = box.classList.contains("expanded") ? "collapse" : "expand";
  });

  // Fetch episode index for this run
  runState.id = runId;
  document.getElementById("ep-list").innerHTML = `<div class="text-sm text-gray-400 py-12 text-center">Loading episodes…</div>`;
  try {
    const resp = await fetch(`/api/runs/${encodeURIComponent(runId)}/index`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    runState.index = await resp.json();
  } catch (e) {
    document.getElementById("ep-list").innerHTML = `<div class="text-sm text-red-600 py-12 text-center">Failed to load episode index: ${escapeHtml(String(e))}</div>`;
    return;
  }
  paintEpisodeList();
}

// ─── Episode list (per-run) ──────────────────────────────────────────────
function correctnessPill(is_correct) {
  if (is_correct === true)  return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-green-100 text-green-700">✓ correct</span>`;
  if (is_correct === false) return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-red-100 text-red-700">✗ incorrect</span>`;
  return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-gray-100 text-gray-600">·</span>`;
}
function rewardBadge(r) {
  if (r == null) return "";
  const cls = r >= 1 ? "bg-green-50 text-green-700 border-green-200"
            : r >  0 ? "bg-amber-50 text-amber-700 border-amber-200"
            : "bg-gray-50 text-gray-600 border-gray-200";
  return `<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[11px] font-medium ${cls} mono">
    <span class="opacity-60">reward</span>${Number(r).toFixed(3)}</span>`;
}
function chevron() {
  return `<svg class="chev w-3.5 h-3.5 text-gray-400" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M7.21 14.77a.75.75 0 010-1.06l3.71-3.71-3.71-3.71a.75.75 0 111.06-1.06l4.24 4.24a.75.75 0 010 1.06l-4.24 4.24a.75.75 0 01-1.06 0z" clip-rule="evenodd"/></svg>`;
}
function fieldBox(label, value) {
  const v = formatValue(value);
  if (v === "") return "";
  return `
    <div class="mt-2" data-field-wrap>
      <div class="flex items-center justify-between">
        <span class="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">${escapeHtml(label)}</span>
        <button class="toggle-expand text-[11px] text-gray-400 hover:text-accent-600">expand</button>
      </div>
      <div class="field-box mt-1 bg-layer-1 border border-gray-200 rounded-md p-2">
        <pre class="text-[12.5px] leading-relaxed text-gray-800">${escapeHtml(v)}</pre>
      </div>
    </div>`;
}
function _firstString(v) {
  if (v == null) return "";
  if (typeof v === "string") return v.replace(/\s+/g, " ").slice(0, 80);
  return JSON.stringify(v).slice(0, 80);
}
function renderChatCompletions(msgs) {
  if (!Array.isArray(msgs) || !msgs.length) return "";
  const roleStyle = {
    system:    "bg-layer-3 border-gray-200 text-gray-700",
    user:      "bg-layer-2 border-gray-200",
    assistant: "bg-white border-accent-200",
    tool:      "bg-amber-50 border-amber-200 text-amber-900",
  };
  const bubbles = msgs.map((m) => {
    const role = m.role || "?";
    const content = typeof m.content === "string" ? m.content : formatValue(m.content);
    const cls = roleStyle[role] || "bg-white border-gray-200";
    return `
      <div class="border ${cls} rounded-md p-2.5">
        <div class="text-[10px] font-semibold uppercase tracking-wider text-gray-500 mb-1">${escapeHtml(role)}</div>
        <pre class="text-[12.5px] leading-relaxed text-gray-800 max-h-80 overflow-auto">${escapeHtml(content)}</pre>
      </div>`;
  }).join("");
  return `
    <details class="mt-2" data-field-wrap>
      <summary class="flex items-center justify-between cursor-pointer">
        <span class="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">chat completions <span class="text-gray-400 normal-case">(${msgs.length} message${msgs.length === 1 ? "" : "s"})</span></span>
        <span class="text-[11px] text-gray-400">click to toggle</span>
      </summary>
      <div class="mt-1 space-y-1.5">${bubbles}</div>
    </details>`;
}
function renderStep(step, idx) {
  const meta = step.metadata && Object.keys(step.metadata).length ? step.metadata : null;
  const doneTag = step.done ? `<span class="px-1.5 py-0.5 rounded bg-layer-2 text-gray-600 text-[11px] font-medium">done</span>` : "";
  // Preview: prefer model_response (Harbor/ATIF), fall back to output/action/input.
  const preview = _firstString(step.model_response) || _firstString(step.output) || _firstString(step.action) || _firstString(step.input) || "";
  return `
    <details class="border border-gray-200 rounded-md bg-white">
      <summary class="px-3 py-2 flex items-center gap-2 hover:bg-layer-1">
        ${chevron()}
        <span class="text-xs font-semibold text-gray-700">Step ${idx}</span>
        ${doneTag}
        ${step.reward != null ? rewardBadge(step.reward) : ""}
        <span class="ml-auto text-[11px] text-gray-400 truncate max-w-[40ch]">${escapeHtml(preview)}</span>
      </summary>
      <div class="px-3 pb-3">
        ${fieldBox("input",          step.input)}
        ${fieldBox("thought",        step.thought)}
        ${fieldBox("model_response", step.model_response)}
        ${fieldBox("output",         step.output)}
        ${fieldBox("action",         step.action)}
        ${fieldBox("observation",    step.observation)}
        ${renderChatCompletions(step.chat_completions)}
        ${meta ? fieldBox("metadata", meta) : ""}
      </div>
    </details>`;
}
function renderTrajectory(traj, ti) {
  const steps = traj.steps || [];
  const sigEntries = Object.entries(traj.signals || {});
  return `
    <details class="border border-gray-200 rounded-md bg-layer-1" open>
      <summary class="px-3 py-2 flex items-center gap-2 hover:bg-layer-2 rounded-md">
        ${chevron()}
        <span class="text-xs font-semibold text-gray-700">Trajectory ${ti} · <span class="text-accent-600">${escapeHtml(traj.name || "agent")}</span></span>
        <span class="text-[11px] text-gray-500">${steps.length} step${steps.length === 1 ? "" : "s"}</span>
        ${traj.reward != null ? rewardBadge(traj.reward) : ""}
        ${sigEntries.length
          ? `<span class="ml-2 text-[11px] text-gray-500 mono">${sigEntries.map(([k,v]) => `${escapeHtml(k)}=${Number(v).toFixed(3)}`).join(" · ")}</span>`
          : ""}
      </summary>
      <div class="px-3 pb-3 space-y-2">
        ${steps.map((s, i) => renderStep(s, i)).join("")}
      </div>
    </details>`;
}
function renderTaskPanel(task) {
  if (!task || typeof task !== "object") return fieldBox("task", task);
  const instruction = task.instruction;
  const metadata = task.metadata && Object.keys(task.metadata).length ? task.metadata : null;
  return `
    <div class="bg-layer-2 border border-gray-200 rounded-md p-3">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Task</span>
        ${task.id ? `<code class="text-[11px] bg-white border border-gray-200 px-1.5 py-0.5 rounded text-gray-700">${escapeHtml(task.id)}</code>` : ""}
      </div>
      ${instruction != null ? `<pre class="text-[12.5px] text-gray-800 max-h-64 overflow-auto">${escapeHtml(formatValue(instruction))}</pre>` : ""}
      ${metadata ? fieldBox("task metadata", metadata) : ""}
    </div>`;
}
function renderEpisodeBody(ep) {
  const parts = [renderTaskPanel(ep.task)];
  for (const [ti, traj] of (ep.trajectories || []).entries()) parts.push(renderTrajectory(traj, ti));
  if (ep.artifacts && Object.keys(ep.artifacts).length) parts.push(fieldBox("artifacts", ep.artifacts));
  if (ep.metrics   && Object.keys(ep.metrics).length)   parts.push(fieldBox("metrics",   ep.metrics));
  if (ep.metadata  && Object.keys(ep.metadata).length)  parts.push(fieldBox("metadata",  ep.metadata));
  return `<div class="px-4 pb-4 space-y-3">${parts.join("")}</div>`;
}
function renderEpisodeHeader(item) {
  const idx = item.eval_idx != null
    ? `<span class="text-[11px] mono text-gray-400 tabular-nums w-10 text-right">#${String(item.eval_idx).padStart(4, "0")}</span>` : "";
  const tid = item.task_id ? `<code class="text-[12px] text-gray-700 mono">${escapeHtml(item.task_id)}</code>` : "";
  const term = item.termination_reason
    ? `<span class="text-[11px] text-gray-500">· ${escapeHtml(item.termination_reason)}</span>` : "";
  const trajCount = item.n_trajectories
    ? `<span class="text-[11px] text-gray-500">· ${item.n_trajectories} traj · ${item.n_steps} step${item.n_steps === 1 ? "" : "s"}</span>` : "";
  return `
    <summary class="px-3 py-2 flex items-center gap-2 hover:bg-layer-1 rounded-md">
      ${chevron()}
      ${idx}
      ${correctnessPill(item.is_correct)}
      ${tid}
      ${item.reward != null ? rewardBadge(item.reward) : ""}
      ${trajCount}
      ${term}
      <span class="ml-2 text-[11px] text-gray-400 truncate flex-1">${escapeHtml(item.instruction_preview || "")}</span>
    </summary>`;
}

async function loadEpisode(runId, filename) {
  const key = `${runId}::${filename}`;
  if (episodeCache.has(key)) return episodeCache.get(key);
  const resp = await fetch(`/api/runs/${encodeURIComponent(runId)}/episodes/${encodeURIComponent(filename)}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  episodeCache.set(key, data);
  return data;
}

function paintEpisodeList() {
  const list = document.getElementById("ep-list");
  if (!list) return;
  const q = runState.search.trim().toLowerCase();
  const visible = runState.index.filter(item => {
    if (runState.filter === "correct"   && item.is_correct !== true)  return false;
    if (runState.filter === "incorrect" && item.is_correct !== false) return false;
    if (q) {
      const hay = `${item.task_id || ""} ${item.instruction_preview || ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  document.getElementById("ep-count").textContent = `${visible.length} / ${runState.index.length} shown`;
  list.innerHTML = visible.map(item => `
    <details class="bg-white border border-gray-200 rounded-md shadow-subtle"
             data-filename="${escapeHtml(item.filename)}">
      ${renderEpisodeHeader(item)}
      <div class="episode-body border-t border-gray-100 text-gray-500 text-xs px-4 py-3" data-pending="1">Loading…</div>
    </details>
  `).join("");

  list.querySelectorAll("details[data-filename]").forEach(det => {
    det.addEventListener("toggle", () => {
      if (!det.open) return;
      const body = det.querySelector(".episode-body");
      if (!body || !body.dataset.pending) return;
      const fname = det.getAttribute("data-filename");
      loadEpisode(runState.id, fname).then(ep => {
        body.removeAttribute("data-pending");
        body.outerHTML = renderEpisodeBody(ep);
      }).catch(err => {
        body.textContent = "Failed to load: " + err;
      });
    });
  });
}

// ─── Boot ───────────────────────────────────────────────────────────────
function route() {
  const r = parseHash();
  if (r.view === "run") renderRunView(r.id);
  else renderListView();
}

if (PRESELECT_RUN && !location.hash) {
  location.hash = `#/run/${encodeURIComponent(PRESELECT_RUN)}`;
} else {
  route();
}
</script>
</body>
</html>
"""


def _render_html(root_path: Path, runs: list[dict], preselect: str | None) -> str:
    title = "rLLM Episode Viewer"
    return (
        _PAGE_HTML.replace("__TITLE__", title)
        .replace("__ROOT_PATH__", str(root_path))
        .replace("__ROOT_PATH_JSON__", json.dumps(str(root_path)))
        .replace("__RUNS_JSON__", json.dumps(runs, ensure_ascii=False))
        .replace("__PRESELECT__", json.dumps(preselect))
    )


# --------------------------------------------------------------------------- #
# HTTP server                                                                 #
# --------------------------------------------------------------------------- #


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler(root_path: Path, html_factory):
    """Build a request handler bound to ``root_path``.

    Routes:
      * ``GET /``                                 → SPA shell
      * ``GET /api/runs``                         → freshly-scanned run list
      * ``GET /api/runs/<id>/index``              → episode index for a run
      * ``GET /api/runs/<id>/episodes/<file>``    → one episode JSON
      * other GETs                                → 404 (no static fall-through)
    """
    safe_id = re.compile(r"^[A-Za-z0-9._-]+$")
    safe_file = re.compile(r"^episode_[A-Za-z0-9._-]+\.json$")
    resolved_root = root_path.resolve()

    def _is_safe_run_id(rid: str) -> bool:
        # Reject empty, "." / ".." themselves, and any path-traversal attempt.
        # Regex permits "." in names (timestamps), so block "../" patterns explicitly.
        if not rid or rid in (".", "..") or ".." in rid or "/" in rid or "\\" in rid:
            return False
        return bool(safe_id.match(rid))

    def _under_root(p: Path) -> bool:
        try:
            p.relative_to(resolved_root)
            return True
        except ValueError:
            return False

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet by default
            pass

        def _send_json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file_bytes(self, path: Path) -> None:
            try:
                data = path.read_bytes()
            except OSError:
                self.send_error(404, "Not Found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]

            if path in ("/", "/index.html"):
                return self._send_html(html_factory())

            if path == "/api/runs":
                return self._send_json(200, _scan_runs(root_path))

            m = re.match(r"^/api/runs/([^/]+)/index$", path)
            if m:
                run_id = m.group(1)
                if not _is_safe_run_id(run_id):
                    return self.send_error(400, "Bad run id")
                run_dir = (root_path / run_id).resolve()
                if not _under_root(run_dir):
                    return self.send_error(400, "Bad run id")
                episodes_dir = _resolve_episodes_dir(run_dir)
                if not episodes_dir.is_dir():
                    return self.send_error(404, "Run not found")
                return self._send_json(200, _build_episode_index(episodes_dir))

            m = re.match(r"^/api/runs/([^/]+)/episodes/([^/]+)$", path)
            if m:
                run_id, fname = m.group(1), m.group(2)
                if not _is_safe_run_id(run_id) or not safe_file.match(fname):
                    return self.send_error(400, "Bad path")
                target = (root_path / run_id / "episodes" / fname).resolve()
                if not _under_root(target):
                    return self.send_error(400, "Bad path")
                if not target.is_file():
                    return self.send_error(404, "Episode not found")
                return self._send_file_bytes(target)

            return self.send_error(404, "Not Found")

    return Handler


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def _eval_results_root() -> Path:
    import os

    rllm_home = os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))
    return Path(rllm_home) / "eval_results"


def launch(
    target: str | Path | None = None,
    server_port: int = 7860,
    host: str = "127.0.0.1",
    open_browser: bool = True,
) -> None:
    """Serve the rLLM episode dashboard.

    Args:
        target: Either ``None`` (serve the whole ``~/.rllm/eval_results/``
            directory at ``#/``), a path to that root, or a path to a
            single run directory inside it (drills straight into ``#/run/<id>``).
        server_port: Local port to bind. ``0`` selects a free port.
        host: Bind host. Defaults to ``127.0.0.1`` (loopback only).
        open_browser: If True, opens the page in the default browser.
    """
    root = _eval_results_root()
    preselect: str | None = None

    if target is not None:
        target_path = Path(target).expanduser().resolve()
        if not target_path.exists():
            raise SystemExit(f"Path does not exist: {target_path}")

        # If the user pointed at a specific run directory, infer root from
        # its parent and preselect that run on load.
        if (target_path / "episodes").is_dir():
            root = target_path.parent
            preselect = target_path.name
        else:
            root = target_path

    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"No eval_results directory at {root}.\nRun `rllm eval <bench>` first to populate it.")

    # Build HTML lazily so reloading the page picks up newly-completed runs.
    def html_factory() -> str:
        runs = _scan_runs(root)
        return _render_html(root, runs, preselect)

    handler = _make_handler(root, html_factory)
    server = _ThreadingServer((host, server_port), handler)
    bound_port = server.server_address[1]
    url = f"http://{host}:{bound_port}/"
    if preselect:
        url += f"#/run/{preselect}"

    print(f"  Serving rLLM episode viewer at {url}")
    print(f"  Root: {root}")
    print("  Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down…")
    finally:
        server.server_close()

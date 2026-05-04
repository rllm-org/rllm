/**
 * Typed fetch helpers for the rLLM Console backend.
 *
 * All endpoints live under {@link API_BASE} (`/console/api` by default).
 * Override at dev time by setting `VITE_API_URL` to point at the backend
 * — Vite's proxy already routes `/console/api/*` to the right place.
 */

export const API_BASE = "/console/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, init);
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new ApiError(r.status, r.statusText, text);
  }
  return r.json() as Promise<T>;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public body: string,
  ) {
    super(`${status} ${statusText}: ${body}`);
  }
}

// ---- Shell --------------------------------------------------------------

export interface PanelDescriptor {
  id: string;
  title: string;
  icon: string;
  nav_order: number;
  placeholder: boolean;
  has_router: boolean;
}

export interface ShellInfo {
  version: string;
  url_prefix: string;
  eval_results_root: string | null;
  panels: PanelDescriptor[];
}

export const fetchShellInfo = () => request<ShellInfo>("/shell/info");

// ---- Sessions panel -----------------------------------------------------

export interface TraceRecord {
  _created_at: number;
  [key: string]: unknown;
}

/**
 * Filter shape for ``GET /panels/sessions/traces``. All keys are
 * optional and are pushed down to SQL via the denormalized columns.
 *
 * Cursor pagination: ``until`` (older-than) for scroll-back,
 * ``since`` (newer-than) for live tail. ``limit`` capped at 500
 * server-side; default 100.
 */
export interface TraceFilters {
  run_id?: string;
  session_id?: string;
  model?: string;
  harness?: string;
  has_error?: boolean;
  latency_min?: number;
  latency_max?: number;
  since?: number;
  until?: number;
  limit?: number;
  order?: "ASC" | "DESC";
}

export interface TraceFacets {
  models: string[];
  harnesses: string[];
  runs: string[];
}

function filtersToQuery(f: TraceFilters): URLSearchParams {
  const q = new URLSearchParams();
  if (f.run_id) q.set("run_id", f.run_id);
  if (f.session_id) q.set("session_id", f.session_id);
  if (f.model) q.set("model", f.model);
  if (f.harness) q.set("harness", f.harness);
  if (f.has_error != null) q.set("has_error", String(f.has_error));
  if (f.latency_min != null) q.set("latency_min", String(f.latency_min));
  if (f.latency_max != null) q.set("latency_max", String(f.latency_max));
  if (f.since != null) q.set("since", String(f.since));
  if (f.until != null) q.set("until", String(f.until));
  if (f.limit != null) q.set("limit", String(f.limit));
  if (f.order) q.set("order", f.order);
  return q;
}

export const fetchTraceFeed = (filters: TraceFilters = {}) =>
  request<TraceRecord[]>(
    `/panels/sessions/traces?${filtersToQuery(filters)}`,
  );

export const fetchTraceFacets = () =>
  request<TraceFacets>("/panels/sessions/facets");

// ---- Runs panel ---------------------------------------------------------

export interface EvalRunSummary {
  id: string;
  benchmark: string;
  model: string;
  agent: string;
  split: string;
  timestamp: string;
  created_at: string | null;
  score: number | null;
  correct: number | null;
  total: number | null;
  errors: number | null;
  n_episodes: number;
  status: "completed" | "incomplete" | "running";
  in_flight: boolean;
  started_at: number | null;
  ended_at: number | null;
}

export interface EpisodeIndexRow {
  filename: string;
  eval_idx: number | null;
  task_id: string | null;
  is_correct: boolean | null;
  termination_reason: string | null;
  n_trajectories: number;
  n_steps: number;
  reward: number | null;
  instruction_preview: string;
}

export interface InFlightTask {
  idx: number;
  session_id: string;
  task_id: string | null;
  instruction: string;
  started_at: number;
  elapsed_s: number | null;
  trace_count: number;
  last_trace_at: number | null;
}

export interface RunSessionSummary {
  session_id: string;
  trace_count: number;
  first_at: number | null;
  last_at: number | null;
}

export interface LivePayload {
  run_id: string;
  started_at: number | null;
  ended_at: number | null;
  in_flight: boolean;
  sessions: RunSessionSummary[];
  in_flight_tasks: InFlightTask[];
  finished_count: number;
  started_count: number;
}

export const fetchEvalRuns = () => request<EvalRunSummary[]>("/panels/runs");
export const fetchEpisodeIndex = (runId: string) =>
  request<EpisodeIndexRow[]>(
    `/panels/runs/${encodeURIComponent(runId)}/index`,
  );
export const fetchLivePayload = (runId: string) =>
  request<LivePayload>(`/panels/runs/${encodeURIComponent(runId)}/live`);

/** Per-run paginated trace feed. Same filter shape as Sessions. */
export const fetchRunTraces = (runId: string, filters: TraceFilters = {}) =>
  request<TraceRecord[]>(
    `/panels/runs/${encodeURIComponent(runId)}/traces?${filtersToQuery(filters)}`,
  );

// ---- Datasets panel -----------------------------------------------------

export interface DatasetCard {
  name: string;
  description: string;
  source: string;
  category: string;
  splits: string[];
  default_agent: string | null;
  reward_fn: string | null;
  eval_split: string | null;
  instruction_field: string | null;
  transform: string | null;
  local_splits: string[];
  is_local: boolean;
}

export interface DatasetSplitInfo {
  name: string;
  is_local: boolean;
  path: string | null;
  n_rows: number | null;
  schema: Record<string, string> | null;
}

export interface DatasetDetail extends DatasetCard {
  splits_detail: DatasetSplitInfo[];
}

export interface DatasetEntries {
  rows: Record<string, unknown>[];
  total: number;
  offset: number;
  limit: number;
  columns: string[];
  n_pages: number;
}

export const fetchDatasets = () =>
  request<{ datasets: DatasetCard[]; categories: string[] }>(
    "/panels/datasets",
  );

export const fetchDatasetDetail = (name: string) =>
  request<DatasetDetail>(`/panels/datasets/${encodeURIComponent(name)}`);

export const fetchDatasetEntries = (
  name: string,
  split: string,
  offset: number,
  limit: number,
) => {
  const q = new URLSearchParams({
    split,
    offset: String(offset),
    limit: String(limit),
  });
  return request<DatasetEntries>(
    `/panels/datasets/${encodeURIComponent(name)}/entries?${q}`,
  );
};

/** SSE event payload yielded by the pull stream. */
export type PullEvent =
  | { type: "start"; name: string }
  | { type: "log"; line: string }
  | { type: "done"; ok: boolean; exit_code: number };

/**
 * POST + read-streaming wrapper for ``rllm dataset pull <name>``.
 *
 * EventSource only supports GET, so we use ``fetch`` + a manual SSE
 * parser over ``response.body``. Yields one PullEvent per ``data:``
 * frame. Caller is responsible for awaiting the iterator to
 * completion (or aborting via ``signal``, which cancels the stream
 * and the subprocess).
 */
export async function* streamDatasetPull(
  name: string,
  signal?: AbortSignal,
): AsyncGenerator<PullEvent> {
  const response = await fetch(
    `${API_BASE}/panels/datasets/${encodeURIComponent(name)}/pull`,
    { method: "POST", signal },
  );
  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    throw new ApiError(response.status, response.statusText, text);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          yield JSON.parse(line.slice(6)) as PullEvent;
        } catch {
          // Ignore malformed frames; keep streaming.
        }
      }
    }
  }
}

// ---- Settings panel -----------------------------------------------------

export interface ConsoleConfig {
  version: string;
  url_prefix: string;
  eval_results_root: string | null;
  rllm_home: string;
  gateway_db_path: string;
  console_env_file: string;
}

export interface EnvVarRow {
  key: string;
  label: string;
  description: string;
  category: string;
  secret: boolean;
  url: string | null;
  is_set: boolean;
  in_console_file: boolean;
  masked_value: string | null;
}

export interface EnvList {
  categories: string[];
  rows: EnvVarRow[];
  console_env_file: string;
}

export const fetchConsoleConfig = () =>
  request<ConsoleConfig>("/panels/settings/config");

export const fetchEnvVars = () => request<EnvList>("/panels/settings/env");

export const revealEnvVar = (key: string) =>
  request<{
    key: string;
    value: string;
    secret: boolean;
    revealed: boolean;
    in_console_file: boolean;
  }>(`/panels/settings/env/${encodeURIComponent(key)}?reveal=true`);

export const setEnvVar = (key: string, value: string) =>
  request<{ key: string; ok: boolean }>("/panels/settings/env", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key, value }),
  });

export const deleteEnvVar = (key: string) =>
  request<{
    key: string;
    removed_from_file: boolean;
    removed_from_env: boolean;
  }>(`/panels/settings/env/${encodeURIComponent(key)}`, { method: "DELETE" });

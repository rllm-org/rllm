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

export interface RunRow {
  run_id: string;
  started_at: number | null;
  ended_at: number | null;
  metadata: Record<string, unknown>;
  session_count: number;
  trace_count: number;
  last_trace_at: number | null;
}

export interface SessionRow {
  session_id: string;
  run_id: string;
  trace_count: number;
  first_at: number | null;
  last_at: number | null;
}

export interface TraceRecord {
  _created_at: number;
  [key: string]: unknown;
}

export const fetchSessionRuns = () =>
  request<RunRow[]>("/panels/sessions/runs");

export const fetchSessions = (runId?: string) =>
  request<SessionRow[]>(
    "/panels/sessions/sessions" +
      (runId ? `?run_id=${encodeURIComponent(runId)}` : ""),
  );

export const fetchTraces = (params: {
  session_id: string;
  run_id?: string;
  since?: number;
  limit?: number;
}) => {
  const q = new URLSearchParams();
  q.set("session_id", params.session_id);
  if (params.run_id) q.set("run_id", params.run_id);
  if (params.since != null) q.set("since", String(params.since));
  if (params.limit != null) q.set("limit", String(params.limit));
  return request<TraceRecord[]>(`/panels/sessions/traces?${q}`);
};

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
  status: "completed" | "incomplete";
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

export interface LivePayload {
  in_flight: InFlightTask[];
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

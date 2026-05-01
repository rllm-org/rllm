import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ChevronRight } from "lucide-react";

import { API_BASE, ApiError } from "~/lib/api";
import { JsonField } from "~/panels/sessions/JsonField";

interface EpisodeJson {
  id?: string;
  eval_idx?: number;
  is_correct?: boolean | null;
  termination_reason?: string | null;
  task?: string | { id?: string; instruction?: string; [k: string]: unknown };
  trajectories?: Trajectory[];
  artifacts?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

interface Trajectory {
  uid?: string;
  name?: string;
  reward?: number | null;
  steps?: Step[];
  output?: unknown;
  signals?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  session_id?: string;
}

interface Step {
  id?: string;
  input?: unknown;
  output?: unknown;
  action?: unknown;
  reward?: number | null;
  done?: boolean;
  metadata?: Record<string, unknown> | null;
}

async function fetchEpisode(
  runId: string,
  filename: string,
): Promise<EpisodeJson> {
  const r = await fetch(
    `${API_BASE}/panels/runs/${encodeURIComponent(runId)}/episodes/${encodeURIComponent(filename)}`,
  );
  if (!r.ok) {
    throw new ApiError(r.status, r.statusText, await r.text().catch(() => ""));
  }
  return r.json();
}

interface Props {
  runId: string;
  filename: string;
  onBack: () => void;
}

export function EpisodeView({ runId, filename, onBack }: Props) {
  const q = useQuery({
    queryKey: ["episode", runId, filename],
    queryFn: () => fetchEpisode(runId, filename),
  });

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-line bg-chrome px-4 py-2.5">
        <button
          type="button"
          onClick={onBack}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-muted hover:bg-active hover:text-strong"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Episodes
        </button>
        <ChevronRight className="h-3.5 w-3.5 text-faint" />
        <div className="truncate font-mono text-xs text-strong">{filename}</div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        {q.isLoading && <div className="text-sm text-subtle">loading…</div>}
        {q.isError && (
          <div className="text-sm text-rose-600 dark:text-rose-400">
            failed to load: {String(q.error)}
          </div>
        )}
        {q.data && <EpisodeBody episode={q.data} />}
      </div>
    </div>
  );
}

function EpisodeBody({ episode }: { episode: EpisodeJson }) {
  const taskId =
    typeof episode.task === "string"
      ? episode.task
      : episode.task?.id ?? "—";
  const instruction =
    typeof episode.task === "object" ? episode.task?.instruction : null;

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-line bg-chrome p-4">
        <div className="grid grid-cols-2 gap-3 text-xs md:grid-cols-4">
          <Field label="Task">{taskId}</Field>
          <Field label="Eval idx">{episode.eval_idx ?? "—"}</Field>
          <Field label="Correct">
            {episode.is_correct == null
              ? "—"
              : episode.is_correct
                ? "yes"
                : "no"}
          </Field>
          <Field label="Termination">{episode.termination_reason ?? "—"}</Field>
        </div>
        {instruction && (
          <div className="mt-3 border-t border-line pt-3">
            <div className="mb-1 text-[10px] uppercase tracking-wide text-subtle">
              Instruction
            </div>
            <pre className="whitespace-pre-wrap break-words font-mono text-xs text-strong">
              {instruction}
            </pre>
          </div>
        )}
      </div>

      {episode.trajectories?.map((t, i) => (
        <TrajectoryCard key={t.uid ?? i} trajectory={t} index={i} />
      ))}

      {episode.metrics && Object.keys(episode.metrics).length > 0 && (
        <JsonField label="metrics" value={episode.metrics} />
      )}
      {episode.metadata && Object.keys(episode.metadata).length > 0 && (
        <JsonField
          label="metadata"
          value={episode.metadata}
          initiallyOpen={false}
        />
      )}
      {episode.artifacts && Object.keys(episode.artifacts).length > 0 && (
        <JsonField
          label="artifacts"
          value={episode.artifacts}
          initiallyOpen={false}
        />
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-subtle">
        {label}
      </div>
      <div className="mt-0.5 truncate font-mono text-strong">{children}</div>
    </div>
  );
}

function TrajectoryCard({
  trajectory,
  index,
}: {
  trajectory: Trajectory;
  index: number;
}) {
  return (
    <div className="rounded-md border border-line bg-chrome">
      <div className="flex items-center gap-3 border-b border-line px-3 py-2">
        <span className="font-mono text-[11px] text-muted">
          trajectory #{index}
        </span>
        {trajectory.name && (
          <span className="font-mono text-xs text-strong">
            {trajectory.name}
          </span>
        )}
        {trajectory.reward != null && (
          <span className="ml-auto font-mono text-xs text-body">
            reward {trajectory.reward.toFixed(2)}
          </span>
        )}
      </div>
      <div className="space-y-2 p-3">
        {trajectory.steps?.map((s, i) => (
          <StepCard key={s.id ?? i} step={s} index={i} />
        ))}
        {trajectory.signals && Object.keys(trajectory.signals).length > 0 && (
          <JsonField
            label="signals"
            value={trajectory.signals}
            initiallyOpen={false}
          />
        )}
      </div>
    </div>
  );
}

function StepCard({ step, index }: { step: Step; index: number }) {
  return (
    <div className="rounded border border-line bg-canvas p-2">
      <div className="mb-2 flex items-center gap-3 text-[10px]">
        <span className="font-mono text-subtle">step #{index}</span>
        {step.reward != null && (
          <span className="font-mono text-muted">
            r={step.reward.toFixed(2)}
          </span>
        )}
        {step.done && (
          <span className="rounded bg-emerald-100 dark:bg-emerald-900/40 px-1.5 py-0.5 font-mono text-emerald-700 dark:text-emerald-300">
            done
          </span>
        )}
      </div>
      <div className="space-y-2">
        {step.input != null && <JsonField label="input" value={step.input} />}
        {step.output != null && (
          <JsonField label="output" value={step.output} />
        )}
        {step.action != null && (
          <JsonField label="action" value={step.action} initiallyOpen={false} />
        )}
        {step.metadata && (
          <JsonField
            label="metadata"
            value={step.metadata}
            initiallyOpen={false}
          />
        )}
      </div>
    </div>
  );
}

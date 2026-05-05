import { useQuery } from "@tanstack/react-query";

import { fetchEvalRuns, type EvalRunSummary } from "~/lib/api";
import { percent, timeAgo } from "~/lib/format";
import { cn } from "~/lib/utils";

interface Props {
  selected: string | null;
  onSelect: (runId: string) => void;
}

/**
 * Left-column run list. Each row is a compact card: id, live/completed
 * dot, benchmark · model, score-or-trace-count, relative timestamp.
 *
 * Polled every 5s — runs change slowly relative to traces, and we
 * already get fast updates inside the right-pane RunDetail.
 */
export function RunsList({ selected, onSelect }: Props) {
  const q = useQuery({
    queryKey: ["eval-runs"],
    queryFn: fetchEvalRuns,
    refetchInterval: 5_000,
  });

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-line bg-chrome px-3 py-2">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-subtle">
          Runs
        </div>
        <div className="font-mono text-[10px] text-subtle">
          {q.data ? q.data.length : ""}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        {q.isLoading && (
          <div className="px-3 py-4 text-xs text-subtle">loading…</div>
        )}
        {q.isError && (
          <div className="px-3 py-4 text-xs text-rose-600 dark:text-rose-400">
            failed to load — start{" "}
            <code className="rounded bg-active px-1 font-mono">rllm view</code>{" "}
            against an eval-results dir
          </div>
        )}
        {q.data?.length === 0 && (
          <div className="px-3 py-4 text-xs text-subtle">
            no runs yet — kick one off with{" "}
            <code className="rounded bg-active px-1 font-mono">
              rllm eval &lt;bench&gt;
            </code>
          </div>
        )}
        {q.data?.map((r) => (
          <RunRow
            key={r.id}
            run={r}
            active={selected === r.id}
            onClick={() => onSelect(r.id)}
          />
        ))}
      </div>
    </div>
  );
}

function RunRow({
  run,
  active,
  onClick,
}: {
  run: EvalRunSummary;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full flex-col gap-0.5 border-b border-line-subtle px-3 py-2 text-left hover:bg-hover",
        active && "bg-active",
      )}
    >
      <div className="flex items-center gap-1.5">
        <StatusDot live={run.in_flight} status={run.status} />
        <div className="truncate font-mono text-xs text-strong">{run.id}</div>
      </div>
      <div className="truncate font-mono text-[10px] text-subtle">
        {run.benchmark}
        {run.model && run.model !== "—" ? ` · ${run.model}` : ""}
      </div>
      <div className="flex items-center justify-between font-mono text-[10px] text-subtle">
        <ScoreCell run={run} />
        <span>
          {run.created_at ? timeAgo(Date.parse(run.created_at) / 1000) : ""}
        </span>
      </div>
    </button>
  );
}

function StatusDot({ live, status }: { live: boolean; status: string }) {
  if (live) {
    return (
      <span
        title="live"
        className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-sky-500"
      />
    );
  }
  if (status === "completed") {
    return (
      <span
        title="completed"
        className="h-2 w-2 shrink-0 rounded-full bg-emerald-500"
      />
    );
  }
  return (
    <span
      title={status}
      className="h-2 w-2 shrink-0 rounded-full bg-amber-500"
    />
  );
}

function ScoreCell({ run }: { run: EvalRunSummary }) {
  if (run.score == null) {
    return (
      <span className="text-faint">
        {run.n_episodes > 0 ? `${run.n_episodes} ep` : "—"}
      </span>
    );
  }
  const tone =
    run.score >= 0.7
      ? "text-emerald-700 dark:text-emerald-400"
      : run.score >= 0.4
        ? "text-amber-600 dark:text-amber-400"
        : "text-rose-600 dark:text-rose-400";
  return (
    <span className={tone}>
      {percent(run.score)}
      {run.correct != null && run.total != null && (
        <span className="ml-1 text-faint">
          {run.correct}/{run.total}
        </span>
      )}
    </span>
  );
}

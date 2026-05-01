import { useQuery } from "@tanstack/react-query";

import { fetchEvalRuns, type EvalRunSummary } from "~/lib/api";
import { compactNumber, percent, timeAgo } from "~/lib/format";
import { cn } from "~/lib/utils";

const COLS: Array<{ key: keyof EvalRunSummary | "_score"; label: string; cls?: string }> = [
  { key: "id", label: "Run", cls: "min-w-[20rem]" },
  { key: "benchmark", label: "Benchmark" },
  { key: "model", label: "Model" },
  { key: "agent", label: "Agent" },
  { key: "_score", label: "Score" },
  { key: "n_episodes", label: "Episodes" },
  { key: "errors", label: "Errors" },
  { key: "status", label: "Status" },
  { key: "created_at", label: "Created" },
];

export function RunsTable({ onPick }: { onPick: (runId: string) => void }) {
  const q = useQuery({
    queryKey: ["eval-runs"],
    queryFn: fetchEvalRuns,
    refetchInterval: 5_000,
  });

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-line bg-chrome px-4 py-2.5">
        <div className="text-sm font-medium text-strong">Runs</div>
        <div className="font-mono text-[11px] text-subtle">
          {q.data ? `${q.data.length} runs` : ""}
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        {q.isLoading && (
          <div className="px-4 py-6 text-sm text-subtle">loading runs…</div>
        )}
        {q.isError && (
          <div className="px-4 py-6 text-sm text-rose-600 dark:text-rose-400">
            failed to load — check that ``rllm view`` was started against an
            eval-results directory
          </div>
        )}
        {q.data?.length === 0 && (
          <div className="px-4 py-6 text-sm text-subtle">
            no eval runs yet — kick one off with{" "}
            <code className="rounded bg-active px-1.5 py-0.5 font-mono text-xs">
              rllm eval &lt;bench&gt;
            </code>
          </div>
        )}
        {q.data && q.data.length > 0 && (
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10 bg-canvas">
              <tr className="border-b border-line text-left">
                {COLS.map((c) => (
                  <th
                    key={c.key}
                    className={cn(
                      "whitespace-nowrap px-3 py-2 text-[10px] font-semibold uppercase tracking-wide text-subtle",
                      c.cls,
                    )}
                  >
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {q.data.map((r) => (
                <tr
                  key={r.id}
                  onClick={() => onPick(r.id)}
                  className="cursor-pointer border-b border-line-subtle hover:bg-hover"
                >
                  <td className="max-w-[24rem] truncate px-3 py-2 font-mono text-xs text-strong">
                    {r.id}
                  </td>
                  <td className="px-3 py-2 text-xs text-body">{r.benchmark}</td>
                  <td className="px-3 py-2 text-xs text-body">{r.model}</td>
                  <td className="px-3 py-2 text-xs text-body">{r.agent}</td>
                  <td className="px-3 py-2 text-xs">
                    <ScoreCell score={r.score} correct={r.correct} total={r.total} />
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-body">
                    {compactNumber(r.n_episodes)}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    <span
                      className={cn(
                        r.errors && r.errors > 0
                          ? "text-rose-600 dark:text-rose-400"
                          : "text-subtle",
                      )}
                    >
                      {compactNumber(r.errors ?? 0)}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs">
                    <StatusBadge status={r.status} />
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-xs text-subtle">
                    {r.created_at
                      ? timeAgo(Date.parse(r.created_at) / 1000)
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function ScoreCell({
  score,
  correct,
  total,
}: {
  score: number | null;
  correct: number | null;
  total: number | null;
}) {
  if (score == null) return <span className="text-subtle">—</span>;
  const tone =
    score >= 0.7
      ? "text-emerald-700 dark:text-emerald-400"
      : score >= 0.4
        ? "text-amber-600 dark:text-amber-400"
        : "text-rose-600 dark:text-rose-400";
  return (
    <span className={cn("font-mono", tone)}>
      {percent(score)}
      {correct != null && total != null && (
        <span className="ml-1 text-[10px] text-subtle">
          {correct}/{total}
        </span>
      )}
    </span>
  );
}

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "completed"
      ? "border-emerald-300 dark:border-emerald-700/60 bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300"
      : "border-amber-300 dark:border-amber-700/60 bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300";
  return (
    <span
      className={cn(
        "rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide",
        tone,
      )}
    >
      {status}
    </span>
  );
}

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ChevronRight, CircleAlert, CircleCheck } from "lucide-react";

import {
  fetchEpisodeIndex,
  fetchLivePayload,
  type EpisodeIndexRow,
} from "~/lib/api";
import { compactNumber, percent, timeAgo } from "~/lib/format";
import { cn } from "~/lib/utils";

interface Props {
  runId: string;
  onBack: () => void;
  onPickEpisode: (filename: string) => void;
}

export function RunDetail({ runId, onBack, onPickEpisode }: Props) {
  const live = useQuery({
    queryKey: ["live", runId],
    queryFn: () => fetchLivePayload(runId),
    refetchInterval: 2_000,
  });

  const idx = useQuery({
    queryKey: ["episode-index", runId],
    queryFn: () => fetchEpisodeIndex(runId),
    refetchInterval: 3_000,
  });

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-line bg-chrome px-4 py-2.5">
        <button
          type="button"
          onClick={onBack}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-muted hover:bg-active hover:text-strong"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Runs
        </button>
        <ChevronRight className="h-3.5 w-3.5 text-faint" />
        <div className="truncate font-mono text-xs text-strong">{runId}</div>
      </div>

      <LiveBanner data={live.data} />

      <div className="flex-1 overflow-auto">
        {idx.isLoading && (
          <div className="px-4 py-6 text-sm text-subtle">loading episodes…</div>
        )}
        {idx.data && idx.data.length === 0 && (
          <div className="px-4 py-6 text-sm text-subtle">
            no episodes written yet
          </div>
        )}
        {idx.data && idx.data.length > 0 && (
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10 bg-canvas">
              <tr className="border-b border-line text-left">
                {[
                  "Idx",
                  "Task",
                  "Correct",
                  "Reward",
                  "Trajectories",
                  "Steps",
                  "Termination",
                  "Instruction",
                ].map((h) => (
                  <th
                    key={h}
                    className="whitespace-nowrap px-3 py-2 text-[10px] font-semibold uppercase tracking-wide text-subtle"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {idx.data.map((e) => (
                <EpisodeRow
                  key={e.filename}
                  e={e}
                  onClick={() => onPickEpisode(e.filename)}
                />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function EpisodeRow({
  e,
  onClick,
}: {
  e: EpisodeIndexRow;
  onClick: () => void;
}) {
  return (
    <tr
      onClick={onClick}
      className="cursor-pointer border-b border-line-subtle hover:bg-hover"
    >
      <td className="px-3 py-2 font-mono text-xs text-body">
        {e.eval_idx ?? "—"}
      </td>
      <td className="max-w-[16rem] truncate px-3 py-2 font-mono text-xs text-strong">
        {e.task_id ?? "—"}
      </td>
      <td className="px-3 py-2 text-xs">
        {e.is_correct == null ? (
          <span className="text-subtle">—</span>
        ) : e.is_correct ? (
          <CircleCheck className="h-4 w-4 text-emerald-700 dark:text-emerald-400" />
        ) : (
          <CircleAlert className="h-4 w-4 text-rose-600 dark:text-rose-400" />
        )}
      </td>
      <td className="px-3 py-2 font-mono text-xs">
        {e.reward == null ? (
          <span className="text-subtle">—</span>
        ) : (
          <span
            className={cn(
              e.reward >= 0.7
                ? "text-emerald-700 dark:text-emerald-400"
                : e.reward >= 0.4
                  ? "text-amber-600 dark:text-amber-400"
                  : "text-rose-600 dark:text-rose-400",
            )}
          >
            {percent(e.reward)}
          </span>
        )}
      </td>
      <td className="px-3 py-2 font-mono text-xs text-body">
        {compactNumber(e.n_trajectories)}
      </td>
      <td className="px-3 py-2 font-mono text-xs text-body">
        {compactNumber(e.n_steps)}
      </td>
      <td className="px-3 py-2 text-xs text-muted">
        {e.termination_reason ?? "—"}
      </td>
      <td className="max-w-[28rem] truncate px-3 py-2 text-xs text-subtle">
        {e.instruction_preview}
      </td>
    </tr>
  );
}

function LiveBanner({ data }: { data: { in_flight: Array<{ idx: number; task_id: string | null; elapsed_s: number | null; trace_count: number; }>; finished_count: number; started_count: number } | undefined }) {
  if (!data || data.in_flight.length === 0) return null;
  return (
    <div className="border-b border-amber-200 dark:border-amber-900/60 bg-amber-50 dark:bg-amber-950/40 px-4 py-2">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">
        Live — {data.in_flight.length} in flight ·{" "}
        {data.finished_count}/{data.started_count} finished
      </div>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-amber-800 dark:text-amber-200">
        {data.in_flight.slice(0, 8).map((t) => (
          <span key={t.idx}>
            #{t.idx} {t.task_id ?? "?"}
            {t.elapsed_s != null && (
              <span className="ml-1 text-amber-700/70 dark:text-amber-600 dark:text-amber-400/70">
                {Math.round(t.elapsed_s)}s
              </span>
            )}
            <span className="ml-1 text-amber-700/70 dark:text-amber-600 dark:text-amber-400/70">
              ({t.trace_count} traces)
            </span>
          </span>
        ))}
        {data.in_flight.length > 8 && (
          <span className="text-amber-700/70 dark:text-amber-600 dark:text-amber-400/70">
            +{data.in_flight.length - 8} more
          </span>
        )}
      </div>
    </div>
  );
}

import { useQuery } from "@tanstack/react-query";

import { fetchSessionRuns } from "~/lib/api";
import { compactNumber, timeAgo } from "~/lib/format";
import { cn } from "~/lib/utils";

interface Props {
  selected: string | null;
  onSelect: (runId: string | null) => void;
}

/**
 * Left pane — every gateway run with derived counts. Auto-refreshes
 * every 5s so a live eval shows up without a manual reload.
 */
export function RunsList({ selected, onSelect }: Props) {
  const q = useQuery({
    queryKey: ["session-runs"],
    queryFn: fetchSessionRuns,
    refetchInterval: 5_000,
  });

  return (
    <div className="flex h-full flex-col border-r border-line">
      <div className="flex items-center justify-between border-b border-line px-3 py-2">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted">
          Runs
        </div>
        <div className="font-mono text-[10px] text-subtle">
          {q.data ? `${q.data.length}` : ""}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        {q.isLoading && (
          <div className="px-3 py-4 text-xs text-subtle">loading…</div>
        )}
        {q.isError && (
          <div className="px-3 py-4 text-xs text-rose-600 dark:text-rose-400">
            failed to load runs
          </div>
        )}
        {q.data?.length === 0 && (
          <div className="px-3 py-4 text-xs text-subtle">
            no runs yet — kick off an eval to populate
          </div>
        )}
        {q.data?.map((r) => {
          const isActive = selected === r.run_id;
          return (
            <button
              key={r.run_id}
              type="button"
              onClick={() => onSelect(isActive ? null : r.run_id)}
              className={cn(
                "block w-full border-b border-line-subtle px-3 py-2 text-left transition-colors",
                isActive
                  ? "bg-active"
                  : "hover:bg-hover",
              )}
            >
              <div className="truncate font-mono text-[11px] text-strong">
                {r.run_id}
              </div>
              <div className="mt-1 flex items-center gap-3 text-[10px] text-subtle">
                <span>{compactNumber(r.session_count)} sessions</span>
                <span>{compactNumber(r.trace_count)} traces</span>
                <span className="ml-auto">{timeAgo(r.last_trace_at)}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

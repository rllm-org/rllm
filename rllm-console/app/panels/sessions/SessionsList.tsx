import { useQuery } from "@tanstack/react-query";

import { fetchSessions } from "~/lib/api";
import { compactNumber, timeAgo } from "~/lib/format";
import { cn } from "~/lib/utils";

interface Props {
  runId: string | null;
  selected: string | null;
  onSelect: (sessionId: string | null) => void;
}

/**
 * Middle pane — sessions within the selected run. Disabled state when
 * no run is picked (sessions cross-run is a less useful default).
 */
export function SessionsList({ runId, selected, onSelect }: Props) {
  const q = useQuery({
    queryKey: ["sessions", runId],
    queryFn: () => fetchSessions(runId ?? undefined),
    enabled: runId !== null,
    refetchInterval: 3_000,
  });

  return (
    <div className="flex h-full flex-col border-r border-line">
      <div className="flex items-center justify-between border-b border-line px-3 py-2">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted">
          Sessions
        </div>
        <div className="font-mono text-[10px] text-subtle">
          {q.data ? `${q.data.length}` : ""}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        {runId === null && (
          <div className="px-3 py-4 text-xs text-subtle">
            pick a run on the left
          </div>
        )}
        {runId !== null && q.isLoading && (
          <div className="px-3 py-4 text-xs text-subtle">loading…</div>
        )}
        {q.data?.length === 0 && (
          <div className="px-3 py-4 text-xs text-subtle">
            no sessions in this run
          </div>
        )}
        {q.data?.map((s) => {
          const isActive = selected === s.session_id;
          return (
            <button
              key={`${s.run_id}::${s.session_id}`}
              type="button"
              onClick={() => onSelect(isActive ? null : s.session_id)}
              className={cn(
                "block w-full border-b border-line-subtle px-3 py-2 text-left transition-colors",
                isActive ? "bg-active" : "hover:bg-hover",
              )}
            >
              <div className="truncate font-mono text-[11px] text-strong">
                {s.session_id}
              </div>
              <div className="mt-1 flex items-center gap-3 text-[10px] text-subtle">
                <span>{compactNumber(s.trace_count)} traces</span>
                <span className="ml-auto">{timeAgo(s.last_at)}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

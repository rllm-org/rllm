import { useQuery } from "@tanstack/react-query";

import { fetchTraces, type TraceRecord } from "~/lib/api";
import { TraceCard } from "./TraceCard";

interface Props {
  runId: string | null;
  sessionId: string | null;
}

/**
 * Right pane — full trace timeline for the selected session. Polls
 * every 2s for live tail; switching sessions rebuilds the cache key.
 */
export function TracesPane({ runId, sessionId }: Props) {
  const q = useQuery({
    queryKey: ["traces", runId, sessionId],
    queryFn: () =>
      sessionId
        ? fetchTraces({ session_id: sessionId, run_id: runId ?? undefined })
        : Promise.resolve([] as TraceRecord[]),
    enabled: sessionId !== null,
    refetchInterval: 2_000,
  });

  if (sessionId === null) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-subtle">
        pick a session
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-line bg-chrome px-4 py-2">
        <div className="truncate font-mono text-xs text-body">
          {sessionId}
        </div>
        <div className="font-mono text-[10px] text-subtle">
          {q.data ? `${q.data.length} traces` : ""}
        </div>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {q.isLoading && (
          <div className="text-xs text-subtle">loading traces…</div>
        )}
        {q.data?.length === 0 && (
          <div className="text-xs text-subtle">
            no traces yet — they appear here as the session emits LLM calls
          </div>
        )}
        {q.data?.map((t, i) => (
          <TraceCard key={i} trace={t} />
        ))}
      </div>
    </div>
  );
}

import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { TraceCard } from "~/components/trace/TraceCard";
import {
  fetchLivePayload,
  fetchRunTraces,
  type RunSessionSummary,
  type TraceRecord,
} from "~/lib/api";
import { compactNumber, timeAgo } from "~/lib/format";
import { cn } from "~/lib/utils";

const PAGE_SIZE = 100;
const SESSIONS_POLL_MS = 3_000;
const LIVE_TAIL_MS = 2_000;

/**
 * Live, Sessions-style view of a single run. Left pane lists sessions
 * (one per task/episode in flight) sorted by most-recent activity;
 * right pane is the trace feed for the selected session, where each
 * trace renders as a Step card.
 *
 * Step semantics: one ``TraceRecord`` == one LLM call == one ``Step``.
 * Multi-call steps would group on ``trace.step_id`` once harnesses
 * stamp it consistently — defer until that lands.
 */
export function RunStepsView({ runId }: { runId: string }) {
  const live = useQuery({
    queryKey: ["run-live", runId],
    queryFn: () => fetchLivePayload(runId),
    refetchInterval: SESSIONS_POLL_MS,
  });

  const sessions = live.data?.sessions ?? [];
  const [selected, setSelected] = useState<string | null>(null);

  // Auto-select the most-recent session when none is picked yet.
  useEffect(() => {
    if (selected !== null) return;
    if (sessions.length === 0) return;
    setSelected(sessions[0].session_id);
  }, [selected, sessions]);

  return (
    <div className="flex h-full overflow-hidden">
      <div className="w-80 shrink-0 border-r border-line">
        <SessionsList
          sessions={sessions}
          selected={selected}
          onSelect={setSelected}
          empty={!live.isLoading && sessions.length === 0}
        />
      </div>
      <div className="min-w-0 flex-1">
        {selected ? (
          <RunTraceFeed runId={runId} sessionId={selected} />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-subtle">
            {live.isLoading ? "loading sessions…" : "pick a session"}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Left pane — sessions list
// ---------------------------------------------------------------------------

function SessionsList({
  sessions,
  selected,
  onSelect,
  empty,
}: {
  sessions: RunSessionSummary[];
  selected: string | null;
  onSelect: (sid: string) => void;
  empty: boolean;
}) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-line bg-chrome px-3 py-2">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-subtle">
          Sessions
        </div>
        <div className="font-mono text-[10px] text-subtle">{sessions.length}</div>
      </div>
      <div className="flex-1 overflow-y-auto">
        {empty && (
          <div className="px-3 py-4 text-xs text-subtle">
            no sessions yet — they appear as the run emits LLM calls
          </div>
        )}
        {sessions.map((s) => (
          <button
            key={s.session_id}
            type="button"
            onClick={() => onSelect(s.session_id)}
            className={cn(
              "flex w-full flex-col gap-0.5 border-b border-line-subtle px-3 py-2 text-left hover:bg-hover",
              selected === s.session_id && "bg-active",
            )}
          >
            <div className="truncate font-mono text-xs text-strong">
              {s.session_id}
            </div>
            <div className="flex items-center gap-2 font-mono text-[10px] text-subtle">
              <span>{compactNumber(s.trace_count)} traces</span>
              {s.last_at != null && <span>· {timeAgo(s.last_at)}</span>}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Right pane — per-session trace feed (Step cards)
// ---------------------------------------------------------------------------

function RunTraceFeed({ runId, sessionId }: { runId: string; sessionId: string }) {
  // Per-session timeline: oldest-first so Step #1 is at the top.
  const back = useInfiniteQuery({
    queryKey: ["run-traces", runId, sessionId],
    initialPageParam: undefined as number | undefined,
    queryFn: ({ pageParam }) =>
      fetchRunTraces(runId, {
        session_id: sessionId,
        since: pageParam,
        limit: PAGE_SIZE,
        order: "ASC",
      }),
    getNextPageParam: (last) =>
      last.length < PAGE_SIZE
        ? undefined
        : (last[last.length - 1]?._created_at ?? undefined),
  });

  const allTraces = useMemo(() => back.data?.pages.flat() ?? [], [back.data]);
  const cursor = allTraces[allTraces.length - 1]?._created_at;

  // Live tail: newer traces append on the bottom (next Step).
  const live = useQuery({
    queryKey: ["run-traces-tail", runId, sessionId, cursor],
    queryFn: () =>
      fetchRunTraces(runId, {
        session_id: sessionId,
        since: cursor ?? undefined,
        limit: PAGE_SIZE,
        order: "ASC",
      }),
    enabled: cursor != null,
    refetchInterval: LIVE_TAIL_MS,
  });

  const liveTraces = live.data ?? [];
  const seen = new Set<unknown>();
  const merged: TraceRecord[] = [];
  for (const t of [...allTraces, ...liveTraces]) {
    const key = t.trace_id ?? `${t._created_at}-${t.session_id ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(t);
  }

  // Auto-scroll on new tail entries so live runs feel "alive".
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const lastLenRef = useRef(merged.length);
  useEffect(() => {
    if (merged.length > lastLenRef.current && scrollRef.current) {
      const el = scrollRef.current;
      // Only auto-stick if user is already near the bottom.
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      if (distanceFromBottom < 120) {
        el.scrollTop = el.scrollHeight;
      }
    }
    lastLenRef.current = merged.length;
  }, [merged.length]);

  // Bottom sentinel for "load more older" — we paginate *forward* in
  // ASC order, so "more" means newer/older relative to the cursor.
  // Actually with ASC + cursor=last, fetchNextPage gets newer rows.
  // For per-session timelines that rarely overflows PAGE_SIZE, this is
  // fine; truly long sessions need a smarter strategy (deferred).
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && back.hasNextPage && !back.isFetchingNextPage) {
          back.fetchNextPage();
        }
      },
      { rootMargin: "200px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [back]);

  if (back.isLoading) {
    return <div className="p-4 text-xs text-subtle">loading steps…</div>;
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-line bg-chrome px-4 py-2 text-[10px] font-mono text-subtle">
        <div className="truncate">{sessionId}</div>
        <div>
          {merged.length} step{merged.length === 1 ? "" : "s"}
        </div>
      </div>
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4">
        {merged.length === 0 && (
          <div className="text-xs text-subtle">no traces in this session yet</div>
        )}
        {merged.map((t, i) => (
          <StepCard key={(t.trace_id as string) ?? i} stepNumber={i + 1} trace={t} />
        ))}
        <div ref={sentinelRef} />
      </div>
    </div>
  );
}

function StepCard({
  stepNumber,
  trace,
}: {
  stepNumber: number;
  trace: TraceRecord;
}) {
  return (
    <div className="space-y-1">
      <div className="font-mono text-[10px] uppercase tracking-wide text-faint">
        Step #{stepNumber}
      </div>
      <TraceCard trace={trace} />
    </div>
  );
}

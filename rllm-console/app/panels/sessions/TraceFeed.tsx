import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { TraceCard } from "~/components/trace/TraceCard";
import {
  fetchTraceFeed,
  type TraceFilters,
  type TraceRecord,
} from "~/lib/api";

const PAGE_SIZE = 100;
// Shared tail-poll interval (ms) for the live "newer than last seen" cursor.
const LIVE_POLL_MS = 2_000;

interface Props {
  filters: TraceFilters;
  onPinSession: (sessionId: string) => void;
}

/**
 * Global trace feed. Two query streams stitched together:
 *
 * 1. ``useInfiniteQuery`` — pages older traces via ``until=lastSeen``
 *    cursor. The user scrolls and we extend the list.
 * 2. ``useQuery`` — polls ``since=newest`` every 2s so newly-arrived
 *    traces prepend without disturbing the scroll-back state.
 *
 * Filters are part of every cache key so a filter change rebuilds both
 * streams from scratch. Sessions can be "pinned" by clicking the
 * session-id chip on a card; the parent owns that state, we just
 * render the click affordance.
 */
export function TraceFeed({ filters, onPinSession }: Props) {
  // Older-pages stream.
  const back = useInfiniteQuery({
    queryKey: ["sessions-feed", filters],
    initialPageParam: undefined as number | undefined,
    queryFn: ({ pageParam }) =>
      fetchTraceFeed({
        ...filters,
        until: pageParam,
        limit: PAGE_SIZE,
      }),
    getNextPageParam: (last) =>
      last.length < PAGE_SIZE
        ? undefined
        : (last[last.length - 1]?._created_at ?? undefined),
  });

  const oldestSeenCursor =
    back.data?.pages.flat().reduce<number | null>((acc, t) => {
      if (acc == null || t._created_at < acc) return t._created_at;
      return acc;
    }, null) ?? null;
  const newestSeenCursor =
    back.data?.pages[0]?.[0]?._created_at ?? null;

  // Live-tail stream — newer than the newest we've rendered.
  const live = useQuery({
    queryKey: ["sessions-feed-tail", filters, newestSeenCursor],
    queryFn: () =>
      fetchTraceFeed({
        ...filters,
        since: newestSeenCursor ?? undefined,
        limit: PAGE_SIZE,
        // ASC because we want oldest-first within "everything since
        // the cursor"; we render newest-first afterwards.
        order: "ASC",
      }),
    enabled: newestSeenCursor != null,
    refetchInterval: LIVE_POLL_MS,
  });

  // Splice live + back. Live is sorted ASC; reverse before merging so
  // the final list stays DESC by created_at.
  const liveDesc = (live.data ?? []).slice().reverse();
  const olderDesc = back.data?.pages.flat() ?? [];
  const seen = new Set<unknown>();
  const all: TraceRecord[] = [];
  for (const t of [...liveDesc, ...olderDesc]) {
    const key = t.trace_id ?? `${t._created_at}-${t.session_id ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    all.push(t);
  }

  // Bottom sentinel for infinite scroll.
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
  }, [back, oldestSeenCursor]);

  if (back.isLoading) {
    return <div className="p-4 text-xs text-subtle">loading traces…</div>;
  }
  if (back.isError) {
    return (
      <div className="p-4 text-xs text-rose-600 dark:text-rose-400">
        failed to load traces — is the gateway db at ~/.rllm/gateway/traces.db?
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-line bg-chrome px-4 py-2 text-[10px] font-mono text-subtle">
        <div>{all.length} traces</div>
        <div>{back.isFetching ? "fetching…" : "live"}</div>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {all.length === 0 && (
          <div className="text-xs text-subtle">
            no traces match — relax filters or wait for the next LLM call to land
          </div>
        )}
        {all.map((t) => (
          <FeedCard
            key={(t.trace_id as string) ?? `${t._created_at}-${t.session_id ?? ""}`}
            trace={t}
            onPinSession={onPinSession}
          />
        ))}
        <div ref={sentinelRef} />
        {back.isFetchingNextPage && (
          <div className="py-2 text-center text-[10px] text-subtle">loading older…</div>
        )}
        {!back.hasNextPage && all.length > 0 && (
          <div className="py-2 text-center text-[10px] text-faint">end of feed</div>
        )}
      </div>
    </div>
  );
}

/** Trace card with a click-to-pin chip for the session_id. */
function FeedCard({
  trace,
  onPinSession,
}: {
  trace: TraceRecord;
  onPinSession: (sessionId: string) => void;
}) {
  const sessionId = trace.session_id as string | undefined;
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2 text-[10px] font-mono text-subtle">
        {trace.run_id ? <span>run: {trace.run_id as string}</span> : null}
        {sessionId ? (
          <button
            type="button"
            onClick={() => onPinSession(sessionId)}
            className="rounded border border-line px-1 hover:bg-hover hover:text-strong"
            title="pin session"
          >
            session: {sessionId}
          </button>
        ) : null}
        {trace.harness ? <span>harness: {trace.harness as string}</span> : null}
      </div>
      <TraceCard trace={trace} />
    </div>
  );
}

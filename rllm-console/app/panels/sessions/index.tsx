import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { useSearchParams } from "react-router";

import { fetchTraceFacets, type TraceFilters } from "~/lib/api";

import { FilterBar } from "./FilterBar";
import { TraceFeed } from "./TraceFeed";

/**
 * Global gateway-trace feed. Top: filter bar. Below: paginated trace
 * feed (newest first, infinite scroll back, live tail prepend).
 *
 * Filter state is encoded in the URL search params so links are
 * shareable. Filters that are unset stay out of the URL.
 */
export function SessionsPanel(_props: { panelId: string }) {
  const [params, setParams] = useSearchParams();

  const filters = useMemo<TraceFilters>(() => parseFilters(params), [params]);

  const facetsQuery = useQuery({
    queryKey: ["sessions-facets"],
    queryFn: fetchTraceFacets,
    refetchInterval: 30_000,
  });

  const onChange = (next: TraceFilters) => {
    setParams(serializeFilters(next), { replace: false });
  };

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <FilterBar
        filters={filters}
        facets={facetsQuery.data}
        onChange={onChange}
      />
      <div className="min-h-0 flex-1">
        <TraceFeed
          filters={filters}
          onPinSession={(sid) => onChange({ ...filters, session_id: sid })}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// URLSearchParams ⇄ TraceFilters
// ---------------------------------------------------------------------------

function parseFilters(p: URLSearchParams): TraceFilters {
  const get = (k: string) => p.get(k) ?? undefined;
  const num = (k: string) => {
    const v = p.get(k);
    if (v == null) return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };
  const bool = (k: string) => {
    const v = p.get(k);
    if (v == null) return undefined;
    return v === "true";
  };
  return {
    run_id: get("run_id"),
    session_id: get("session_id"),
    model: get("model"),
    harness: get("harness"),
    has_error: bool("has_error"),
    latency_min: num("latency_min"),
    latency_max: num("latency_max"),
    since: num("since"),
    until: num("until"),
  };
}

function serializeFilters(f: TraceFilters): URLSearchParams {
  const p = new URLSearchParams();
  if (f.run_id) p.set("run_id", f.run_id);
  if (f.session_id) p.set("session_id", f.session_id);
  if (f.model) p.set("model", f.model);
  if (f.harness) p.set("harness", f.harness);
  if (f.has_error != null) p.set("has_error", String(f.has_error));
  if (f.latency_min != null) p.set("latency_min", String(f.latency_min));
  if (f.latency_max != null) p.set("latency_max", String(f.latency_max));
  if (f.since != null) p.set("since", String(f.since));
  if (f.until != null) p.set("until", String(f.until));
  return p;
}

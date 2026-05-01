import { useEffect, useState } from "react";

import { RunsList } from "./RunsList";

// "SessionsPanel" is what the registry imports — keep this the
// canonical export from the index module.
import { SessionsList } from "./SessionsList";
import { TracesPane } from "./TracesPane";

/**
 * Three-pane layout: Runs (cross-run gateway list) → Sessions (within
 * selected run) → Traces (timeline for selected session).
 *
 * State flows top-down. Selecting a different run auto-clears session
 * selection so the right pane doesn't show stale traces.
 */
export function SessionsPanel(_props: { panelId: string }) {
  const [runId, setRunId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    setSessionId(null);
  }, [runId]);

  // Flex over grid: with `grid h-full grid-cols-[…]`, the implicit
  // single row sizes to content (auto), so children's `h-full` resolves
  // against an unconstrained track and the inner `overflow-y-auto` has
  // nothing to scroll against. Horizontal flex respects parent height
  // via `align-items: stretch`, so each pane gets the right height and
  // its inner scroll container actually scrolls. `min-w-0` on the wide
  // pane lets long trace cards shrink instead of pushing the layout.
  return (
    <div className="flex h-full overflow-hidden">
      <div className="w-72 shrink-0">
        <RunsList selected={runId} onSelect={setRunId} />
      </div>
      <div className="w-[22rem] shrink-0">
        <SessionsList
          runId={runId}
          selected={sessionId}
          onSelect={setSessionId}
        />
      </div>
      <div className="min-w-0 flex-1">
        <TracesPane runId={runId} sessionId={sessionId} />
      </div>
    </div>
  );
}

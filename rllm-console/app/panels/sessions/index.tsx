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

  return (
    <div className="grid h-full grid-cols-[18rem_22rem_1fr]">
      <RunsList selected={runId} onSelect={setRunId} />
      <SessionsList
        runId={runId}
        selected={sessionId}
        onSelect={setSessionId}
      />
      <TracesPane runId={runId} sessionId={sessionId} />
    </div>
  );
}

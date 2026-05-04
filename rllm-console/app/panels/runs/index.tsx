import { useEffect, useState } from "react";

import { EpisodeView } from "./EpisodeView";
import { RunDetail } from "./RunDetail";
import { RunsList } from "./RunsList";

/**
 * Two-pane layout: run list (left) + run detail (right). Mirrors the
 * Sessions panel's old shape: pick a run on the left, see everything
 * about it on the right without losing the list.
 *
 * Right pane has a third state — drilling into a single episode JSON
 * — which takes over the pane with a back affordance. The list stays
 * visible throughout, so jumping between runs is one click.
 */
export function RunsPanel(_props: { panelId: string }) {
  const [runId, setRunId] = useState<string | null>(null);
  const [episodeFile, setEpisodeFile] = useState<string | null>(null);

  // Switching runs auto-clears the episode drill-down so the right
  // pane doesn't show a stale episode from the previous run.
  useEffect(() => {
    setEpisodeFile(null);
  }, [runId]);

  return (
    <div className="flex h-full overflow-hidden">
      <div className="w-80 shrink-0 border-r border-line">
        <RunsList selected={runId} onSelect={setRunId} />
      </div>
      <div className="min-w-0 flex-1">
        {runId == null ? (
          <Empty />
        ) : episodeFile != null ? (
          <EpisodeView
            runId={runId}
            filename={episodeFile}
            onBack={() => setEpisodeFile(null)}
          />
        ) : (
          <RunDetail runId={runId} onPickEpisode={setEpisodeFile} />
        )}
      </div>
    </div>
  );
}

function Empty() {
  return (
    <div className="flex h-full items-center justify-center text-xs text-subtle">
      pick a run
    </div>
  );
}

import { useState } from "react";

import { EpisodeView } from "./EpisodeView";
import { RunDetail } from "./RunDetail";
import { RunsTable } from "./RunsTable";

/**
 * Three-stage navigation:
 *   1. RunsTable        — every eval run as a sortable grid
 *   2. RunDetail        — episode list + live in-flight banner
 *   3. EpisodeView      — task → trajectories → steps
 *
 * State is shallow on purpose; the URL doesn't track this yet (revisit
 * once we settle on bookmarkable links).
 */
export function RunsPanel(_props: { panelId: string }) {
  const [runId, setRunId] = useState<string | null>(null);
  const [episodeFile, setEpisodeFile] = useState<string | null>(null);

  if (runId && episodeFile) {
    return (
      <EpisodeView
        runId={runId}
        filename={episodeFile}
        onBack={() => setEpisodeFile(null)}
      />
    );
  }

  if (runId) {
    return (
      <RunDetail
        runId={runId}
        onBack={() => {
          setRunId(null);
          setEpisodeFile(null);
        }}
        onPickEpisode={setEpisodeFile}
      />
    );
  }

  return <RunsTable onPick={setRunId} />;
}

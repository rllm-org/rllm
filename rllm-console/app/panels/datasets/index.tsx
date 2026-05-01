import { useState } from "react";

import { DatasetDetail } from "./DatasetDetail";
import { DatasetGrid } from "./DatasetGrid";

/**
 * Two-stage navigation:
 *   1. DatasetGrid  — every registered dataset as a card (filterable
 *                     by category, searchable by name).
 *   2. DatasetDetail — header + per-split metadata + paginated entry
 *                      browser. Entries only renderable when the
 *                      split has a local parquet cache.
 *
 * State is shallow on purpose; URL bookmarking can come later.
 */
export function DatasetsPanel(_props: { panelId: string }) {
  const [name, setName] = useState<string | null>(null);
  if (name) {
    return <DatasetDetail name={name} onBack={() => setName(null)} />;
  }
  return <DatasetGrid onPick={setName} />;
}

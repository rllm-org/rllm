/**
 * Client-side panel registry.
 *
 * The backend's `/api/shell/info` is the source of truth for *which*
 * panels exist; this module maps those ids to React render components.
 * Adding a new panel = drop it under `app/panels/<id>/` and wire it here.
 *
 * If a backend-registered panel has no client entry, the shell renders
 * the `Placeholder` fallback.
 */
import type { ComponentType } from "react";

import { DatasetsPanel } from "./datasets";
import { Placeholder } from "./Placeholder";
import { SessionsPanel } from "./sessions";
import { RunsPanel } from "./runs";
import { SettingsPanel } from "./settings";

export type PanelComponent = ComponentType<{ panelId: string }>;

const REGISTRY: Record<string, PanelComponent> = {
  datasets: DatasetsPanel,
  sessions: SessionsPanel,
  runs: RunsPanel,
  settings: SettingsPanel,
};

export function getPanelComponent(id: string): PanelComponent {
  return REGISTRY[id] ?? Placeholder;
}

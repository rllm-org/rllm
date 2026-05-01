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

import { Placeholder } from "./Placeholder";
import { SessionsPanel } from "./sessions";
import { RunsPanel } from "./runs";

export type PanelComponent = ComponentType<{ panelId: string }>;

const REGISTRY: Record<string, PanelComponent> = {
  sessions: SessionsPanel,
  runs: RunsPanel,
};

export function getPanelComponent(id: string): PanelComponent {
  return REGISTRY[id] ?? Placeholder;
}

import { ConfigSection } from "./ConfigSection";
import { EnvVarsSection } from "./EnvVarsSection";

/**
 * Two-section page: read-only Config (paths, versions) on top,
 * editable Env Vars below. Env vars persist to ``~/.rllm/console.env``
 * and load on next ``rllm view`` startup.
 */
export function SettingsPanel(_props: { panelId: string }) {
  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-4xl space-y-6 p-6">
        <ConfigSection />
        <EnvVarsSection />
      </div>
    </div>
  );
}

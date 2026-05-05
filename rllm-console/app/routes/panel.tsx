import { useParams } from "react-router";

import { ConsoleLayout } from "~/components/Layout";
import { getPanelComponent } from "~/panels/registry";

export default function PanelRoute() {
  const { panelId } = useParams<{ panelId: string }>();
  if (!panelId) {
    return (
      <ConsoleLayout>
        <div className="flex h-full items-center justify-center text-sm text-subtle">
          no panel selected
        </div>
      </ConsoleLayout>
    );
  }
  const Panel = getPanelComponent(panelId);
  return (
    <ConsoleLayout>
      <Panel panelId={panelId} />
    </ConsoleLayout>
  );
}

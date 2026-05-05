import { useQuery } from "@tanstack/react-query";

import { fetchShellInfo } from "~/lib/api";
import { iconForPanel } from "~/lib/icons";

export function Placeholder({ panelId }: { panelId: string }) {
  const shell = useQuery({ queryKey: ["shell-info"], queryFn: fetchShellInfo });
  const panel = shell.data?.panels.find((p) => p.id === panelId);
  const Icon = iconForPanel(panel?.icon ?? "circle");
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-subtle">
      <Icon className="h-10 w-10" />
      <h2 className="text-lg font-medium text-body">
        {panel?.title ?? panelId}
      </h2>
      <p className="max-w-md text-center text-sm">
        This panel is registered but doesn&rsquo;t have a UI yet. Build one in{" "}
        <code className="rounded bg-active px-1.5 py-0.5 font-mono text-xs">
          rllm-console/app/panels/{panelId}/
        </code>{" "}
        and add it to{" "}
        <code className="rounded bg-active px-1.5 py-0.5 font-mono text-xs">
          panels/registry.tsx
        </code>
        .
      </p>
    </div>
  );
}

import { useQuery } from "@tanstack/react-query";
import { Navigate } from "react-router";

import { ConsoleLayout } from "~/components/Layout";
import { fetchShellInfo } from "~/lib/api";

/**
 * On open, send the user to the first non-placeholder panel — or the
 * first panel of any kind if every one is a placeholder. Avoids a stale
 * landing screen with no obvious next click.
 */
export default function Home() {
  const shell = useQuery({
    queryKey: ["shell-info"],
    queryFn: fetchShellInfo,
    staleTime: 60_000,
  });

  if (shell.isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-subtle">
        loading shell…
      </div>
    );
  }

  const panels = shell.data?.panels ?? [];
  const target =
    panels.find((p) => !p.placeholder)?.id ?? panels[0]?.id ?? null;
  if (target) return <Navigate to={`/p/${target}`} replace />;

  return (
    <ConsoleLayout>
      <div className="flex h-full items-center justify-center text-sm text-subtle">
        no panels registered
      </div>
    </ConsoleLayout>
  );
}

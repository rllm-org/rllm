import { useQuery } from "@tanstack/react-query";

import { fetchShellInfo } from "~/lib/api";
import { Sidebar } from "./Sidebar";

export function ConsoleLayout({ children }: { children: React.ReactNode }) {
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

  if (shell.isError || !shell.data) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-rose-600 dark:text-rose-400">
        could not reach rLLM Console backend
      </div>
    );
  }

  return (
    <div className="flex h-full">
      <Sidebar panels={shell.data.panels} version={shell.data.version} />
      <main className="flex-1 overflow-hidden">{children}</main>
    </div>
  );
}

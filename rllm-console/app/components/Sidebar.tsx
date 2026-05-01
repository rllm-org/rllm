import { Moon, Sun } from "lucide-react";
import { NavLink } from "react-router";

import type { PanelDescriptor } from "~/lib/api";
import { iconForPanel } from "~/lib/icons";
import { useTheme } from "~/lib/theme";
import { cn } from "~/lib/utils";

interface SidebarProps {
  panels: PanelDescriptor[];
  version: string;
}

export function Sidebar({ panels, version }: SidebarProps) {
  const { theme, toggle } = useTheme();
  return (
    <aside className="flex h-full w-56 flex-col border-r border-line bg-chrome">
      <div className="px-4 py-5">
        <div className="text-sm font-semibold tracking-wide text-strong">
          rLLM Console
        </div>
        <div className="mt-0.5 font-mono text-[10px] text-subtle">
          v{version}
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto px-2 py-2">
        {panels.map((p) => {
          const Icon = iconForPanel(p.icon);
          return (
            <NavLink
              key={p.id}
              to={`/p/${p.id}`}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors",
                  isActive
                    ? "bg-active text-strong"
                    : "text-muted hover:bg-active hover:text-strong",
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="truncate">{p.title}</span>
              {p.placeholder && (
                <span className="ml-auto text-[9px] uppercase text-faint">
                  soon
                </span>
              )}
            </NavLink>
          );
        })}
      </nav>
      <div className="border-t border-line px-2 py-2">
        <button
          type="button"
          onClick={toggle}
          aria-label={`switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title={`Theme: ${theme} — click to switch`}
          className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm text-muted transition-colors hover:bg-active hover:text-strong"
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4 shrink-0" />
          ) : (
            <Moon className="h-4 w-4 shrink-0" />
          )}
          <span className="truncate capitalize">{theme} mode</span>
        </button>
      </div>
    </aside>
  );
}

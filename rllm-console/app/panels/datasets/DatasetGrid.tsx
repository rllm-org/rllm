import { useQuery } from "@tanstack/react-query";
import { HardDrive, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { fetchDatasets, type DatasetCard as DatasetCardType } from "~/lib/api";
import { cn } from "~/lib/utils";

interface Props {
  onPick: (name: string) => void;
}

export function DatasetGrid({ onPick }: Props) {
  const q = useQuery({
    queryKey: ["datasets"],
    queryFn: fetchDatasets,
    staleTime: 60_000,
  });

  const [filter, setFilter] = useState("");
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [localOnly, setLocalOnly] = useState(false);

  const filtered = useMemo(() => {
    if (!q.data) return [];
    const f = filter.trim().toLowerCase();
    return q.data.datasets.filter((d) => {
      if (activeCategory && d.category !== activeCategory) return false;
      if (localOnly && !d.is_local) return false;
      if (!f) return true;
      return (
        d.name.toLowerCase().includes(f) ||
        d.description.toLowerCase().includes(f) ||
        d.source.toLowerCase().includes(f)
      );
    });
  }, [q.data, filter, activeCategory, localOnly]);

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-line bg-chrome">
        <div className="flex items-center justify-between px-4 py-2.5">
          <div className="text-sm font-medium text-strong">Datasets</div>
          <div className="font-mono text-[11px] text-subtle">
            {q.data
              ? `${filtered.length} of ${q.data.datasets.length}`
              : "loading…"}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-4 py-2">
          <div className="flex items-center gap-2 rounded border border-line bg-canvas px-2 py-1">
            <Search className="h-3.5 w-3.5 text-faint" />
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="search name, source, description…"
              className="w-64 bg-transparent text-xs text-strong placeholder:text-faint focus:outline-none"
            />
          </div>
          <button
            type="button"
            onClick={() => setActiveCategory(null)}
            className={cn(
              "rounded border px-2 py-1 text-[11px] transition-colors",
              activeCategory === null
                ? "border-line bg-active text-strong"
                : "border-line text-muted hover:bg-active hover:text-strong",
            )}
          >
            all
          </button>
          {q.data?.categories.map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setActiveCategory(activeCategory === c ? null : c)}
              className={cn(
                "rounded border px-2 py-1 text-[11px] transition-colors",
                activeCategory === c
                  ? "border-line bg-active text-strong"
                  : "border-line text-muted hover:bg-active hover:text-strong",
              )}
            >
              {c}
            </button>
          ))}
          <span className="ml-auto" />
          <label className="flex items-center gap-1.5 text-[11px] text-muted">
            <input
              type="checkbox"
              checked={localOnly}
              onChange={(e) => setLocalOnly(e.target.checked)}
              className="accent-zinc-500"
            />
            local only
          </label>
        </div>
      </header>

      <div className="flex-1 overflow-auto p-4">
        {q.isLoading && (
          <div className="text-sm text-subtle">loading datasets…</div>
        )}
        {filtered.length === 0 && q.data && (
          <div className="text-sm text-subtle">
            no datasets match the current filters
          </div>
        )}
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
          {filtered.map((d) => (
            <DatasetCard key={d.name} dataset={d} onClick={() => onPick(d.name)} />
          ))}
        </div>
      </div>
    </div>
  );
}

function DatasetCard({
  dataset,
  onClick,
}: {
  dataset: DatasetCardType;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex flex-col gap-2 rounded-md border border-line bg-chrome p-3 text-left transition-colors hover:border-line hover:bg-hover"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="font-mono text-xs font-semibold text-strong group-hover:text-strong">
          {dataset.name}
        </div>
        {dataset.is_local && (
          <span
            className="flex shrink-0 items-center gap-1 rounded bg-emerald-50 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
            title={`Cached locally: ${dataset.local_splits.join(", ")}`}
          >
            <HardDrive className="h-2.5 w-2.5" />
            local
          </span>
        )}
      </div>
      <p className="line-clamp-2 text-xs text-muted">
        {dataset.description || (
          <span className="text-faint">(no description)</span>
        )}
      </p>
      <div className="flex flex-wrap items-center gap-1 text-[10px]">
        <span className="rounded border border-line px-1.5 py-0.5 font-mono uppercase tracking-wide text-subtle">
          {dataset.category}
        </span>
        {dataset.splits.map((s) => (
          <span
            key={s}
            className={cn(
              "rounded px-1.5 py-0.5 font-mono",
              dataset.local_splits.includes(s)
                ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                : "bg-active text-faint",
            )}
            title={
              dataset.local_splits.includes(s)
                ? "downloaded"
                : "not downloaded — only metadata available"
            }
          >
            {s}
          </span>
        ))}
      </div>
      <div className="truncate font-mono text-[10px] text-faint">
        {dataset.source}
      </div>
    </button>
  );
}

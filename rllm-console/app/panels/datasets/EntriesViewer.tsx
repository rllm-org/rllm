import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useState } from "react";

import { fetchDatasetEntries } from "~/lib/api";
import { compactNumber } from "~/lib/format";
import { cn } from "~/lib/utils";

const PAGE_SIZE = 25;

interface Props {
  name: string;
  split: string;
}

/**
 * Paginated entry browser for a downloaded split. Each entry renders
 * as a card with key/value rows; long string values get a 2-line clamp
 * with click-to-expand. JSON-y values render as collapsible nested
 * blocks so deeply structured rows (chat-style benchmarks, hf_image
 * fields) stay scannable.
 */
export function EntriesViewer({ name, split }: Props) {
  const [page, setPage] = useState(0);
  const offset = page * PAGE_SIZE;

  const q = useQuery({
    queryKey: ["dataset-entries", name, split, offset, PAGE_SIZE],
    queryFn: () => fetchDatasetEntries(name, split, offset, PAGE_SIZE),
    staleTime: 60_000,
  });

  return (
    <section className="rounded-md border border-line bg-chrome">
      <header className="flex items-center justify-between border-b border-line px-4 py-2.5">
        <div className="text-xs text-muted">
          Entries{" "}
          {q.data && (
            <>
              <span className="font-mono text-strong">
                {compactNumber(offset + 1)}–
                {compactNumber(Math.min(offset + PAGE_SIZE, q.data.total))}
              </span>{" "}
              <span className="text-faint">of {compactNumber(q.data.total)}</span>
            </>
          )}
        </div>
        <Pager
          page={page}
          totalPages={q.data?.n_pages ?? 0}
          onChange={setPage}
        />
      </header>
      <div className="space-y-2 p-3">
        {q.isLoading && (
          <div className="text-xs text-subtle">loading entries…</div>
        )}
        {q.isError && (
          <div className="text-xs text-rose-600 dark:text-rose-400">
            failed: {String(q.error)}
          </div>
        )}
        {q.data?.rows.map((row, i) => (
          <EntryCard key={offset + i} row={row} index={offset + i} />
        ))}
      </div>
    </section>
  );
}

function Pager({
  page,
  totalPages,
  onChange,
}: {
  page: number;
  totalPages: number;
  onChange: (p: number) => void;
}) {
  if (totalPages <= 1) return null;
  return (
    <div className="flex items-center gap-1 text-xs">
      <button
        type="button"
        onClick={() => onChange(Math.max(0, page - 1))}
        disabled={page === 0}
        className="rounded p-1 text-muted hover:bg-active hover:text-strong disabled:opacity-30"
      >
        <ChevronLeft className="h-3.5 w-3.5" />
      </button>
      <span className="font-mono text-[11px] text-subtle">
        {page + 1} / {totalPages}
      </span>
      <button
        type="button"
        onClick={() => onChange(Math.min(totalPages - 1, page + 1))}
        disabled={page >= totalPages - 1}
        className="rounded p-1 text-muted hover:bg-active hover:text-strong disabled:opacity-30"
      >
        <ChevronRight className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

function EntryCard({
  row,
  index,
}: {
  row: Record<string, unknown>;
  index: number;
}) {
  return (
    <div className="rounded border border-line bg-canvas">
      <div className="flex items-center justify-between border-b border-line px-3 py-1.5">
        <span className="font-mono text-[11px] text-subtle">#{index}</span>
        <span className="font-mono text-[10px] text-faint">
          {Object.keys(row).length} fields
        </span>
      </div>
      <dl className="divide-y divide-line">
        {Object.entries(row).map(([k, v]) => (
          <FieldRow key={k} k={k} v={v} />
        ))}
      </dl>
    </div>
  );
}

function FieldRow({ k, v }: { k: string; v: unknown }) {
  const [expanded, setExpanded] = useState(false);
  const text = formatValue(v);
  const isLong = typeof v === "string" ? v.length > 200 : text.length > 200;

  return (
    <div className="grid grid-cols-[10rem_1fr] items-start gap-3 px-3 py-1.5">
      <dt className="truncate font-mono text-[11px] text-muted" title={k}>
        {k}
      </dt>
      <dd
        className={cn(
          "min-w-0 break-words font-mono text-[11px] text-strong",
          !expanded && "line-clamp-3 cursor-pointer",
          expanded && "whitespace-pre-wrap",
        )}
        onClick={() => isLong && setExpanded((e) => !e)}
      >
        {text}
        {!expanded && isLong && (
          <span className="ml-2 text-faint">click to expand</span>
        )}
      </dd>
    </div>
  );
}

function formatValue(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

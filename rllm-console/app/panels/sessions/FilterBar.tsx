import { X } from "lucide-react";
import type { ChangeEvent } from "react";

import type { TraceFacets, TraceFilters } from "~/lib/api";
import { cn } from "~/lib/utils";

const TIME_PRESETS: { label: string; seconds: number | null }[] = [
  { label: "all time", seconds: null },
  { label: "5m", seconds: 5 * 60 },
  { label: "15m", seconds: 15 * 60 },
  { label: "1h", seconds: 60 * 60 },
  { label: "24h", seconds: 24 * 60 * 60 },
];

interface Props {
  filters: TraceFilters;
  facets: TraceFacets | undefined;
  onChange: (next: TraceFilters) => void;
}

/**
 * Sticky filter row over the global trace feed. Every control is a
 * controlled component reading from + writing to the parent's filter
 * object, which lives in URL search params so the URL is shareable.
 *
 * The "time range" presets convert to a ``since`` cursor (now − N
 * seconds) at filter-change time; we don't recompute as the clock
 * advances, so a preset is a snapshot, not a sliding window. Trade-off
 * is intentional: live tail is handled separately by polling with a
 * ``since=lastSeen`` cursor.
 */
export function FilterBar({ filters, facets, onChange }: Props) {
  const set = (patch: Partial<TraceFilters>) => onChange({ ...filters, ...patch });
  const clear = () => onChange({});

  const activePreset = derivePresetLabel(filters.since);
  const hasFilters = Object.values(filters).some(
    (v) => v !== undefined && v !== null && v !== "",
  );

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-line bg-chrome px-3 py-2 text-xs">
      <Select
        label="run"
        value={filters.run_id ?? ""}
        options={facets?.runs ?? []}
        onChange={(v) => set({ run_id: v || undefined })}
      />
      <Select
        label="model"
        value={filters.model ?? ""}
        options={facets?.models ?? []}
        onChange={(v) => set({ model: v || undefined })}
      />
      <Select
        label="harness"
        value={filters.harness ?? ""}
        options={facets?.harnesses ?? []}
        onChange={(v) => set({ harness: v || undefined })}
      />
      <ToggleChip
        active={filters.has_error === true}
        onClick={() =>
          set({
            has_error: filters.has_error === true ? undefined : true,
          })
        }
      >
        errors only
      </ToggleChip>
      <NumberField
        label="min ms"
        value={filters.latency_min}
        onChange={(v) => set({ latency_min: v })}
      />
      <div className="flex items-center gap-1 rounded border border-line bg-base px-1.5 py-0.5">
        <span className="text-faint">time:</span>
        {TIME_PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            className={cn(
              "rounded px-1.5 py-0.5 font-mono uppercase tracking-wide",
              activePreset === p.label
                ? "bg-active text-strong"
                : "text-muted hover:bg-hover hover:text-strong",
            )}
            onClick={() => set({ since: presetToSince(p.seconds) })}
          >
            {p.label}
          </button>
        ))}
      </div>
      {filters.session_id && (
        <Pin label={`session: ${filters.session_id}`} onClear={() => set({ session_id: undefined })} />
      )}
      {hasFilters && (
        <button
          type="button"
          onClick={clear}
          className="ml-auto rounded border border-line px-2 py-0.5 font-mono text-faint hover:bg-hover hover:text-strong"
        >
          clear all
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-controls
// ---------------------------------------------------------------------------

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  const handle = (e: ChangeEvent<HTMLSelectElement>) => onChange(e.target.value);
  return (
    <label className="flex items-center gap-1 rounded border border-line bg-base px-1.5 py-0.5">
      <span className="text-faint">{label}:</span>
      <select
        value={value}
        onChange={handle}
        className="bg-transparent font-mono text-strong outline-none"
      >
        <option value="">any</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number | undefined;
  onChange: (v: number | undefined) => void;
}) {
  const handle = (e: ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    if (raw === "") return onChange(undefined);
    const n = Number(raw);
    if (Number.isFinite(n)) onChange(n);
  };
  return (
    <label className="flex items-center gap-1 rounded border border-line bg-base px-1.5 py-0.5">
      <span className="text-faint">{label}:</span>
      <input
        type="number"
        min={0}
        value={value ?? ""}
        onChange={handle}
        className="w-16 bg-transparent font-mono text-strong outline-none"
        placeholder="any"
      />
    </label>
  );
}

function ToggleChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded border px-2 py-0.5 font-mono uppercase tracking-wide",
        active
          ? "border-line bg-active text-strong"
          : "border-line text-muted hover:bg-hover hover:text-strong",
      )}
    >
      {children}
    </button>
  );
}

function Pin({ label, onClear }: { label: string; onClear: () => void }) {
  return (
    <span className="flex items-center gap-1 rounded border border-line bg-active px-2 py-0.5 font-mono text-strong">
      {label}
      <button
        type="button"
        onClick={onClear}
        className="text-subtle hover:text-strong"
        aria-label="clear pin"
      >
        <X className="h-3 w-3" />
      </button>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Time-range helpers
// ---------------------------------------------------------------------------

function presetToSince(seconds: number | null): number | undefined {
  if (seconds == null) return undefined;
  return Date.now() / 1000 - seconds;
}

function derivePresetLabel(since: number | undefined): string {
  if (since == null) return "all time";
  const ago = Date.now() / 1000 - since;
  // Match within a 30s slop so the highlight survives small drifts.
  for (const p of TIME_PRESETS) {
    if (p.seconds == null) continue;
    if (Math.abs(ago - p.seconds) < 30) return p.label;
  }
  return "";
}

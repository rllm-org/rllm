import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  ChevronRight,
  Download,
  ExternalLink,
  HardDrive,
} from "lucide-react";
import { useEffect, useState } from "react";

import {
  fetchDatasetDetail,
  type DatasetSplitInfo,
} from "~/lib/api";
import { compactNumber } from "~/lib/format";
import { cn } from "~/lib/utils";

import { EntriesViewer } from "./EntriesViewer";
import { PullDialog } from "./PullDialog";

interface Props {
  name: string;
  onBack: () => void;
}

export function DatasetDetail({ name, onBack }: Props) {
  const q = useQuery({
    queryKey: ["dataset-detail", name],
    queryFn: () => fetchDatasetDetail(name),
    staleTime: 60_000,
  });

  // Pick the first locally-cached split as the default focus when the
  // detail loads, falling back to the first split otherwise.
  const [activeSplit, setActiveSplit] = useState<string | null>(null);
  useEffect(() => {
    if (!q.data || activeSplit !== null) return;
    const local = q.data.splits_detail.find((s) => s.is_local);
    setActiveSplit(local?.name ?? q.data.splits_detail[0]?.name ?? null);
  }, [q.data, activeSplit]);

  const [pulling, setPulling] = useState(false);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-line bg-chrome px-4 py-2.5">
        <button
          type="button"
          onClick={onBack}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-muted hover:bg-active hover:text-strong"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Datasets
        </button>
        <ChevronRight className="h-3.5 w-3.5 text-faint" />
        <div className="truncate font-mono text-xs text-strong">{name}</div>
      </div>

      <div className="flex-1 overflow-auto">
        {q.isLoading && (
          <div className="px-4 py-6 text-sm text-subtle">loading…</div>
        )}
        {q.isError && (
          <div className="px-4 py-6 text-sm text-rose-600 dark:text-rose-400">
            failed to load: {String(q.error)}
          </div>
        )}
        {q.data && (
          <div className="mx-auto max-w-5xl space-y-4 p-4">
            <DatasetHeader dataset={q.data} />
            <SplitTabs
              splits={q.data.splits_detail}
              active={activeSplit}
              onChange={setActiveSplit}
            />
            {activeSplit &&
              (q.data.splits_detail.find((s) => s.name === activeSplit)
                ?.is_local ? (
                <EntriesViewer name={name} split={activeSplit} />
              ) : (
                <NotDownloadedHint
                  name={name}
                  split={activeSplit}
                  onPull={() => setPulling(true)}
                />
              ))}
          </div>
        )}
      </div>
      {pulling && (
        <PullDialog name={name} onClose={() => setPulling(false)} />
      )}
    </div>
  );
}

function DatasetHeader({
  dataset,
}: {
  dataset: {
    name: string;
    description: string;
    source: string;
    category: string;
    default_agent: string | null;
    reward_fn: string | null;
    eval_split: string | null;
    instruction_field: string | null;
    transform: string | null;
    is_local: boolean;
  };
}) {
  return (
    <div className="rounded-md border border-line bg-chrome">
      <div className="border-b border-line px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="truncate font-mono text-base font-semibold text-strong">
                {dataset.name}
              </h2>
              <span className="rounded border border-line px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-subtle">
                {dataset.category}
              </span>
              {dataset.is_local && (
                <span className="flex items-center gap-1 rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
                  <HardDrive className="h-3 w-3" />
                  local
                </span>
              )}
            </div>
            <p className="mt-1 text-sm text-body">{dataset.description}</p>
          </div>
          {dataset.source && (
            <a
              href={`https://huggingface.co/datasets/${dataset.source}`}
              target="_blank"
              rel="noreferrer"
              className="flex shrink-0 items-center gap-1 text-xs text-muted hover:text-strong"
              title="Open on Hugging Face"
            >
              <span className="font-mono">{dataset.source}</span>
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>
      </div>
      <dl className="grid grid-cols-[10rem_1fr] gap-x-4 gap-y-2 px-4 py-3 text-xs">
        <Row label="Default agent" value={dataset.default_agent} mono />
        <Row label="Reward fn" value={dataset.reward_fn} mono />
        <Row label="Eval split" value={dataset.eval_split} mono />
        <Row label="Instruction field" value={dataset.instruction_field} mono />
        <Row label="Transform" value={dataset.transform} mono />
      </dl>
    </div>
  );
}

function Row({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string | null | undefined;
  mono?: boolean;
}) {
  return (
    <>
      <dt className="text-muted">{label}</dt>
      <dd className={mono ? "truncate font-mono text-strong" : "text-strong"}>
        {value ?? <span className="text-faint">—</span>}
      </dd>
    </>
  );
}

function SplitTabs({
  splits,
  active,
  onChange,
}: {
  splits: DatasetSplitInfo[];
  active: string | null;
  onChange: (name: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1 border-b border-line">
      {splits.map((s) => (
        <button
          key={s.name}
          type="button"
          onClick={() => onChange(s.name)}
          className={cn(
            "border-b-2 px-3 py-2 text-xs transition-colors",
            active === s.name
              ? "border-strong text-strong"
              : "border-transparent text-muted hover:text-strong",
          )}
        >
          <span className="font-mono">{s.name}</span>
          {s.n_rows != null && (
            <span className="ml-1.5 text-[10px] text-faint">
              · {compactNumber(s.n_rows)} rows
            </span>
          )}
          {!s.is_local && (
            <span className="ml-1.5 text-[10px] text-faint">· remote</span>
          )}
        </button>
      ))}
    </div>
  );
}

function NotDownloadedHint({
  name,
  split,
  onPull,
}: {
  name: string;
  split: string;
  onPull: () => void;
}) {
  return (
    <div className="rounded-md border border-line bg-chrome p-6 text-center">
      <p className="text-sm text-muted">
        Split{" "}
        <code className="rounded bg-active px-1.5 py-0.5 font-mono text-xs text-strong">
          {split}
        </code>{" "}
        is not cached locally — entry preview is unavailable.
      </p>
      <div className="mt-3 flex flex-col items-center gap-2">
        <button
          type="button"
          onClick={onPull}
          className="flex items-center gap-1.5 rounded border border-line bg-canvas px-3 py-1.5 text-xs text-strong transition-colors hover:bg-active"
        >
          <Download className="h-3.5 w-3.5" />
          Pull dataset
        </button>
        <p className="text-[10px] text-faint">
          equivalent to{" "}
          <code className="rounded bg-active px-1 py-0.5 font-mono">
            rllm dataset pull {name}
          </code>
        </p>
      </div>
    </div>
  );
}

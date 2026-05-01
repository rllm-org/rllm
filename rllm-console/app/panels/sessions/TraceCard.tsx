import { ChevronRight } from "lucide-react";
import { useState } from "react";

import type { TraceRecord } from "~/lib/api";
import { timeAgo } from "~/lib/format";
import { cn } from "~/lib/utils";

import { JsonField } from "./JsonField";

/**
 * Single LLM-call card. Header is always visible (model, timing,
 * usage); body expands to show the request payload, the response, and
 * any extra metadata.
 *
 * The shape is permissive — TraceRecord is gateway-internal and varies
 * across providers (OpenAI vs Anthropic, completion vs chat, sync vs
 * stream). We surface a few well-known fields and dump the rest as
 * a collapsible JSON blob so nothing is hidden by accident.
 */
export function TraceCard({ trace }: { trace: TraceRecord }) {
  const [open, setOpen] = useState(false);

  const data = trace as Record<string, unknown>;
  const model = (data.model as string | undefined) ?? "?";
  const status = (data.status_code as number | undefined) ?? null;
  const latencyMs = (data.duration_ms as number | undefined) ?? null;
  const usage = data.usage as
    | { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
    | undefined;

  const request = data.request ?? data.input ?? data.body ?? null;
  const response = data.response ?? data.output ?? null;

  return (
    <div className="rounded-md border border-line bg-chrome">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-hover"
      >
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-subtle transition-transform",
            open && "rotate-90",
          )}
        />
        <span className="truncate font-mono text-xs text-strong">
          {model}
        </span>
        {status != null && (
          <span
            className={cn(
              "font-mono text-[10px]",
              status >= 400 ? "text-rose-600 dark:text-rose-400" : "text-emerald-700 dark:text-emerald-400",
            )}
          >
            {status}
          </span>
        )}
        {latencyMs != null && (
          <span className="font-mono text-[10px] text-subtle">
            {(latencyMs / 1000).toFixed(2)}s
          </span>
        )}
        {usage?.total_tokens != null && (
          <span className="font-mono text-[10px] text-subtle">
            {usage.total_tokens} tok
          </span>
        )}
        <span className="ml-auto font-mono text-[10px] text-subtle">
          {timeAgo(trace._created_at)}
        </span>
      </button>
      {open && (
        <div className="space-y-3 border-t border-line p-3">
          {request != null && <JsonField label="request" value={request} />}
          {response != null && <JsonField label="response" value={response} />}
          <JsonField label="raw" value={data} initiallyOpen={false} />
        </div>
      )}
    </div>
  );
}

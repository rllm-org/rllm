import { ChevronRight } from "lucide-react";
import { useState } from "react";

import type { TraceRecord } from "~/lib/api";
import { timeAgo } from "~/lib/format";
import { cn } from "~/lib/utils";

import { Conversation } from "./Conversation";
import { JsonField } from "./JsonField";
import type { ChatMessage } from "./MessageBubble";

type ViewMode = "chat" | "raw";

/**
 * Single LLM-call card. Header is always visible (model, timing,
 * usage); body expands to a chat view (request messages + response
 * bubble) by default, with a "raw" toggle that swaps in collapsible
 * JSON for debugging.
 *
 * The trace shape varies across providers and gateway versions, but
 * the gateway flattens to a stable triple:
 *
 *   * ``messages``         — list of input chat messages
 *   * ``response_message`` — single assistant turn from the model
 *   * ``raw_request`` / ``raw_response`` — verbatim provider payloads
 *
 * Older traces (no ``messages`` flattening) fall back to ``request`` /
 * ``response`` / ``input`` / ``output`` fields, then to the raw view.
 */
export function TraceCard({ trace }: { trace: TraceRecord }) {
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<ViewMode>("chat");

  const data = trace as Record<string, unknown>;
  const model = (data.model as string | undefined) ?? "?";
  const status =
    (data.status_code as number | undefined) ??
    (data.finish_reason ? null : null);
  const finishReason = data.finish_reason as string | undefined;
  const latencyMs =
    (data.duration_ms as number | undefined) ??
    (data.latency_ms as number | undefined) ??
    null;

  const tokenCounts = data.token_counts as
    | { prompt?: number; completion?: number }
    | undefined;
  const usage = data.usage as
    | { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
    | undefined;
  const totalTokens =
    usage?.total_tokens ??
    ((tokenCounts?.prompt ?? 0) + (tokenCounts?.completion ?? 0) || null);

  const messages = pickMessages(data);
  const responseMessage = pickResponse(data);
  const hasChatShape = messages !== null || responseMessage !== null;

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
        <span className="truncate font-mono text-xs text-strong">{model}</span>
        {status != null && (
          <span
            className={cn(
              "font-mono text-[10px]",
              status >= 400
                ? "text-rose-600 dark:text-rose-400"
                : "text-emerald-700 dark:text-emerald-400",
            )}
          >
            {status}
          </span>
        )}
        {finishReason && (
          <span className="font-mono text-[10px] text-subtle">
            {finishReason}
          </span>
        )}
        {latencyMs != null && (
          <span className="font-mono text-[10px] text-subtle">
            {(latencyMs / 1000).toFixed(2)}s
          </span>
        )}
        {totalTokens != null && (
          <span className="font-mono text-[10px] text-subtle">
            {totalTokens} tok
          </span>
        )}
        <span className="ml-auto font-mono text-[10px] text-subtle">
          {timeAgo(trace._created_at)}
        </span>
      </button>
      {open && (
        <div className="space-y-3 border-t border-line p-3">
          {hasChatShape && (
            <div className="flex items-center justify-end gap-1 text-[10px]">
              <span className="text-faint">view:</span>
              <ToggleButton
                active={view === "chat"}
                onClick={() => setView("chat")}
              >
                chat
              </ToggleButton>
              <ToggleButton
                active={view === "raw"}
                onClick={() => setView("raw")}
              >
                raw
              </ToggleButton>
            </div>
          )}
          {hasChatShape && view === "chat" ? (
            <Conversation messages={messages} response={responseMessage} />
          ) : (
            <RawView data={data} />
          )}
        </div>
      )}
    </div>
  );
}

function ToggleButton({
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
        "rounded border px-2 py-0.5 font-mono uppercase tracking-wide transition-colors",
        active
          ? "border-line bg-active text-strong"
          : "border-line text-muted hover:bg-active hover:text-strong",
      )}
    >
      {children}
    </button>
  );
}

function RawView({ data }: { data: Record<string, unknown> }) {
  // Surface the most useful payloads up-top, raw blob below.
  const request = data.raw_request ?? data.request ?? data.input ?? data.body;
  const response = data.raw_response ?? data.response ?? data.output;
  return (
    <div className="space-y-3">
      {request != null && <JsonField label="request" value={request} />}
      {response != null && <JsonField label="response" value={response} />}
      <JsonField label="raw" value={data} initiallyOpen={false} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shape extraction
// ---------------------------------------------------------------------------

function pickMessages(data: Record<string, unknown>): ChatMessage[] | null {
  // Flattened shape (current gateway).
  const flat = data.messages;
  if (Array.isArray(flat) && flat.every((m) => isChatMessage(m))) {
    return flat as ChatMessage[];
  }
  // Provider-native shapes inside raw_request.
  const raw = data.raw_request as Record<string, unknown> | undefined;
  if (raw) {
    const m = raw.messages;
    if (Array.isArray(m) && m.every((x) => isChatMessage(x))) {
      return m as ChatMessage[];
    }
  }
  // Older traces: ``request: {messages: [...]}``.
  const req = data.request as Record<string, unknown> | undefined;
  if (req) {
    const m = req.messages;
    if (Array.isArray(m) && m.every((x) => isChatMessage(x))) {
      return m as ChatMessage[];
    }
  }
  return null;
}

function pickResponse(data: Record<string, unknown>): ChatMessage | null {
  // Gateway-flattened single response message.
  const rm = data.response_message;
  if (isChatMessage(rm)) return rm as ChatMessage;

  // OpenAI-shaped raw_response.choices[0].message
  const raw = data.raw_response as Record<string, unknown> | undefined;
  if (raw) {
    const choices = raw.choices;
    if (Array.isArray(choices) && choices.length > 0) {
      const first = choices[0] as Record<string, unknown> | undefined;
      const msg = first?.message;
      if (isChatMessage(msg)) return msg as ChatMessage;
    }
    // Anthropic-shaped: top-level role/content.
    if (isChatMessage(raw)) return raw as ChatMessage;
  }
  // Legacy `response: { message: ... }`.
  const resp = data.response as Record<string, unknown> | undefined;
  if (resp) {
    const m = resp.message;
    if (isChatMessage(m)) return m as ChatMessage;
    if (isChatMessage(resp)) return resp as ChatMessage;
  }
  return null;
}

function isChatMessage(v: unknown): boolean {
  return (
    typeof v === "object" &&
    v !== null &&
    "role" in (v as object) &&
    typeof (v as { role: unknown }).role === "string"
  );
}

import { Bot, Cpu, User, Wrench } from "lucide-react";
import { type ComponentType } from "react";

import { cn } from "~/lib/utils";

import { JsonField } from "./JsonField";
import { Markdown } from "./Markdown";

/**
 * Generic chat-message shape — covers both OpenAI and Anthropic
 * conventions. ``content`` may be a plain string, a list of content
 * parts (text/image_url/tool_use), or null for assistant turns that
 * are pure tool calls.
 */
export interface ChatMessage {
  role?: string;
  content?: unknown;
  tool_calls?: ToolCall[] | null;
  tool_call_id?: string | null;
  name?: string;
  refusal?: string | null;
  [k: string]: unknown;
}

export interface ToolCall {
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string | object };
}

const ROLE_STYLE: Record<
  string,
  { label: string; icon: ComponentType<{ className?: string }>; tone: string }
> = {
  system: {
    label: "system",
    icon: Cpu,
    tone: "border-amber-300 bg-amber-50 dark:border-amber-900/60 dark:bg-amber-950/40",
  },
  user: {
    label: "user",
    icon: User,
    tone: "border-line bg-chrome",
  },
  assistant: {
    label: "assistant",
    icon: Bot,
    tone: "border-emerald-200 bg-emerald-50 dark:border-emerald-700/40 dark:bg-emerald-950/30",
  },
  tool: {
    label: "tool",
    icon: Wrench,
    tone: "border-line bg-active",
  },
};

const FALLBACK_STYLE = {
  label: "?",
  icon: Bot,
  tone: "border-line bg-chrome",
};

export function MessageBubble({
  message,
  index,
}: {
  message: ChatMessage;
  index?: number;
}) {
  const role = (message.role ?? "?").toLowerCase();
  const style = ROLE_STYLE[role] ?? { ...FALLBACK_STYLE, label: role };
  const Icon = style.icon;
  const hasRefusal =
    typeof message.refusal === "string" && message.refusal.length > 0;
  const hasToolCalls = Array.isArray(message.tool_calls) && message.tool_calls.length > 0;

  return (
    <div className={cn("rounded-md border p-3", style.tone)}>
      <div className="mb-2 flex items-center gap-2">
        <Icon className="h-3.5 w-3.5 text-subtle" />
        <span className="font-mono text-[10px] uppercase tracking-wide text-subtle">
          {style.label}
        </span>
        {message.name && (
          <span className="font-mono text-[10px] text-subtle">
            · {String(message.name)}
          </span>
        )}
        {message.tool_call_id && (
          <span className="font-mono text-[10px] text-subtle">
            · id={String(message.tool_call_id).slice(-8)}
          </span>
        )}
        {index != null && (
          <span className="ml-auto font-mono text-[10px] text-faint">
            #{index}
          </span>
        )}
      </div>
      <ContentRenderer content={message.content} />
      {hasRefusal && (
        <div className="mt-2 rounded border border-rose-300 bg-rose-50 p-2 text-xs text-rose-700 dark:border-rose-900/60 dark:bg-rose-950/30 dark:text-rose-300">
          <span className="font-semibold">refusal:</span>{" "}
          {String(message.refusal)}
        </div>
      )}
      {hasToolCalls && (
        <div className="mt-2 space-y-2">
          {(message.tool_calls as ToolCall[]).map((tc, i) => (
            <ToolCallView key={tc.id ?? i} call={tc} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}

function ContentRenderer({ content }: { content: unknown }) {
  if (content == null || content === "") {
    return <span className="text-xs italic text-faint">(empty)</span>;
  }
  if (typeof content === "string") {
    return <Markdown>{content}</Markdown>;
  }
  // OpenAI / Anthropic multi-part content: an array of typed parts.
  if (Array.isArray(content)) {
    return (
      <div className="space-y-2">
        {content.map((part, i) => (
          <ContentPart key={i} part={part} />
        ))}
      </div>
    );
  }
  // Anything else (object) — fall back to JSON.
  return <JsonField label="content" value={content} initiallyOpen />;
}

interface ContentPart {
  type?: string;
  text?: string;
  image_url?: { url?: string } | string;
  source?: unknown;
  [k: string]: unknown;
}

function ContentPart({ part }: { part: unknown }) {
  if (typeof part !== "object" || part == null) {
    return <Markdown>{String(part)}</Markdown>;
  }
  const p = part as ContentPart;
  if (p.type === "text" && typeof p.text === "string") {
    return <Markdown>{p.text}</Markdown>;
  }
  if (p.type === "image_url") {
    const url = typeof p.image_url === "string" ? p.image_url : p.image_url?.url;
    if (typeof url === "string") {
      return (
        <div className="rounded border border-line bg-canvas p-2">
          <img
            src={url}
            alt=""
            className="max-h-64 max-w-full rounded"
            loading="lazy"
          />
          <div className="mt-1 truncate font-mono text-[10px] text-faint">
            {url}
          </div>
        </div>
      );
    }
  }
  // Anthropic-shape image / unknown part — JSON fallback.
  return <JsonField label={p.type ?? "part"} value={p} initiallyOpen />;
}

function ToolCallView({ call, index }: { call: ToolCall; index: number }) {
  const fnName = call.function?.name ?? "(unknown)";
  let args = call.function?.arguments;
  let parsed: unknown = null;
  if (typeof args === "string") {
    try {
      parsed = JSON.parse(args);
    } catch {
      parsed = null;
    }
  } else {
    parsed = args ?? null;
  }
  return (
    <div className="rounded border border-line bg-canvas p-2">
      <div className="flex items-center gap-2">
        <Wrench className="h-3 w-3 text-subtle" />
        <span className="font-mono text-[11px] text-strong">
          {fnName}
        </span>
        {call.id && (
          <span className="font-mono text-[10px] text-faint">
            id={call.id.slice(-8)}
          </span>
        )}
        <span className="ml-auto font-mono text-[10px] text-faint">
          #{index}
        </span>
      </div>
      {parsed != null ? (
        <JsonField label="arguments" value={parsed} initiallyOpen />
      ) : args != null ? (
        <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-base">
          {String(args)}
        </pre>
      ) : null}
    </div>
  );
}

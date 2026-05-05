import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Download, XCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { streamDatasetPull } from "~/lib/api";
import { cn } from "~/lib/utils";

interface Props {
  name: string;
  onClose: () => void;
}

type PullState = "running" | "ok" | "failed";

/**
 * Modal that streams ``rllm dataset pull <name>`` output line-by-line.
 *
 * Lifecycle:
 *   - Mounts → POSTs to /pull → consumes the SSE stream
 *   - On ``{type:"done", ok:true}`` → invalidates dataset queries so
 *     the rest of the UI refreshes (cards flip to is_local=true,
 *     detail view picks up the new split metadata).
 *   - On ``{type:"done", ok:false}`` → shows the failure but keeps
 *     the log visible so the user can copy the error.
 *   - On unmount before completion → AbortController cancels the
 *     fetch, server tears down the subprocess.
 */
export function PullDialog({ name, onClose }: Props) {
  const qc = useQueryClient();
  const [lines, setLines] = useState<string[]>([]);
  const [state, setState] = useState<PullState>("running");
  const [exitCode, setExitCode] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const tailRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const abort = new AbortController();
    let cancelled = false;
    (async () => {
      try {
        for await (const ev of streamDatasetPull(name, abort.signal)) {
          if (cancelled) return;
          if (ev.type === "log") {
            setLines((prev) => [...prev, ev.line]);
          } else if (ev.type === "done") {
            setExitCode(ev.exit_code);
            setState(ev.ok ? "ok" : "failed");
            if (ev.ok) {
              qc.invalidateQueries({ queryKey: ["datasets"] });
              qc.invalidateQueries({ queryKey: ["dataset-detail", name] });
            }
          }
        }
      } catch (e) {
        if (!cancelled) {
          setError(String(e));
          setState("failed");
        }
      }
    })();
    return () => {
      cancelled = true;
      abort.abort();
    };
  }, [name, qc]);

  // Auto-scroll log to bottom as new lines come in.
  useEffect(() => {
    tailRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
  }, [lines.length]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-2xl rounded-md border border-line bg-canvas shadow-lg">
        <header className="flex items-center gap-2 border-b border-line px-4 py-3">
          <Download className="h-4 w-4 text-muted" />
          <h3 className="text-sm font-semibold text-strong">
            Pulling{" "}
            <code className="rounded bg-active px-1.5 py-0.5 font-mono text-xs">
              {name}
            </code>
          </h3>
          <span className="ml-auto">
            <StatusPill state={state} exitCode={exitCode} />
          </span>
        </header>
        <div className="max-h-96 overflow-y-auto bg-chrome p-3 font-mono text-[11px] leading-relaxed">
          {lines.length === 0 && state === "running" && (
            <div className="text-faint">starting…</div>
          )}
          {lines.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-words text-body">
              {line}
            </div>
          ))}
          {error && (
            <div className="mt-2 text-rose-700 dark:text-rose-400">
              {error}
            </div>
          )}
          <div ref={tailRef} />
        </div>
        <footer className="flex items-center justify-end gap-2 border-t border-line px-4 py-3 text-xs">
          <button
            type="button"
            onClick={onClose}
            className={cn(
              "rounded px-3 py-1",
              state === "running"
                ? "text-muted hover:bg-active hover:text-strong"
                : "bg-active text-strong",
            )}
          >
            {state === "running" ? "Cancel" : "Close"}
          </button>
        </footer>
      </div>
    </div>
  );
}

function StatusPill({
  state,
  exitCode,
}: {
  state: PullState;
  exitCode: number | null;
}) {
  if (state === "running") {
    return (
      <span className="font-mono text-[10px] uppercase tracking-wide text-subtle">
        running…
      </span>
    );
  }
  if (state === "ok") {
    return (
      <span className="flex items-center gap-1 rounded bg-emerald-50 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
        <CheckCircle2 className="h-3 w-3" />
        done
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1 rounded bg-rose-50 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
      <XCircle className="h-3 w-3" />
      failed{exitCode != null && ` · exit ${exitCode}`}
    </span>
  );
}

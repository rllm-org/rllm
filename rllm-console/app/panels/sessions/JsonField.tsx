import { ChevronRight } from "lucide-react";
import { useState } from "react";

import { cn } from "~/lib/utils";

interface Props {
  label: string;
  value: unknown;
  initiallyOpen?: boolean;
}

/**
 * Collapsible JSON blob with a header. ``initiallyOpen`` defaults to
 * true for "first-class" fields (request/response) so users see content
 * without clicking; the ``raw`` dump defaults to closed to keep noise
 * down.
 */
export function JsonField({ label, value, initiallyOpen = true }: Props) {
  const [open, setOpen] = useState(initiallyOpen);
  const text =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);

  return (
    <div className="rounded border border-line">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 bg-hover px-2 py-1 text-left"
      >
        <ChevronRight
          className={cn(
            "h-3 w-3 text-subtle transition-transform",
            open && "rotate-90",
          )}
        />
        <span className="font-mono text-[10px] uppercase tracking-wide text-subtle">
          {label}
        </span>
      </button>
      {open && (
        <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-[11px] leading-relaxed text-body">
          {text}
        </pre>
      )}
    </div>
  );
}

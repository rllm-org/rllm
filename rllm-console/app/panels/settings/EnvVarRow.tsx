import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, ExternalLink, Eye, EyeOff, Pencil, Trash2, X } from "lucide-react";
import { useState } from "react";

import {
  deleteEnvVar,
  revealEnvVar,
  type EnvVarRow as EnvVarRowType,
} from "~/lib/api";
import { cn } from "~/lib/utils";

interface Props {
  row: EnvVarRowType;
  onSave: (value: string) => Promise<unknown>;
}

/**
 * One env var line. Three states:
 *   1. unset       — small "+ set value" button on the right
 *   2. set         — masked value + edit/reveal/delete controls
 *   3. editing     — input + save/cancel; reveals plaintext via the
 *                    /env/{key}?reveal=true endpoint
 */
export function EnvVarRowComponent({ row, onSave }: Props) {
  const qc = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [revealed, setRevealed] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const delMut = useMutation({
    mutationFn: () => deleteEnvVar(row.key),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["env-vars"] }),
  });

  async function startEdit() {
    setError(null);
    if (row.is_set) {
      try {
        const r = await revealEnvVar(row.key);
        setDraft(r.value);
      } catch {
        setDraft("");
      }
    } else {
      setDraft("");
    }
    setEditing(true);
  }

  async function save() {
    try {
      await onSave(draft);
      setEditing(false);
      setRevealed(null);
    } catch (e) {
      setError(String(e));
    }
  }

  async function toggleReveal() {
    if (revealed) {
      setRevealed(null);
      return;
    }
    try {
      const r = await revealEnvVar(row.key);
      setRevealed(r.value);
    } catch {
      // Silent — masked stays.
    }
  }

  return (
    <div className="flex items-start gap-3 rounded px-2 py-1.5 hover:bg-hover">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <code className="font-mono text-xs text-strong">{row.key}</code>
          {row.url && (
            <a
              href={row.url}
              target="_blank"
              rel="noreferrer"
              className="text-faint hover:text-muted"
              title="get a key"
            >
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
          <span className="text-[10px] text-faint">{row.label}</span>
          {row.is_set && !row.in_console_file && (
            <span
              title="Set in shell environment but not managed by the console — values you save here will override the shell."
              className="rounded border border-line px-1 py-0 font-mono text-[9px] uppercase tracking-wide text-subtle"
            >
              shell
            </span>
          )}
        </div>
        {!editing && (
          <div className="mt-0.5 truncate font-mono text-[11px] text-muted">
            {row.is_set ? (
              revealed ?? row.masked_value ?? ""
            ) : (
              <span className="text-faint">{row.description}</span>
            )}
          </div>
        )}
        {editing && (
          <div className="mt-1 flex items-center gap-2">
            <input
              type="text"
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void save();
                if (e.key === "Escape") setEditing(false);
              }}
              className="flex-1 rounded border border-line bg-canvas px-2 py-1 font-mono text-xs text-strong outline-none focus:border-zinc-500"
              placeholder={row.description}
            />
            <button
              type="button"
              onClick={save}
              className="rounded p-1 text-emerald-700 hover:bg-active dark:text-emerald-400"
              title="Save (Enter)"
            >
              <Check className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={() => setEditing(false)}
              className="rounded p-1 text-muted hover:bg-active hover:text-strong"
              title="Cancel (Esc)"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
        {error && (
          <div className="mt-1 text-[10px] text-rose-600 dark:text-rose-400">
            {error}
          </div>
        )}
      </div>

      {!editing && (
        <div className="flex items-center gap-0.5">
          {row.is_set && row.secret && (
            <button
              type="button"
              onClick={toggleReveal}
              className={cn(
                "rounded p-1 text-faint hover:bg-active hover:text-strong",
                revealed && "text-strong",
              )}
              title={revealed ? "Hide" : "Reveal"}
            >
              {revealed ? (
                <EyeOff className="h-3.5 w-3.5" />
              ) : (
                <Eye className="h-3.5 w-3.5" />
              )}
            </button>
          )}
          <button
            type="button"
            onClick={() => void startEdit()}
            className="rounded p-1 text-faint hover:bg-active hover:text-strong"
            title={row.is_set ? "Edit" : "Set value"}
          >
            <Pencil className="h-3.5 w-3.5" />
          </button>
          {row.in_console_file && (
            <button
              type="button"
              onClick={() => {
                if (confirm(`Remove ${row.key}?`)) delMut.mutate();
              }}
              className="rounded p-1 text-faint hover:bg-active hover:text-rose-600 dark:hover:text-rose-400"
              title="Remove"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      )}
    </div>
  );
}

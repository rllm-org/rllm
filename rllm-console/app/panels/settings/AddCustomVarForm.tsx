import { useState } from "react";

interface Props {
  onSave: (key: string, value: string) => Promise<void>;
  onCancel: () => void;
}

/**
 * Inline form for adding an env var that isn't in the well-known list.
 * Validates the key against the same regex the backend uses so the
 * user gets immediate feedback (the server also enforces it).
 */
export function AddCustomVarForm({ onSave, onCancel }: Props) {
  const [key, setKey] = useState("");
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);

  const keyValid = /^[A-Za-z_][A-Za-z0-9_]*$/.test(key);

  async function submit() {
    if (!keyValid) {
      setError("Key must start with a letter or _, and contain only [A-Z0-9_].");
      return;
    }
    try {
      await onSave(key, value);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="border-b border-line bg-canvas px-4 py-3">
      <div className="flex items-center gap-2">
        <input
          type="text"
          autoFocus
          placeholder="VAR_NAME"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          className="w-48 rounded border border-line bg-canvas px-2 py-1 font-mono text-xs text-strong outline-none focus:border-zinc-500"
        />
        <span className="font-mono text-xs text-faint">=</span>
        <input
          type="text"
          placeholder="value"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void submit();
            if (e.key === "Escape") onCancel();
          }}
          className="flex-1 rounded border border-line bg-canvas px-2 py-1 font-mono text-xs text-strong outline-none focus:border-zinc-500"
        />
        <button
          type="button"
          disabled={!keyValid || !value}
          onClick={() => void submit()}
          className="rounded bg-active px-3 py-1 text-xs text-strong disabled:opacity-40"
        >
          Save
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded px-3 py-1 text-xs text-muted hover:bg-active hover:text-strong"
        >
          Cancel
        </button>
      </div>
      {error && (
        <div className="mt-2 text-[11px] text-rose-600 dark:text-rose-400">
          {error}
        </div>
      )}
    </div>
  );
}

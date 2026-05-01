import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ExternalLink, Plus } from "lucide-react";
import { useState } from "react";

import { fetchEnvVars, setEnvVar, type EnvVarRow } from "~/lib/api";

import { AddCustomVarForm } from "./AddCustomVarForm";
import { EnvVarRowComponent } from "./EnvVarRow";

/**
 * Editable list of env vars, grouped by category. Mutations write
 * through to ``~/.rllm/console.env`` and apply to ``os.environ`` of
 * the running gateway, so changes take effect without restart.
 */
export function EnvVarsSection() {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["env-vars"],
    queryFn: fetchEnvVars,
    staleTime: 5_000,
  });

  const setMut = useMutation({
    mutationFn: (vars: { key: string; value: string }) =>
      setEnvVar(vars.key, vars.value),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["env-vars"] }),
  });

  const [showAdd, setShowAdd] = useState(false);

  if (q.isLoading) {
    return (
      <section className="rounded-md border border-line bg-chrome p-4 text-xs text-subtle">
        loading env vars…
      </section>
    );
  }
  if (!q.data) return null;

  // Group rows by category, preserving the order returned by the API.
  const byCategory = new Map<string, EnvVarRow[]>();
  for (const cat of q.data.categories) byCategory.set(cat, []);
  for (const row of q.data.rows) {
    const bucket = byCategory.get(row.category) ?? [];
    bucket.push(row);
    byCategory.set(row.category, bucket);
  }

  return (
    <section className="rounded-md border border-line bg-chrome">
      <header className="flex items-center justify-between border-b border-line px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-strong">
            Environment variables
          </h2>
          <p className="mt-0.5 text-xs text-subtle">
            Persisted to{" "}
            <code className="rounded bg-active px-1 py-0.5 font-mono text-[10px]">
              {q.data.console_env_file}
            </code>{" "}
            and applied to the running gateway.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowAdd((v) => !v)}
          className="flex items-center gap-1.5 rounded border border-line px-2.5 py-1 text-xs text-muted transition-colors hover:bg-active hover:text-strong"
        >
          <Plus className="h-3.5 w-3.5" />
          {showAdd ? "Cancel" : "Add custom"}
        </button>
      </header>

      {showAdd && (
        <AddCustomVarForm
          onSave={async (key, value) => {
            await setMut.mutateAsync({ key, value });
            setShowAdd(false);
          }}
          onCancel={() => setShowAdd(false)}
        />
      )}

      <div className="divide-y divide-line">
        {[...byCategory.entries()]
          .filter(([, rows]) => rows.length > 0)
          .map(([category, rows]) => (
            <div key={category} className="px-4 py-3">
              <div className="mb-2 flex items-center gap-2">
                <h3 className="text-[10px] font-semibold uppercase tracking-wide text-subtle">
                  {category}
                </h3>
                <span className="text-[10px] text-faint">·</span>
                <span className="text-[10px] text-faint">
                  {rows.filter((r) => r.is_set).length} / {rows.length} set
                </span>
              </div>
              <div className="space-y-1">
                {rows.map((row) => (
                  <EnvVarRowComponent
                    key={row.key}
                    row={row}
                    onSave={(value) =>
                      setMut.mutateAsync({ key: row.key, value })
                    }
                  />
                ))}
              </div>
            </div>
          ))}
      </div>
    </section>
  );
}

// Keeps the import set tidy: the icon is used by EnvVarRow directly,
// re-export here so the panel index doesn't need to know about it.
export { ExternalLink };

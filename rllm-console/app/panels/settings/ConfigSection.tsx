import { useQuery } from "@tanstack/react-query";

import { fetchConsoleConfig } from "~/lib/api";

export function ConfigSection() {
  const q = useQuery({
    queryKey: ["console-config"],
    queryFn: fetchConsoleConfig,
    staleTime: 60_000,
  });

  return (
    <section className="rounded-md border border-line bg-chrome">
      <header className="border-b border-line px-4 py-3">
        <h2 className="text-sm font-semibold text-strong">Config</h2>
        <p className="mt-0.5 text-xs text-subtle">
          Read-only — paths and versions resolved at startup.
        </p>
      </header>
      <div className="p-4">
        {q.isLoading && <div className="text-xs text-subtle">loading…</div>}
        {q.data && (
          <dl className="grid grid-cols-[12rem_1fr] gap-x-4 gap-y-2 text-xs">
            <Row label="Console version" value={q.data.version} />
            <Row label="URL prefix" value={q.data.url_prefix} mono />
            <Row label="rLLM home" value={q.data.rllm_home} mono />
            <Row label="Eval results root" value={q.data.eval_results_root} mono />
            <Row label="Gateway DB" value={q.data.gateway_db_path} mono />
            <Row label="Env file" value={q.data.console_env_file} mono />
          </dl>
        )}
      </div>
    </section>
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
        {value ?? <span className="text-faint">— not set</span>}
      </dd>
    </>
  );
}

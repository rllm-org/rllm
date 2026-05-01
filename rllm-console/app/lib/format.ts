/** Format a unix timestamp (seconds) as a relative string ("2m ago"). */
export function timeAgo(ts: number | null | undefined): string {
  if (!ts) return "—";
  const ms = Date.now() - ts * 1000;
  if (ms < 0) return "now";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

/** Compact integer formatter (1.2k, 3.4M). */
export function compactNumber(n: number | null | undefined): string {
  if (n == null) return "—";
  if (Math.abs(n) < 1000) return String(n);
  if (Math.abs(n) < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

/** Format a score in [0,1] as "12.3%" or "—" when null. */
export function percent(score: number | null | undefined): string {
  if (score == null) return "—";
  return `${(score * 100).toFixed(1)}%`;
}

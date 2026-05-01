import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "~/lib/utils";

/**
 * Markdown wrapper tuned for LLM-generated content. Uses Tailwind
 * utility classes per element rather than the `@tailwindcss/typography`
 * plugin, since prose styling on top of our custom dark/light tokens is
 * fiddly and we only render a small subset of markdown in practice.
 *
 * Special handling:
 *
 * * Code blocks render as monospace cards with theme-aware bg.
 * * Inline code gets a subtle pill background.
 * * Tables are scrollable so wide tool outputs don't blow the layout.
 * * LaTeX delimiters (\[..\], \(..\), $..$) are *not* rendered — they
 *   pass through as text. Adding KaTeX is ~250KB; revisit if there's
 *   demand.
 */
export function Markdown({
  children,
  className,
}: {
  children: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "text-sm leading-relaxed text-strong",
        "[&>*]:my-2 first:[&>*]:mt-0 last:[&>*]:mb-0",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (p) => <h1 className="text-base font-semibold" {...p} />,
          h2: (p) => <h2 className="text-base font-semibold" {...p} />,
          h3: (p) => <h3 className="text-sm font-semibold" {...p} />,
          p: (p) => <p className="whitespace-pre-wrap" {...p} />,
          ul: (p) => <ul className="ml-4 list-disc space-y-1" {...p} />,
          ol: (p) => <ol className="ml-4 list-decimal space-y-1" {...p} />,
          li: (p) => <li {...p} />,
          a: (p) => (
            <a
              className="text-emerald-700 underline-offset-2 hover:underline dark:text-emerald-400"
              target="_blank"
              rel="noreferrer"
              {...p}
            />
          ),
          strong: (p) => <strong className="font-semibold text-strong" {...p} />,
          em: (p) => <em className="italic" {...p} />,
          blockquote: (p) => (
            <blockquote
              className="border-l-2 border-line pl-3 text-muted"
              {...p}
            />
          ),
          hr: () => <hr className="my-3 border-line" />,
          table: (p) => (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs" {...p} />
            </div>
          ),
          th: (p) => (
            <th
              className="border border-line bg-active px-2 py-1 text-left font-semibold"
              {...p}
            />
          ),
          td: (p) => <td className="border border-line px-2 py-1" {...p} />,
          code: ({ className: cls, children, ...rest }) => {
            const isBlock = /language-/.test(cls ?? "");
            if (isBlock) {
              return (
                <code
                  className={cn(
                    "block overflow-x-auto rounded border border-line bg-canvas p-3 font-mono text-[11px]",
                    cls,
                  )}
                  {...rest}
                >
                  {children}
                </code>
              );
            }
            return (
              <code
                className="rounded bg-active px-1 py-0.5 font-mono text-[11px]"
                {...rest}
              >
                {children}
              </code>
            );
          },
          pre: (p) => <pre className="my-2" {...p} />,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

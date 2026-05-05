import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  isRouteErrorResponse,
} from "react-router";

import "./app.css";
import { THEME_STORAGE_KEY } from "./lib/theme";

import type { Route } from "./+types/root";

export const links: Route.LinksFunction = () => [
  { rel: "icon", href: "/console/favicon.svg", type: "image/svg+xml" },
];

// Runs synchronously before paint to apply the persisted theme,
// preventing a flash of dark on a light-default load (or vice versa).
// Failure here is silent — the SPA still works at the default theme.
const THEME_PREAMBLE = `(function(){try{var k=${JSON.stringify(THEME_STORAGE_KEY)};var t=localStorage.getItem(k);if(t==="dark")document.documentElement.classList.add("dark");}catch(e){}})();`;

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
        <script dangerouslySetInnerHTML={{ __html: THEME_PREAMBLE }} />
      </head>
      <body className="h-full bg-canvas text-strong font-sans antialiased">
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

export default function App() {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 5_000 },
        },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <Outlet />
    </QueryClientProvider>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  let title = "Something broke";
  let detail = "";
  if (isRouteErrorResponse(error)) {
    title = `${error.status} ${error.statusText}`;
    detail = typeof error.data === "string" ? error.data : "";
  } else if (error instanceof Error) {
    title = error.message;
    detail = error.stack ?? "";
  }
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-8">
      <h1 className="text-2xl font-semibold">{title}</h1>
      {detail && (
        <pre className="max-h-96 max-w-3xl overflow-auto rounded bg-chrome p-4 text-xs text-body">
          {detail}
        </pre>
      )}
    </div>
  );
}

/**
 * Tiny theme manager — light/dark via a class on <html>, persisted to
 * localStorage. Default is light. SSR-safe (no-ops without `window`).
 *
 * The matching FOUC-prevention inline script lives in ``root.tsx`` so
 * the right class is applied before React paints.
 */

import { useEffect, useState, useCallback } from "react";

export type Theme = "light" | "dark";

export const THEME_STORAGE_KEY = "rllm-console-theme";
const DEFAULT_THEME: Theme = "light";

function readStoredTheme(): Theme {
  if (typeof window === "undefined") return DEFAULT_THEME;
  try {
    const v = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* ignore — private mode, etc. */
  }
  return DEFAULT_THEME;
}

function applyTheme(theme: Theme) {
  if (typeof document === "undefined") return;
  document.documentElement.classList.toggle("dark", theme === "dark");
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* ignore */
  }
}

/**
 * Hook for the toggle button. Reads the current class on `<html>` so
 * it's always in sync with the FOUC-prevention script.
 */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(readStoredTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  return { theme, setTheme, toggle };
}

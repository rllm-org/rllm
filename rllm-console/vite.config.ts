import { reactRouter } from "@react-router/dev/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";

// The console is mounted under /console/ in production (FastAPI serves
// it at <gateway>/console/...). Vite's `base` has to match so emitted
// asset URLs in index.html resolve correctly. RR's `basename` (in
// react-router.config.ts) handles the routing side.
const BASE = process.env.RLLM_CONSOLE_BASE_PATH || "/console/";

export default defineConfig({
  base: BASE,
  plugins: [tailwindcss(), reactRouter(), tsconfigPaths()],
  server: {
    proxy: {
      // In `bun run dev` we serve the SPA at :5173 and proxy /console/api/*
      // to a backend running rllm view (typical port 7860). Override with
      // VITE_API_URL when pointing at a different host.
      "/console/api": {
        target: process.env.VITE_API_URL || "http://127.0.0.1:7860",
        changeOrigin: true,
      },
    },
  },
});

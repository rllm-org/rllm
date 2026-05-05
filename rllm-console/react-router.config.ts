import type { Config } from "@react-router/dev/config";

export default {
  // The console is served as static assets from a Python wheel — no SSR.
  ssr: false,
  // Build under url_prefix so React Router knows it's mounted at /console/.
  // Override at build time with RLLM_CONSOLE_BASE_PATH if the gateway uses
  // a different prefix.
  basename: process.env.RLLM_CONSOLE_BASE_PATH || "/console/",
} satisfies Config;

import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));

/** vite-node config for the headless §H generation runner. */
export default defineConfig({
  resolve: {
    alias: {
      "@tauri-apps/api/core": path.join(here, "tauri-core-shim.ts"),
    },
  },
});

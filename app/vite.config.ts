import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: ["es2022", "chrome110", "safari16"],
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_DEBUG,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("echarts") || id.includes("zrender")) return "echarts";
          if (id.includes("maplibre-gl")) return "maplibre";
          if (id.includes("@deck.gl") || id.includes("deck.gl")) return "deckgl";
          if (id.includes("@deck.gl/geo-layers")) return "deckgl";
          if (id.includes("@monaco-editor") || id.includes("monaco-editor")) return "monaco";
          if (id.includes("@duckdb/duckdb-wasm")) return "duckdb";
          if (id.includes("three") || id.includes("@react-three")) return "r3f";
          if (id.includes("sigma") || id.includes("graphology")) return "sigma";
        },
      },
    },
  },
});

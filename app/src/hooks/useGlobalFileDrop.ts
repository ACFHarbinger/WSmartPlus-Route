import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { toast } from "sonner";
import { useAppStore } from "../store/app";
import { useLayoutStore } from "../store/layout";
import type { WsrouteExtractResult } from "../types";
import { useFileDrop } from "./useFileDrop";

/** App-wide drop handler for `.wsroute` bundles and `.jsonl` logs (§G.8 / §G.14). */
export function useGlobalFileDrop() {
  const { projectRoot, setMode, setPendingLogPath } = useAppStore();
  const setFileDropDragging = useLayoutStore((s) => s.setFileDropDragging);

  const handleDrop = useCallback(
    async (paths: string[]) => {
      const wsroute = paths.find((p) => p.toLowerCase().endsWith(".wsroute"));
      const jsonl = paths.find((p) => /\.(jsonl|log)$/i.test(p));

      if (wsroute) {
        if (!projectRoot) {
          toast.error("Set project root in Settings before importing bundles");
          return;
        }
        const destDir = `${projectRoot}/assets/output/.imports/${Date.now()}`;
        try {
          const result = await invoke<WsrouteExtractResult>("extract_wsroute_bundle", {
            path: wsroute,
            destDir,
          });
          toast.success(`Extracted ${result.extracted_files.length} files`);
          if (result.log_path) {
            setPendingLogPath(result.log_path);
            setMode("simulation_summary");
          } else {
            setMode("output_browser");
            toast.info("No .jsonl log in bundle — browse extracted files in Output Browser");
          }
        } catch (err) {
          toast.error("Failed to import bundle", { description: String(err) });
        }
        return;
      }

      if (jsonl) {
        setPendingLogPath(jsonl);
        setMode("simulation_summary");
        toast.success("Log loaded", { description: jsonl.split("/").pop() });
        return;
      }

      toast.error("Drop a .wsroute bundle or .jsonl log file");
    },
    [projectRoot, setMode, setPendingLogPath]
  );

  return useFileDrop(handleDrop, true, setFileDropDragging);
}

import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { toast } from "sonner";
import type { WsrouteExtractResult } from "../../types";
import { useRecentHandoff } from "./useRecentHandoff";

/** Pick a `.wsroute` bundle, extract it, and open the log in Simulation Summary. */
export function useWsrouteImport() {
  const { projectRoot, setMode, handoff } = useRecentHandoff();

  return useCallback(async () => {
    const bundlePath = (await open({
      filters: [{ name: "WSmart-Route Bundle", extensions: ["wsroute"] }],
      title: "Select .wsroute bundle",
    })) as string | null;
    if (!bundlePath) return;

    const destDir = (await open({
      directory: true,
      title: "Extract bundle to…",
    })) as string | null;
    if (!destDir) return;

    try {
      const result = await invoke<WsrouteExtractResult>("extract_wsroute_bundle", {
        path: bundlePath,
        destDir,
      });
      toast.success(`Extracted ${result.extracted_files.length} files`);
      if (result.log_path) {
        handoff(result.log_path, "log");
      } else {
        setMode("output_browser");
        toast.info("No .jsonl log found — browse extracted files in Output Browser");
      }
    } catch (err) {
      toast.error("Failed to import bundle", { description: String(err) });
    }
  }, [projectRoot, setMode, handoff]);
}

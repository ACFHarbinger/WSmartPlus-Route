import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { toast } from "sonner";
import { useAppStore } from "../store/app";
import { useRecentFilesStore } from "../store/recentFiles";
import type { WsrouteExtractResult } from "../types";
import { applyRecentHandoff, type RecentPendingSetters } from "../utils/recentHandoff";

/** Pick a `.wsroute` bundle, extract it, and open the log in Simulation Summary. */
export function useWsrouteImport() {
  const {
    projectRoot,
    setMode,
    setPendingLogPath,
    setPendingRunPath,
    setPendingCsvPath,
    setPendingTrainingRunPath,
    setPendingCheckpoint,
    setPendingConfigPath,
  } = useAppStore();
  const pushRecent = useRecentFilesStore((s) => s.pushRecent);

  const pendingSetters: RecentPendingSetters = {
    pendingLogPath: setPendingLogPath,
    pendingRunPath: setPendingRunPath,
    pendingCsvPath: setPendingCsvPath,
    pendingTrainingRunPath: setPendingTrainingRunPath,
    pendingCheckpoint: setPendingCheckpoint,
    pendingConfigPath: setPendingConfigPath,
  };

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
        applyRecentHandoff({
          path: result.log_path,
          kind: "log",
          projectRoot,
          pushRecent,
          setMode,
          pendingSetters,
        });
      } else {
        setMode("output_browser");
        toast.info("No .jsonl log found — browse extracted files in Output Browser");
      }
    } catch (err) {
      toast.error("Failed to import bundle", { description: String(err) });
    }
    // pending setters are stable Zustand actions
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectRoot, pushRecent, setMode]);
}

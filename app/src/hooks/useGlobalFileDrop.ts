import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { toast } from "sonner";
import { useAppStore } from "../store/app";
import { useLayoutStore } from "../store/layout";
import { useRecentFilesStore, type RecentFileKind } from "../store/recentFiles";
import type { WsrouteExtractResult } from "../types";
import { portfolioRunLabel } from "../utils/arrowPipeline";
import { useFileDrop } from "./useFileDrop";

function findPath(paths: string[], re: RegExp): string | undefined {
  return paths.find((p) => re.test(p));
}

/**
 * App-wide drop handler for Studio artefacts (§G.8 / §G.6 / §G.12 / §G.13 / §G.14).
 * Routes logs, CSVs, checkpoints, configs, and `.wsroute` bundles through the same
 * ``portfolioRunLabel`` + pending-path handoffs used by Command Palette recents.
 */
export function useGlobalFileDrop() {
  const {
    projectRoot,
    setMode,
    setPendingLogPath,
    setPendingCsvPath,
    setPendingCheckpoint,
    setPendingConfigPath,
  } = useAppStore();
  const pushRecent = useRecentFilesStore((s) => s.pushRecent);
  const setFileDropDragging = useLayoutStore((s) => s.setFileDropDragging);

  const handoffRecent = useCallback(
    (
      path: string,
      kind: RecentFileKind,
      setPending: (path: string | null) => void,
      mode: Parameters<typeof setMode>[0],
      successLabel: string
    ) => {
      pushRecent({
        path,
        label: portfolioRunLabel(path, undefined, projectRoot),
        kind,
      });
      setPending(path);
      setMode(mode);
      toast.success(successLabel, { description: path.split(/[/\\]/).pop() });
    },
    [projectRoot, pushRecent, setMode]
  );

  const handleDrop = useCallback(
    async (paths: string[]) => {
      const wsroute = findPath(paths, /\.wsroute$/i);
      const jsonl = findPath(paths, /\.jsonl$/i);
      const logFile = findPath(paths, /\.log$/i);
      const checkpoint = findPath(paths, /\.(pt|ckpt|pth)$/i);
      const config = findPath(paths, /\.(ya?ml|toml|cfg|ini)$/i);
      const csv = findPath(paths, /\.csv$/i);

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
            pushRecent({
              path: result.log_path,
              label: portfolioRunLabel(result.log_path, undefined, projectRoot),
              kind: "log",
            });
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

      const logPath = jsonl ?? logFile;
      if (logPath) {
        handoffRecent(logPath, "log", setPendingLogPath, "simulation_summary", "Log loaded");
        return;
      }

      if (checkpoint) {
        handoffRecent(
          checkpoint,
          "checkpoint",
          setPendingCheckpoint,
          "eval_runner",
          "Checkpoint loaded in Eval Runner"
        );
        return;
      }

      if (config) {
        handoffRecent(
          config,
          "config",
          setPendingConfigPath,
          "config_editor",
          "Config opened in Config Editor"
        );
        return;
      }

      if (csv) {
        handoffRecent(csv, "csv", setPendingCsvPath, "data_explorer", "CSV loaded in Data Explorer");
        return;
      }

      toast.error(
        "Drop a .wsroute, .jsonl, .csv, checkpoint (.pt/.ckpt/.pth), or config (.yaml/…) file"
      );
    },
    [
      projectRoot,
      pushRecent,
      setMode,
      setPendingLogPath,
      setPendingCsvPath,
      setPendingCheckpoint,
      setPendingConfigPath,
      handoffRecent,
    ]
  );

  return useFileDrop(handleDrop, true, setFileDropDragging);
}

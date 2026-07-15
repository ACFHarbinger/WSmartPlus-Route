import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { toast } from "sonner";
import { useAppStore } from "../store/app";
import { useLayoutStore } from "../store/layout";
import { useRecentFilesStore, recentKindFromPath, type RecentFileKind } from "../store/recentFiles";
import type { WsrouteExtractResult } from "../types";
import {
  highestPriorityKind,
  makeRecentEntry,
  recentHandoffSpec,
  type RecentPendingKey,
} from "../utils/recentHandoff";
import { useFileDrop } from "./useFileDrop";

function findPath(paths: string[], re: RegExp): string | undefined {
  return paths.find((p) => re.test(p));
}

type PendingSetters = Record<RecentPendingKey, (path: string | null) => void>;

/**
 * App-wide drop handler for Studio artefacts (§G.8 / §G.6 / §G.7 / §G.12–§G.14 / §G.17).
 * Routes logs, CSVs, checkpoints, configs, training/run directories, and `.wsroute`
 * bundles through shared ``recentKindFromPath`` + ``portfolioRunLabel`` handoffs.
 */
export function useGlobalFileDrop() {
  const {
    projectRoot,
    setMode,
    setPendingLogPath,
    setPendingCsvPath,
    setPendingCheckpoint,
    setPendingConfigPath,
    setPendingRunPath,
    setPendingTrainingRunPath,
  } = useAppStore();
  const pushRecent = useRecentFilesStore((s) => s.pushRecent);
  const setFileDropDragging = useLayoutStore((s) => s.setFileDropDragging);

  const pendingSetters: PendingSetters = {
    pendingLogPath: setPendingLogPath,
    pendingRunPath: setPendingRunPath,
    pendingCsvPath: setPendingCsvPath,
    pendingTrainingRunPath: setPendingTrainingRunPath,
    pendingCheckpoint: setPendingCheckpoint,
    pendingConfigPath: setPendingConfigPath,
  };

  const handoffKind = useCallback(
    (path: string, kind: RecentFileKind, toastOnSuccess = true) => {
      const spec = recentHandoffSpec(kind);
      pushRecent(makeRecentEntry(path, kind, projectRoot));
      pendingSetters[spec.pendingKey](path);
      setMode(spec.mode);
      if (toastOnSuccess) {
        toast.success(spec.successLabel, {
          description: path.split(/[/\\]/).pop(),
        });
      }
    },
    // pending setters are stable Zustand actions
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [projectRoot, pushRecent, setMode]
  );

  const handleDrop = useCallback(
    async (paths: string[]) => {
      const wsroute = findPath(paths, /\.wsroute$/i);

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
            handoffKind(result.log_path, "log", false);
            toast.success("Log loaded", {
              description: result.log_path.split(/[/\\]/).pop(),
            });
          } else {
            setMode("output_browser");
            toast.info("No .jsonl log in bundle — browse extracted files in Output Browser");
          }
        } catch (err) {
          toast.error("Failed to import bundle", { description: String(err) });
        }
        return;
      }

      // Classify every path; push all known kinds, navigate to highest priority.
      const classified: Array<{ path: string; kind: RecentFileKind }> = [];
      for (const path of paths) {
        const kind = recentKindFromPath(path);
        if (kind) classified.push({ path, kind });
      }

      if (classified.length === 0) {
        toast.error(
          "Drop a .wsroute, .jsonl, .csv, checkpoint, config, training logs/, or assets/output run"
        );
        return;
      }

      // Push every classified path for Command Palette recents parity.
      for (const { path, kind } of classified) {
        pushRecent(makeRecentEntry(path, kind, projectRoot));
      }

      const primaryKind = highestPriorityKind(classified.map((c) => c.kind));
      const primary = primaryKind
        ? classified.find((c) => c.kind === primaryKind)
        : undefined;
      if (!primary) return;

      const spec = recentHandoffSpec(primary.kind);
      pendingSetters[spec.pendingKey](primary.path);
      setMode(spec.mode);
      const extra =
        classified.length > 1 ? ` (+${classified.length - 1} more in Recent)` : "";
      toast.success(`${spec.successLabel}${extra}`, {
        description: primary.path.split(/[/\\]/).pop(),
      });
    },
    // pending setters are stable Zustand actions
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [projectRoot, pushRecent, setMode, handoffKind]
  );

  return useFileDrop(handleDrop, true, setFileDropDragging);
}

/**
 * Shared recent-file handoff wiring for nav meshes, Command Palette, drop, and pages
 * (§G.7 / §G.8 / §D.7).
 */
import { useCallback, useMemo } from "react";
import { useAppStore } from "../store/app";
import {
  useRecentFilesStore,
  type RecentFileKind,
} from "../store/recentFiles";
import {
  applyRecentHandoff,
  type ApplyRecentHandoffArgs,
  type RecentHandoffSpec,
  type RecentPendingSetters,
} from "../utils/recentHandoff";

/** Stable map of pending-path setters for ``applyRecentHandoff``. */
export function useRecentPendingSetters(): RecentPendingSetters {
  const setPendingLogPath = useAppStore((s) => s.setPendingLogPath);
  const setPendingRunPath = useAppStore((s) => s.setPendingRunPath);
  const setPendingCsvPath = useAppStore((s) => s.setPendingCsvPath);
  const setPendingTrainingRunPath = useAppStore((s) => s.setPendingTrainingRunPath);
  const setPendingCheckpoint = useAppStore((s) => s.setPendingCheckpoint);
  const setPendingConfigPath = useAppStore((s) => s.setPendingConfigPath);

  return useMemo(
    () => ({
      pendingLogPath: setPendingLogPath,
      pendingRunPath: setPendingRunPath,
      pendingCsvPath: setPendingCsvPath,
      pendingTrainingRunPath: setPendingTrainingRunPath,
      pendingCheckpoint: setPendingCheckpoint,
      pendingConfigPath: setPendingConfigPath,
    }),
    [
      setPendingLogPath,
      setPendingRunPath,
      setPendingCsvPath,
      setPendingTrainingRunPath,
      setPendingCheckpoint,
      setPendingConfigPath,
    ]
  );
}

export interface UseRecentHandoffResult {
  projectRoot: string;
  setMode: ReturnType<typeof useAppStore.getState>["setMode"];
  pushRecent: ReturnType<typeof useRecentFilesStore.getState>["pushRecent"];
  pendingSetters: RecentPendingSetters;
  /**
   * Push a recent entry and navigate via pending-path handoff.
   * Pass ``navigate: false`` to only update recents (local file pickers).
   */
  handoff: (
    path: string,
    kind: RecentFileKind,
    opts?: Pick<ApplyRecentHandoffArgs, "storedLabel" | "navigate">
  ) => RecentHandoffSpec;
}

/** Bundle projectRoot, pushRecent, pending setters, and ``applyRecentHandoff``. */
export function useRecentHandoff(): UseRecentHandoffResult {
  const projectRoot = useAppStore((s) => s.projectRoot);
  const setMode = useAppStore((s) => s.setMode);
  const pushRecent = useRecentFilesStore((s) => s.pushRecent);
  const pendingSetters = useRecentPendingSetters();

  const handoff = useCallback(
    (
      path: string,
      kind: RecentFileKind,
      opts?: Pick<ApplyRecentHandoffArgs, "storedLabel" | "navigate">
    ): RecentHandoffSpec =>
      applyRecentHandoff({
        path,
        kind,
        projectRoot,
        pushRecent,
        setMode,
        pendingSetters,
        storedLabel: opts?.storedLabel,
        navigate: opts?.navigate,
      }),
    [projectRoot, pushRecent, setMode, pendingSetters]
  );

  return { projectRoot, setMode, pushRecent, pendingSetters, handoff };
}

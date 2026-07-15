/**
 * Shared recent-file / drop handoff specs for Command Palette + global drop (§G.7 / §G.8 / §D.7).
 */
import type { AppMode } from "../types";
import type { RecentFile, RecentFileKind } from "../store/recentFiles";
import { portfolioRunLabel } from "./arrowPipeline";

export type RecentPendingKey =
  | "pendingLogPath"
  | "pendingRunPath"
  | "pendingCsvPath"
  | "pendingTrainingRunPath"
  | "pendingCheckpoint"
  | "pendingConfigPath";

export interface RecentHandoffSpec {
  mode: AppMode;
  pendingKey: RecentPendingKey;
  successLabel: string;
}

export type RecentPendingSetters = Record<RecentPendingKey, (path: string | null) => void>;

const HANDOFF: Record<RecentFileKind, RecentHandoffSpec> = {
  log: {
    mode: "simulation_summary",
    pendingKey: "pendingLogPath",
    successLabel: "Log loaded",
  },
  run: {
    mode: "output_browser",
    pendingKey: "pendingRunPath",
    successLabel: "Run opened in Output Browser",
  },
  csv: {
    mode: "data_explorer",
    pendingKey: "pendingCsvPath",
    successLabel: "CSV loaded in Data Explorer",
  },
  training: {
    mode: "training",
    pendingKey: "pendingTrainingRunPath",
    successLabel: "Training run opened in Training Monitor",
  },
  checkpoint: {
    mode: "eval_runner",
    pendingKey: "pendingCheckpoint",
    successLabel: "Checkpoint loaded in Eval Runner",
  },
  config: {
    mode: "config_editor",
    pendingKey: "pendingConfigPath",
    successLabel: "Config opened in Config Editor",
  },
};

/** Resolve navigation mode + pending-path key for a recent-file kind. */
export function recentHandoffSpec(kind: RecentFileKind): RecentHandoffSpec {
  return HANDOFF[kind];
}

/** Build a recent-file entry with ``portfolioRunLabel`` for brush/SQL parity. */
export function makeRecentEntry(
  path: string,
  kind: RecentFileKind,
  projectRoot?: string | null,
  storedLabel?: string
): Omit<RecentFile, "openedAt"> {
  return {
    path,
    kind,
    label: portfolioRunLabel(path, storedLabel, projectRoot),
  };
}

export interface ApplyRecentHandoffArgs {
  path: string;
  kind: RecentFileKind;
  projectRoot?: string | null;
  /** Prefer stored/run name when re-pushing an existing recent entry. */
  storedLabel?: string;
  pushRecent: (file: Omit<RecentFile, "openedAt">) => void;
  setMode: (mode: AppMode) => void;
  pendingSetters: RecentPendingSetters;
  /**
   * When false, only push the recent entry (no mode/pending navigation).
   * Useful when the caller already has local open logic (file pickers, etc.).
   */
  navigate?: boolean;
  /**
   * Override the default destination mode for this kind (e.g. open a log in
   * Simulation Monitor instead of Simulation Summary). Pending-path key is
   * unchanged so both Summary and Monitor still consume ``pendingLogPath``.
   */
  mode?: AppMode;
}

/**
 * Push a recent-file entry and optionally navigate via pending-path handoff.
 * Returns the handoff spec for toast/label reuse at call sites.
 */
export function applyRecentHandoff(args: ApplyRecentHandoffArgs): RecentHandoffSpec {
  const {
    path,
    kind,
    projectRoot,
    storedLabel,
    pushRecent,
    setMode,
    pendingSetters,
    navigate = true,
    mode,
  } = args;
  pushRecent(makeRecentEntry(path, kind, projectRoot, storedLabel));
  const spec = recentHandoffSpec(kind);
  if (navigate) {
    pendingSetters[spec.pendingKey](path);
    setMode(mode ?? spec.mode);
  }
  return { ...spec, mode: mode ?? spec.mode };
}

/**
 * Drop/open priority when multiple artefacts are present (highest first).
 * Bundles are handled separately before this list.
 */
export const RECENT_DROP_PRIORITY: RecentFileKind[] = [
  "log",
  "checkpoint",
  "training",
  "run",
  "config",
  "csv",
];

/** Pick the highest-priority kind present in a multiset of classified paths. */
export function highestPriorityKind(
  kinds: Iterable<RecentFileKind>
): RecentFileKind | null {
  const set = new Set(kinds);
  for (const kind of RECENT_DROP_PRIORITY) {
    if (set.has(kind)) return kind;
  }
  return null;
}

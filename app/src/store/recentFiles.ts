import { create } from "zustand";
import { persist } from "zustand/middleware";
import { portfolioRunLabel } from "../utils/arrowPipeline";

export type RecentFileKind = "log" | "run" | "csv" | "training" | "checkpoint";

export interface RecentFile {
  path: string;
  label: string;
  kind: RecentFileKind;
  openedAt: number;
}

interface RecentFilesState {
  files: RecentFile[];
  pushRecent: (file: Omit<RecentFile, "openedAt">) => void;
  refreshRecentLabels: (projectRoot?: string | null) => void;
  removeRecent: (path: string) => void;
  clearRecent: () => void;
}

const MAX_RECENT = 12;

export const useRecentFilesStore = create<RecentFilesState>()(
  persist(
    (set) => ({
      files: [],
      pushRecent: (file) =>
        set((s) => {
          const openedAt = Date.now();
          const next = [
            { ...file, openedAt },
            ...s.files.filter((f) => f.path !== file.path),
          ].slice(0, MAX_RECENT);
          return { files: next };
        }),
      refreshRecentLabels: (projectRoot) =>
        set((s) => {
          if (s.files.length === 0) return s;
          const files = s.files.map((f) => ({
            ...f,
            label: portfolioRunLabel(f.path, f.label, projectRoot),
          }));
          const changed = files.some((f, i) => f.label !== s.files[i]!.label);
          return changed ? { files } : s;
        }),
      removeRecent: (path) =>
        set((s) => ({ files: s.files.filter((f) => f.path !== path) })),
      clearRecent: () => set({ files: [] }),
    }),
    { name: "wsmart-studio-recent-files" }
  )
);

export function recentFileLabel(path: string): string {
  return path.split("/").pop() ?? path;
}

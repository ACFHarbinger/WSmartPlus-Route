import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AppMode } from "../types";

interface AppState {
  mode: AppMode;
  theme: "dark" | "light";
  projectRoot: string;
  pythonPath: string;
  // Ephemeral — set by TrainingMonitor checkpoint browser to pre-populate EvaluationRunner
  pendingCheckpoint: string | null;
  setMode: (mode: AppMode) => void;
  setTheme: (theme: "dark" | "light") => void;
  setProjectRoot: (root: string) => void;
  setPythonPath: (path: string) => void;
  setPendingCheckpoint: (path: string | null) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      mode: "simulation",
      theme: "dark",
      projectRoot: "",
      pythonPath: "",
      pendingCheckpoint: null,
      setMode: (mode) => set({ mode }),
      setTheme: (theme) => {
        if (theme === "dark") {
          document.documentElement.classList.add("dark");
        } else {
          document.documentElement.classList.remove("dark");
        }
        set({ theme });
      },
      setProjectRoot: (projectRoot) => set({ projectRoot }),
      setPythonPath: (pythonPath) => set({ pythonPath }),
      setPendingCheckpoint: (pendingCheckpoint) => set({ pendingCheckpoint }),
    }),
    {
      name: "wsmart-studio-app",
      partialize: (s) => ({
        mode: s.mode,
        theme: s.theme,
        projectRoot: s.projectRoot,
        pythonPath: s.pythonPath,
      }),
    }
  )
);

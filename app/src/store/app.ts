import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AppMode } from "../types";

interface AppState {
  mode: AppMode;
  theme: "dark" | "light";
  projectRoot: string;
  pythonPath: string;
  setMode: (mode: AppMode) => void;
  setTheme: (theme: "dark" | "light") => void;
  setProjectRoot: (root: string) => void;
  setPythonPath: (path: string) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      mode: "simulation",
      theme: "dark",
      projectRoot: "",
      pythonPath: "",
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

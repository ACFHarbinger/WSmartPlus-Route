import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AppMode, BenchmarkLogRef, EvalAnalyticsRow, PendingMapCompare } from "../types";
import {
  applyDomTheme,
  resolveEffectiveTheme,
  type EffectiveTheme,
  type ThemePreference,
} from "../utils/theme";

interface AppState {
  mode: AppMode;
  theme: ThemePreference;
  effectiveTheme: EffectiveTheme;
  projectRoot: string;
  pythonPath: string;
  // Ephemeral — set by TrainingMonitor checkpoint browser to pre-populate EvaluationRunner
  pendingCheckpoint: string | null;
  // Ephemeral — set by OutputBrowser to auto-load a log file in SimulationSummary
  pendingLogPath: string | null;
  // Ephemeral — set by EvaluationRunner to pre-load results in BenchmarkAnalysis
  pendingEvalResults: EvalAnalyticsRow[] | null;
  pendingBenchmarkLogs: BenchmarkLogRef[] | null;
  pendingRunPath: string | null;
  pendingCsvPath: string | null;
  pendingTrainingRunPath: string | null;
  pendingMapCompare: PendingMapCompare | null;
  setMode: (mode: AppMode) => void;
  setTheme: (theme: ThemePreference) => void;
  setEffectiveTheme: (effective: EffectiveTheme) => void;
  setProjectRoot: (root: string) => void;
  setPythonPath: (path: string) => void;
  setPendingCheckpoint: (path: string | null) => void;
  setPendingLogPath: (path: string | null) => void;
  setPendingEvalResults: (rows: EvalAnalyticsRow[] | null) => void;
  setPendingBenchmarkLogs: (logs: BenchmarkLogRef[] | null) => void;
  setPendingRunPath: (path: string | null) => void;
  setPendingCsvPath: (path: string | null) => void;
  setPendingTrainingRunPath: (path: string | null) => void;
  setPendingMapCompare: (compare: PendingMapCompare | null) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      mode: "simulation",
      theme: "dark",
      effectiveTheme: "dark",
      projectRoot: "",
      pythonPath: "",
      pendingCheckpoint: null,
      pendingLogPath: null,
      pendingEvalResults: null,
      pendingBenchmarkLogs: null,
      pendingRunPath: null,
      pendingCsvPath: null,
      pendingTrainingRunPath: null,
      pendingMapCompare: null,
      setMode: (mode) => set({ mode }),
      setTheme: (theme) => {
        const effective = resolveEffectiveTheme(theme);
        applyDomTheme(effective);
        set({ theme, effectiveTheme: effective });
      },
      setEffectiveTheme: (effectiveTheme) => set({ effectiveTheme }),
      setProjectRoot: (projectRoot) => set({ projectRoot }),
      setPythonPath: (pythonPath) => set({ pythonPath }),
      setPendingCheckpoint: (pendingCheckpoint) => set({ pendingCheckpoint }),
      setPendingLogPath: (pendingLogPath) => set({ pendingLogPath }),
      setPendingEvalResults: (pendingEvalResults: EvalAnalyticsRow[] | null) =>
        set({ pendingEvalResults }),
      setPendingBenchmarkLogs: (pendingBenchmarkLogs: BenchmarkLogRef[] | null) =>
        set({ pendingBenchmarkLogs }),
      setPendingRunPath: (pendingRunPath) => set({ pendingRunPath }),
      setPendingCsvPath: (pendingCsvPath) => set({ pendingCsvPath }),
      setPendingTrainingRunPath: (pendingTrainingRunPath) => set({ pendingTrainingRunPath }),
      setPendingMapCompare: (pendingMapCompare) => set({ pendingMapCompare }),
    }),
    {
      name: "wsmart-studio-app",
      partialize: (s) => ({
        mode: s.mode,
        theme: s.theme,
        projectRoot: s.projectRoot,
        pythonPath: s.pythonPath,
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        const effective = resolveEffectiveTheme(state.theme);
        applyDomTheme(effective);
        state.effectiveTheme = effective;
      },
    }
  )
);

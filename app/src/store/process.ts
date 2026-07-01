import { create } from "zustand";
import type { ProcessEntry, ProcessStatus } from "../types";

interface ProcessState {
  processes: Record<string, ProcessEntry>;
  addProcess: (entry: Omit<ProcessEntry, "logLines">) => void;
  appendLog: (id: string, line: string) => void;
  updateStatus: (id: string, status: ProcessStatus, exitCode?: number) => void;
  removeProcess: (id: string) => void;
}

export const useProcessStore = create<ProcessState>((set) => ({
  processes: {},

  addProcess: (entry) =>
    set((s) => ({
      processes: {
        ...s.processes,
        [entry.id]: { ...entry, logLines: [] },
      },
    })),

  appendLog: (id, line) =>
    set((s) => {
      const proc = s.processes[id];
      if (!proc) return s;
      return {
        processes: {
          ...s.processes,
          [id]: {
            ...proc,
            // Keep the last 2000 lines to avoid unbounded memory growth
            logLines: [...proc.logLines.slice(-1999), line],
          },
        },
      };
    }),

  updateStatus: (id, status, exitCode) =>
    set((s) => {
      const proc = s.processes[id];
      if (!proc) return s;
      return {
        processes: {
          ...s.processes,
          [id]: { ...proc, status, exitCode },
        },
      };
    }),

  removeProcess: (id) =>
    set((s) => {
      const next = { ...s.processes };
      delete next[id];
      return { processes: next };
    }),
}));

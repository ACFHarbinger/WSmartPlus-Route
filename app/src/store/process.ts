import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ProcessEntry, ProcessStatus } from "../types";

interface ProcessState {
  processes: Record<string, ProcessEntry>;
  addProcess: (entry: Omit<ProcessEntry, "logLines">) => void;
  appendLog: (id: string, line: string) => void;
  updateStatus: (id: string, status: ProcessStatus, exitCode?: number) => void;
  removeProcess: (id: string) => void;
  clearCompleted: () => void;
}

export const useProcessStore = create<ProcessState>()(
  persist(
    (set) => ({
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

      clearCompleted: () =>
        set((s) => ({
          processes: Object.fromEntries(
            Object.entries(s.processes).filter(([, p]) => p.status === "running")
          ),
        })),
    }),
    {
      name: "wsmart-studio-processes",
      // Persist only the last 50 completed processes, stripping log lines
      // (logs are volatile and can be very large).
      partialize: (s) => ({
        processes: Object.fromEntries(
          Object.entries(s.processes)
            .filter(([, p]) => p.status !== "running")
            .slice(-50)
            .map(([id, p]) => [id, { ...p, logLines: [] }])
        ),
      }),
    }
  )
);

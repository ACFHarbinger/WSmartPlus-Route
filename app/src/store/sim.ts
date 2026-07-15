import { create } from "zustand";
import type { DayLogEntry, PolicyVizEntry } from "../types";

interface SimState {
  // All entries from the current log file
  entries: DayLogEntry[];
  policyVizEntries: PolicyVizEntry[];
  // Selected filters (mirrors Streamlit sidebar controls)
  selectedPolicy: string | null;
  selectedSample: number | null;
  selectedDay: number | null;
  // Active log file being watched
  watchPath: string | null;
  isWatching: boolean;

  addEntry: (entry: DayLogEntry) => void;
  addPolicyVizEntry: (entry: PolicyVizEntry) => void;
  loadEntries: (entries: DayLogEntry[]) => void;
  loadPolicyVizEntries: (entries: PolicyVizEntry[]) => void;
  setSelectedPolicy: (policy: string | null) => void;
  setSelectedSample: (sample: number | null) => void;
  setSelectedDay: (day: number | null) => void;
  setWatchPath: (path: string | null) => void;
  setWatching: (watching: boolean) => void;
  reset: () => void;
}

export const useSimStore = create<SimState>((set) => ({
  entries: [],
  policyVizEntries: [],
  selectedPolicy: null,
  selectedSample: null,
  selectedDay: null,
  watchPath: null,
  isWatching: false,

  addEntry: (entry) =>
    set((s) => {
      // Deduplicate by (policy, sample_id, day)
      const exists = s.entries.some(
        (e) =>
          e.policy === entry.policy &&
          e.sample_id === entry.sample_id &&
          e.day === entry.day
      );
      return exists ? s : { entries: [...s.entries, entry] };
    }),

  addPolicyVizEntry: (entry) =>
    set((s) => {
      const exists = s.policyVizEntries.some(
        (e) =>
          e.policy === entry.policy &&
          e.sample_id === entry.sample_id &&
          e.day === entry.day &&
          e.policy_type === entry.policy_type
      );
      return exists ? s : { policyVizEntries: [...s.policyVizEntries, entry] };
    }),

  loadEntries: (entries) => set({ entries }),

  loadPolicyVizEntries: (entries) => set({ policyVizEntries: entries }),

  setSelectedPolicy: (selectedPolicy) => set({ selectedPolicy }),
  setSelectedSample: (selectedSample) => set({ selectedSample }),
  setSelectedDay: (selectedDay) => set({ selectedDay }),
  setWatchPath: (watchPath) => set({ watchPath }),
  setWatching: (isWatching) => set({ isWatching }),

  reset: () =>
    set({
      entries: [],
      policyVizEntries: [],
      selectedPolicy: null,
      selectedSample: null,
      selectedDay: null,
      watchPath: null,
      isWatching: false,
    }),
}));

// Derived selectors
export function uniquePolicies(entries: DayLogEntry[]): string[] {
  return [...new Set(entries.map((e) => e.policy))].sort();
}

export function uniqueSamples(entries: DayLogEntry[]): number[] {
  return [...new Set(entries.map((e) => e.sample_id))].sort((a, b) => a - b);
}

export function filterEntries(
  entries: DayLogEntry[],
  policy: string | null,
  sampleId: number | null
): DayLogEntry[] {
  return entries.filter(
    (e) =>
      (policy === null || e.policy === policy) &&
      (sampleId === null || e.sample_id === sampleId)
  );
}

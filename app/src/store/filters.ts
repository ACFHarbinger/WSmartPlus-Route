import { create } from "zustand";

/** App-wide policy/sample filters — propagate across analytics views (§G.7). */
interface GlobalFiltersState {
  policy: string | null;
  sampleId: number | null;
  setPolicy: (policy: string | null) => void;
  setSampleId: (sampleId: number | null) => void;
  reset: () => void;
}

export const useGlobalFiltersStore = create<GlobalFiltersState>((set) => ({
  policy: null,
  sampleId: null,
  setPolicy: (policy) => set({ policy }),
  setSampleId: (sampleId) => set({ sampleId }),
  reset: () => set({ policy: null, sampleId: null }),
}));

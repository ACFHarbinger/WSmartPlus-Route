import { create } from "zustand";

/** App-wide policy/sample/run filters — propagate across analytics views (§G.7). */
interface GlobalFiltersState {
  policy: string | null;
  sampleId: number | null;
  /** Portfolio ``run_label`` brush for multi-run SQL sync (§G.6). */
  runLabel: string | null;
  /** Portfolio city/scale group brush — expands to all runs in the group (§G.6). */
  brushedCity: string | null;
  /** Global log-scale y-axis toggle for overflow/profit bar charts (§G.1 / §G.7). */
  logScale: boolean;
  setPolicy: (policy: string | null) => void;
  setSampleId: (sampleId: number | null) => void;
  setRunLabel: (runLabel: string | null) => void;
  setBrushedCity: (city: string | null) => void;
  setLogScale: (logScale: boolean) => void;
  reset: () => void;
}

export const useGlobalFiltersStore = create<GlobalFiltersState>((set) => ({
  policy: null,
  sampleId: null,
  runLabel: null,
  brushedCity: null,
  logScale: false,
  setPolicy: (policy) => set({ policy }),
  setSampleId: (sampleId) => set({ sampleId }),
  setRunLabel: (runLabel) => set({ runLabel }),
  setBrushedCity: (city) => set({ brushedCity: city }),
  setLogScale: (logScale) => set({ logScale }),
  reset: () =>
    set({ policy: null, sampleId: null, runLabel: null, brushedCity: null, logScale: false }),
}));

import { create } from "zustand";
import type { ArrowPipelineTiming } from "../utils/arrowPipeline";

interface DuckDbState {
  ready: boolean;
  loading: boolean;
  lastPipeline: ArrowPipelineTiming | null;
  error: string | null;
  setReady: (ready: boolean) => void;
  setLoading: (loading: boolean) => void;
  setLastPipeline: (timing: ArrowPipelineTiming | null) => void;
  setError: (error: string | null) => void;
}

export const useDuckDbStore = create<DuckDbState>((set) => ({
  ready: false,
  loading: false,
  lastPipeline: null,
  error: null,
  setReady: (ready) => set({ ready }),
  setLoading: (loading) => set({ loading }),
  setLastPipeline: (lastPipeline) => set({ lastPipeline }),
  setError: (error) => set({ error }),
}));

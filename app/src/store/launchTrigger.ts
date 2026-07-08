import { create } from "zustand";

/** Nonce counters — increment to request a launch from the active launcher page (§G.7 Ctrl+R). */
interface LaunchTriggerState {
  simNonce: number;
  trainNonce: number;
  dataGenNonce: number;
  evalNonce: number;
  triggerSim: () => void;
  triggerTrain: () => void;
  triggerDataGen: () => void;
  triggerEval: () => void;
}

export const useLaunchTriggerStore = create<LaunchTriggerState>((set) => ({
  simNonce: 0,
  trainNonce: 0,
  dataGenNonce: 0,
  evalNonce: 0,
  triggerSim: () => set((s) => ({ simNonce: s.simNonce + 1 })),
  triggerTrain: () => set((s) => ({ trainNonce: s.trainNonce + 1 })),
  triggerDataGen: () => set((s) => ({ dataGenNonce: s.dataGenNonce + 1 })),
  triggerEval: () => set((s) => ({ evalNonce: s.evalNonce + 1 })),
}));

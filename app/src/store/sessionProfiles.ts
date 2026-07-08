/**
 * Named session profiles — snapshots of all three launcher form states (§G.14 / §D.4 Option C).
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { useSimLauncherStore } from "./launchers";
import { useTrainHubStore } from "./launchers";
import { useDataGenStore } from "./launchers";

export interface SessionProfileSnapshot {
  sim: ReturnType<typeof useSimLauncherStore.getState>;
  train: ReturnType<typeof useTrainHubStore.getState>;
  dataGen: ReturnType<typeof useDataGenStore.getState>;
}

export interface SessionProfile {
  id: string;
  name: string;
  createdAt: number;
  snapshot: {
    sim: Omit<SessionProfileSnapshot["sim"], "patch">;
    train: Omit<SessionProfileSnapshot["train"], "patch">;
    dataGen: Omit<SessionProfileSnapshot["dataGen"], "patch">;
  };
}

interface SessionProfilesState {
  profiles: SessionProfile[];
  saveProfile: (name: string) => void;
  loadProfile: (id: string) => void;
  deleteProfile: (id: string) => void;
}

function stripPatch<T extends { patch?: unknown }>(state: T): Omit<T, "patch"> {
  const { patch: _patch, ...rest } = state;
  return rest;
}

export function captureLauncherSnapshot(): SessionProfile["snapshot"] {
  return {
    sim: stripPatch(useSimLauncherStore.getState()),
    train: stripPatch(useTrainHubStore.getState()),
    dataGen: stripPatch(useDataGenStore.getState()),
  };
}

export function applyLauncherSnapshot(snapshot: SessionProfile["snapshot"]) {
  const { patch: simPatch } = useSimLauncherStore.getState();
  const { patch: trainPatch } = useTrainHubStore.getState();
  const { patch: dataPatch } = useDataGenStore.getState();
  simPatch(snapshot.sim);
  trainPatch(snapshot.train);
  dataPatch(snapshot.dataGen);
}

export const useSessionProfilesStore = create<SessionProfilesState>()(
  persist(
    (set) => ({
      profiles: [],
      saveProfile: (name) => {
        const profile: SessionProfile = {
          id: `profile_${Date.now()}`,
          name: name.trim() || `Profile ${Date.now()}`,
          createdAt: Date.now(),
          snapshot: captureLauncherSnapshot(),
        };
        set((s) => ({ profiles: [profile, ...s.profiles].slice(0, 20) }));
      },
      loadProfile: (id) => {
        const profile = useSessionProfilesStore.getState().profiles.find((p) => p.id === id);
        if (profile) applyLauncherSnapshot(profile.snapshot);
      },
      deleteProfile: (id) => {
        set((s) => ({ profiles: s.profiles.filter((p) => p.id !== id) }));
      },
    }),
    { name: "wsroute-session-profiles" }
  )
);

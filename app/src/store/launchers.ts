/**
 * Persisted form state for the three launcher pages (§D.4 session persistence).
 *
 * Each store uses a single `patch` action so components can update any subset
 * of fields without per-field setters.  Ephemeral runtime state (process ID,
 * live metrics, run status) stays in local component state.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";

// ── Simulation Launcher (§G.9)

interface SimLauncherState {
  selectedPolicies: string[];
  area: string;
  numLoc: number;
  samples: number;
  nCores: number;
  seed: number;
  distribution: string;
  extraOverrides: string;
  patch: (updates: Partial<Omit<SimLauncherState, "patch">>) => void;
}

export const useSimLauncherStore = create<SimLauncherState>()(
  persist(
    (set) => ({
      selectedPolicies: ["aco_hh", "alns", "bpc", "hgs", "pg_clns", "psoma", "sans", "swc_tcf"],
      area: "figueiradafoz",
      numLoc: 350,
      samples: 1,
      nCores: 4,
      seed: 42,
      distribution: "emp",
      extraOverrides: "",
      patch: (updates) => set(updates as Partial<SimLauncherState>),
    }),
    { name: "wsroute-sim-launcher" }
  )
);

// ── Training Hub (§G.10)

interface TrainHubState {
  trainMode: "train" | "hpo" | "meta" | "eval";
  problem: string;
  seed: number;
  wandb: boolean;
  extraOverrides: string;
  model: string;
  encoder: string;
  batchSize: number;
  epochs: number;
  hpoTrials: number;
  hpoMethod: string;
  hpoWorkers: number;
  metaStrategy: string;
  metaLr: number;
  metaHistoryLength: number;
  mrlBatchSize: number;
  mrlStep: number;
  checkpointPath: string;
  evalDataset: string;
  evalSamples: number;
  evalStrategy: string;
  patch: (updates: Partial<Omit<TrainHubState, "patch">>) => void;
}

export const useTrainHubStore = create<TrainHubState>()(
  persist(
    (set) => ({
      trainMode: "train",
      problem: "vrpp",
      seed: 42,
      wandb: false,
      extraOverrides: "",
      model: "am",
      encoder: "gat",
      batchSize: 64,
      epochs: 100,
      hpoTrials: 50,
      hpoMethod: "nsgaii",
      hpoWorkers: 1,
      metaStrategy: "hrl",
      metaLr: 0.000005,
      metaHistoryLength: 10,
      mrlBatchSize: 256,
      mrlStep: 10,
      checkpointPath: "",
      evalDataset: "",
      evalSamples: 10,
      evalStrategy: "greedy",
      patch: (updates) => set(updates as Partial<TrainHubState>),
    }),
    { name: "wsroute-train-hub" }
  )
);

// ── Data Generation Wizard (§G.11)

interface DataGenState {
  dataSource: "synthetic" | "tsplib" | "sensor";
  tsplibPath: string;
  sensorCsvPath: string;
  problem: string;
  distributions: string[];
  datasetType: string;
  seed: number;
  overwrite: boolean;
  area: string;
  numLoc: number;
  nSamples: number;
  nDays: number;
  extraOverrides: string;
  patch: (updates: Partial<Omit<DataGenState, "patch">>) => void;
}

export const useDataGenStore = create<DataGenState>()(
  persist(
    (set) => ({
      dataSource: "synthetic",
      tsplibPath: "",
      sensorCsvPath: "",
      problem: "vrpp",
      distributions: ["gamma3"],
      datasetType: "test_simulator",
      seed: 42,
      overwrite: true,
      area: "figueiradafoz",
      numLoc: 350,
      nSamples: 1,
      nDays: 30,
      extraOverrides: "",
      patch: (updates) => set(updates as Partial<DataGenState>),
    }),
    { name: "wsroute-data-gen" }
  )
);

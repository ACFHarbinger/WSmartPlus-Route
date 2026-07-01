// ── Simulation log types ─────────────────────────────────────────────────────
// Mirror of Python DayLogEntry / SimDayData from logic/src/ui/services/log_parser.py

export interface BinCoord {
  id: number;
  dataset_id?: number;
  lat?: number;
  lng?: number;
}

export interface SimDayData {
  tour?: Array<BinCoord | number>;
  bin_state_c?: number[];
  bin_state_collected?: boolean[];
  mandatory?: number[];
  tour_indices?: number[];
  all_bin_coords?: BinCoord[];
  // KPI metrics — matches _PRIMARY_KPI_MAP + _SECONDARY_KPI_MAP in kpi.py
  overflows?: number;
  kg?: number;
  km?: number;
  "kg/km"?: number;
  cost?: number;
  profit?: number;
  ncol?: number;
  kg_lost?: number;
}

export interface DayLogEntry {
  policy: string;
  sample_id: number;
  day: number;
  data: SimDayData;
}

// ── Training log types ───────────────────────────────────────────────────────

export interface TrainingRun {
  name: string;
  path: string;
  has_metrics: boolean;
  has_hparams: boolean;
}

export interface TrainingMetricsRow {
  epoch?: number;
  step?: number;
  train_loss?: number;
  val_loss?: number;
  reward?: number;
  entropy?: number;
  grad_norm?: number;
  lr?: number;
  [key: string]: number | undefined;
}

// ── Process management ───────────────────────────────────────────────────────

export type ProcessStatus = "running" | "completed" | "cancelled" | "failed";

export interface ProcessEntry {
  id: string;
  command: string;
  pid: number;
  status: ProcessStatus;
  startTime: number;
  exitCode?: number;
  logLines: string[];
}

export interface StdoutLine {
  id: string;
  line: string;
}

export interface StatusUpdate {
  id: string;
  status: ProcessStatus;
  exit_code?: number;
}

// ── App navigation ───────────────────────────────────────────────────────────

export type AppMode =
  // Monitor modes (Streamlit parity)
  | "simulation"        // Simulation Digital Twin  — logic/src/ui/pages/simulation/
  | "training"          // Training Monitor         — logic/src/ui/pages/training.py
  | "simulation_summary"// Simulation Summary       — logic/src/ui/pages/simulation/summary.py
  | "benchmark"         // Benchmark Analysis       — logic/src/ui/components/benchmark_charts.py
  | "data_explorer"     // Data Explorer            — logic/src/ui/pages/data_explorer.py
  | "experiment_tracker"// Experiment Tracker       — logic/src/ui/pages/experiment_tracker.py
  | "algorithms"        // Algorithms Registry      — logic/src/ui/pages/algorithms.py
  | "hpo_tracker"       // HPO Tracker              — logic/src/ui/pages/hpo_tracker.py
  // Studio-only modes
  | "process_monitor"   // Process Monitor          — §G.15
  | "sim_launcher"      // Simulation Launcher      — §G.9
  | "training_hub"      // Training & HPO Hub       — §G.10
  | "data_gen"          // Data Generation Wizard   — §G.11
  | "config_editor"     // Configuration Editor     — §G.13
  | "output_browser";   // Output Browser           — §G.14

export interface NavSection {
  label: string;
  items: NavItem[];
}

export interface NavItem {
  mode: AppMode;
  label: string;
  streamlitEquivalent?: string;
}

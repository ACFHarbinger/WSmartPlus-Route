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

export interface ProcessSpawned {
  id: string;
  command: string;
  pid: number;
  start_time: number;
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

export interface DirEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size_bytes: number;
  extension: string;
}

export interface OutputDir {
  name: string;
  path: string;
  created_at: string;
  size_bytes: number;
}

// ── Policy registry (§G.9) ─────────────────────────────────────────────────

export interface SimPolicyEntry {
  id: string;
  config_key: string;
}

// ── Eval analytics (§G.12) ─────────────────────────────────────────────────

export interface BenchmarkLogRef {
  path: string;
  label: string;
}

export interface EvalAnalyticsRow {
  checkpoint: string;
  cost?: number;
  gap?: number;
  time?: number;
  policy?: string;
  [key: string]: number | string | undefined;
}

// ── Optuna HPO (§G.18) ───────────────────────────────────────────────────────

export interface OptunaStudySummary {
  name: string;
  n_trials: number;
  n_complete: number;
  best_value: number | null;
}

export interface OptunaTrial {
  number: number;
  value: number | null;
  state: string;
  params: Record<string, string | number | boolean>;
}

export interface DatasetPreviewStats {
  num_instances: number;
  num_nodes: number | null;
  demand_mean: number | null;
  demand_std: number | null;
  demand_histogram: number[];
  distance_mean: number | null;
  file_size_bytes: number;
}

export interface OptunaStudyData {
  name: string;
  trials: OptunaTrial[];
  importances: Record<string, number>;
  best_value: number | null;
  best_params: Record<string, string | number | boolean>;
}

// ── MLflow experiment tracking (§G.18) ─────────────────────────────────────

export interface MlflowRun {
  run_id: string;
  run_name: string;
  experiment_id: string;
  status: string;
  start_time: number | null;
  end_time: number | null;
  artifact_uri: string;
  params: Record<string, string>;
  metrics: Record<string, number>;
  tags: Record<string, string>;
}

export interface MlflowMetricPoint {
  step: number;
  value: number;
}

// ── ZenML pipeline tracking (§G.18) ──────────────────────────────────────────

export interface ZenmlPipelineRun {
  id: string;
  pipeline: string;
  status: string;
  created: string;
  updated: string;
  stack: string;
}

export interface ZenmlPipelineStep {
  name: string;
  status: string;
  created: string;
  updated: string;
  duration_seconds: number | null;
}

// ── Studio data bundles (§G.8) ───────────────────────────────────────────────

export interface WsrouteBundleFile {
  path: string;
  size_bytes: number;
}

export interface WsrouteBundleInfo {
  path: string;
  version: string | null;
  created_at: string | null;
  files: WsrouteBundleFile[];
}

export interface WsrouteExtractResult {
  dest_dir: string;
  extracted_files: string[];
  log_path: string | null;
}

// ── App navigation ───────────────────────────────────────────────────────────

/** Ephemeral map compare intent from Algorithm Comparison → Simulation Monitor. */
export interface PendingMapCompare {
  policies: string[];
  layout: "overlay" | "split";
  mapMode?: "echarts" | "deckgl";
}

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
  | "output_browser"    // Output Browser           — §G.14
  | "eval_runner"       // Evaluation Runner        — §G.12
  | "settings";         // App Settings             — §G.19

export interface NavSection {
  label: string;
  items: NavItem[];
}

export interface NavItem {
  mode: AppMode;
  label: string;
  streamlitEquivalent?: string;
}

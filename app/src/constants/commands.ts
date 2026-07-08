import type { AppMode } from "../types";

export type PaletteAction = "toggle_theme" | "shortcuts_help" | "import_wsroute";

export interface PaletteCommand {
  id: string;
  label: string;
  section: string;
  keywords?: string;
  mode?: AppMode;
  action?: PaletteAction;
}

/** Searchable command palette entries (§G.7). */
export const PALETTE_COMMANDS: PaletteCommand[] = [
  { id: "simulation", label: "Simulation Digital Twin", section: "Monitor", mode: "simulation", keywords: "map geospatial twin" },
  { id: "training", label: "Training Monitor", section: "Monitor", mode: "training", keywords: "lightning metrics" },
  { id: "process_monitor", label: "Process Monitor", section: "Monitor", mode: "process_monitor", keywords: "logs spawn" },
  { id: "simulation_summary", label: "Simulation Summary", section: "Analysis", mode: "simulation_summary", keywords: "overview ranking" },
  { id: "benchmark", label: "Benchmark Analysis", section: "Analysis", mode: "benchmark", keywords: "compare runs" },
  { id: "data_explorer", label: "Data Explorer", section: "Analysis", mode: "data_explorer", keywords: "csv table" },
  { id: "experiment_tracker", label: "Experiment Tracker", section: "Analysis", mode: "experiment_tracker", keywords: "mlflow zenml" },
  { id: "algorithms", label: "Algorithm Registry", section: "Analysis", mode: "algorithms", keywords: "policies constructors" },
  { id: "hpo_tracker", label: "HPO Tracker", section: "Analysis", mode: "hpo_tracker", keywords: "optuna trials" },
  { id: "sim_launcher", label: "Simulation Launcher", section: "Launch", mode: "sim_launcher", keywords: "test_sim hydra" },
  { id: "training_hub", label: "Training & HPO Hub", section: "Launch", mode: "training_hub", keywords: "train eval" },
  { id: "data_gen", label: "Data Generation", section: "Launch", mode: "data_gen", keywords: "dataset bins" },
  { id: "eval_runner", label: "Evaluation Runner", section: "Launch", mode: "eval_runner", keywords: "checkpoint eval" },
  { id: "output_browser", label: "Output Browser", section: "Files", mode: "output_browser", keywords: "artefacts wsroute" },
  { id: "config_editor", label: "Config Editor", section: "Files", mode: "config_editor", keywords: "yaml hydra" },
  { id: "settings", label: "Settings", section: "App", mode: "settings", keywords: "project root theme" },
  { id: "toggle_theme", label: "Toggle Dark / Light Theme", section: "Actions", action: "toggle_theme", keywords: "appearance" },
  { id: "shortcuts_help", label: "Keyboard Shortcuts Help", section: "Actions", action: "shortcuts_help", keywords: "hotkeys" },
  { id: "import_wsroute", label: "Import .wsroute Bundle", section: "Actions", action: "import_wsroute", keywords: "extract package zip" },
];

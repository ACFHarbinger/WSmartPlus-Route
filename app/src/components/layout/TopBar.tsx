import { Moon, Sun } from "lucide-react";
import { useAppStore } from "../../store/app";
import { useProcessStore } from "../../store/process";

const TITLES: Record<string, string> = {
  simulation: "Simulation Digital Twin",
  training: "Training Monitor",
  process_monitor: "Process Monitor",
  simulation_summary: "Simulation Summary",
  benchmark: "Benchmark Analysis",
  data_explorer: "Data Explorer",
  experiment_tracker: "Experiment Tracker",
  algorithms: "Algorithm Registry",
  hpo_tracker: "HPO Tracker",
  sim_launcher: "Simulation Launcher",
  training_hub: "Training & HPO Hub",
  data_gen: "Data Generation Wizard",
  config_editor: "Configuration Editor",
  output_browser: "Output Browser",
};

export function TopBar() {
  const { mode, theme, setTheme } = useAppStore();
  const processes = useProcessStore((s) => s.processes);
  const running = Object.values(processes).filter((p) => p.status === "running").length;

  return (
    <header className="h-11 shrink-0 flex items-center justify-between px-5 bg-canvas-surface border-b border-canvas-border">
      <h1 className="text-sm font-semibold text-gray-100">
        {TITLES[mode] ?? mode}
      </h1>

      <div className="flex items-center gap-3">
        {running > 0 && (
          <span className="flex items-center gap-1.5 text-xs text-accent-success">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-success animate-pulse" />
            {running} running
          </span>
        )}

        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="btn-ghost p-1.5"
          title="Toggle theme"
        >
          {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
        </button>
      </div>
    </header>
  );
}

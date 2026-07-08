import {
  Activity,
  BarChart2,
  Bot,
  Cpu,
  Database,
  FileText,
  FlaskConical,
  GitBranch,
  Layers,
  Map,
  Play,
  Settings,
  Sliders,
  Terminal,
  Zap,
  FolderOpen,
  ClipboardList,
} from "lucide-react";
import { useAppStore } from "../../store/app";
import { useLayoutStore } from "../../store/layout";
import type { AppMode, NavSection } from "../../types";
import { prefetchPage } from "../../utils/pagePrefetch";

const NAV: NavSection[] = [
  {
    label: "Monitor",
    items: [
      { mode: "simulation", label: "Simulation Digital Twin", streamlitEquivalent: "simulation" },
      { mode: "training", label: "Training Monitor", streamlitEquivalent: "training" },
      { mode: "process_monitor", label: "Process Monitor" },
    ],
  },
  {
    label: "Analysis",
    items: [
      { mode: "simulation_summary", label: "Simulation Summary", streamlitEquivalent: "simulation_summary" },
      { mode: "benchmark", label: "Benchmark Analysis", streamlitEquivalent: "benchmark" },
      { mode: "data_explorer", label: "Data Explorer", streamlitEquivalent: "data_explorer" },
      { mode: "experiment_tracker", label: "Experiment Tracker", streamlitEquivalent: "experiment_tracker" },
      { mode: "algorithms", label: "Algorithm Registry", streamlitEquivalent: "algorithms" },
      { mode: "hpo_tracker", label: "HPO Tracker", streamlitEquivalent: "hpo_tracker" },
    ],
  },
  {
    label: "Launch",
    items: [
      { mode: "sim_launcher", label: "Simulation Launcher" },
      { mode: "training_hub", label: "Training & HPO Hub" },
      { mode: "data_gen", label: "Data Generation" },
      { mode: "eval_runner", label: "Evaluation Runner" },
    ],
  },
  {
    label: "Files",
    items: [
      { mode: "output_browser", label: "Output Browser" },
      { mode: "config_editor", label: "Config Editor" },
    ],
  },
  {
    label: "App",
    items: [
      { mode: "settings", label: "Settings" },
    ],
  },
];

const ICON: Record<AppMode, React.ReactNode> = {
  simulation: <Map size={15} />,
  training: <Activity size={15} />,
  process_monitor: <Terminal size={15} />,
  simulation_summary: <BarChart2 size={15} />,
  benchmark: <Layers size={15} />,
  data_explorer: <Database size={15} />,
  experiment_tracker: <GitBranch size={15} />,
  algorithms: <Bot size={15} />,
  hpo_tracker: <Sliders size={15} />,
  sim_launcher: <Play size={15} />,
  training_hub: <Cpu size={15} />,
  data_gen: <FlaskConical size={15} />,
  config_editor: <FileText size={15} />,
  output_browser: <FolderOpen size={15} />,
  eval_runner: <ClipboardList size={15} />,
  settings: <Settings size={15} />,
};

export function Sidebar() {
  const { mode, setMode } = useAppStore();
  const sidebarOpen = useLayoutStore((s) => s.sidebarOpen);
  const setSidebarOpen = useLayoutStore((s) => s.setSidebarOpen);

  return (
    <aside
      data-tour="sidebar"
      className={[
        "shrink-0 flex flex-col bg-canvas-surface border-r border-canvas-border h-screen overflow-y-auto z-30",
        "w-56 transition-transform duration-200",
        "fixed lg:static inset-y-0 left-0",
        sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0 lg:w-0 lg:border-0 lg:overflow-hidden",
      ].join(" ")}
    >
      {/* Brand */}
      <div className="px-4 py-4 border-b border-canvas-border">
        <div className="flex items-center gap-2">
          <Zap size={18} className="text-accent-primary" />
          <span className="font-bold text-sm tracking-tight text-white">
            WSmart-Route
          </span>
        </div>
        <p className="text-[10px] text-canvas-muted mt-0.5 ml-6">Studio</p>
      </div>

      {/* Nav sections */}
      <nav className="flex-1 px-2 py-3 space-y-4">
        {NAV.map((section) => (
          <div key={section.label}>
            <p className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-widest text-canvas-muted">
              {section.label}
            </p>
            <ul className="space-y-0.5">
              {section.items.map((item) => {
                const active = mode === item.mode;
                return (
                  <li key={item.mode}>
                    <button
                      data-tour={
                        item.mode === "simulation"
                          ? "simulation"
                          : item.mode === "output_browser"
                          ? "output-browser"
                          : item.mode === "sim_launcher"
                          ? "launch"
                          : undefined
                      }
                      onMouseEnter={() => prefetchPage(item.mode)}
                      onClick={() => {
                        setMode(item.mode);
                        if (window.innerWidth < 1024) setSidebarOpen(false);
                      }}
                      className={[
                        "w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-sm transition-colors text-left",
                        active
                          ? "bg-accent-primary/20 text-accent-secondary font-medium"
                          : "text-gray-400 hover:text-gray-100 hover:bg-canvas-hover",
                      ].join(" ")}
                    >
                      <span className={active ? "text-accent-primary" : "text-canvas-muted"}>
                        {ICON[item.mode]}
                      </span>
                      {item.label}
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-canvas-border">
        <p className="text-[10px] text-canvas-muted">v0.1.0</p>
      </div>
    </aside>
  );
}

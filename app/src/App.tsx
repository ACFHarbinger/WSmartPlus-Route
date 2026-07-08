import { useEffect } from "react";
import { Toaster } from "sonner";
import { Layout } from "./components/layout/Layout";
import { useProcessMonitor } from "./hooks/useProcessMonitor";
import { useAppStore } from "./store/app";
import type { AppMode } from "./types";

// pages/monitor — real-time process and training views
import { SimulationMonitor } from "./pages/monitor/SimulationMonitor";
import { TrainingMonitor } from "./pages/monitor/TrainingMonitor";
import { ProcessMonitor } from "./pages/monitor/ProcessMonitor";
// pages/analysis — post-run analytics and exploration
import { SimulationSummary } from "./pages/analysis/SimulationSummary";
import { BenchmarkAnalysis } from "./pages/analysis/BenchmarkAnalysis";
import { DataExplorer } from "./pages/analysis/DataExplorer";
import { ExperimentTracker } from "./pages/analysis/ExperimentTracker";
import { AlgorithmComparison } from "./pages/analysis/AlgorithmComparison";
import { HPOTracker } from "./pages/analysis/HPOTracker";
// pages/launch — process launchers
import { SimulationLauncher } from "./pages/launch/SimulationLauncher";
import { TrainingHub } from "./pages/launch/TrainingHub";
import { DataGeneration } from "./pages/launch/DataGeneration";
import { EvaluationRunner } from "./pages/launch/EvaluationRunner";
// pages/files — file and config management
import { ConfigEditor } from "./pages/files/ConfigEditor";
import { OutputBrowser } from "./pages/files/OutputBrowser";
// pages/app — application settings
import { Settings } from "./pages/app/Settings";

function ActivePage() {
  const mode = useAppStore((s) => s.mode);

  switch (mode) {
    case "simulation":
      return <SimulationMonitor />;
    case "training":
      return <TrainingMonitor />;
    case "simulation_summary":
      return <SimulationSummary />;
    case "benchmark":
      return <BenchmarkAnalysis />;
    case "data_explorer":
      return <DataExplorer />;
    case "experiment_tracker":
      return <ExperimentTracker />;
    case "algorithms":
      return <AlgorithmComparison />;
    case "hpo_tracker":
      return <HPOTracker />;
    case "process_monitor":
      return <ProcessMonitor />;
    case "sim_launcher":
      return <SimulationLauncher />;
    case "training_hub":
      return <TrainingHub />;
    case "data_gen":
      return <DataGeneration />;
    case "config_editor":
      return <ConfigEditor />;
    case "output_browser":
      return <OutputBrowser />;
    case "eval_runner":
      return <EvaluationRunner />;
    case "settings":
      return <Settings />;
    default:
      return <SimulationMonitor />;
  }
}

// Keyboard shortcut map — digit keys (no modifiers) for quick navigation
const DIGIT_MODES: AppMode[] = [
  "simulation",         // 1
  "simulation_summary", // 2
  "training",           // 3
  "benchmark",          // 4
  "sim_launcher",       // 5
  "training_hub",       // 6
  "process_monitor",    // 7
  "settings",           // 8
];

export default function App() {
  const { theme, setMode } = useAppStore();

  // Sync theme class on mount
  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);

  // Global keyboard shortcuts (skip when focus is in a text field)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target.isContentEditable
      ) return;

      // Ctrl+, → Settings
      if ((e.ctrlKey || e.metaKey) && e.key === ",") {
        e.preventDefault();
        setMode("settings");
        return;
      }
      // Ctrl+Shift+P → Process Monitor
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "P") {
        e.preventDefault();
        setMode("process_monitor");
        return;
      }
      // G → Geospatial (placeholder — navigates to simulation monitor for now)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "g") {
        e.preventDefault();
        setMode("simulation");
        return;
      }
      // Q → HPO query tracker
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "q") {
        e.preventDefault();
        setMode("hpo_tracker");
        return;
      }
      // Digit 1-8 → quick mode switch
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
        const n = parseInt(e.key);
        if (n >= 1 && n <= DIGIT_MODES.length) {
          e.preventDefault();
          setMode(DIGIT_MODES[n - 1]);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setMode]);

  // Global process event listener
  useProcessMonitor();

  return (
    <>
      <Layout>
        <ActivePage />
      </Layout>
      <Toaster
        theme={theme}
        position="bottom-right"
        toastOptions={{
          classNames: {
            toast: "bg-canvas-elevated border-canvas-border text-gray-100",
          },
        }}
      />
    </>
  );
}

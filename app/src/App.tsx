import { useEffect } from "react";
import { Toaster } from "sonner";
import { Layout } from "./components/layout/Layout";
import { useProcessMonitor } from "./hooks/useProcessMonitor";
import { useAppStore } from "./store/app";

// Pages — Streamlit parity
import { SimulationMonitor } from "./pages/SimulationMonitor";
import { TrainingMonitor } from "./pages/TrainingMonitor";
import { SimulationSummary } from "./pages/SimulationSummary";
import { BenchmarkAnalysis } from "./pages/BenchmarkAnalysis";
import { DataExplorer } from "./pages/DataExplorer";
import { ExperimentTracker } from "./pages/ExperimentTracker";
import { AlgorithmComparison } from "./pages/AlgorithmComparison";
import { HPOTracker } from "./pages/HPOTracker";
// Pages — Studio-only
import { ProcessMonitor } from "./pages/ProcessMonitor";
import { SimulationLauncher } from "./pages/SimulationLauncher";
import { TrainingHub } from "./pages/TrainingHub";
import { DataGeneration } from "./pages/DataGeneration";
import { ConfigEditor } from "./pages/ConfigEditor";
import { OutputBrowser } from "./pages/OutputBrowser";
import { EvaluationRunner } from "./pages/EvaluationRunner";
import { Settings } from "./pages/Settings";

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

export default function App() {
  const { theme } = useAppStore();

  // Sync theme class on mount
  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);

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

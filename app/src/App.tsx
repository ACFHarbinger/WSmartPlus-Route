import { lazy, Suspense, useEffect } from "react";
import { RefreshCw } from "lucide-react";
import { Toaster } from "sonner";
import { Layout } from "./components/layout/Layout";
import { useHashSync } from "./hooks/useHashSync";
import { useProcessMonitor } from "./hooks/useProcessMonitor";
import { useAppStore } from "./store/app";
import { useLaunchTriggerStore } from "./store/launchTrigger";
import { useLayoutStore } from "./store/layout";
import { useDuckDbInit } from "./hooks/useDuckDbInit";
import { prefetchPage } from "./utils/pagePrefetch";
import { markStartup } from "./utils/startupTiming";
import type { AppMode } from "./types";

// Lazy-loaded pages (§G.7 performance)
const SimulationMonitor = lazy(() =>
  import("./pages/monitor/SimulationMonitor").then((m) => ({ default: m.SimulationMonitor }))
);
const TrainingMonitor = lazy(() =>
  import("./pages/monitor/TrainingMonitor").then((m) => ({ default: m.TrainingMonitor }))
);
const ProcessMonitor = lazy(() =>
  import("./pages/monitor/ProcessMonitor").then((m) => ({ default: m.ProcessMonitor }))
);
const SimulationSummary = lazy(() =>
  import("./pages/analysis/SimulationSummary").then((m) => ({ default: m.SimulationSummary }))
);
const BenchmarkAnalysis = lazy(() =>
  import("./pages/analysis/BenchmarkAnalysis").then((m) => ({ default: m.BenchmarkAnalysis }))
);
const CityComparison = lazy(() =>
  import("./pages/analysis/CityComparison").then((m) => ({ default: m.CityComparison }))
);
const OlapExplorer = lazy(() =>
  import("./pages/analysis/OlapExplorer").then((m) => ({ default: m.OlapExplorer }))
);
const DataExplorer = lazy(() =>
  import("./pages/analysis/DataExplorer").then((m) => ({ default: m.DataExplorer }))
);
const ExperimentTracker = lazy(() =>
  import("./pages/analysis/ExperimentTracker").then((m) => ({ default: m.ExperimentTracker }))
);
const AlgorithmComparison = lazy(() =>
  import("./pages/analysis/AlgorithmComparison").then((m) => ({ default: m.AlgorithmComparison }))
);
const HPOTracker = lazy(() =>
  import("./pages/analysis/HPOTracker").then((m) => ({ default: m.HPOTracker }))
);
const SimulationLauncher = lazy(() =>
  import("./pages/launch/SimulationLauncher").then((m) => ({ default: m.SimulationLauncher }))
);
const TrainingHub = lazy(() =>
  import("./pages/launch/TrainingHub").then((m) => ({ default: m.TrainingHub }))
);
const DataGeneration = lazy(() =>
  import("./pages/launch/DataGeneration").then((m) => ({ default: m.DataGeneration }))
);
const EvaluationRunner = lazy(() =>
  import("./pages/launch/EvaluationRunner").then((m) => ({ default: m.EvaluationRunner }))
);
const ConfigEditor = lazy(() =>
  import("./pages/files/ConfigEditor").then((m) => ({ default: m.ConfigEditor }))
);
const OutputBrowser = lazy(() =>
  import("./pages/files/OutputBrowser").then((m) => ({ default: m.OutputBrowser }))
);
const Settings = lazy(() =>
  import("./pages/app/Settings").then((m) => ({ default: m.Settings }))
);

function PageFallback() {
  return (
    <div className="flex items-center justify-center h-64 gap-2 text-canvas-muted text-sm">
      <RefreshCw size={14} className="animate-spin" />
      Loading…
    </div>
  );
}

function ActivePage() {
  const mode = useAppStore((s) => s.mode);

  let page: React.ReactNode;
  switch (mode) {
    case "simulation":
      page = <SimulationMonitor />;
      break;
    case "training":
      page = <TrainingMonitor />;
      break;
    case "simulation_summary":
      page = <SimulationSummary />;
      break;
    case "benchmark":
      page = <BenchmarkAnalysis />;
      break;
    case "city_comparison":
      page = <CityComparison />;
      break;
    case "olap_explorer":
      page = <OlapExplorer />;
      break;
    case "data_explorer":
      page = <DataExplorer />;
      break;
    case "experiment_tracker":
      page = <ExperimentTracker />;
      break;
    case "algorithms":
      page = <AlgorithmComparison />;
      break;
    case "hpo_tracker":
      page = <HPOTracker />;
      break;
    case "process_monitor":
      page = <ProcessMonitor />;
      break;
    case "sim_launcher":
      page = <SimulationLauncher />;
      break;
    case "training_hub":
      page = <TrainingHub />;
      break;
    case "data_gen":
      page = <DataGeneration />;
      break;
    case "config_editor":
      page = <ConfigEditor />;
      break;
    case "output_browser":
      page = <OutputBrowser />;
      break;
    case "eval_runner":
      page = <EvaluationRunner />;
      break;
    case "settings":
      page = <Settings />;
      break;
    default:
      page = <SimulationMonitor />;
  }

  return <Suspense fallback={<PageFallback />}>{page}</Suspense>;
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
  const { theme, setMode, mode } = useAppStore();
  const commandPaletteOpen = useLayoutStore((s) => s.commandPaletteOpen);
  const setCommandPaletteOpen = useLayoutStore((s) => s.setCommandPaletteOpen);
  const setShortcutsOpen = useLayoutStore((s) => s.setShortcutsOpen);
  const setGuidedTourOpen = useLayoutStore((s) => s.setGuidedTourOpen);
  const setGuidedTourStep = useLayoutStore((s) => s.setGuidedTourStep);
  const guidedTourOpen = useLayoutStore((s) => s.guidedTourOpen);
  const { triggerSim, triggerTrain, triggerDataGen, triggerEval } = useLaunchTriggerStore();

  useHashSync();
  useDuckDbInit();

  // Warm critical route chunks on startup (§G.7)
  useEffect(() => {
    void Promise.all([
      import("./pages/monitor/SimulationMonitor"),
      import("./pages/analysis/SimulationSummary"),
      import("./pages/analysis/BenchmarkAnalysis"),
      import("./pages/analysis/CityComparison"),
      import("./pages/analysis/AlgorithmComparison"),
      import("./pages/analysis/OlapExplorer"),
      import("./pages/monitor/ProcessMonitor"),
      import("./pages/files/OutputBrowser"),
      import("echarts-for-react"),
      import("./components/maps/DeckRouteMap"),
      import("maplibre-gl"),
      import("@deck.gl/react"),
      import("@monaco-editor/react"),
    ]).then(() => markStartup("prefetchDone"));
    prefetchPage("simulation");
    prefetchPage("simulation_summary");
    prefetchPage("benchmark");
    prefetchPage("city_comparison");
    prefetchPage("algorithms");
    prefetchPage("olap_explorer");
    prefetchPage("process_monitor");
    prefetchPage("output_browser");
  }, []);

  // Sync theme class on mount
  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);

  // Global keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ctrl+K → command palette (§G.7) — works even inside text fields
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setCommandPaletteOpen(!commandPaletteOpen);
        return;
      }

      const target = e.target as HTMLElement;
      const inTextField =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target.isContentEditable;

      // Escape → close overlays
      if (e.key === "Escape") {
        if (commandPaletteOpen) setCommandPaletteOpen(false);
        else if (guidedTourOpen) setGuidedTourOpen(false);
        else setShortcutsOpen(false);
        return;
      }

      // Ctrl+Shift+/ → guided tour (§G.19)
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "/") {
        e.preventDefault();
        setGuidedTourStep(0);
        setGuidedTourOpen(true);
        return;
      }

      if (inTextField) return;

      // ? → keyboard shortcuts help overlay (§G.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key === "?") {
        e.preventDefault();
        setShortcutsOpen(true);
        return;
      }
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
      // P → Process monitor (§G.7 / §D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "p") {
        e.preventDefault();
        setMode("process_monitor");
        return;
      }
      // M → Map / simulation digital twin (§G.7 / §D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "m") {
        e.preventDefault();
        setMode("simulation");
        return;
      }
      // Ctrl+R → launch on active launcher page (§G.7)
      if ((e.ctrlKey || e.metaKey) && e.key === "r") {
        e.preventDefault();
        if (mode === "sim_launcher") triggerSim();
        else if (mode === "training_hub") triggerTrain();
        else if (mode === "data_gen") triggerDataGen();
        else if (mode === "eval_runner") triggerEval();
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
  }, [
    setMode,
    mode,
    triggerSim,
    triggerTrain,
    triggerDataGen,
    triggerEval,
    setShortcutsOpen,
    setCommandPaletteOpen,
    commandPaletteOpen,
    guidedTourOpen,
    setGuidedTourOpen,
    setGuidedTourStep,
  ]);

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

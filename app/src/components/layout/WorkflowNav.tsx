/**
 * Analytical workflow strip — Overview → Drill-Down → Geospatial → … → Launch (§G.7).
 */
import { ChevronRight } from "lucide-react";
import { useAppStore } from "../../store/app";
import type { AppMode } from "../../types";

const WORKFLOW_STEPS: Array<{ label: string; mode: AppMode }> = [
  { label: "Overview", mode: "simulation_summary" },
  { label: "Drill-Down", mode: "benchmark" },
  { label: "Geospatial", mode: "simulation" },
  { label: "Registry", mode: "algorithms" },
  { label: "ML", mode: "experiment_tracker" },
  { label: "HPO", mode: "hpo_tracker" },
  { label: "Launch", mode: "sim_launcher" },
];

export function WorkflowNav() {
  const { mode, setMode } = useAppStore();

  return (
    <nav
      className="flex items-center gap-0.5 px-5 py-1.5 bg-canvas-bg border-b border-canvas-border overflow-x-auto"
      aria-label="Analytical workflow"
    >
      {WORKFLOW_STEPS.map((step, i) => {
        const active = mode === step.mode;
        return (
          <span key={step.mode} className="flex items-center shrink-0">
            {i > 0 && <ChevronRight size={10} className="text-canvas-muted mx-0.5" />}
            <button
              onClick={() => setMode(step.mode)}
              className={`text-[10px] px-2 py-0.5 rounded-full transition-colors whitespace-nowrap ${
                active
                  ? "bg-accent-primary/25 text-accent-secondary font-medium"
                  : "text-canvas-muted hover:text-gray-200 hover:bg-canvas-hover"
              }`}
            >
              {step.label}
            </button>
          </span>
        );
      })}
    </nav>
  );
}

import { useCallback } from "react";
import { ChevronLeft, ChevronRight, Compass, X } from "lucide-react";
import { useLayoutStore } from "../../store/layout";

const STEPS = [
  {
    title: "Navigate views",
    body: "Use the sidebar to switch between Monitor, Analysis, Launch, and Files. The workflow strip above each page shows where you are in the analytical flow.",
  },
  {
    title: "Command palette",
    body: "Press Ctrl+K (or click the search icon) to fuzzy-search any view, action, or recent file. Import .wsroute bundles and jump to Simulation Summary from here.",
  },
  {
    title: "Simulation Digital Twin",
    body: "Open a .jsonl log to watch KPIs, bin-fill strips, and route maps update day-by-day. Use the day scrubber and playback controls to replay a run.",
  },
  {
    title: "Output Browser",
    body: "Browse assets/output/, inspect run configs, export .wsroute bundles, or drag-drop a bundle onto the file viewer to inspect its manifest.",
  },
  {
    title: "Launch & monitor",
    body: "Spawn simulations and training jobs from the Launch section. Process Monitor streams stdout in real time and fires toast notifications on completion.",
  },
];

export function GuidedTour() {
  const {
    guidedTourOpen,
    guidedTourStep,
    setGuidedTourOpen,
    setGuidedTourStep,
    setGuidedTourDismissed,
  } = useLayoutStore();

  const close = useCallback(() => {
    setGuidedTourOpen(false);
    setGuidedTourStep(0);
  }, [setGuidedTourOpen, setGuidedTourStep]);

  const finish = useCallback(() => {
    setGuidedTourDismissed(true);
    close();
  }, [setGuidedTourDismissed, close]);

  if (!guidedTourOpen) return null;

  const step = STEPS[guidedTourStep];
  const isLast = guidedTourStep === STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="card max-w-md w-full space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <Compass size={16} className="text-accent-primary" />
            <h2 className="text-sm font-semibold text-gray-100">Studio guided tour</h2>
          </div>
          <button onClick={close} className="btn-ghost p-1" title="Close tour">
            <X size={14} />
          </button>
        </div>

        <p className="text-[10px] text-canvas-muted uppercase tracking-wide">
          Step {guidedTourStep + 1} of {STEPS.length}
        </p>
        <h3 className="text-sm font-medium text-gray-200">{step.title}</h3>
        <p className="text-xs text-canvas-muted leading-relaxed">{step.body}</p>

        <div className="flex gap-1">
          {STEPS.map((_, i) => (
            <span
              key={i}
              className={`h-1 flex-1 rounded-full ${
                i === guidedTourStep ? "bg-accent-primary" : "bg-canvas-border"
              }`}
            />
          ))}
        </div>

        <div className="flex items-center justify-between pt-1">
          <button
            onClick={() => setGuidedTourStep(Math.max(0, guidedTourStep - 1))}
            disabled={guidedTourStep === 0}
            className="btn-ghost text-xs flex items-center gap-1 disabled:opacity-30"
          >
            <ChevronLeft size={12} />
            Back
          </button>
          {isLast ? (
            <button onClick={finish} className="btn-primary text-sm">
              Finish
            </button>
          ) : (
            <button
              onClick={() => setGuidedTourStep(guidedTourStep + 1)}
              className="btn-primary text-sm flex items-center gap-1"
            >
              Next
              <ChevronRight size={12} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

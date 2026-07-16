/**
 * Cross-page train/HPO workflow shortcuts (§A.2 / §A.4 / §G.10 / §G.15 / §G.17 / §G.18).
 */
import { PathHandoffButtons } from "../common/PathHandoffButtons";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";

export interface TrainHpoNavMeshProps {
  /** Hide Training Hub shortcut (on the hub page itself). */
  hideHub?: boolean;
  /** Show HPO Tracker + Experiment Tracker shortcuts during live HPO. */
  showHpoLinks?: boolean;
  /** Show Output Browser shortcut after a successful run. */
  showOutputBrowser?: boolean;
  /** Auto-select this run in Output Browser via ``pendingRunPath``. */
  outputRunPath?: string | null;
  /** Auto-select this run in Training Monitor via ``pendingTrainingRunPath``. */
  trainingRunPath?: string | null;
  /** When false, only Process Monitor (+ optional output browser) are shown. */
  showTrainLinks?: boolean;
  className?: string;
}

export function TrainHpoNavMesh({
  hideHub = false,
  showHpoLinks = false,
  showOutputBrowser = false,
  outputRunPath = null,
  trainingRunPath = null,
  showTrainLinks = true,
  className = "",
}: TrainHpoNavMeshProps) {
  const { setMode } = useRecentHandoff();

  return (
    <div className={`flex items-center gap-2 flex-wrap ${className}`}>
      {showOutputBrowser && (
        <PathHandoffButtons
          path={outputRunPath}
          kind="run"
          labeled
          iconSize={12}
        />
      )}
      {showTrainLinks && (
        <>
          {!hideHub && (
            <button
              onClick={() => setMode("training_hub")}
              className="btn-ghost text-xs text-canvas-muted"
            >
              Training Hub →
            </button>
          )}
          <PathHandoffButtons
            path={trainingRunPath}
            kind="training"
            labeled
            iconSize={12}
          />
          {showHpoLinks && (
            <>
              <button
                onClick={() => setMode("hpo_tracker")}
                className="btn-ghost text-xs text-canvas-muted"
              >
                HPO Tracker →
              </button>
              <button
                onClick={() => setMode("experiment_tracker")}
                className="btn-ghost text-xs text-canvas-muted"
              >
                Experiment Tracker →
              </button>
            </>
          )}
        </>
      )}
      <button
        onClick={() => setMode("process_monitor")}
        className="btn-ghost text-xs text-canvas-muted"
      >
        Process Monitor →
      </button>
    </div>
  );
}

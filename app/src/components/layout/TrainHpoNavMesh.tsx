/**
 * Cross-page train/HPO workflow shortcuts (§A.2 / §A.4 / §G.10 / §G.15 / §G.17 / §G.18).
 */
import { useAppStore } from "../../store/app";
import { useRecentFilesStore } from "../../store/recentFiles";
import { applyRecentHandoff, type RecentPendingSetters } from "../../utils/recentHandoff";

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
  const {
    projectRoot,
    setMode,
    setPendingLogPath,
    setPendingRunPath,
    setPendingCsvPath,
    setPendingTrainingRunPath,
    setPendingCheckpoint,
    setPendingConfigPath,
  } = useAppStore();
  const pushRecent = useRecentFilesStore((s) => s.pushRecent);

  const pendingSetters: RecentPendingSetters = {
    pendingLogPath: setPendingLogPath,
    pendingRunPath: setPendingRunPath,
    pendingCsvPath: setPendingCsvPath,
    pendingTrainingRunPath: setPendingTrainingRunPath,
    pendingCheckpoint: setPendingCheckpoint,
    pendingConfigPath: setPendingConfigPath,
  };

  return (
    <div className={`flex items-center gap-2 flex-wrap ${className}`}>
      {showOutputBrowser && (
        <button
          onClick={() => {
            if (outputRunPath) {
              applyRecentHandoff({
                path: outputRunPath,
                kind: "run",
                projectRoot,
                pushRecent,
                setMode,
                pendingSetters,
              });
            } else {
              setMode("output_browser");
            }
          }}
          className="btn-ghost text-xs text-accent-success"
        >
          Output Browser →
        </button>
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
          <button
            onClick={() => {
              if (trainingRunPath) {
                applyRecentHandoff({
                  path: trainingRunPath,
                  kind: "training",
                  projectRoot,
                  pushRecent,
                  setMode,
                  pendingSetters,
                });
              } else {
                setMode("training");
              }
            }}
            className="btn-ghost text-xs text-canvas-muted"
          >
            Training Monitor →
          </button>
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

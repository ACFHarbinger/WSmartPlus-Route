/**
 * Cross-page sim / data-gen / eval launcher shortcuts (§D.7 / §G.9 / §G.11 / §G.12).
 */
import { useAppStore } from "../../store/app";
import { useRecentFilesStore } from "../../store/recentFiles";
import { portfolioRunLabel } from "../../utils/arrowPipeline";
import type { LauncherKind } from "../../utils/launcherProcess";

export interface LauncherNavMeshProps {
  kind: LauncherKind;
  /** Hide the current launcher page shortcut. */
  hideSelf?: boolean;
  /** Hide Training Hub shortcut (on the hub eval panel itself). */
  hideHub?: boolean;
  /** Show post-run analytics shortcuts (summary, benchmark, data explorer). */
  showPostRun?: boolean;
  /** Eval-only: open Benchmark Analysis with current results. */
  onOpenAnalytics?: () => void;
  /** Eval-only: pre-populate Evaluation Runner with this checkpoint path. */
  checkpointPath?: string | null;
  /** Show Output Browser shortcut after a successful run. */
  showOutputBrowser?: boolean;
  /** Auto-select this run in Output Browser via ``pendingRunPath``. */
  outputRunPath?: string | null;
  className?: string;
}

const LAUNCHER_MODE: Record<LauncherKind, "sim_launcher" | "data_gen" | "eval_runner"> = {
  sim: "sim_launcher",
  data_gen: "data_gen",
  eval: "eval_runner",
};

const LAUNCHER_LABEL: Record<LauncherKind, string> = {
  sim: "Simulation Launcher",
  data_gen: "Data Generation",
  eval: "Evaluation Runner",
};

export function LauncherNavMesh({
  kind,
  hideSelf = false,
  hideHub = false,
  showPostRun = false,
  onOpenAnalytics,
  checkpointPath,
  showOutputBrowser = false,
  outputRunPath = null,
  className = "",
}: LauncherNavMeshProps) {
  const { projectRoot, setMode, setPendingCheckpoint, setPendingRunPath } = useAppStore();
  const pushRecent = useRecentFilesStore((s) => s.pushRecent);

  return (
    <div className={`flex items-center gap-2 flex-wrap ${className}`}>
      {!hideSelf && (
        <button
          onClick={() => setMode(LAUNCHER_MODE[kind])}
          className="btn-ghost text-xs text-canvas-muted"
        >
          {LAUNCHER_LABEL[kind]} →
        </button>
      )}

      {kind === "sim" && (
        <>
          <button
            onClick={() => setMode("simulation")}
            className="btn-ghost text-xs text-canvas-muted"
          >
            Simulation Monitor →
          </button>
          {showPostRun && (
            <button
              onClick={() => setMode("simulation_summary")}
              className="btn-ghost text-xs text-accent-primary"
            >
              Simulation Summary →
            </button>
          )}
        </>
      )}

      {kind === "data_gen" && showPostRun && (
        <button
          onClick={() => setMode("data_explorer")}
          className="btn-ghost text-xs text-accent-primary"
        >
          Data Explorer →
        </button>
      )}

      {kind === "eval" && (
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
            onClick={() => setMode("training")}
            className="btn-ghost text-xs text-canvas-muted"
          >
            Training Monitor →
          </button>
          {showPostRun && checkpointPath && (
            <button
              onClick={() => {
                pushRecent({
                  path: checkpointPath,
                  label: portfolioRunLabel(checkpointPath, undefined, projectRoot),
                  kind: "checkpoint",
                });
                setPendingCheckpoint(checkpointPath);
                setMode("eval_runner");
              }}
              className="btn-ghost text-xs text-accent-secondary"
            >
              Load in Eval Runner →
            </button>
          )}
          {showPostRun && onOpenAnalytics && (
            <button
              onClick={onOpenAnalytics}
              className="btn-ghost text-xs text-accent-primary"
            >
              Benchmark Analysis →
            </button>
          )}
        </>
      )}

      {showOutputBrowser && (
        <button
          onClick={() => {
            if (outputRunPath) {
              pushRecent({
                path: outputRunPath,
                label: portfolioRunLabel(outputRunPath, undefined, projectRoot),
                kind: "run",
              });
              setPendingRunPath(outputRunPath);
            }
            setMode("output_browser");
          }}
          className="btn-ghost text-xs text-accent-success"
        >
          Output Browser →
        </button>
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

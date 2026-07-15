/**
 * Cross-page sim / data-gen / eval launcher shortcuts (§D.7 / §G.9 / §G.11 / §G.12).
 */
import { LogHandoffButtons } from "../common/LogHandoffButtons";
import { PathHandoffButtons } from "../common/PathHandoffButtons";
import { useRecentHandoff } from "../../hooks/useRecentHandoff";
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
  /**
   * Post-run sim log (``.jsonl``) for Simulation Summary / Monitor handoff via
   * ``pendingLogPath`` (§G.9 / §G.1 / §G.16 / §D.7). When unset, the shared
   * log-handoff control still navigates to each destination mode.
   */
  simLogPath?: string | null;
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
  simLogPath = null,
  className = "",
}: LauncherNavMeshProps) {
  const { setMode } = useRecentHandoff();

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
        <LogHandoffButtons
          path={simLogPath}
          labeled
          iconSize={12}
          targets={showPostRun ? ["summary", "monitor"] : ["monitor"]}
        />
      )}

      {kind === "data_gen" && showPostRun && (
        <PathHandoffButtons kind="csv" labeled iconSize={12} />
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
          <PathHandoffButtons kind="training" labeled iconSize={12} />
          {showPostRun && checkpointPath && (
            <PathHandoffButtons
              path={checkpointPath}
              kind="checkpoint"
              labeled
              iconSize={12}
            />
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
        <PathHandoffButtons
          path={outputRunPath}
          kind="run"
          labeled
          iconSize={12}
        />
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

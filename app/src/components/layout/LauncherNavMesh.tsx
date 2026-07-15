/**
 * Cross-page sim / data-gen / eval launcher shortcuts (§D.7 / §G.9 / §G.11 / §G.12).
 */
import { useAppStore } from "../../store/app";
import type { LauncherKind } from "../../utils/launcherProcess";

export interface LauncherNavMeshProps {
  kind: LauncherKind;
  /** Hide the current launcher page shortcut. */
  hideSelf?: boolean;
  /** Show post-run analytics shortcuts (summary, benchmark, data explorer). */
  showPostRun?: boolean;
  /** Eval-only: open Benchmark Analysis with current results. */
  onOpenAnalytics?: () => void;
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
  showPostRun = false,
  onOpenAnalytics,
  className = "",
}: LauncherNavMeshProps) {
  const setMode = useAppStore((s) => s.setMode);

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
          <button
            onClick={() => setMode("training")}
            className="btn-ghost text-xs text-canvas-muted"
          >
            Training Monitor →
          </button>
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

      <button
        onClick={() => setMode("process_monitor")}
        className="btn-ghost text-xs text-canvas-muted"
      >
        Process Monitor →
      </button>
    </div>
  );
}

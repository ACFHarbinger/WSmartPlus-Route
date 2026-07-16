/**
 * Shared per-checkpoint live eval row for Evaluation Runner and Process Monitor
 * (§G.12 / §G.15 / §D.7).
 */
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import type { EvalResult } from "../../../utils/benchmark/evalResults";
import { hasEvalMetrics } from "../../../utils/benchmark/evalResults";
import { parentRunBrushLabelFromCheckpointPath } from "../../../utils/training/checkpoints";
import { EvalResultKpiRow } from "./EvalResultKpiRow";
import { LiveTrainProgressBar } from "../live/LiveTrainProgressBar";
import { ProcessLogTail } from "../process/ProcessLogTail";

export interface EvalCheckpointLiveCardProps {
  procId: string;
  checkpointName: string;
  /** Hydra ``eval.policy.model.load_path`` when known — enables path-chip brush (§G.12 / §D.7). */
  checkpointPath?: string | null;
  projectRoot?: string | null;
  status?: string;
  isRunning: boolean;
  result?: EvalResult;
  logLines?: string[];
  maxLines?: number;
  /** When false, parent ``LauncherLivePanel`` renders the shared log tail (§G.12 / §D.7). */
  showLogTail?: boolean;
  className?: string;
}

export function EvalCheckpointLiveCard({
  procId,
  checkpointName,
  checkpointPath,
  projectRoot,
  status,
  isRunning,
  result,
  logLines = [],
  maxLines,
  showLogTail = true,
  className = "",
}: EvalCheckpointLiveCardProps) {
  return (
    <div
      className={`rounded-lg border border-canvas-border/60 bg-canvas-bg/40 p-2 space-y-1.5 ${className}`.trim()}
    >
      <div className="flex items-center justify-between gap-2">
        {checkpointPath ? (
          <OpenPathToolbar
            path={checkpointPath}
            projectRoot={projectRoot}
            kind="checkpoint"
            label={checkpointName}
            brushLabel={parentRunBrushLabelFromCheckpointPath(checkpointPath, projectRoot)}
            chipClassName="max-w-full"
            className="min-w-0 flex-1"
          />
        ) : (
          <span className="text-xs font-mono text-gray-300 truncate">{checkpointName}</span>
        )}
        {!isRunning && status && (
          <span
            className={`text-[10px] shrink-0 ${
              status === "completed" ? "text-accent-success" : "text-accent-danger"
            }`}
          >
            {status}
          </span>
        )}
      </div>
      {result && hasEvalMetrics(result) && (
        <EvalResultKpiRow result={result} size="compact" showPolicy={false} />
      )}
      {isRunning && <LiveTrainProgressBar processId={procId} />}
      {showLogTail && (
        <ProcessLogTail
          logLines={logLines}
          maxLines={maxLines}
          waiting={isRunning}
          variant="compact"
        />
      )}
    </div>
  );
}

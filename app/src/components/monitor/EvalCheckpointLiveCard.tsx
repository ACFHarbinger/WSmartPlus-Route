/**
 * Shared per-checkpoint live eval row for Evaluation Runner and Process Monitor
 * (§G.12 / §G.15 / §D.7).
 */
import type { EvalResult } from "../../utils/evalResults";
import { hasEvalMetrics } from "../../utils/evalResults";
import { EvalResultKpiRow } from "./EvalResultKpiRow";
import { LiveTrainProgressBar } from "./LiveTrainProgressBar";
import { ProcessLogTail } from "./ProcessLogTail";

export interface EvalCheckpointLiveCardProps {
  procId: string;
  checkpointName: string;
  status?: string;
  isRunning: boolean;
  result?: EvalResult;
  logLines?: string[];
  maxLines?: number;
  className?: string;
}

export function EvalCheckpointLiveCard({
  procId,
  checkpointName,
  status,
  isRunning,
  result,
  logLines = [],
  maxLines,
  className = "",
}: EvalCheckpointLiveCardProps) {
  return (
    <div
      className={`rounded-lg border border-canvas-border/60 bg-canvas-bg/40 p-2 space-y-1.5 ${className}`.trim()}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-mono text-gray-300 truncate">{checkpointName}</span>
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
      <ProcessLogTail
        logLines={logLines}
        maxLines={maxLines}
        waiting={isRunning}
        variant="compact"
      />
    </div>
  );
}

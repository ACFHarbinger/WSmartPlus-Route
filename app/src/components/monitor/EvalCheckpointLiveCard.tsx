/**
 * Shared per-checkpoint live eval row for Evaluation Runner (§G.12 / §D.7).
 */
import type { EvalResult } from "../../utils/evalResults";
import { hasEvalMetrics } from "../../utils/evalResults";
import { EvalResultKpiRow } from "./EvalResultKpiRow";
import { LiveTrainProgressBar } from "./LiveTrainProgressBar";

export interface EvalCheckpointLiveCardProps {
  procId: string;
  checkpointName: string;
  status?: string;
  isRunning: boolean;
  result?: EvalResult;
  tail: string[];
  className?: string;
}

export function EvalCheckpointLiveCard({
  procId,
  checkpointName,
  status,
  isRunning,
  result,
  tail,
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
      {tail.length > 0 && (
        <div className="space-y-0.5 max-h-20 overflow-auto">
          {tail.map((line, i) => (
            <p
              key={i}
              className="text-[10px] font-mono text-gray-400 leading-snug truncate"
            >
              {line}
            </p>
          ))}
        </div>
      )}
      {isRunning && tail.length === 0 && (
        <p className="text-[10px] text-canvas-muted">Waiting for output…</p>
      )}
    </div>
  );
}

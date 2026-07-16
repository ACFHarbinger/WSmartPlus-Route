/**
 * Shared eval result card for Process Monitor embedded panels (§G.12 / §G.15 / §D.7).
 */
import { BarChart3 } from "lucide-react";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import type { EvalResult } from "../../../utils/benchmark/evalResults";
import { parentRunBrushLabelFromCheckpointPath } from "../../../utils/training/checkpoints";
import { EvalResultKpiRow } from "./EvalResultKpiRow";

export interface EvalResultCardProps {
  result: EvalResult;
  projectRoot?: string | null;
  onOpenAnalytics?: () => void;
  className?: string;
}

export function EvalResultCard({
  result,
  projectRoot,
  onOpenAnalytics,
  className = "",
}: EvalResultCardProps) {
  return (
    <div className={`card space-y-2 ${className}`.trim()}>
      <div className="flex items-center justify-between gap-2">
        {result.checkpointPath ? (
          <OpenPathToolbar
            path={result.checkpointPath}
            projectRoot={projectRoot}
            kind="checkpoint"
            label={result.checkpointName}
            brushLabel={parentRunBrushLabelFromCheckpointPath(
              result.checkpointPath,
              projectRoot
            )}
            chipClassName="font-semibold text-gray-200 max-w-full"
            className="min-w-0 flex-1"
          />
        ) : (
          <h3 className="text-xs font-semibold text-gray-200">{result.checkpointName}</h3>
        )}
        {onOpenAnalytics && (
          <button
            onClick={onOpenAnalytics}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <BarChart3 size={12} />
            Open in Analytics →
          </button>
        )}
      </div>
      <EvalResultKpiRow result={result} />
    </div>
  );
}

/**
 * Shared eval result KPI row for Evaluation Runner and Process Monitor (§G.12 / §G.15 / §D.7).
 */
import type { EvalResult } from "../../utils/evalResults";

export interface EvalResultKpiRowProps {
  result: EvalResult;
  /** compact = inline per-checkpoint rows; default = Process Monitor card body. */
  size?: "compact" | "default";
  showPolicy?: boolean;
  className?: string;
}

function metricClass(size: "compact" | "default") {
  return size === "compact" ? "text-[10px]" : "text-xs";
}

function valueClass(size: "compact" | "default") {
  return size === "compact" ? "font-mono text-gray-300" : "font-mono text-gray-200";
}

export function EvalResultKpiRow({
  result,
  size = "default",
  showPolicy = true,
  className = "",
}: EvalResultKpiRowProps) {
  const textClass = metricClass(size);
  const valueTone = valueClass(size);
  const gapClass = size === "compact" ? "gap-3" : "gap-4";
  const Wrapper = size === "compact" ? "div" : "div";

  return (
    <Wrapper className={`flex flex-wrap ${gapClass} ${textClass} ${className}`.trim()}>
      {result.cost != null && (
        size === "compact" ? (
          <span>
            <span className="text-canvas-muted">Cost </span>
            <span className={valueTone}>{result.cost.toFixed(4)}</span>
          </span>
        ) : (
          <div>
            <span className="text-canvas-muted">Cost </span>
            <span className={valueTone}>{result.cost.toFixed(4)}</span>
          </div>
        )
      )}
      {result.gap != null && (
        size === "compact" ? (
          <span>
            <span className="text-canvas-muted">Gap </span>
            <span className={valueTone}>{result.gap.toFixed(4)}%</span>
          </span>
        ) : (
          <div>
            <span className="text-canvas-muted">Gap </span>
            <span className={valueTone}>{result.gap.toFixed(4)}%</span>
          </div>
        )
      )}
      {result.time != null && (
        size === "compact" ? (
          <span>
            <span className="text-canvas-muted">Time </span>
            <span className={valueTone}>{result.time.toFixed(3)}s</span>
          </span>
        ) : (
          <div>
            <span className="text-canvas-muted">Time </span>
            <span className={valueTone}>{result.time.toFixed(3)}s</span>
          </div>
        )
      )}
      {showPolicy && result.policy != null && (
        <div>
          <span className="text-canvas-muted">Policy </span>
          <span className={valueTone}>{result.policy}</span>
        </div>
      )}
    </Wrapper>
  );
}

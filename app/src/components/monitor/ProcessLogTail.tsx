/**
 * Shared stdout/stderr tail display for launcher live panels (§G.11 / §G.12 / §G.15 / §D.7).
 */
import { useMemo } from "react";
import { processLogTail } from "../../utils/processLog";

export interface ProcessLogTailProps {
  /** Pre-formatted tail lines (from ``processLogTail``). */
  lines?: string[];
  /** Raw process log lines — formatted via ``processLogTail`` when ``lines`` omitted. */
  logLines?: string[];
  maxLines?: number;
  /** Show the waiting placeholder when there is no output yet. */
  waiting?: boolean;
  variant?: "compact" | "default";
  className?: string;
}

export function ProcessLogTail({
  lines: linesProp,
  logLines,
  maxLines = 10,
  waiting = false,
  variant = "default",
  className = "",
}: ProcessLogTailProps) {
  const lines = useMemo(() => {
    if (linesProp !== undefined) return linesProp;
    return processLogTail(logLines ?? [], maxLines);
  }, [linesProp, logLines, maxLines]);

  if (lines.length === 0) {
    if (!waiting) return null;
    return (
      <p
        className={`text-canvas-muted ${
          variant === "compact" ? "text-[10px]" : "text-xs"
        } ${className}`.trim()}
      >
        Waiting for output…
      </p>
    );
  }

  const containerClass =
    variant === "compact"
      ? "space-y-0.5 max-h-20 overflow-auto"
      : "bg-canvas-bg rounded-lg p-2 space-y-0.5 max-h-36 overflow-auto";

  const lineClass =
    variant === "compact"
      ? "text-[10px] font-mono text-gray-400 leading-snug truncate"
      : "text-xs font-mono text-gray-400 leading-snug";

  return (
    <div className={`${containerClass} ${className}`.trim()}>
      {lines.map((line, i) => (
        <p key={i} className={lineClass}>
          {line}
        </p>
      ))}
    </div>
  );
}

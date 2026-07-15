/**
 * Clickable path chip with run-label brush ring highlight (§G.14–§G.16 / §D.7).
 */
import type { ReactNode } from "react";
import { useRunLabelBrushToggle } from "../../hooks/useRunLabelBrushToggle";
import { runLabelFromPath } from "../../utils/policyTelemetryTrends";

interface Props {
  path: string;
  className?: string;
  trailing?: ReactNode;
}

export function PathRunLabelChip({ path, className = "", trailing }: Props) {
  const { handleRunLabelClick, isBrushActive } = useRunLabelBrushToggle();
  const runLabel = runLabelFromPath(path);
  const brushActive = isBrushActive(runLabel);

  return (
    <button
      type="button"
      onClick={() => handleRunLabelClick(runLabel)}
      className={`flex items-center gap-1.5 text-xs text-canvas-muted truncate max-w-md rounded-lg px-1.5 py-0.5 transition-colors hover:text-gray-200 ${
        brushActive ? "ring-1 ring-accent-secondary/40" : ""
      } ${className}`}
      title={path}
    >
      <span className="truncate font-mono">{path.split(/[/\\]/).pop()}</span>
      {trailing}
    </button>
  );
}

/**
 * Portfolio / comparison loaded-run row with ``PathRunLabelChip`` brush parity (§G.1 / §G.14 / §D.7).
 */
import type { ReactNode } from "react";
import { X } from "lucide-react";
import { PathRunLabelChip } from "./PathRunLabelChip";
import { runLabelFromPath } from "../../utils/policyTelemetryTrends";

interface Props {
  path: string;
  /** Display label for portfolio single-run highlight; defaults to path stem. */
  label?: string;
  activeRunLabel?: string | null;
  selected?: boolean;
  onRemove?: () => void;
  leading?: ReactNode;
  trailing?: ReactNode;
  className?: string;
}

export function LoadedRunRow({
  path,
  label,
  activeRunLabel = null,
  selected = false,
  onRemove,
  leading,
  trailing,
  className = "",
}: Props) {
  const runLabel = label ?? runLabelFromPath(path);
  const portfolioActive = Boolean(activeRunLabel && activeRunLabel === runLabel);

  return (
    <div
      className={`flex items-center gap-2 text-xs text-gray-300 rounded px-1 -mx-1 ${
        selected || portfolioActive ? "bg-accent-primary/15" : ""
      } ${className}`}
    >
      {leading}
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          className="text-canvas-muted hover:text-accent-danger shrink-0"
        >
          <X size={12} />
        </button>
      )}
      <PathRunLabelChip path={path} className="flex-1 min-w-0" trailing={trailing} />
    </div>
  );
}

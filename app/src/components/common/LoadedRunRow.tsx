/**
 * Portfolio / comparison loaded-run row with ``PathRunLabelChip`` brush parity (§G.1 / §G.14 / §D.7).
 */
import { useMemo, type ReactNode } from "react";
import { X } from "lucide-react";
import { PathRunLabelChip } from "./PathRunLabelChip";
import { useAppStore } from "../../store/app";
import { resolveLocalProjectPath } from "../../utils/outputRunPath";
import { runLabelFromPath } from "../../utils/policyTelemetryTrends";
import type { RecentFileKind } from "../../store/recentFiles";

interface Props {
  path: string;
  /** Resolve relative paths against project root before brush (§G.1 / §G.14–§G.17 / §D.7). */
  projectRoot?: string | null;
  /** Display label for portfolio single-run highlight; defaults to path stem. */
  label?: string;
  activeRunLabel?: string | null;
  selected?: boolean;
  onRemove?: () => void;
  leading?: ReactNode;
  trailing?: ReactNode;
  /**
   * Show kind-aware path handoffs via ``PathHandoffButtons`` (``.jsonl`` dual Summary /
   * Monitor; training / run / csv / checkpoint / config single-icon) (§G.1 / §G.7 / §D.7).
   */
  pathHandoffs?: boolean;
  /**
   * Alias for ``pathHandoffs`` kept for portfolio ``.jsonl`` call sites (§G.1 / §G.16).
   */
  logHandoffs?: boolean;
  /** Explicit handoff kind when path classification is ambiguous. */
  handoffKind?: RecentFileKind | null;
  className?: string;
}

export function LoadedRunRow({
  path,
  projectRoot,
  label,
  activeRunLabel = null,
  selected = false,
  onRemove,
  leading,
  trailing,
  pathHandoffs = false,
  logHandoffs = false,
  handoffKind = null,
  className = "",
}: Props) {
  const storeProjectRoot = useAppStore((s) => s.projectRoot);
  const effectiveProjectRoot = projectRoot ?? storeProjectRoot;
  const resolvedPath = useMemo(
    () => resolveLocalProjectPath(path, effectiveProjectRoot) ?? path,
    [path, effectiveProjectRoot]
  );
  const runLabel = label ?? runLabelFromPath(resolvedPath);
  const portfolioActive = Boolean(activeRunLabel && activeRunLabel === runLabel);
  const showPathHandoffs = pathHandoffs || logHandoffs;

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
      <PathRunLabelChip
        path={path}
        projectRoot={effectiveProjectRoot}
        label={label}
        className="flex-1 min-w-0"
        handoff={showPathHandoffs ? (handoffKind ?? true) : false}
        handoffStoredLabel={label}
        trailing={trailing}
      />
    </div>
  );
}

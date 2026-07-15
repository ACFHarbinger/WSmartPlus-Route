/**
 * Portfolio / comparison loaded-run row with ``PathRunLabelChip`` brush parity (§G.1 / §G.14 / §D.7).
 */
import { useMemo, type ReactNode } from "react";
import { BarChart2, Map as MapIcon, X } from "lucide-react";
import { PathRunLabelChip } from "./PathRunLabelChip";
import { useRecentHandoff } from "../../hooks/useRecentHandoff";
import { useAppStore } from "../../store/app";
import { resolveLocalProjectPath } from "../../utils/outputRunPath";
import { runLabelFromPath } from "../../utils/policyTelemetryTrends";

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
   * Show Summary + Simulation Monitor handoff buttons for ``.jsonl`` portfolio rows
   * using the shared mode override (§G.1 / §G.16 / §D.7).
   */
  logHandoffs?: boolean;
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
  logHandoffs = false,
  className = "",
}: Props) {
  const storeProjectRoot = useAppStore((s) => s.projectRoot);
  const { handoff } = useRecentHandoff();
  const effectiveProjectRoot = projectRoot ?? storeProjectRoot;
  const resolvedPath = useMemo(
    () => resolveLocalProjectPath(path, effectiveProjectRoot) ?? path,
    [path, effectiveProjectRoot]
  );
  const runLabel = label ?? runLabelFromPath(resolvedPath);
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
      <PathRunLabelChip
        path={path}
        projectRoot={effectiveProjectRoot}
        label={label}
        className="flex-1 min-w-0"
        trailing={
          logHandoffs || trailing ? (
            <>
              {logHandoffs && (
                <span className="flex items-center gap-0.5 shrink-0">
                  <button
                    type="button"
                    title="Open in Simulation Summary"
                    onClick={(e) => {
                      e.stopPropagation();
                      handoff(path, "log", { storedLabel: label });
                    }}
                    className="btn-ghost p-0.5 text-accent-primary"
                  >
                    <BarChart2 size={11} />
                  </button>
                  <button
                    type="button"
                    title="Open in Simulation Monitor"
                    onClick={(e) => {
                      e.stopPropagation();
                      handoff(path, "log", { storedLabel: label, mode: "simulation" });
                    }}
                    className="btn-ghost p-0.5 text-accent-secondary"
                  >
                    <MapIcon size={11} />
                  </button>
                </span>
              )}
              {trailing}
            </>
          ) : undefined
        }
      />
    </div>
  );
}

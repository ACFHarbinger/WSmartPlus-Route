/**
 * Shared process-id footer row for launcher and train/HPO live panels (§G.9–§G.12 / §G.10 / §G.15 / §G.17 / §G.18 / §D.7).
 */
import { PathRunLabelChip } from "../common/PathRunLabelChip";
import { useAppStore } from "../../store/app";

export interface ProcessIdFooterProps {
  processId?: string;
  processIds?: string[];
  /** When set, renders ``PathRunLabelChip`` for click-to-brush parity with panel headers (§G.9–§G.18 / §D.7). */
  logPath?: string | null;
  /** Resolve relative log paths against project root before brush (§G.9–§G.18 / §D.7). */
  projectRoot?: string | null;
  className?: string;
}

export function ProcessIdFooter({
  processId,
  processIds,
  logPath,
  projectRoot,
  className = "",
}: ProcessIdFooterProps) {
  const storeProjectRoot = useAppStore((s) => s.projectRoot);
  const effectiveProjectRoot = projectRoot ?? storeProjectRoot;
  const ids =
    processIds ?? (processId != null && processId !== "" ? [processId] : []);
  if (ids.length === 0 && !logPath) return null;

  if (logPath) {
    return (
      <div className={`flex items-center gap-2 min-w-0 ${className}`.trim()}>
        <PathRunLabelChip
          path={logPath}
          projectRoot={effectiveProjectRoot}
          className="flex-1 min-w-0"
          handoff
        />
        {ids.length === 1 && (
          <span className="text-[10px] text-canvas-muted font-mono shrink-0 truncate max-w-[8rem]">
            {ids[0]}
          </span>
        )}
      </div>
    );
  }

  if (ids.length === 1) {
    return (
      <p className={`text-xs text-canvas-muted font-mono truncate ${className}`.trim()}>
        {ids[0]}
      </p>
    );
  }

  return (
    <div className={`space-y-0.5 ${className}`.trim()}>
      {ids.map((id) => (
        <p key={id} className="text-xs text-canvas-muted font-mono truncate">
          {id}
        </p>
      ))}
    </div>
  );
}

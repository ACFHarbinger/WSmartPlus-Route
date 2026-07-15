/**
 * Shared process-id footer row for launcher and train/HPO live panels (§G.9–§G.12 / §G.10 / §G.15 / §G.17 / §G.18 / §D.7).
 */
import { OpenPathToolbar } from "../common/OpenPathToolbar";
import { useAppStore } from "../../store/app";

export interface ProcessIdFooterProps {
  processId?: string;
  processIds?: string[];
  /** When set, renders ``OpenPathToolbar`` for click-to-brush + path handoffs (§G.9–§G.18 / §D.7). */
  logPath?: string | null;
  /**
   * Process-derived ``run_label`` for chip brush parity with live headers / GlobalFilterBar
   * when the path stem would otherwise diverge (train dirs, process-id fallback) (§D.7).
   */
  runLabel?: string | null;
  /** Resolve relative log paths against project root before brush (§G.9–§G.18 / §D.7). */
  projectRoot?: string | null;
  className?: string;
}

export function ProcessIdFooter({
  processId,
  processIds,
  logPath,
  runLabel,
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
      <OpenPathToolbar
        path={logPath}
        projectRoot={effectiveProjectRoot}
        brushLabel={runLabel?.trim() ? runLabel : undefined}
        chipClassName="flex-1 min-w-0 max-w-none"
        className={`flex-1 min-w-0 ${className}`.trim()}
      >
        {ids.length === 1 ? (
          <span className="text-[10px] text-canvas-muted font-mono shrink-0 truncate max-w-[8rem]">
            {ids[0]}
          </span>
        ) : ids.length > 1 ? (
          <span
            className="text-[10px] text-canvas-muted font-mono shrink-0 truncate max-w-[14rem]"
            title={ids.join(", ")}
          >
            {ids.length} processes
          </span>
        ) : null}
      </OpenPathToolbar>
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

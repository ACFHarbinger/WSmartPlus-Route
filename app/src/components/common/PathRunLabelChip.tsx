/**
 * Clickable path chip with run-label brush ring highlight (§G.14–§G.16 / §D.7).
 *
 * Optional ``handoff`` appends kind-aware ``PathHandoffButtons`` (Summary / Monitor dual
 * for ``.jsonl``, single-icon destinations for other recent-file kinds) so live headers,
 * footers, and open-path chips share one control surface (§G.7 / §D.7).
 *
 * Prefer ``OpenPathToolbar`` / ``RunLabelHeaderSuffix`` at call sites; this chip remains
 * the atomic path+brush control composed by those shared shells.
 */
import { useMemo, type ReactNode } from "react";
import { useRunLabelBrushToggle } from "../../hooks/useRunLabelBrushToggle";
import { useAppStore } from "../../store/app";
import type { RecentFileKind } from "../../store/recentFiles";
import { resolveLocalProjectPath } from "../../utils/outputRunPath";
import { runLabelFromPath } from "../../utils/policyTelemetryTrends";
import type { LogHandoffTarget } from "./LogHandoffButtons";
import { PathHandoffButtons } from "./PathHandoffButtons";

interface Props {
  path: string;
  /** Resolve relative paths against project root before brush (§G.10–§G.13 / §D.7). */
  projectRoot?: string | null;
  /** Override display text (defaults to path stem). */
  label?: string;
  /** Override brush run_label when display label should differ (defaults to ``label`` or path stem). */
  brushLabel?: string;
  className?: string;
  trailing?: ReactNode;
  /**
   * When true, auto-classify ``path`` and show ``PathHandoffButtons``.
   * Pass an explicit ``RecentFileKind`` to force the destination (e.g. checkpoint).
   * Composes with ``trailing`` (handoff icons first, then custom trailing).
   */
  handoff?: boolean | RecentFileKind;
  /** Optional stored recent-file label for handoff push (§G.7 / §D.7). */
  handoffStoredLabel?: string;
  /** Log-kind only: Summary / Monitor target filter (§G.1 / §G.16 / §D.7). */
  handoffTargets?: LogHandoffTarget[];
  /** Icon size for auto handoff buttons (default 11). */
  handoffIconSize?: number;
  /** Called after a handoff (e.g. close Command Palette). */
  handoffOnAfterOpen?: () => void;
}

export function PathRunLabelChip({
  path,
  projectRoot,
  label,
  brushLabel,
  className = "",
  trailing,
  handoff,
  handoffStoredLabel,
  handoffTargets,
  handoffIconSize = 11,
  handoffOnAfterOpen,
}: Props) {
  const { handleRunLabelClick, isBrushActive } = useRunLabelBrushToggle();
  const storeProjectRoot = useAppStore((s) => s.projectRoot);
  const effectiveProjectRoot = projectRoot ?? storeProjectRoot;
  const resolvedPath = useMemo(
    () => resolveLocalProjectPath(path, effectiveProjectRoot) ?? path,
    [path, effectiveProjectRoot]
  );
  const runLabel = brushLabel ?? label ?? runLabelFromPath(resolvedPath);
  const brushActive = isBrushActive(runLabel);
  const displayText = label ?? resolvedPath.split(/[/\\]/).pop() ?? resolvedPath;

  const handoffNode = handoff ? (
    <PathHandoffButtons
      path={resolvedPath}
      kind={typeof handoff === "string" ? handoff : undefined}
      storedLabel={handoffStoredLabel}
      targets={handoffTargets}
      iconSize={handoffIconSize}
      onAfterOpen={handoffOnAfterOpen}
    />
  ) : null;
  const autoTrailing =
    handoffNode || trailing ? (
      <>
        {handoffNode}
        {trailing}
      </>
    ) : undefined;

  return (
    <button
      type="button"
      onClick={() => handleRunLabelClick(runLabel)}
      className={`flex items-center gap-1.5 text-xs text-canvas-muted truncate max-w-md rounded-lg px-1.5 py-0.5 transition-colors hover:text-gray-200 ${
        brushActive ? "ring-1 ring-accent-secondary/40" : ""
      } ${className}`}
      title={resolvedPath}
    >
      <span className="truncate font-mono">{displayText}</span>
      {autoTrailing}
    </button>
  );
}

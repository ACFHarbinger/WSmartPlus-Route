/**
 * Clickable path chip with run-label brush ring highlight (§G.14–§G.16 / §D.7).
 */
import { useMemo, type ReactNode } from "react";
import { useRunLabelBrushToggle } from "../../hooks/useRunLabelBrushToggle";
import { resolveLocalProjectPath } from "../../utils/outputRunPath";
import { runLabelFromPath } from "../../utils/policyTelemetryTrends";

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
}

export function PathRunLabelChip({
  path,
  projectRoot,
  label,
  brushLabel,
  className = "",
  trailing,
}: Props) {
  const { handleRunLabelClick, isBrushActive } = useRunLabelBrushToggle();
  const resolvedPath = useMemo(
    () => resolveLocalProjectPath(path, projectRoot) ?? path,
    [path, projectRoot]
  );
  const runLabel = brushLabel ?? label ?? runLabelFromPath(resolvedPath);
  const brushActive = isBrushActive(runLabel);
  const displayText = label ?? resolvedPath.split(/[/\\]/).pop() ?? resolvedPath;

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
      {trailing}
    </button>
  );
}

interface HeaderSuffixProps {
  logPath?: string | null;
  runLabel?: string | null;
  /** embedded / muted headers use accent-secondary without font-normal. */
  tone?: "default" | "muted";
  chipClassName?: string;
}

/** Inline run-label suffix for live panel headers — chip when path known, else plain text (§G.9–§G.18 / §D.7). */
export function RunLabelHeaderSuffix({
  logPath,
  runLabel,
  tone = "default",
  chipClassName = "ml-2",
}: HeaderSuffixProps) {
  if (logPath) {
    return <PathRunLabelChip path={logPath} className={chipClassName} />;
  }
  if (!runLabel) return null;
  const textClass =
    tone === "muted"
      ? "ml-2 text-accent-secondary"
      : "ml-2 text-xs font-normal text-accent-secondary";
  return <span className={textClass}>· {runLabel}</span>;
}

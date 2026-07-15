/**
 * Open-path toolbar cluster: optional labeled reverse-handoff buttons + path chip
 * with icon handoffs (§G.1 / §G.4–§G.7 / §G.9–§G.19 / §D.7).
 *
 * Collapses the repeated dual-control pattern on analytics open-log toolbars,
 * OLAP / Data Explorer ingest paths, Config Editor YAML paths, Output Browser
 * run headers / file viewers / checkpoint sidebars / ``.wsroute`` members,
 * Training Monitor logs discovery + run panels + checkpoint browsers, HPO /
 * Experiment tracker paths (storage URI, reports, trial log dirs, MLflow run
 * dirs), ML introspection archives, Settings path previews, launcher
 * selected-path previews (eval / train / data-gen), eval / benchmark
 * results-table checkpoints, Process Monitor process rows, live-panel
 * ``ProcessIdFooter`` chips, portfolio ``LoadedRunRow`` lists, Command Palette
 * recent-file rows, live-panel ``RunLabelHeaderSuffix`` chips, Graph Topology
 * distance-matrix paths, and residual open-path surfaces after pass 227 put
 * chip handoffs on ``PathRunLabelChip``.
 */
import type { ReactNode } from "react";
import type { RecentFileKind } from "../../store/recentFiles";
import type { LogHandoffTarget } from "./LogHandoffButtons";
import { PathHandoffButtons } from "./PathHandoffButtons";
import { PathRunLabelChip } from "./PathRunLabelChip";

export interface OpenPathToolbarProps {
  path: string;
  /** Resolve relative paths against project root before brush (§G.1–§G.18 / §D.7). */
  projectRoot?: string | null;
  /**
   * Explicit kind for labeled buttons and default chip handoff.
   * When omitted, both auto-classify via ``recentKindFromPath``.
   */
  kind?: RecentFileKind | null;
  /** Optional stored recent-file label (portfolio / run name). */
  storedLabel?: string;
  /** Show labeled ``PathHandoffButtons`` beside the chip (default false). */
  labeled?: boolean;
  /**
   * Log-kind target filter for labeled buttons (e.g. host pages pass the reverse
   * destination only — Summary → Monitor, Monitor → Summary).
   */
  labeledTargets?: LogHandoffTarget[];
  /** Icon size for labeled buttons (default 14 for toolbars). */
  labeledIconSize?: number;
  /**
   * Chip handoff: ``true`` auto-classifies, ``false`` disables, or an explicit
   * ``RecentFileKind``. Defaults to ``kind ?? true``.
   */
  handoff?: boolean | RecentFileKind;
  /**
   * Chip icon targets. Defaults to ``labeledTargets`` when set so host pages keep
   * reverse-destination parity between labeled and chip controls.
   */
  handoffTargets?: LogHandoffTarget[];
  /** Called after a chip/labeled handoff (e.g. close Command Palette). */
  handoffOnAfterOpen?: () => void;
  /** Chip trailing meta (pipeline timing, row counts, live spinners). */
  trailing?: ReactNode;
  /** Chip display label override. */
  label?: string;
  /** Chip brush run_label override. */
  brushLabel?: string;
  chipClassName?: string;
  className?: string;
  /** Order of labeled vs chip (default labeled first). */
  order?: "labeled-first" | "chip-first";
  /** Extra toolbar controls after the path cluster (export buttons, etc.). */
  children?: ReactNode;
}

/**
 * Labeled reverse-handoff + path chip with icon handoffs for open-path toolbars.
 * Returns ``null`` when ``path`` is empty.
 */
export function OpenPathToolbar({
  path,
  projectRoot,
  kind = null,
  storedLabel,
  labeled = false,
  labeledTargets,
  labeledIconSize = 14,
  handoff,
  handoffTargets,
  handoffOnAfterOpen,
  trailing,
  label,
  brushLabel,
  chipClassName = "",
  className = "",
  order = "labeled-first",
  children,
}: OpenPathToolbarProps) {
  const trimmed = path.trim();
  if (!trimmed) return null;

  const chipHandoff = handoff === undefined ? (kind ?? true) : handoff;
  const chipTargets = handoffTargets ?? labeledTargets;

  const labeledNode = labeled ? (
    <PathHandoffButtons
      path={trimmed}
      kind={kind}
      storedLabel={storedLabel}
      targets={labeledTargets}
      labeled
      iconSize={labeledIconSize}
      className="shrink-0"
      onAfterOpen={handoffOnAfterOpen}
    />
  ) : null;

  const chipNode = (
    <PathRunLabelChip
      path={trimmed}
      projectRoot={projectRoot}
      label={label}
      brushLabel={brushLabel}
      className={chipClassName}
      handoff={chipHandoff}
      handoffStoredLabel={storedLabel}
      handoffTargets={chipTargets}
      handoffOnAfterOpen={handoffOnAfterOpen}
      trailing={trailing}
    />
  );

  return (
    <span className={`flex items-center gap-3 flex-wrap min-w-0 ${className}`}>
      {order === "chip-first" ? (
        <>
          {chipNode}
          {labeledNode}
        </>
      ) : (
        <>
          {labeledNode}
          {chipNode}
        </>
      )}
      {children}
    </span>
  );
}

interface HeaderSuffixProps {
  logPath?: string | null;
  runLabel?: string | null;
  /** Resolve relative log paths against project root before brush (§G.9–§G.18 / §D.7). */
  projectRoot?: string | null;
  /** embedded / muted headers use accent-secondary without font-normal. */
  tone?: "default" | "muted";
  chipClassName?: string;
  /**
   * Show path-kind handoffs on the header chip (default true when ``logPath`` is set).
   * Covers Simulation Launcher / Process Monitor / train-HPO live headers (§G.9–§G.18 / §D.7).
   */
  handoff?: boolean | RecentFileKind;
}

/**
 * Inline run-label suffix for live panel headers — ``OpenPathToolbar`` when path
 * known, else plain text (§G.9–§G.18 / §D.7).
 *
 * When both ``logPath`` and process-derived ``runLabel`` are set, the chip brushes
 * ``runLabel`` so live headers stay in sync with ``GlobalFilterBar`` (§D.7).
 */
export function RunLabelHeaderSuffix({
  logPath,
  runLabel,
  projectRoot,
  tone = "default",
  chipClassName = "ml-2",
  handoff = true,
}: HeaderSuffixProps) {
  if (logPath) {
    return (
      <OpenPathToolbar
        path={logPath}
        projectRoot={projectRoot}
        chipClassName={chipClassName}
        className="inline-flex items-center gap-1.5 min-w-0"
        handoff={handoff}
        brushLabel={runLabel?.trim() ? runLabel : undefined}
      />
    );
  }
  if (!runLabel) return null;
  const textClass =
    tone === "muted"
      ? "ml-2 text-accent-secondary"
      : "ml-2 text-xs font-normal text-accent-secondary";
  return <span className={textClass}>· {runLabel}</span>;
}

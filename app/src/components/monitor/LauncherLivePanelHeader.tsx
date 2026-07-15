/**
 * Shared live/post-run panel header for sim / data-gen / eval launcher workflows (§G.9 / §G.11 / §G.12 / §G.15).
 */
import type { ReactNode } from "react";
import { Activity, CheckCircle, XCircle } from "lucide-react";
import { RunLabelHeaderSuffix } from "../common/PathRunLabelChip";
import { useAppStore } from "../../store/app";
import { LauncherNavMesh, type LauncherNavMeshProps } from "../layout/LauncherNavMesh";

export type LauncherLiveStatus = "running" | "completed" | "failed" | string;

export interface LauncherLivePanelHeaderProps {
  status: LauncherLiveStatus;
  title: ReactNode;
  navMesh: LauncherNavMeshProps;
  /** card = launcher pages; embedded = Process Monitor analytics section. */
  variant?: "card" | "embedded";
  /** Process Monitor: accent-secondary run label suffix. */
  runLabel?: string | null;
  /** When set, renders ``PathRunLabelChip`` for click-to-brush parity (§G.9–§G.15 / §D.7). */
  logPath?: string | null;
  /** Resolve relative log paths against project root before brush (§G.9–§G.15 / §D.7). */
  projectRoot?: string | null;
  /** Embedded variant: append · live when status is running. */
  showLiveSuffix?: boolean;
  /** Card variant: optional trailing content beside nav mesh (e.g. sim countdown). */
  navTrailing?: ReactNode;
  className?: string;
}

function StatusIcon({ status, size }: { status: LauncherLiveStatus; size: number }) {
  if (status === "running") {
    return <Activity size={size} className="text-accent-primary animate-pulse" />;
  }
  if (status === "completed") {
    return <CheckCircle size={size} className="text-accent-success" />;
  }
  return <XCircle size={size} className="text-accent-danger" />;
}

export function LauncherLivePanelHeader({
  status,
  title,
  navMesh,
  variant = "card",
  runLabel,
  logPath,
  projectRoot,
  showLiveSuffix = true,
  navTrailing,
  className = "",
}: LauncherLivePanelHeaderProps) {
  const storeProjectRoot = useAppStore((s) => s.projectRoot);
  const effectiveProjectRoot = projectRoot ?? storeProjectRoot;
  const nav = <LauncherNavMesh {...navMesh} />;

  if (variant === "embedded") {
    return (
      <div className={`flex items-center gap-2 flex-wrap ${className}`}>
        <p className="text-xs text-canvas-muted flex-1 min-w-0 flex items-center flex-wrap gap-x-1">
          <span>{title}</span>
          <RunLabelHeaderSuffix
            logPath={logPath}
            runLabel={runLabel}
            projectRoot={effectiveProjectRoot}
            tone="muted"
          />
          {showLiveSuffix && status === "running" && (
            <span className="ml-2 text-accent-success">· live</span>
          )}
        </p>
        {nav}
      </div>
    );
  }

  return (
    <div className={`flex items-center justify-between gap-2 flex-wrap ${className}`}>
      <div className="flex items-center gap-2 min-w-0 flex-wrap">
        <StatusIcon status={status} size={14} />
        <h2 className="text-sm font-semibold text-gray-200 flex items-center flex-wrap gap-x-1">
          <span>{title}</span>
          <RunLabelHeaderSuffix
            logPath={logPath}
            runLabel={runLabel}
            projectRoot={effectiveProjectRoot}
          />
          {showLiveSuffix && status === "running" && (
            <span className="ml-2 text-xs font-normal text-accent-success">· live</span>
          )}
        </h2>
      </div>
      <div className="flex items-center gap-2 flex-wrap justify-end">
        {navTrailing}
        {nav}
      </div>
    </div>
  );
}

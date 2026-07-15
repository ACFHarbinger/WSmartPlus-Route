/**
 * Shared live/post-run panel header for train/HPO workflow pages (§G.10 / §G.15 / §G.17 / §G.18).
 */
import type { ReactNode } from "react";
import { Activity, CheckCircle, Radio, XCircle } from "lucide-react";
import { TrainHpoNavMesh, type TrainHpoNavMeshProps } from "../layout/TrainHpoNavMesh";
import { TrainHpoRehydrationBadges } from "./TrainHpoRehydrationBadges";

export type TrainHpoLiveStatus = "running" | "completed" | "failed" | string;

export interface TrainHpoLivePanelHeaderProps {
  status: TrainHpoLiveStatus;
  title: ReactNode;
  processId?: string;
  metricCount?: number;
  healthCount?: number;
  attentionCount?: number;
  navMesh: TrainHpoNavMeshProps;
  /** Pulse icon while running: radio (monitor pages) or activity (Training Hub). */
  runningIcon?: "radio" | "activity";
  /** inline = flex-wrap row; split = justify-between title group vs nav (Training Hub). */
  layout?: "inline" | "split";
  /** mono = accent-success font-mono; heading = semibold gray-200; muted = analytics subtitle. */
  titleTone?: "mono" | "heading" | "muted";
  /** Append · live suffix when status is running (Process Monitor analytics header). */
  showLiveSuffix?: boolean;
  className?: string;
}

function StatusIcon({
  status,
  runningIcon,
  size,
}: {
  status: TrainHpoLiveStatus;
  runningIcon: "radio" | "activity";
  size: number;
}) {
  if (status === "running") {
    if (runningIcon === "activity") {
      return <Activity size={size} className="text-accent-primary animate-pulse" />;
    }
    return <Radio size={size} className="text-accent-success animate-pulse shrink-0" />;
  }
  if (status === "completed") {
    return <CheckCircle size={size} className="text-accent-success shrink-0" />;
  }
  return <XCircle size={size} className="text-accent-danger shrink-0" />;
}

export function TrainHpoLivePanelHeader({
  status,
  title,
  processId,
  metricCount = 0,
  healthCount = 0,
  attentionCount = 0,
  navMesh,
  runningIcon = "radio",
  layout = "inline",
  titleTone = "mono",
  showLiveSuffix = false,
  className = "",
}: TrainHpoLivePanelHeaderProps) {
  const iconSize = titleTone === "heading" ? 14 : 13;
  const badges = (
    <TrainHpoRehydrationBadges
      metricCount={metricCount}
      healthCount={healthCount}
      attentionCount={attentionCount}
    />
  );
  const nav = <TrainHpoNavMesh {...navMesh} />;

  if (layout === "split") {
    return (
      <div className={`flex items-center justify-between ${className}`}>
        <div className="flex items-center gap-2">
          <StatusIcon status={status} runningIcon={runningIcon} size={iconSize} />
          <h2 className="text-sm font-semibold text-gray-200">{title}</h2>
          {badges}
        </div>
        {nav}
      </div>
    );
  }

  if (titleTone === "muted") {
    return (
      <div className={`flex items-center gap-2 flex-wrap ${className}`}>
        <p className="text-xs text-canvas-muted flex-1 min-w-0">
          {title}
          {showLiveSuffix && status === "running" && (
            <span className="ml-2 text-accent-success">· live</span>
          )}
        </p>
        {badges}
        {nav}
      </div>
    );
  }

  return (
    <div className={`flex items-center gap-2 flex-wrap ${className}`}>
      <StatusIcon status={status} runningIcon={runningIcon} size={iconSize} />
      <p className="text-sm text-accent-success font-mono">{title}</p>
      {processId && (
        <span className="text-xs text-canvas-muted font-mono truncate flex-1">
          {processId}
        </span>
      )}
      {badges}
      {nav}
    </div>
  );
}

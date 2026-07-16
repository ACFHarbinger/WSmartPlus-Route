/**
 * Shared live/post-run panel header for train/HPO workflow pages (§G.10 / §G.15 / §G.17 / §G.18).
 */
import type { ReactNode } from "react";
import { Activity, CheckCircle, Radio, XCircle } from "lucide-react";
import { RunLabelHeaderSuffix } from "../../common/OpenPathToolbar";
import { useAppStore } from "../../../store/app";
import { TrainHpoNavMesh, type TrainHpoNavMeshProps } from "../../layout/TrainHpoNavMesh";
import { TrainHpoRehydrationBadges } from "./TrainHpoRehydrationBadges";

export type TrainHpoLiveStatus = "running" | "completed" | "failed" | string;

export interface TrainHpoOverlaySelect {
  checked: boolean;
  onChange: () => void;
}

export interface TrainHpoLivePanelHeaderProps {
  status: TrainHpoLiveStatus;
  title: ReactNode;
  processId?: string;
  metricCount?: number;
  healthCount?: number;
  attentionCount?: number;
  navMesh: TrainHpoNavMeshProps;
  /** Optional overlay-chart checkbox (Training Monitor LIVE_KEY selection). */
  overlaySelect?: TrainHpoOverlaySelect;
  /** Pulse icon while running: radio (monitor pages) or activity (Training Hub). */
  runningIcon?: "radio" | "activity";
  /** inline = flex-wrap row; split = justify-between title group vs nav (Training Hub). */
  layout?: "inline" | "split";
  /** mono = accent-success font-mono; heading = semibold gray-200; muted = analytics subtitle. */
  titleTone?: "mono" | "heading" | "muted";
  /** Append · live suffix when status is running (Process Monitor analytics header). */
  showLiveSuffix?: boolean;
  /** Process Monitor: accent-secondary run label suffix (parity with LauncherLivePanelHeader). */
  runLabel?: string | null;
  /** When set, renders ``OpenPathToolbar`` via ``RunLabelHeaderSuffix`` for click-to-brush parity (§G.10–§G.18 / §D.7). */
  logPath?: string | null;
  /** Resolve relative log paths against project root before brush (§G.10–§G.18 / §D.7). */
  projectRoot?: string | null;
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

function InlineTitleRow({
  status,
  title,
  titleSuffix,
  processId,
  badges,
  runningIcon,
  iconSize,
}: {
  status: TrainHpoLiveStatus;
  title: ReactNode;
  titleSuffix: ReactNode;
  processId?: string;
  badges: ReactNode;
  runningIcon: "radio" | "activity";
  iconSize: number;
}) {
  return (
    <>
      <StatusIcon status={status} runningIcon={runningIcon} size={iconSize} />
      <p className="text-sm text-accent-success font-mono">
        {title}
        {titleSuffix}
      </p>
      {processId && (
        <span className="text-xs text-canvas-muted font-mono truncate max-w-xs">
          {processId}
        </span>
      )}
      {badges}
    </>
  );
}

export function TrainHpoLivePanelHeader({
  status,
  title,
  processId,
  metricCount = 0,
  healthCount = 0,
  attentionCount = 0,
  navMesh,
  overlaySelect,
  runningIcon = "radio",
  layout = "inline",
  titleTone = "mono",
  showLiveSuffix = false,
  runLabel,
  logPath,
  projectRoot,
  className = "",
}: TrainHpoLivePanelHeaderProps) {
  const storeProjectRoot = useAppStore((s) => s.projectRoot);
  const effectiveProjectRoot = projectRoot ?? storeProjectRoot;
  const iconSize = titleTone === "heading" ? 14 : 13;
  const badges = (
    <TrainHpoRehydrationBadges
      metricCount={metricCount}
      healthCount={healthCount}
      attentionCount={attentionCount}
    />
  );
  const nav = <TrainHpoNavMesh {...navMesh} />;

  const titleSuffix = (
    <>
      <RunLabelHeaderSuffix
        logPath={logPath}
        runLabel={runLabel}
        projectRoot={effectiveProjectRoot}
      />
      {showLiveSuffix && status === "running" && (
        <span className="ml-2 text-xs font-normal text-accent-success">· live</span>
      )}
    </>
  );

  if (layout === "split") {
    return (
      <div className={`flex items-center justify-between ${className}`}>
        <div className="flex items-center gap-2 flex-wrap min-w-0">
          <StatusIcon status={status} runningIcon={runningIcon} size={iconSize} />
          <h2 className="text-sm font-semibold text-gray-200">
            {title}
            {titleSuffix}
          </h2>
          {badges}
        </div>
        {nav}
      </div>
    );
  }

  if (titleTone === "muted") {
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
        {badges}
        {nav}
      </div>
    );
  }

  const inlineRow = (
    <InlineTitleRow
      status={status}
      title={title}
      titleSuffix={titleSuffix}
      processId={processId}
      badges={badges}
      runningIcon={runningIcon}
      iconSize={iconSize}
    />
  );

  return (
    <div className={`flex items-center gap-2 flex-wrap ${className}`}>
      {overlaySelect ? (
        <label className="flex items-center gap-3 py-1 px-1 rounded-lg cursor-pointer flex-1 min-w-0">
          <input
            type="checkbox"
            checked={overlaySelect.checked}
            onChange={overlaySelect.onChange}
            className="accent-accent-primary"
          />
          {inlineRow}
        </label>
      ) : (
        inlineRow
      )}
      {nav}
    </div>
  );
}

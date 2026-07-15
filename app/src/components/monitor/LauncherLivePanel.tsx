/**
 * Shared sim / data-gen / eval live/post-run panel shell (§G.9 / §G.11 / §G.12 / §G.15).
 * Optional ``logLines`` renders ``ProcessLogTail`` below children (§D.7).
 */
import type { ReactNode } from "react";
import { LiveTrainProgressBar, type LiveTrainProgressBarProps } from "./LiveTrainProgressBar";
import {
  LauncherLivePanelHeader,
  type LauncherLivePanelHeaderProps,
} from "./LauncherLivePanelHeader";
import { ProcessLogTail } from "./ProcessLogTail";

export interface LauncherLivePanelProgress extends LiveTrainProgressBarProps {
  /** When false, omit the progress bar even if other progress props are set. */
  show?: boolean;
}

export interface LauncherLivePanelProps {
  header: LauncherLivePanelHeaderProps;
  progress?: LauncherLivePanelProgress;
  /** card = standalone bordered panel; embedded = border-top section inside Process Monitor. */
  variant?: "card" | "embedded";
  cardClassName?: string;
  children?: ReactNode;
  footer?: ReactNode;
  /** Raw process stdout/stderr lines for the shared log tail (§G.9 / §G.11 / §G.15 / §D.7). */
  logLines?: string[];
  logTailMaxLines?: number;
  logTailWaiting?: boolean;
}

export function LauncherLivePanel({
  header,
  progress,
  variant = "card",
  cardClassName = "",
  children,
  footer,
  logLines,
  logTailMaxLines = 20,
  logTailWaiting = false,
}: LauncherLivePanelProps) {
  const showProgress = progress?.show !== false && progress?.processId != null;
  const panelVariant = variant;
  const headerVariant = header.variant ?? panelVariant;

  const body = (
    <>
      <LauncherLivePanelHeader {...header} variant={headerVariant} />
      {showProgress && progress && (
        <LiveTrainProgressBar
          processId={progress.processId}
          fallbackTotal={progress.fallbackTotal}
          fallbackValue={progress.fallbackValue}
          className={progress.className}
        />
      )}
      {children}
      {logLines !== undefined && (
        <ProcessLogTail
          logLines={logLines}
          maxLines={logTailMaxLines}
          waiting={logTailWaiting}
          variant="default"
        />
      )}
      {footer}
    </>
  );

  if (panelVariant === "embedded") {
    return (
      <div className={`space-y-3 pt-2 border-t border-canvas-border ${cardClassName}`}>
        {body}
      </div>
    );
  }

  return <div className={`card space-y-3 ${cardClassName}`.trim()}>{body}</div>;
}

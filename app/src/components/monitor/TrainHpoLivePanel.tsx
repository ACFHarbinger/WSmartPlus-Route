/**
 * Shared train/HPO live/post-run panel shell: header + progress + analytics strip (§G.10 / §G.15 / §G.17 / §G.18).
 */
import type { ReactNode } from "react";
import { LiveTrainProgressBar, type LiveTrainProgressBarProps } from "./LiveTrainProgressBar";
import {
  TrainHpoAnalyticsStrip,
  type TrainHpoAnalyticsStripProps,
} from "./TrainHpoAnalyticsStrip";
import {
  TrainHpoLivePanelHeader,
  type TrainHpoLivePanelHeaderProps,
} from "./TrainHpoLivePanelHeader";

export interface TrainHpoLivePanelProgress extends LiveTrainProgressBarProps {
  /** When false, omit the progress bar even if other progress props are set. */
  show?: boolean;
}

export interface TrainHpoLivePanelProps {
  header: TrainHpoLivePanelHeaderProps;
  analytics: TrainHpoAnalyticsStripProps;
  progress?: TrainHpoLivePanelProgress;
  /** card = standalone bordered panel; embedded = border-top section inside Process Monitor. */
  variant?: "card" | "embedded";
  cardClassName?: string;
  /** Optional trailing row (e.g. Training Hub process id). */
  footer?: ReactNode;
  /** When false, omit the analytics strip (Training Hub non-train/HPO modes). */
  showAnalytics?: boolean;
  analyticsWrapperClassName?: string;
}

export function TrainHpoLivePanel({
  header,
  analytics,
  progress,
  variant = "card",
  cardClassName = "",
  footer,
  showAnalytics = true,
  analyticsWrapperClassName,
}: TrainHpoLivePanelProps) {
  const showProgress = progress?.show !== false && progress?.processId != null;

  const body = (
    <>
      <TrainHpoLivePanelHeader {...header} />
      {showProgress && progress && (
        <LiveTrainProgressBar
          processId={progress.processId}
          fallbackTotal={progress.fallbackTotal}
          fallbackValue={progress.fallbackValue}
          className={progress.className}
        />
      )}
      {showAnalytics && (
        <div className={analyticsWrapperClassName}>
          <TrainHpoAnalyticsStrip {...analytics} />
        </div>
      )}
      {footer}
    </>
  );

  if (variant === "embedded") {
    return (
      <div className={`space-y-3 pt-2 border-t border-canvas-border ${cardClassName}`}>
        {body}
      </div>
    );
  }

  return <div className={`card space-y-3 ${cardClassName}`.trim()}>{body}</div>;
}

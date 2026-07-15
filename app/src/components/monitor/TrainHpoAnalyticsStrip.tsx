/**
 * Shared train/HPO analytics strip: metric snapshot, sparklines, health, attention (§G.10 / §G.15 / §G.17 / §G.18).
 */
import type { ReactNode } from "react";
import { Activity } from "lucide-react";
import { RuntimeAttentionPanel } from "../analysis/RuntimeAttentionPanel";
import { TrainingHealthPanel } from "../analysis/TrainingHealthPanel";
import {
  GradNormSparkline,
  LrSparkline,
  TrainingMetricSnapshot,
} from "./TrainingMetricSparklines";
import { postRunTrainingRehydrationMessage } from "../../utils/trainingMetrics";
import type {
  AttentionVizEntry,
  TrainingHealthEntry,
  TrainingMetricsRow,
} from "../../types";

export interface TrainHpoAnalyticsStripProps {
  metrics: TrainingMetricsRow[];
  healthEntries?: TrainingHealthEntry[];
  attentionEntries?: AttentionVizEntry[];
  logScale: boolean;
  theme?: "dark" | "light";
  exportNamePrefix: string;
  /** When true, render the post-run rehydration banner above the snapshot. */
  isPostRun?: boolean;
  postRunFallback?: string;
  /** Include health + attention panels (default true). */
  showHealthAttention?: boolean;
  /** Optional content between snapshot and sparklines (e.g. Training Hub live chart). */
  middleContent?: ReactNode;
}

export function TrainHpoAnalyticsStrip({
  metrics,
  healthEntries = [],
  attentionEntries = [],
  logScale,
  theme = "dark",
  exportNamePrefix,
  isPostRun = false,
  postRunFallback = "Post-run shortcuts — open Training Monitor or Output Browser for this run",
  showHealthAttention = true,
  middleContent,
}: TrainHpoAnalyticsStripProps) {
  const latestMetric = metrics[metrics.length - 1];

  return (
    <div className="space-y-3">
      {isPostRun && (
        <div className="flex items-center gap-2 text-xs text-canvas-muted">
          <Activity size={12} />
          {postRunTrainingRehydrationMessage({
            metricCount: metrics.length,
            healthCount: healthEntries.length,
            attentionCount: attentionEntries.length,
            fallback: postRunFallback,
          })}
        </div>
      )}

      {latestMetric && <TrainingMetricSnapshot metric={latestMetric} />}

      {middleContent}

      {metrics.length >= 2 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <GradNormSparkline
            metrics={metrics}
            logScale={logScale}
            exportName={`${exportNamePrefix}-grad-norm`}
          />
          <LrSparkline
            metrics={metrics}
            logScale={logScale}
            exportName={`${exportNamePrefix}-lr`}
          />
        </div>
      )}

      {showHealthAttention && (
        <>
          <TrainingHealthPanel entries={healthEntries} />
          <RuntimeAttentionPanel
            entries={attentionEntries}
            theme={theme}
            logScale={logScale}
          />
        </>
      )}
    </div>
  );
}

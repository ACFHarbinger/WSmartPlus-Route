/**
 * Shared grad-norm / LR sparklines for train/HPO live and post-run panels (§G.15 / §G.17 / §G.18).
 */
import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import type { TrainingMetricsRow } from "../../../types";

function MetricSparkline({
  label,
  data,
  color,
  exportName,
  logScale = false,
}: {
  label: string;
  data: [number, number][];
  color: string;
  exportName?: string;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  if (data.length === 0) return null;
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <p className="text-xs text-canvas-muted">{label}</p>
        {exportName && (
          <ChartExportButtons
            chartRef={{ current: chartRef.current }}
            filenameStem={exportName}
            size={10}
          />
        )}
      </div>
      <ReactECharts
        ref={chartRef}
        option={{
          backgroundColor: "transparent",
          grid: { left: 40, right: 10, top: 8, bottom: 28 },
          xAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 9 } },
          yAxis: {
            type: (logScale ? "log" : "value") as "log" | "value",
            logBase: 10,
            axisLabel: { color: "#9090b0", fontSize: 9 },
            minorSplitLine: { show: false },
          },
          series: [{
            type: "line",
            data: logScale ? data.map(([x, y]) => [x, Math.max(y, 1e-8)]) : data,
            smooth: false,
            lineStyle: { color, width: 1.5 },
            areaStyle: { color: `${color}1a` },
            symbol: "none",
          }],
          tooltip: { trigger: "axis" },
        }}
        style={{ height: 80 }}
      />
    </div>
  );
}

export function GradNormSparkline({
  metrics,
  logScale = false,
  exportName = "training-grad-norm",
}: {
  metrics: TrainingMetricsRow[];
  logScale?: boolean;
  exportName?: string;
}) {
  const data = metrics
    .filter((r) => r.grad_norm != null)
    .map((r): [number, number] => [r.epoch ?? r.step ?? 0, r.grad_norm!]);
  return (
    <MetricSparkline
      label={logScale ? "Gradient Norm (log)" : "Gradient Norm"}
      data={data}
      color="#f87171"
      exportName={exportName}
      logScale={logScale}
    />
  );
}

export function LrSparkline({
  metrics,
  logScale = false,
  exportName = "training-lr",
}: {
  metrics: TrainingMetricsRow[];
  logScale?: boolean;
  exportName?: string;
}) {
  const data = metrics
    .filter((r) => r.lr != null)
    .map((r): [number, number] => [r.step ?? r.epoch ?? 0, r.lr!]);
  return (
    <MetricSparkline
      label={logScale ? "Learning Rate (log)" : "Learning Rate"}
      data={data}
      color="#fbbf24"
      exportName={exportName}
      logScale={logScale}
    />
  );
}

export function TrainingMetricSnapshot({
  metric,
}: {
  metric: TrainingMetricsRow;
}) {
  return (
    <div className="flex flex-wrap gap-4 text-xs">
      {metric.epoch != null && (
        <div>
          <span className="text-canvas-muted">Epoch </span>
          <span className="font-mono text-gray-200">{metric.epoch}</span>
        </div>
      )}
      {metric.train_loss != null && (
        <div>
          <span className="text-canvas-muted">Train loss </span>
          <span className="font-mono text-gray-200">{metric.train_loss.toFixed(4)}</span>
        </div>
      )}
      {metric.val_loss != null && (
        <div>
          <span className="text-canvas-muted">Val loss </span>
          <span className="font-mono text-gray-200">{metric.val_loss.toFixed(4)}</span>
        </div>
      )}
      {metric.reward != null && (
        <div>
          <span className="text-canvas-muted">Reward </span>
          <span className="font-mono text-accent-success">{metric.reward.toFixed(4)}</span>
        </div>
      )}
      {metric.grad_norm != null && (
        <div>
          <span className="text-canvas-muted">‖∇‖ </span>
          <span className="font-mono text-gray-200">{metric.grad_norm.toFixed(3)}</span>
        </div>
      )}
    </div>
  );
}

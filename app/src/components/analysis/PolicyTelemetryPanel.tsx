/**
 * Policy iteration telemetry dashboard (§A.3).
 *
 * Renders ECharts panels from ``PolicyVizMixin`` data streamed via
 * ``POLICY_VIZ_START:`` log markers.
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Activity } from "lucide-react";
import { ChartExportButtons } from "../common/ChartExportButtons";
import {
  filterPolicyVizEntries,
  policyVizChartOptions,
  policyVizTypeLabel,
} from "../../utils/policyTelemetry";
import type { PolicyVizEntry } from "../../types";

interface Props {
  entries: PolicyVizEntry[];
  policy: string | null;
  sampleId: number | null;
  day: number | null;
  theme: "dark" | "light";
  logScale?: boolean;
}

export function PolicyTelemetryPanel({
  entries,
  policy,
  sampleId,
  day,
  theme,
  logScale = false,
}: Props) {
  const chartRefs = useRef<Array<EChartsReact | null>>([]);

  const active = useMemo(
    () => filterPolicyVizEntries(entries, policy, sampleId, day),
    [entries, policy, sampleId, day]
  );

  const latest = active.length ? active[active.length - 1]! : null;
  const charts = useMemo(
    () => (latest ? policyVizChartOptions(latest, theme, logScale) : []),
    [latest, theme, logScale]
  );

  if (!latest || charts.length === 0) {
    return (
      <div className="card p-4 text-xs text-canvas-muted">
        <p className="flex items-center gap-2 font-medium text-gray-400">
          <Activity size={14} />
          Policy Telemetry
        </p>
        <p className="mt-2">
          No iteration telemetry for the selected policy/day. Run a simulation with an
          ALNS, HGS, ACO, or other ``PolicyVizMixin`` solver to populate this panel.
        </p>
      </div>
    );
  }

  const metricCount = Object.keys(latest.data).length;
  const iterCount = Math.max(...Object.values(latest.data).map((v) => v.length), 0);

  return (
    <div className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="flex items-center gap-2 text-sm font-semibold text-gray-200">
            <Activity size={14} className="text-accent-primary" />
            Policy Telemetry
            <span className="text-xs font-normal text-canvas-muted">
              {policyVizTypeLabel(latest.policy_type)} · day {latest.day} · {iterCount} steps ·{" "}
              {metricCount} metrics
            </span>
          </p>
          <p className="text-[10px] text-canvas-muted mt-0.5 truncate max-w-xl">{latest.policy}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {charts.map((option, idx) => (
          <div key={idx} className="relative rounded-lg border border-canvas-border bg-canvas-elevated/40 p-2">
            <div className="absolute right-2 top-2 z-10">
              <ChartExportButtons
                chartRef={{ current: chartRefs.current[idx] ?? null }}
                filenameStem={`policy-viz-${latest.policy_type}-${idx}`}
              />
            </div>
            <ReactECharts
              ref={(el) => {
                chartRefs.current[idx] = el;
              }}
              option={option}
              style={{ height: 220, width: "100%" }}
              notMerge
              lazyUpdate
            />
          </div>
        ))}
      </div>
    </div>
  );
}

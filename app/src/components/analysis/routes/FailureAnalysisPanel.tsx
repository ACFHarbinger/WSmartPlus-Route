/**
 * Simulation failure root-cause dashboard (§A.6).
 *
 * Renders structured summaries from ``FailureAnalyzer`` streamed via
 * ``SIM_FAILURE_START:`` log markers or embedded in day log payloads.
 */
import { useMemo } from "react";
import { AlertOctagon, Search } from "lucide-react";
import {
  simFailureCauseLabel,
  simFailureSeverityColor,
} from "../../../utils/sim/simFailure";
import type { SimFailureEntry, SimFailureSummary } from "../../../types";

interface Props {
  entry: SimFailureEntry | null;
  embedded?: SimFailureSummary | null;
}

export function FailureAnalysisPanel({ entry, embedded }: Props) {
  const summary = entry?.data ?? embedded ?? null;

  const causes = useMemo(() => summary?.root_causes ?? [], [summary]);

  if (!summary?.has_failure) {
    return (
      <div className="card p-4 text-xs text-canvas-muted">
        <p className="flex items-center gap-2 font-medium text-gray-400">
          <Search size={14} />
          Failure Analysis
        </p>
        <p className="mt-2">
          No failure signals for this day. The analyzer flags overflows, waste
          loss, negative profit, fill-rate spikes, and skipped high-fill bins.
        </p>
      </div>
    );
  }

  const color = simFailureSeverityColor(summary.severity);

  return (
    <div className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-2 text-sm font-semibold text-gray-200">
          <AlertOctagon size={14} style={{ color }} />
          Failure Analysis
          <span className="text-xs font-normal uppercase tracking-wide" style={{ color }}>
            {summary.severity}
          </span>
        </p>
        <div className="flex flex-wrap gap-2">
          {causes.map((code) => (
            <span
              key={code}
              className="text-[10px] px-2 py-0.5 rounded-full border border-canvas-border text-canvas-muted"
            >
              {simFailureCauseLabel(code)}
            </span>
          ))}
        </div>
      </div>

      <p className="text-xs text-canvas-muted">{summary.summary}</p>

      {summary.metrics && (
        <div className="flex flex-wrap gap-4 text-[10px] font-mono text-canvas-muted">
          <span>overflows: {summary.metrics.new_overflows}</span>
          <span>kg lost: {summary.metrics.kg_lost?.toFixed?.(1) ?? summary.metrics.kg_lost}</span>
          <span>profit: {summary.metrics.profit?.toFixed?.(2) ?? summary.metrics.profit} €</span>
        </div>
      )}

      {summary.overflow_bins && summary.overflow_bins.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-canvas-muted">Overflow bins</p>
          <div className="max-h-40 overflow-y-auto rounded-lg border border-canvas-border text-[10px]">
            <table className="w-full">
              <thead className="text-canvas-muted border-b border-canvas-border">
                <tr>
                  <th className="px-2 py-1 text-left">Bin</th>
                  <th className="px-2 py-1 text-right">Pred</th>
                  <th className="px-2 py-1 text-right">Actual</th>
                  <th className="px-2 py-1 text-center">Tour</th>
                  <th className="px-2 py-1 text-center">Spike</th>
                </tr>
              </thead>
              <tbody>
                {summary.overflow_bins.map((bin) => (
                  <tr key={bin.bin_index} className="border-b border-canvas-border/30">
                    <td className="px-2 py-1 font-mono">{bin.bin_id}</td>
                    <td className="px-2 py-1 text-right">{bin.predicted_fill.toFixed(1)}</td>
                    <td className="px-2 py-1 text-right">{bin.actual_fill.toFixed(1)}</td>
                    <td className="px-2 py-1 text-center">{bin.in_tour ? "✓" : "—"}</td>
                    <td className="px-2 py-1 text-center">{bin.fill_spike ? "!" : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {summary.skipped_high_fill_bins && summary.skipped_high_fill_bins.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-canvas-muted">Skipped high-fill bins</p>
          <div className="flex flex-wrap gap-2">
            {summary.skipped_high_fill_bins.map((bin) => (
              <span
                key={bin.bin_index}
                className="text-[10px] px-2 py-0.5 rounded border border-canvas-border font-mono"
              >
                #{bin.bin_id} · {bin.fill_level.toFixed(0)}%
                {bin.mandatory ? " · mandatory" : ""}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

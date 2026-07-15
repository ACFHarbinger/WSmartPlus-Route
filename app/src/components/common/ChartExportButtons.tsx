import { Download } from "lucide-react";
import type { RefObject } from "react";
import type ReactECharts from "echarts-for-react";
import { exportChartPngWithToast, exportChartSvgWithToast } from "../../utils/chartExport";

export type ChartExportRef =
  | RefObject<ReactECharts | null>
  | { current: ReactECharts | null };

interface ChartExportButtonsProps {
  chartRef: ChartExportRef;
  filenameStem: string;
  size?: number;
  className?: string;
}

/** Paired PNG + SVG export buttons with Sonner toast feedback (§G.7). */
export function ChartExportButtons({
  chartRef,
  filenameStem,
  size = 12,
  className = "",
}: ChartExportButtonsProps) {
  const ref = chartRef as RefObject<ReactECharts | null>;

  return (
    <div className={`flex items-center gap-1 ${className}`}>
      <button
        type="button"
        onClick={() => exportChartPngWithToast(ref, `${filenameStem}.png`)}
        className="btn-ghost text-xs flex items-center gap-1"
        title={`Export ${filenameStem} as PNG`}
      >
        <Download size={size} />
        PNG
      </button>
      <button
        type="button"
        onClick={() => exportChartSvgWithToast(ref, `${filenameStem}.svg`)}
        className="btn-ghost text-xs flex items-center gap-1"
        title={`Export ${filenameStem} as SVG`}
      >
        <Download size={size} />
        SVG
      </button>
    </div>
  );
}

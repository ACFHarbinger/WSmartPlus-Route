import { Download } from "lucide-react";
import {
  exportCanvasPngWithToast,
  exportContainerCanvasPngWithToast,
} from "../../utils/charts/chartExport";

interface CanvasExportButtonProps {
  filename: string;
  label?: string;
  size?: number;
  className?: string;
  canvas?: () => HTMLCanvasElement | null | undefined;
  container?: () => HTMLElement | null | undefined;
}

/** WebGL/canvas PNG export button with Sonner toast feedback (§G.7). */
export function CanvasExportButton({
  filename,
  label = "PNG",
  size = 12,
  className = "",
  canvas,
  container,
}: CanvasExportButtonProps) {
  const handleExport = () => {
    if (canvas) {
      exportCanvasPngWithToast(canvas(), filename);
      return;
    }
    if (container) {
      exportContainerCanvasPngWithToast(container(), filename);
    }
  };

  return (
    <button
      type="button"
      onClick={handleExport}
      className={`btn-ghost text-xs flex items-center gap-1 ${className}`}
      title={`Export ${filename}`}
    >
      <Download size={size} />
      {label}
    </button>
  );
}

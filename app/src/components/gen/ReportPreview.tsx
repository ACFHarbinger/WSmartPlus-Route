/**
 * Report preview panel (§H.7) — scrolling in-app preview of a generated
 * analysis report, rendered through the same HTML builder as the §H.5
 * standalone export (figures inlined), inside a sandboxed iframe so the
 * report's document styling stays isolated from the app.
 */
import { useCallback, useState } from "react";
import { Eye, RefreshCw, X } from "lucide-react";
import { buildHtmlReportPage } from "../../gen/report/htmlReport";

export function ReportPreview({
  projectRoot,
  mdRel,
  onLog,
}: {
  projectRoot: string;
  /** Project-relative path of the generated markdown report. */
  mdRel: string;
  onLog?: (msg: string) => void;
}) {
  const [page, setPage] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);
  const [open, setOpen] = useState(false);

  const build = useCallback(async () => {
    setBuilding(true);
    setOpen(true);
    try {
      setPage(await buildHtmlReportPage(projectRoot, mdRel, onLog));
    } catch (err) {
      onLog?.(`Preview failed: ${err instanceof Error ? err.message : String(err)}`);
      setPage(null);
    } finally {
      setBuilding(false);
    }
  }, [projectRoot, mdRel, onLog]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <button
          onClick={open ? () => setOpen(false) : build}
          disabled={building}
          className="btn-ghost text-xs py-1 px-2 flex items-center gap-1"
        >
          {open ? <X size={12} /> : <Eye size={12} />}
          {building ? "Rendering…" : open ? "Close preview" : "Preview report"}
        </button>
        {open && page && (
          <button onClick={build} disabled={building} className="btn-ghost text-xs py-1 px-2 flex items-center gap-1">
            <RefreshCw size={12} className={building ? "animate-spin" : ""} />
            Refresh
          </button>
        )}
      </div>
      {open && (
        <div className="rounded-lg overflow-hidden border border-canvas-border bg-white">
          {page ? (
            <iframe
              title={`Report preview — ${mdRel}`}
              srcDoc={page}
              sandbox=""
              className="w-full"
              style={{ height: "70vh", border: "none" }}
            />
          ) : (
            <div className="h-24 flex items-center justify-center text-xs text-canvas-muted bg-canvas-bg">
              {building ? "Rendering report…" : "Preview unavailable."}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

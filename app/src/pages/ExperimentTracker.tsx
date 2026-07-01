/**
 * Experiment Tracker — run history browsing.
 * Ports Streamlit `experiment_tracker` mode.
 * Full MLflow / ZenML embedding planned in §G.18.
 */
import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { RefreshCw } from "lucide-react";
import { useAppStore } from "../store/app";

interface OutputDir {
  name: string;
  path: string;
  created_at: string;
  size_bytes: number;
}

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

export function ExperimentTracker() {
  const { projectRoot } = useAppStore();
  const [dirs, setDirs] = useState<OutputDir[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    try {
      const found = await invoke<OutputDir[]>("list_output_dirs", {
        outputPath: `${projectRoot}/assets/output`,
      });
      setDirs(found.sort((a, b) => b.created_at.localeCompare(a.created_at)));
    } finally {
      setLoading(false);
    }
  }, [projectRoot]);

  useEffect(() => {
    if (projectRoot) refresh();
  }, [projectRoot, refresh]);

  if (!projectRoot) {
    return (
      <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
        Set project root to browse experiment outputs.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={refresh}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
        <span className="text-xs text-canvas-muted">{dirs.length} run(s)</span>
      </div>

      <div className="card overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-canvas-border">
              <th className="text-left py-2 px-3 text-canvas-muted font-medium">Run Name</th>
              <th className="text-left py-2 px-3 text-canvas-muted font-medium">Created</th>
              <th className="text-right py-2 px-3 text-canvas-muted font-medium">Size</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-canvas-border">
            {dirs.map((d) => (
              <tr key={d.path} className="hover:bg-canvas-hover">
                <td className="py-2 px-3 font-mono text-gray-300">{d.name}</td>
                <td className="py-2 px-3 text-canvas-muted">
                  {new Date(d.created_at).toLocaleString()}
                </td>
                <td className="py-2 px-3 text-right text-canvas-muted">{formatBytes(d.size_bytes)}</td>
              </tr>
            ))}
            {dirs.length === 0 && !loading && (
              <tr>
                <td colSpan={3} className="py-8 text-center text-canvas-muted">
                  No experiment outputs found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-canvas-muted">
        Full MLflow / ZenML dashboard embedding planned in Phase 18.
      </p>
    </div>
  );
}

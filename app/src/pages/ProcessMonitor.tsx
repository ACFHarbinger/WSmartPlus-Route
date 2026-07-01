/**
 * Process Monitor — live view of all spawned Python processes.
 * Ports Streamlit `process_monitor` and Studio §G.15.
 * Registered processes come from the Rust PROCESS_REGISTRY via useProcessMonitor.
 */
import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Square, ChevronDown, ChevronUp } from "lucide-react";
import { useProcessStore } from "../store/process";
import { StatusPill } from "../components/ui/StatusPill";

export function ProcessMonitor() {
  const { processes } = useProcessStore();
  const [expanded, setExpanded] = useState<string | null>(null);

  const cancel = useCallback(async (id: string) => {
    await invoke("cancel_process", { id });
  }, []);

  const ids = Object.keys(processes).sort(
    (a, b) => (processes[b].startTime ?? 0) - (processes[a].startTime ?? 0)
  );

  if (ids.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-canvas-muted text-sm">
        No processes running. Launch a simulation or training run to see it here.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {ids.map((id) => {
        const proc = processes[id];
        const isExpanded = expanded === id;
        return (
          <div key={id} className="card">
            <div className="flex items-center gap-3">
              <StatusPill status={proc.status} />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-gray-200 font-mono truncate">{id}</p>
                <p className="text-xs text-canvas-muted truncate">{proc.command}</p>
              </div>
              {proc.status === "running" && (
                <button
                  onClick={() => cancel(id)}
                  className="btn-ghost text-accent-danger flex items-center gap-1 text-xs"
                >
                  <Square size={12} />
                  Cancel
                </button>
              )}
              <button
                className="btn-ghost text-canvas-muted p-1"
                onClick={() => setExpanded(isExpanded ? null : id)}
              >
                {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
            </div>

            {isExpanded && (
              <div className="mt-3 bg-canvas-bg rounded-lg p-3 font-mono text-xs text-green-400 h-48 overflow-y-auto">
                {proc.logLines.slice(-50).map((line, i) => (
                  <div key={i} className="log-line">
                    {line}
                  </div>
                ))}
                {proc.logLines.length === 0 && (
                  <span className="text-canvas-muted">No output yet…</span>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

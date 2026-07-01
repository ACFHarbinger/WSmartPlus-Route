/**
 * Simulation Launcher — configure and start a simulation run (§G.9).
 * Ports the PySide6 simulation launch tab.
 * Uses the Rust `spawn_python_process` command so stdout streams to ProcessMonitor.
 */
import { useCallback, useState } from "react";
import { Play } from "lucide-react";
import { useAppStore } from "../store/app";
import { useSpawnProcess } from "../hooks/useSpawnProcess";

const DEFAULT_OVERRIDES = `seed=42\ntask=test_sim\n`;

export function SimulationLauncher() {
  const { projectRoot } = useAppStore();
  const [overrides, setOverrides] = useState(DEFAULT_OVERRIDES);
  const { spawn, launching } = useSpawnProcess();

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const overrideArgs = overrides.split("\n").map((l) => l.trim()).filter(Boolean);
    await spawn({
      id: `sim_${Date.now()}`,
      pythonArgs: ["main.py", "test_sim", ...overrideArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, overrides, spawn]);

  return (
    <div className="space-y-4 max-w-2xl">
      <div className="card space-y-4">
        <div>
          <label className="block text-xs text-canvas-muted mb-1.5">
            Hydra Overrides (one per line)
          </label>
          <textarea
            className="input-base w-full font-mono text-xs h-32 resize-y"
            value={overrides}
            onChange={(e) => setOverrides(e.target.value)}
            spellCheck={false}
            placeholder="seed=42&#10;task=test_sim"
          />
        </div>

        {!projectRoot && (
          <p className="text-xs text-accent-warning">
            Project root is not set. Configure it in settings before launching.
          </p>
        )}

        <button
          onClick={launch}
          disabled={launching || !projectRoot}
          className="btn-primary flex items-center gap-2 w-fit"
        >
          <Play size={14} />
          {launching ? "Launching…" : "Launch Simulation"}
        </button>
      </div>

      <p className="text-xs text-canvas-muted">
        Process output will appear in the Process Monitor. The Simulation Monitor will
        auto-update as new days are logged.
      </p>
    </div>
  );
}

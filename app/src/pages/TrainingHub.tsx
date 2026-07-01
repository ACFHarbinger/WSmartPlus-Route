/**
 * Training Hub — configure and start training / HPO runs (§G.10).
 * Ports the PySide6 RL training tab.
 */
import { useCallback, useState } from "react";
import { Play } from "lucide-react";
import { useAppStore } from "../store/app";
import { useSpawnProcess } from "../hooks/useSpawnProcess";

const MODES = ["train", "hpo", "eval"] as const;
type Mode = (typeof MODES)[number];

const DEFAULT_ARGS: Record<Mode, string> = {
  train: "seed=42\ntracker.enabled=false\n",
  hpo: "seed=42\nhpo.n_trials=50\n",
  eval: "seed=42\n",
};

export function TrainingHub() {
  const { projectRoot } = useAppStore();
  const [mode, setMode] = useState<Mode>("train");
  const [overrides, setOverrides] = useState(DEFAULT_ARGS.train);
  const { spawn, launching } = useSpawnProcess();

  const onModeChange = (m: Mode) => {
    setMode(m);
    setOverrides(DEFAULT_ARGS[m]);
  };

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const overrideArgs = overrides.split("\n").map((l) => l.trim()).filter(Boolean);
    await spawn({
      id: `${mode}_${Date.now()}`,
      pythonArgs: ["main.py", mode, ...overrideArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, mode, overrides, spawn]);

  return (
    <div className="space-y-4 max-w-2xl">
      <div className="card space-y-4">
        {/* Mode selector */}
        <div>
          <label className="block text-xs text-canvas-muted mb-1.5">Mode</label>
          <div className="flex gap-2">
            {MODES.map((m) => (
              <button
                key={m}
                onClick={() => onModeChange(m)}
                className={
                  mode === m
                    ? "btn-primary text-xs py-1 px-3"
                    : "btn-ghost text-xs py-1 px-3"
                }
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-xs text-canvas-muted mb-1.5">
            Hydra Overrides (one per line)
          </label>
          <textarea
            className="input-base w-full font-mono text-xs h-32 resize-y"
            value={overrides}
            onChange={(e) => setOverrides(e.target.value)}
            spellCheck={false}
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
          {launching ? "Launching…" : `Start ${mode}`}
        </button>
      </div>

      <p className="text-xs text-canvas-muted">
        Process output streams to the Process Monitor. Training metrics appear in the
        Training Monitor as Lightning CSV files are written.
      </p>
    </div>
  );
}

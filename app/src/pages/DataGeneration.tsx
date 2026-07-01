/**
 * Data Generation Wizard — run dataset generation scripts (§G.11).
 * Ports the PySide6 data generation tab.
 */
import { useCallback, useState } from "react";
import { Play } from "lucide-react";
import { useAppStore } from "../store/app";
import { useSpawnProcess } from "../hooks/useSpawnProcess";

const SCRIPTS = [
  { id: "gen_dataset", label: "Generate Dataset", args: ["scripts/generate_dataset.py"] },
  { id: "gen_bins", label: "Generate Bins", args: ["scripts/generate_bins.py"] },
  { id: "gen_routes", label: "Generate Routes", args: ["scripts/generate_routes.py"] },
] as const;

export function DataGeneration() {
  const { projectRoot } = useAppStore();
  const [selected, setSelected] = useState<string>(SCRIPTS[0].id);
  const [extraArgs, setExtraArgs] = useState("");
  const { spawn, launching } = useSpawnProcess();

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const script = SCRIPTS.find((s) => s.id === selected);
    if (!script) return;
    const extra = extraArgs.split("\n").map((l) => l.trim()).filter(Boolean);
    await spawn({
      id: `${selected}_${Date.now()}`,
      pythonArgs: [...script.args, ...extra],
      workingDir: projectRoot,
    });
  }, [projectRoot, selected, extraArgs, spawn]);

  return (
    <div className="space-y-4 max-w-2xl">
      <div className="card space-y-4">
        <div>
          <label className="block text-xs text-canvas-muted mb-1.5">Script</label>
          <div className="flex flex-col gap-1">
            {SCRIPTS.map((s) => (
              <label
                key={s.id}
                className="flex items-center gap-3 py-1.5 px-2 rounded-lg hover:bg-canvas-hover cursor-pointer"
              >
                <input
                  type="radio"
                  name="script"
                  value={s.id}
                  checked={selected === s.id}
                  onChange={() => setSelected(s.id)}
                  className="accent-accent-primary"
                />
                <span className="text-sm text-gray-300">{s.label}</span>
                <code className="ml-auto text-xs text-canvas-muted font-mono">
                  {s.args.join(" ")}
                </code>
              </label>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-xs text-canvas-muted mb-1.5">
            Extra CLI Arguments (one per line)
          </label>
          <textarea
            className="input-base w-full font-mono text-xs h-24 resize-y"
            value={extraArgs}
            onChange={(e) => setExtraArgs(e.target.value)}
            spellCheck={false}
            placeholder="--city figueira&#10;--n-bins 350"
          />
        </div>

        {!projectRoot && (
          <p className="text-xs text-accent-warning">
            Project root is not set. Configure it in settings before running.
          </p>
        )}

        <button
          onClick={launch}
          disabled={launching || !projectRoot}
          className="btn-primary flex items-center gap-2 w-fit"
        >
          <Play size={14} />
          {launching ? "Running…" : "Run Script"}
        </button>
      </div>

      <p className="text-xs text-canvas-muted">
        Script output streams to the Process Monitor. The Data Explorer can be used to inspect
        generated CSV files once the script completes.
      </p>
    </div>
  );
}

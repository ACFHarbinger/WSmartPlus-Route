/**
 * Data Generation Wizard — full Hydra form for `main.py gen_data` (§G.11).
 *
 * Mirrors the controller justfile gen-data recipe and gen_data.yaml config.
 * Key Hydra args: data.problem, data.data_distributions, data.dataset_type,
 * data.seed, plus per-graph overrides via "Advanced" panel.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Play, ChevronDown, ChevronUp, Terminal, Activity, CheckCircle, XCircle } from "lucide-react";
import { listen } from "@tauri-apps/api/event";
import { useAppStore } from "../../store/app";
import { useSpawnProcess } from "../../hooks/useSpawnProcess";
import type { StdoutLine, StatusUpdate, ProcessStatus } from "../../types";

const PROBLEMS = ["vrpp", "wcvrp", "scwcvrp", "all"] as const;
const DISTRIBUTIONS = ["gamma3", "emp"] as const;
const DATASET_TYPES = [
  { value: "test_simulator", label: "Test Simulator" },
  { value: "train", label: "Training" },
  { value: "train_time", label: "Training (timed)" },
] as const;
const AREAS = ["figueiradafoz", "riomaior"] as const;

const DIST_LABELS: Record<string, string> = {
  gamma3: "Gamma-3",
  emp: "Empirical",
};

export function DataGeneration() {
  const { projectRoot, setMode } = useAppStore();
  const { spawn, launching } = useSpawnProcess();

  // Core params
  const [problem, setProblem] = useState<string>("vrpp");
  const [distributions, setDistributions] = useState<string[]>(["gamma3"]);
  const [datasetType, setDatasetType] = useState("test_simulator");
  const [seed, setSeed] = useState(42);
  const [overwrite, setOverwrite] = useState(true);

  // Graph params
  const [area, setArea] = useState("figueiradafoz");
  const [numLoc, setNumLoc] = useState(350);
  const [nSamples, setNSamples] = useState(1);
  const [nDays, setNDays] = useState(30);

  // Advanced
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [extraOverrides, setExtraOverrides] = useState("");

  // Live progress state
  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<ProcessStatus | null>(null);
  const [logTail, setLogTail] = useState<string[]>([]);

  useEffect(() => {
    if (!liveProcessId) return;
    let unlistenOut: (() => void) | null = null;
    let unlistenStatus: (() => void) | null = null;

    listen<StdoutLine>("process:stdout", (event) => {
      const { id, line } = event.payload;
      if (id !== liveProcessId) return;
      const text = line.startsWith("[stderr]") ? line.slice(8) : line;
      setLogTail((prev) => [...prev.slice(-19), text]);
    }).then((fn) => { unlistenOut = fn; });

    listen<StatusUpdate>("process:status", (event) => {
      if (event.payload.id === liveProcessId) setRunStatus(event.payload.status);
    }).then((fn) => { unlistenStatus = fn; });

    return () => { unlistenOut?.(); unlistenStatus?.(); };
  }, [liveProcessId]);

  const toggleDist = (d: string) => {
    setDistributions((prev) =>
      prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d]
    );
  };

  const hydraArgs = useMemo(() => {
    const distList = distributions.length > 0 ? distributions.join(",") : "gamma3";
    const args = [
      `data.problem=${problem}`,
      `data.data_distributions=[${distList}]`,
      `data.dataset_type=${datasetType}`,
      `data.overwrite=${overwrite}`,
      `data.graphs.0.area=${area}`,
      `data.graphs.0.num_loc=${numLoc}`,
      `data.graphs.0.n_samples=${nSamples}`,
      `data.graphs.0.n_days=${nDays}`,
      `seed=${seed}`,
    ];
    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    return [...args, ...extra];
  }, [problem, distributions, datasetType, overwrite, area, numLoc, nSamples, nDays, seed, extraOverrides]);

  const commandPreview = `python main.py gen_data \\\n  ${hydraArgs.join(" \\\n  ")}`;

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const procId = `gen_data_${Date.now()}`;
    setLiveProcessId(procId);
    setRunStatus(null);
    setLogTail([]);
    await spawn({
      id: procId,
      pythonArgs: ["main.py", "gen_data", ...hydraArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, hydraArgs, spawn]);

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Problem + distribution */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Dataset</h2>

        <div className="flex flex-wrap gap-6">
          {/* Problem */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Problem</label>
            <select
              className="select-base w-36"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
            >
              {PROBLEMS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>

          {/* Dataset type */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Dataset Type</label>
            <select
              className="select-base w-44"
              value={datasetType}
              onChange={(e) => setDatasetType(e.target.value)}
            >
              {DATASET_TYPES.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>

          {/* Seed */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Seed</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={seed}
              min={0}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Distributions */}
        <div>
          <label className="block text-xs text-canvas-muted mb-2">Distributions</label>
          <div className="flex gap-4">
            {DISTRIBUTIONS.map((d) => (
              <label key={d} className="flex items-center gap-2 cursor-pointer text-sm text-gray-300">
                <input
                  type="checkbox"
                  className="accent-accent-primary"
                  checked={distributions.includes(d)}
                  onChange={() => toggleDist(d)}
                />
                {DIST_LABELS[d]}
              </label>
            ))}
          </div>
          {distributions.length === 0 && (
            <p className="text-xs text-accent-warning mt-1">Select at least one distribution.</p>
          )}
        </div>

        {/* Overwrite toggle */}
        <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-300 w-fit">
          <input
            type="checkbox"
            className="accent-accent-primary"
            checked={overwrite}
            onChange={(e) => setOverwrite(e.target.checked)}
          />
          Overwrite existing files
        </label>
      </div>

      {/* Graph configuration */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Graph</h2>
        <div className="flex flex-wrap gap-4">
          {/* Area */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Area</label>
            <select
              className="select-base w-44"
              value={area}
              onChange={(e) => setArea(e.target.value)}
            >
              {AREAS.map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </div>

          {/* num_loc */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Locations</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={numLoc}
              min={1}
              onChange={(e) => setNumLoc(Number(e.target.value))}
            />
          </div>

          {/* n_samples */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Samples</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={nSamples}
              min={1}
              onChange={(e) => setNSamples(Number(e.target.value))}
            />
          </div>

          {/* n_days */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Days</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={nDays}
              min={1}
              onChange={(e) => setNDays(Number(e.target.value))}
            />
          </div>
        </div>

        <p className="text-xs text-canvas-muted">
          Configures <code>data.graphs[0]</code>. For multi-graph generation use Advanced Overrides.
        </p>
      </div>

      {/* Advanced overrides */}
      <div className="card">
        <button
          className="w-full flex items-center justify-between text-sm font-medium text-gray-300"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          <span>Extra Hydra Overrides</span>
          {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {showAdvanced && (
          <textarea
            className="input-base w-full font-mono text-xs h-20 resize-y mt-3"
            value={extraOverrides}
            onChange={(e) => setExtraOverrides(e.target.value)}
            placeholder="data.penalty_factor=2.5"
            spellCheck={false}
          />
        )}
      </div>

      {/* Command preview */}
      <div className="card space-y-2">
        <div className="flex items-center gap-2 text-xs text-canvas-muted">
          <Terminal size={12} />
          Command preview
        </div>
        <pre className="font-mono text-xs text-accent-secondary whitespace-pre-wrap bg-canvas-bg rounded-lg p-3">
          {commandPreview}
        </pre>
      </div>

      {/* Launch */}
      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">Configure Project Root in Settings first.</p>
        )}
        <button
          onClick={launch}
          disabled={launching || !projectRoot || distributions.length === 0}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching ? "Generating…" : "Generate Dataset"}
        </button>
      </div>

      {/* Live progress panel */}
      {liveProcessId && (() => {
        const isDone = runStatus !== null && runStatus !== "running";
        return (
          <div className="card space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {isDone ? (
                  runStatus === "completed"
                    ? <CheckCircle size={14} className="text-accent-success" />
                    : <XCircle size={14} className="text-accent-danger" />
                ) : (
                  <Activity size={14} className="text-accent-primary animate-pulse" />
                )}
                <h2 className="text-sm font-semibold text-gray-200">
                  {isDone
                    ? runStatus === "completed" ? "Generation Complete" : `Generation ${runStatus}`
                    : "Generating…"}
                </h2>
              </div>
              <button
                onClick={() => setMode("process_monitor")}
                className="btn-ghost text-xs text-canvas-muted"
              >
                Process Monitor
              </button>
            </div>
            {logTail.length > 0 && (
              <div className="bg-canvas-bg rounded-lg p-2 space-y-0.5 max-h-36 overflow-auto">
                {logTail.map((line, i) => (
                  <p key={i} className="text-xs font-mono text-gray-400 leading-snug">{line}</p>
                ))}
              </div>
            )}
            {logTail.length === 0 && !isDone && (
              <p className="text-xs text-canvas-muted">Waiting for output…</p>
            )}
          </div>
        );
      })()}
    </div>
  );
}

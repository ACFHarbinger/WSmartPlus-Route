/**
 * Simulation Launcher — full-featured replacement for `just test-sim` (§G.9).
 *
 * Form parameters mirror the controller justfile recipe exactly:
 *   sim.policies=[...], sim.n_samples, sim.graph.area, sim.graph.num_loc,
 *   sim.data_distribution, sim.cpu_cores, seed
 *
 * After launch a live-status panel subscribes to `process:stdout` events and
 * parses `GUI_DAY_LOG_START:` markers — the same protocol as SimulationMonitor —
 * to show a real-time per-policy KPI snapshot while the run is executing.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Play, ChevronDown, ChevronUp, Terminal, Activity, CheckCircle, XCircle } from "lucide-react";
import { listen } from "@tauri-apps/api/event";
import { useAppStore } from "../../store/app";
import { useSpawnProcess } from "../../hooks/useSpawnProcess";
import type { DayLogEntry, StdoutLine, StatusUpdate, ProcessStatus } from "../../types";

const ALL_POLICIES = [
  "aco_hh",
  "alns",
  "bpc",
  "hgs",
  "pg_clns",
  "psoma",
  "sans",
  "swc_tcf",
] as const;

const DISTRIBUTIONS = [
  { value: "emp", label: "Empirical" },
  { value: "gamma3", label: "Gamma-3" },
];

const GUI_MARKER = "GUI_DAY_LOG_START:";

function NumberField({
  label,
  value,
  onChange,
  min = 1,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-canvas-muted">{label}</label>
      <input
        type="number"
        className="input-base font-mono text-sm w-32"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function fmt(n: number | undefined, decimals = 1): string {
  if (n == null || isNaN(n)) return "—";
  return n.toFixed(decimals);
}

/** Live per-policy KPI card shown while a simulation is running. */
function PolicyLiveCard({ policy, entry }: { policy: string; entry: DayLogEntry }) {
  const d = entry.data;
  return (
    <div className="rounded-lg border border-canvas-border bg-canvas-surface px-3 py-2 space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono font-semibold text-accent-secondary">{policy}</span>
        <span className="text-xs text-canvas-muted">Day {entry.day}</span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
        <span className="text-canvas-muted">Profit</span>
        <span className="font-mono text-gray-200">{fmt(d.profit)} €</span>
        <span className="text-canvas-muted">Distance</span>
        <span className="font-mono text-gray-200">{fmt(d.km)} km</span>
        <span className="text-canvas-muted">Overflows</span>
        <span className={`font-mono ${(d.overflows ?? 0) > 0 ? "text-accent-danger" : "text-accent-success"}`}>
          {d.overflows ?? "—"}
        </span>
        <span className="text-canvas-muted">Waste</span>
        <span className="font-mono text-gray-200">{fmt(d.kg)} kg</span>
      </div>
    </div>
  );
}

export function SimulationLauncher() {
  const { projectRoot, setMode } = useAppStore();
  const { spawn, launching } = useSpawnProcess();

  // Form state — mirrors justfile defaults
  const [selectedPolicies, setSelectedPolicies] = useState<string[]>([
    "aco_hh", "alns", "bpc", "hgs", "pg_clns", "psoma", "sans", "swc_tcf",
  ]);
  const [area, setArea] = useState("figueiradafoz");
  const [numLoc, setNumLoc] = useState(350);
  const [samples, setSamples] = useState(1);
  const [nCores, setNCores] = useState(4);
  const [distribution, setDistribution] = useState("emp");
  const [seed, setSeed] = useState(42);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [extraOverrides, setExtraOverrides] = useState("");

  // Live status tracking
  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const [simStatus, setSimStatus] = useState<ProcessStatus | null>(null);
  // Latest DayLogEntry per policy (keyed by `${policy}::${sample_id}`)
  const [latestByPolicy, setLatestByPolicy] = useState<Record<string, DayLogEntry>>({});

  const togglePolicy = (policy: string) =>
    setSelectedPolicies((prev) =>
      prev.includes(policy) ? prev.filter((p) => p !== policy) : [...prev, policy]
    );

  const selectAll = () => setSelectedPolicies([...ALL_POLICIES]);
  const clearAll = () => setSelectedPolicies([]);

  const hydraArgs = useMemo(() => {
    const args = [
      `sim.policies=[${selectedPolicies.join(",")}]`,
      `sim.n_samples=${samples}`,
      `sim.graph.area=${area}`,
      `sim.graph.num_loc=${numLoc}`,
      `sim.data_distribution=${distribution}`,
      `sim.cpu_cores=${nCores}`,
      `seed=${seed}`,
    ];
    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    return [...args, ...extra];
  }, [selectedPolicies, area, numLoc, samples, nCores, distribution, seed, extraOverrides]);

  const commandPreview = `python main.py test_sim \\\n  ${hydraArgs.join(" \\\n  ")}`;

  // Subscribe to live stdout/status events for the active process
  useEffect(() => {
    if (!liveProcessId) return;

    let unlistenOut: (() => void) | null = null;
    let unlistenStatus: (() => void) | null = null;

    listen<StdoutLine>("process:stdout", (event) => {
      const { id, line } = event.payload;
      if (id !== liveProcessId) return;
      const markerIdx = line.indexOf(GUI_MARKER);
      if (markerIdx === -1) return;
      const jsonStr = line.slice(markerIdx + GUI_MARKER.length).trim();
      try {
        const entry = JSON.parse(jsonStr) as DayLogEntry;
        setLatestByPolicy((prev) => ({
          ...prev,
          [`${entry.policy}::${entry.sample_id}`]: entry,
        }));
      } catch {
        // ignore malformed marker lines
      }
    }).then((fn) => {
      unlistenOut = fn;
    });

    listen<StatusUpdate>("process:status", (event) => {
      if (event.payload.id === liveProcessId) {
        setSimStatus(event.payload.status);
      }
    }).then((fn) => {
      unlistenStatus = fn;
    });

    return () => {
      unlistenOut?.();
      unlistenStatus?.();
    };
  }, [liveProcessId]);

  const launch = useCallback(async () => {
    if (!projectRoot || selectedPolicies.length === 0) return;
    const procId = `sim_${area}_${Date.now()}`;
    setLiveProcessId(procId);
    setLatestByPolicy({});
    setSimStatus(null);
    await spawn({
      id: procId,
      pythonArgs: ["main.py", "test_sim", ...hydraArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, selectedPolicies, hydraArgs, area, spawn]);

  const isRunning = simStatus === null && liveProcessId !== null && !launching;
  const isDone = simStatus === "completed" || simStatus === "failed" || simStatus === "cancelled";
  const liveEntries = Object.values(latestByPolicy);

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Policy selection */}
      <div className="card space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">Policies</h2>
          <div className="flex gap-2 text-xs">
            <button onClick={selectAll} className="btn-ghost py-0.5 px-2">All</button>
            <button onClick={clearAll} className="btn-ghost py-0.5 px-2">None</button>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-1.5">
          {ALL_POLICIES.map((p) => (
            <label
              key={p}
              className={`flex items-center gap-2 py-1.5 px-2 rounded-lg border cursor-pointer text-xs transition-colors ${
                selectedPolicies.includes(p)
                  ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                  : "border-canvas-border text-gray-400 hover:border-canvas-muted"
              }`}
            >
              <input
                type="checkbox"
                checked={selectedPolicies.includes(p)}
                onChange={() => togglePolicy(p)}
                className="sr-only"
              />
              <span
                className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                  selectedPolicies.includes(p) ? "bg-accent-primary" : "bg-canvas-border"
                }`}
              />
              {p}
            </label>
          ))}
        </div>
        {selectedPolicies.length === 0 && (
          <p className="text-xs text-accent-warning">Select at least one policy.</p>
        )}
      </div>

      {/* Dataset & simulation params */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Dataset & Parameters</h2>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-canvas-muted">Area (<code className="font-mono">sim.graph.area</code>)</label>
          <input
            type="text"
            className="input-base font-mono text-sm w-full"
            value={area}
            onChange={(e) => setArea(e.target.value)}
            placeholder="figueiradafoz"
          />
        </div>

        <div className="flex flex-wrap gap-4">
          <NumberField label="Num Locations" value={numLoc} onChange={setNumLoc} min={10} max={10000} step={50} />
          <NumberField label="Samples" value={samples} onChange={setSamples} min={1} max={100} />
          <NumberField label="CPU Cores" value={nCores} onChange={setNCores} min={1} max={64} />
          <NumberField label="Seed" value={seed} onChange={setSeed} min={0} max={99999} />
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-canvas-muted">Distribution (<code className="font-mono">sim.data_distribution</code>)</label>
          <div className="flex gap-2">
            {DISTRIBUTIONS.map((d) => (
              <label
                key={d.value}
                className={`flex items-center gap-2 py-1.5 px-3 rounded-lg border cursor-pointer text-xs transition-colors ${
                  distribution === d.value
                    ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                    : "border-canvas-border text-gray-400 hover:border-canvas-muted"
                }`}
              >
                <input
                  type="radio"
                  name="dist"
                  value={d.value}
                  checked={distribution === d.value}
                  onChange={() => setDistribution(d.value)}
                  className="sr-only"
                />
                {d.label}
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Advanced overrides */}
      <div className="card">
        <button
          className="w-full flex items-center justify-between text-sm font-medium text-gray-300"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          <span>Advanced Overrides</span>
          {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {showAdvanced && (
          <div className="mt-3 space-y-1">
            <p className="text-xs text-canvas-muted">Extra Hydra overrides, one per line.</p>
            <textarea
              className="input-base w-full font-mono text-xs h-20 resize-y mt-1"
              value={extraOverrides}
              onChange={(e) => setExtraOverrides(e.target.value)}
              placeholder="sim.something=value"
              spellCheck={false}
            />
          </div>
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
          <p className="text-xs text-accent-warning">
            Configure Project Root in Settings first.
          </p>
        )}
        <button
          onClick={launch}
          disabled={launching || !projectRoot || selectedPolicies.length === 0}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching ? "Launching…" : "Launch Simulation"}
        </button>
      </div>

      {/* Live status panel — shown once a process has been spawned */}
      {liveProcessId && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {isDone ? (
                simStatus === "completed" ? (
                  <CheckCircle size={14} className="text-accent-success" />
                ) : (
                  <XCircle size={14} className="text-accent-danger" />
                )
              ) : (
                <Activity size={14} className="text-accent-primary animate-pulse" />
              )}
              <h2 className="text-sm font-semibold text-gray-200">
                {isDone
                  ? simStatus === "completed" ? "Run Complete" : `Run ${simStatus}`
                  : "Live Status"}
              </h2>
            </div>
            <div className="flex items-center gap-2">
              {isDone && (
                <button
                  onClick={() => setMode("simulation_summary")}
                  className="btn-ghost text-xs text-accent-primary"
                >
                  View Summary →
                </button>
              )}
              <button
                onClick={() => setMode("process_monitor")}
                className="btn-ghost text-xs text-canvas-muted"
              >
                Process Monitor
              </button>
            </div>
          </div>

          {liveEntries.length === 0 ? (
            <p className="text-xs text-canvas-muted">
              {isDone ? "No day entries received." : "Waiting for first day log…"}
            </p>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {liveEntries
                .sort((a, b) => a.policy.localeCompare(b.policy))
                .map((entry) => (
                  <PolicyLiveCard
                    key={`${entry.policy}::${entry.sample_id}`}
                    policy={`${entry.policy}${liveEntries.filter(e => e.policy === entry.policy).length > 1 ? ` #${entry.sample_id}` : ""}`}
                    entry={entry}
                  />
                ))}
            </div>
          )}

          <p className="text-xs text-canvas-muted font-mono truncate">{liveProcessId}</p>
        </div>
      )}
    </div>
  );
}

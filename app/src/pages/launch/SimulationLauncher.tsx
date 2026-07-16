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
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Play, ChevronDown, ChevronUp, Terminal, RefreshCw } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { LauncherLivePanel } from "../../components/monitor/live/LauncherLivePanel";
import { ProcessIdFooter } from "../../components/monitor/process/ProcessIdFooter";
import { PolicyTelemetryPanel } from "../../components/analysis/telemetry/PolicyTelemetryPanel";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/telemetry/PolicyTelemetryTrendsPanel";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useLaunchTriggerStore } from "../../store/launchTrigger";
import { useSimLauncherStore } from "../../store/launchers";
import { useProcessStore } from "../../store/process";
import { useSpawnProcess } from "../../hooks/process/useSpawnProcess";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import {
  collectPolicyVizFromLogLines,
  uniquePolicyVizPolicies,
} from "../../utils/benchmark/policyTelemetry";
import { extractJsonlPathFromLogLines } from "../../utils/benchmark/policyTelemetryTrends";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { brushLogPathFromProcessLines, outputRunPathFromLogLines } from "../../utils/runs/outputRunPath";
import { collectLatestDayLogsByPolicy } from "../../utils/sim/dayLog";
import { findRecentLauncherProcessId, simLivePanelTitle } from "../../utils/process/launcherProcess";
import type { DayLogEntry, SimPolicyEntry, ProcessStatus } from "../../types";

/** Fallback when project root is unset or registry load fails. */
const FALLBACK_POLICIES = [
  "aco_hh", "alns", "bpc", "hgs", "pg_clns", "psoma", "sans", "swc_tcf",
] as const;

const DISTRIBUTIONS = [
  { value: "emp", label: "Empirical" },
  { value: "gamma3", label: "Gamma-3" },
];

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
function PolicyLiveCard({
  policy,
  entry,
  brushed,
  onBrush,
}: {
  policy: string;
  entry: DayLogEntry;
  brushed?: boolean;
  onBrush?: () => void;
}) {
  const d = entry.data;
  return (
    <div className="rounded-lg border border-canvas-border bg-canvas-surface px-3 py-2 space-y-1">
      <div className="flex items-center justify-between">
        <span
          className={`text-xs font-mono font-semibold ${
            brushed ? "text-accent-secondary" : "text-gray-200"
          } ${onBrush ? "cursor-pointer hover:text-accent-primary" : ""}`}
          title={onBrush ? `${policy} — click to brush policy` : undefined}
          onClick={onBrush}
        >
          {policy}
        </span>
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
  const { projectRoot, effectiveTheme: theme } = useAppStore();
  const { setMode, handoff } = useRecentHandoff();
  const {
    policy: activePolicy,
    runLabel: activeRunLabel,
    logScale,
    setPolicy,
  } = useGlobalFiltersStore();
  const { spawn, launching } = useSpawnProcess();

  // Persisted form state (§D.4 session persistence)
  const {
    selectedPolicies, area, numLoc, samples, nCores, distribution, seed, extraOverrides, patch,
  } = useSimLauncherStore();

  const setSelectedPolicies = (v: string[]) => patch({ selectedPolicies: v });
  const setArea = (v: string) => patch({ area: v });
  const setNumLoc = (v: number) => patch({ numLoc: v });
  const setSamples = (v: number) => patch({ samples: v });
  const setNCores = (v: number) => patch({ nCores: v });
  const setDistribution = (v: string) => patch({ distribution: v });
  const setSeed = (v: number) => patch({ seed: v });
  const setExtraOverrides = (v: string) => patch({ extraOverrides: v });

  // Policy registry loaded from test_sim.yaml (§G.9)
  const [availablePolicies, setAvailablePolicies] = useState<SimPolicyEntry[]>(
    FALLBACK_POLICIES.map((id) => ({ id, config_key: `policy_${id}` }))
  );
  const [policiesLoading, setPoliciesLoading] = useState(false);

  // Ephemeral UI state
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Live status tracking (local id set on launch; falls back to process store after navigation)
  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const processes = useProcessStore((s) => s.processes);
  const displayProcessId = useMemo(
    () => liveProcessId ?? findRecentLauncherProcessId(processes, "sim"),
    [liveProcessId, processes]
  );
  const displayProc = displayProcessId ? processes[displayProcessId] : null;
  const simStatus = displayProc?.status ?? null;
  // Auto-navigate countdown: counts down from 5 when run completes, navigates on 0
  const [navCountdown, setNavCountdown] = useState<number | null>(null);
  const prevSimStatusRef = useRef<ProcessStatus | null>(null);

  const togglePolicy = (policy: string) => {
    const next = selectedPolicies.includes(policy)
      ? selectedPolicies.filter((p) => p !== policy)
      : [...selectedPolicies, policy];
    setSelectedPolicies(next);
  };

  const policyIds = useMemo(() => availablePolicies.map((p) => p.id), [availablePolicies]);

  const loadPolicies = useCallback(async () => {
    if (!projectRoot) return;
    setPoliciesLoading(true);
    try {
      const entries = await invoke<SimPolicyEntry[]>("list_sim_policies", {
        projectRoot,
      });
      if (entries.length > 0) {
        setAvailablePolicies(entries);
        // Drop selections that no longer exist in the registry
        const valid = new Set(entries.map((e) => e.id));
        const filtered = selectedPolicies.filter((p) => valid.has(p));
        if (filtered.length !== selectedPolicies.length) {
          setSelectedPolicies(filtered);
        }
      }
    } catch (err) {
      console.error("Failed to load policy registry:", err);
    } finally {
      setPoliciesLoading(false);
    }
  }, [projectRoot, selectedPolicies, setSelectedPolicies]);

  useEffect(() => {
    loadPolicies();
  }, [projectRoot]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectAll = () => setSelectedPolicies([...policyIds]);
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

  // Start countdown only on running → completed transition (not when rehydrating)
  useEffect(() => {
    if (simStatus === "completed" && prevSimStatusRef.current === "running") {
      setNavCountdown(5);
    } else if (simStatus !== "completed") {
      setNavCountdown(null);
    }
    prevSimStatusRef.current = simStatus;
  }, [simStatus]);

  const liveLogLines = displayProc?.logLines ?? [];
  const simJsonlPath = useMemo(
    () => extractJsonlPathFromLogLines(liveLogLines),
    [liveLogLines]
  );

  useEffect(() => {
    if (navCountdown === null) return;
    if (navCountdown <= 0) {
      // Prefer handing off the completed ``.jsonl`` so Summary auto-loads (§G.9 / §G.1).
      if (simJsonlPath) {
        handoff(simJsonlPath, "log");
      } else {
        setMode("simulation_summary");
      }
      return;
    }
    const id = setTimeout(() => setNavCountdown((n) => (n !== null ? n - 1 : null)), 1000);
    return () => clearTimeout(id);
  }, [navCountdown, setMode, handoff, simJsonlPath]);

  const launch = useCallback(async () => {
    if (!projectRoot || selectedPolicies.length === 0) return;
    const procId = `sim_${area}_${Date.now()}`;
    setLiveProcessId(procId);
    await spawn({
      id: procId,
      pythonArgs: ["main.py", "test_sim", ...hydraArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, selectedPolicies, hydraArgs, area, spawn]);

  const simNonce = useLaunchTriggerStore((s) => s.simNonce);
  useEffect(() => {
    if (simNonce > 0) launch();
  }, [simNonce, launch]);

  const liveProcStatus = displayProc?.status ?? null;
  const latestByPolicy = useMemo(
    () => collectLatestDayLogsByPolicy(liveLogLines),
    [liveLogLines]
  );
  const isDone = simStatus === "completed" || simStatus === "failed" || simStatus === "cancelled";
  const liveEntries = Object.values(latestByPolicy);

  const policyVizEntries = useMemo(
    () => collectPolicyVizFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const vizPolicies = useMemo(
    () => uniquePolicyVizPolicies(policyVizEntries),
    [policyVizEntries]
  );
  const selectedPolicy = activePolicy ?? vizPolicies[0] ?? null;
  const liveRunLabel = useProcessRunLabelBrush(displayProcessId, liveLogLines);
  const outputRunPath = useMemo(
    () => outputRunPathFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const liveLogPath = useMemo(
    () => brushLogPathFromProcessLines(liveLogLines, "sim"),
    [liveLogLines]
  );
  const policyVizLive = liveProcStatus === "running";
  const [telemetryTrendsKey, setTelemetryTrendsKey] = useState(0);

  useEffect(() => {
    if (policyVizEntries.length > 0) {
      setTelemetryTrendsKey((k) => k + 1);
    }
  }, [policyVizEntries.length, liveLogLines.length]);

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Policy selection */}
      <div className="card space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">
            Policies
            <span className="ml-2 text-xs font-normal text-canvas-muted">
              ({availablePolicies.length} registered)
            </span>
          </h2>
          <div className="flex gap-2 text-xs items-center">
            <button
              onClick={loadPolicies}
              disabled={policiesLoading || !projectRoot}
              className="btn-ghost py-0.5 px-2 flex items-center gap-1"
              title="Reload from test_sim.yaml"
            >
              <RefreshCw size={11} className={policiesLoading ? "animate-spin" : ""} />
            </button>
            <button onClick={selectAll} className="btn-ghost py-0.5 px-2">All</button>
            <button onClick={clearAll} className="btn-ghost py-0.5 px-2">None</button>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-1.5 max-h-48 overflow-y-auto">
          {availablePolicies.map((entry) => {
            const p = entry.id;
            return (
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
            );
          })}
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
      {displayProcessId && (
        <div className="space-y-3">
          <GlobalFilterBar
            policies={vizPolicies.length > 0 ? vizPolicies : undefined}
            runLabels={liveRunLabel ? [liveRunLabel] : []}
            showLogScale
          />

        <LauncherLivePanel
          header={{
            status: isDone ? (simStatus ?? "running") : "running",
            title: simLivePanelTitle({
              isRunning: !isDone,
              status: simStatus ?? undefined,
            }),
            runLabel: liveRunLabel,
            logPath: liveLogPath,
            projectRoot,
            navMesh: {
              kind: "sim",
              hideSelf: true,
              showPostRun: isDone && simStatus === "completed",
              showOutputBrowser: isDone && simStatus === "completed",
              outputRunPath,
              simLogPath: simJsonlPath,
            },
            navTrailing:
              isDone && simStatus === "completed" && navCountdown !== null ? (
                <span className="text-xs text-canvas-muted">
                  Auto Summary in {navCountdown}s —{" "}
                  <button
                    className="underline hover:text-gray-200"
                    onClick={() => setNavCountdown(null)}
                  >
                    cancel
                  </button>
                </span>
              ) : undefined,
          }}
          progress={
            !isDone && displayProcessId
              ? { processId: displayProcessId }
              : undefined
          }
          footer={
            <ProcessIdFooter
              processId={displayProcessId}
              logPath={liveLogPath}
              runLabel={liveRunLabel}
              projectRoot={projectRoot}
            />
          }
          logLines={liveLogLines}
          logTailWaiting={!isDone}
        >
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
                    brushed={activePolicy === entry.policy}
                    onBrush={() => setPolicy(activePolicy === entry.policy ? null : entry.policy)}
                  />
                ))}
            </div>
          )}
        </LauncherLivePanel>

          {policyVizEntries.length > 0 && (
            <div className="space-y-3">
              {vizPolicies.length > 1 && (
                <div className="flex flex-wrap gap-1.5">
                  {vizPolicies.map((pol) => (
                    <button
                      key={pol}
                      onClick={() => setPolicy(activePolicy === pol ? null : pol)}
                      className={`text-[10px] font-mono px-2 py-0.5 rounded border transition-colors ${
                        selectedPolicy === pol
                          ? "border-accent-secondary text-accent-secondary bg-accent-secondary/10"
                          : "border-canvas-border text-gray-400 hover:text-gray-200"
                      }`}
                    >
                      {pol}
                    </button>
                  ))}
                </div>
              )}

              <PolicyTelemetryPanel
                entries={policyVizEntries}
                policy={selectedPolicy}
                sampleId={null}
                day={null}
                theme={theme}
                logScale={logScale}
                live={policyVizLive}
              />

              <PolicyTelemetryTrendsPanel
                theme={theme}
                logScale={logScale}
                refreshKey={telemetryTrendsKey}
                initialPolicy={selectedPolicy}
                initialRunLabel={liveRunLabel ?? activeRunLabel}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

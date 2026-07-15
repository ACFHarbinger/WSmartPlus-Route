/**
 * Process Monitor — tabular view of all spawned processes with inline log viewer (§G.15).
 *
 * Each process row shows: status badge, ID, command, PID, start time, live duration.
 * Selecting a row expands the scrollable log viewer with auto-scroll toggle.
 * Cancel button sends SIGTERM via Rust's `cancel_process` command.
 *
 * §A.3 Option C: ``test_sim`` processes surface live + cross-run policy telemetry panels.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Square, ChevronDown, ChevronUp, Terminal, ArrowDown, Trash2 } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { PolicyTelemetryPanel } from "../../components/analysis/PolicyTelemetryPanel";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/PolicyTelemetryTrendsPanel";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import { StatusPill } from "../../components/ui/StatusPill";
import {
  collectPolicyVizFromLogLines,
  uniquePolicyVizPolicies,
} from "../../utils/policyTelemetry";
import { runLabelFromLogLines } from "../../utils/policyTelemetryTrends";

/**
 * Try to parse a log line as structured JSON (e.g. Python's structlog or loguru JSON sink).
 * Renders level-coloured output when successful; falls back to plain text.
 */
function LogLine({ line }: { line: string }) {
  const isStderr = line.startsWith("[stderr]");
  const text = isStderr ? line.slice(8) : line;

  try {
    const json = JSON.parse(text) as Record<string, unknown>;
    const level = String(json.level ?? json.levelname ?? json.severity ?? "").toUpperCase();
    const msg = String(json.msg ?? json.message ?? json.text ?? text);
    const ts = String(json.timestamp ?? json.time ?? json.ts ?? json.t ?? "");

    const levelColor =
      level.startsWith("ERR") || level.startsWith("CRIT") || level.startsWith("FATAL")
        ? "text-accent-danger"
        : level.startsWith("WARN")
        ? "text-accent-warning"
        : level.startsWith("DEBUG")
        ? "text-canvas-muted"
        : "text-gray-400";

    return (
      <span>
        {ts && <span className="text-canvas-muted mr-2 text-[10px]">{ts.slice(0, 19)}</span>}
        {level && (
          <span className={`font-semibold mr-2 ${levelColor}`}>[{level.slice(0, 5)}]</span>
        )}
        <span className="text-gray-300">{msg}</span>
      </span>
    );
  } catch {
    return (
      <span className={isStderr ? "text-accent-warning" : "text-gray-400"}>{line}</span>
    );
  }
}

// Scan the last 30 log lines for a PROGRESS:{...} marker emitted by Python
const PROGRESS_MARKER = "PROGRESS:";

interface ProgressInfo {
  value: number;
  total?: number;
  label?: string;
}

function getLatestProgress(logLines: string[]): ProgressInfo | null {
  const start = Math.max(0, logLines.length - 30);
  for (let i = logLines.length - 1; i >= start; i--) {
    const line = logLines[i];
    const idx = line.indexOf(PROGRESS_MARKER);
    if (idx === -1) continue;
    try {
      const raw = JSON.parse(line.slice(idx + PROGRESS_MARKER.length).trim()) as Record<string, unknown>;
      const value = typeof raw.value === "number" ? raw.value
        : typeof raw.current === "number" ? raw.current : null;
      if (value === null) continue;
      return {
        value,
        total: typeof raw.total === "number" ? raw.total : undefined,
        label: typeof raw.label === "string" ? raw.label : undefined,
      };
    } catch {}
  }
  return null;
}

function useLiveDuration(startTime: number, stopped: boolean): string {
  const [elapsed, setElapsed] = useState(Date.now() - startTime);

  useEffect(() => {
    if (stopped) return;
    const id = setInterval(() => setElapsed(Date.now() - startTime), 1000);
    return () => clearInterval(id);
  }, [startTime, stopped]);

  const s = Math.floor(elapsed / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

function ProcessRow({
  id,
  selected,
  runBrushActive,
  onSelect,
}: {
  id: string;
  selected: boolean;
  runBrushActive?: boolean;
  onSelect: () => void;
}) {
  const proc = useProcessStore((s) => s.processes[id]);
  const removeProcess = useProcessStore((s) => s.removeProcess);
  const [expanded, setExpanded] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  const stopped = proc.status !== "running";
  const duration = useLiveDuration(proc.startTime, stopped);

  const cancel = useCallback(async () => {
    await invoke("cancel_process", { id });
  }, [id]);

  // Auto-scroll log to bottom when new lines arrive
  useEffect(() => {
    if (expanded && autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [proc.logLines.length, expanded, autoScroll]);

  if (!proc) return null;

  return (
    <div
      className={`border rounded-xl overflow-hidden ${
        selected ? "border-accent-secondary/60" : "border-canvas-border"
      } ${runBrushActive ? "ring-1 ring-accent-secondary/40" : ""}`}
    >
      {/* Table row */}
      <div
        className={`flex items-center gap-3 px-4 py-2.5 hover:bg-canvas-hover cursor-pointer ${
          selected ? "bg-accent-primary/10" : "bg-canvas-surface"
        }`}
        onClick={onSelect}
      >
        <StatusPill status={proc.status} />

        <div className="flex-1 min-w-0">
          <p className="text-xs font-mono text-gray-200 truncate">{id}</p>
          <p className="text-xs text-canvas-muted truncate">{proc.command}</p>
        </div>

        <div className="hidden sm:flex items-center gap-4 text-xs text-canvas-muted shrink-0">
          <span title="PID">PID {proc.pid}</span>
          <span title="Elapsed">{duration}</span>
          {proc.exitCode !== undefined && (
            <span
              className={proc.exitCode === 0 ? "text-accent-success" : "text-accent-danger"}
            >
              exit {proc.exitCode}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {proc.status === "running" && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                void cancel();
              }}
              className="flex items-center gap-1 text-xs text-accent-danger hover:text-accent-danger/80 btn-ghost py-1 px-2"
            >
              <Square size={11} />
              Cancel
            </button>
          )}
          {stopped && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                removeProcess(id);
              }}
              className="btn-ghost p-1.5 text-canvas-muted hover:text-accent-danger"
              title="Remove from history"
            >
              <Trash2 size={12} />
            </button>
          )}
          <button
            className="btn-ghost p-1.5 text-canvas-muted"
            onClick={(e) => {
              e.stopPropagation();
              setExpanded((v) => !v);
            }}
            title={expanded ? "Collapse log" : "Expand log"}
          >
            {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>
        </div>
      </div>

      {/* Progress bar — shown when process:stdout emits PROGRESS:{...} markers */}
      {proc.status === "running" && (() => {
        const prog = getLatestProgress(proc.logLines);
        if (!prog) return null;
        const pct = prog.total ? Math.min(100, (prog.value / prog.total) * 100) : null;
        return (
          <div className="px-4 py-1.5 bg-canvas-bg border-t border-canvas-border/40 flex items-center gap-3">
            <div className="flex-1 h-1.5 bg-canvas-elevated rounded-full overflow-hidden">
              {pct !== null ? (
                <div
                  className="h-full bg-accent-primary rounded-full transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
              ) : (
                <div className="h-full bg-accent-primary/40 rounded-full animate-pulse w-full" />
              )}
            </div>
            <span className="text-[10px] font-mono text-canvas-muted shrink-0">
              {pct !== null
                ? `${pct.toFixed(0)}%`
                : prog.label ?? `${prog.value}`}
            </span>
            {prog.label && pct !== null && (
              <span className="text-[10px] text-canvas-muted shrink-0">{prog.label}</span>
            )}
          </div>
        );
      })()}

      {/* Log viewer */}
      {expanded && (
        <div className="bg-canvas-bg border-t border-canvas-border">
          <div className="flex items-center gap-2 px-4 py-1.5 border-b border-canvas-border/50">
            <Terminal size={11} className="text-canvas-muted" />
            <span className="text-xs text-canvas-muted flex-1">
              {proc.logLines.length} lines
            </span>
            <label className="flex items-center gap-1.5 text-xs text-canvas-muted cursor-pointer">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="accent-accent-primary"
              />
              <ArrowDown size={10} />
              Auto-scroll
            </label>
          </div>
          <div
            ref={logRef}
            className="h-48 overflow-y-auto px-4 py-2 font-mono text-xs"
          >
            {proc.logLines.length === 0 && (
              <span className="text-canvas-muted">No output yet…</span>
            )}
            {proc.logLines.map((line, i) => (
              <div key={i} className="py-px">
                <LogLine line={line} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function isTestSimProcess(command: string): boolean {
  return /\btest_sim\b/.test(command);
}

export function ProcessMonitor() {
  const processes = useProcessStore((s) => s.processes);
  const clearCompleted = useProcessStore((s) => s.clearCompleted);
  const { effectiveTheme: theme } = useAppStore();
  const {
    policy: activePolicy,
    runLabel: activeRunLabel,
    logScale,
    setPolicy,
    setRunLabel,
  } = useGlobalFiltersStore();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [telemetryTrendsKey, setTelemetryTrendsKey] = useState(0);

  const ids = Object.keys(processes).sort(
    (a, b) => (processes[b].startTime ?? 0) - (processes[a].startTime ?? 0)
  );

  const running = ids.filter((id) => processes[id].status === "running").length;
  const completed = ids.length - running;

  useEffect(() => {
    if (selectedId && !processes[selectedId]) {
      setSelectedId(null);
    }
  }, [selectedId, processes]);

  const selectedProc = selectedId ? processes[selectedId] : null;
  const selectedIsSim = selectedProc ? isTestSimProcess(selectedProc.command) : false;

  const policyVizEntries = useMemo(
    () => (selectedProc ? collectPolicyVizFromLogLines(selectedProc.logLines) : []),
    [selectedProc]
  );

  const vizPolicies = useMemo(
    () => uniquePolicyVizPolicies(policyVizEntries),
    [policyVizEntries]
  );

  const selectedPolicy = activePolicy ?? vizPolicies[0] ?? null;

  const processRunLabel = useMemo(() => {
    if (!selectedProc) return null;
    return runLabelFromLogLines(selectedProc.logLines, selectedProc.id);
  }, [selectedProc]);

  const processRunBrushById = useMemo(() => {
    const map: Record<string, string> = {};
    for (const id of ids) {
      const proc = processes[id];
      if (!proc || !isTestSimProcess(proc.command)) continue;
      map[id] = runLabelFromLogLines(proc.logLines, proc.id);
    }
    return map;
  }, [ids, processes]);

  useEffect(() => {
    if (!selectedIsSim || !processRunLabel) return;
    setRunLabel(processRunLabel);
  }, [selectedIsSim, processRunLabel, setRunLabel]);

  useEffect(() => {
    if (policyVizEntries.length > 0) {
      setTelemetryTrendsKey((k) => k + 1);
    }
  }, [policyVizEntries.length, selectedProc?.logLines.length]);

  const policyVizLive = selectedProc?.status === "running";

  if (ids.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-canvas-muted gap-2">
        <Terminal size={28} className="opacity-30" />
        <p className="text-sm">No processes yet.</p>
        <p className="text-xs">Launch a simulation, training run, or data generation to see it here.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <GlobalFilterBar
        policies={vizPolicies.length > 0 ? vizPolicies : undefined}
        runLabels={processRunLabel ? [processRunLabel] : []}
        showLogScale
      />

      <div className="flex items-center gap-3">
        <p className="text-sm text-canvas-muted">
          {ids.length} process{ids.length !== 1 ? "es" : ""}
          {running > 0 && (
            <span className="ml-2 text-accent-success">· {running} running</span>
          )}
        </p>
        {completed > 0 && (
          <button
            onClick={clearCompleted}
            className="btn-ghost text-xs text-canvas-muted ml-auto"
          >
            Clear completed ({completed})
          </button>
        )}
      </div>

      {ids.map((id) => (
        <ProcessRow
          key={id}
          id={id}
          selected={selectedId === id}
          runBrushActive={
            Boolean(activeRunLabel) && processRunBrushById[id] === activeRunLabel
          }
          onSelect={() => setSelectedId((prev) => (prev === id ? null : id))}
        />
      ))}

      {selectedIsSim && selectedProc && (
        <div className="space-y-3 pt-2 border-t border-canvas-border">
          <p className="text-xs text-canvas-muted">
            Policy telemetry for <span className="font-mono text-gray-300">{selectedProc.id}</span>
            {processRunLabel && (
              <span className="ml-2 text-accent-secondary">· {processRunLabel}</span>
            )}
          </p>

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
            initialRunLabel={processRunLabel}
          />
        </div>
      )}
    </div>
  );
}

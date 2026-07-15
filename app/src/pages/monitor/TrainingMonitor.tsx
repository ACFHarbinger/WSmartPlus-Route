/**
 * Training Monitor — live and historical training metrics (§G.17).
 *
 * Ports the Streamlit `training` mode from:
 *   logic/src/ui/pages/training.py
 *   logic/src/ui/pages/training_charts.py
 *   logic/src/ui/services/data_loader.py (discover_training_runs, load_training_metrics)
 *
 * §G.17 additions in this pass:
 *   - Multi-run overlay chart: all selected runs on one ECharts canvas, colour-coded
 *   - Hyperparameter panel: reads hparams.yaml via read_text_file, flat key-value table
 *   - Gradient norm sparkline: separate chart for the grad_norm column
 *   - Checkpoint browser: lists checkpoints/ dir, "Load in Eval Runner" button
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { ChevronDown, ChevronRight, FolderOpen, Radio, RefreshCw } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { TrainHpoNavMesh } from "../../components/layout/TrainHpoNavMesh";
import { LiveTrainProgressBar } from "../../components/monitor/LiveTrainProgressBar";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { RuntimeAttentionPanel } from "../../components/analysis/RuntimeAttentionPanel";
import { TrainingHealthPanel } from "../../components/analysis/TrainingHealthPanel";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import { parseAttentionVizLine } from "../../utils/attentionViz";
import { filterCheckpointEntries } from "../../utils/checkpoints";
import { parseTrainingHealthLine } from "../../utils/trainingHealth";
import {
  findActiveLiveTrainProcessId,
  isHpoProcess,
  liveTrainProcessLabel,
} from "../../utils/trainingProcess";
import type {
  AttentionVizEntry,
  DirEntry,
  StdoutLine,
  TrainingHealthEntry,
  TrainingRun,
  TrainingMetricsRow,
} from "../../types";

// Signal keys that identify a line as a training metric (covers Lightning column variants)
const METRIC_SIGNAL_KEYS = [
  "train_loss", "train/rl_loss", "train/il_loss",
  "val_loss", "val/cost", "val_cost",
  "reward", "grad_norm", "entropy", "epoch", "step",
];

// Virtual key used for the live-process entry in metricsMap
const LIVE_KEY = "__live__";
const LIVE_HEALTH_KEY = "__live_health__";
const LIVE_ATTENTION_KEY = "__live_attention__";

/**
 * Parse a stdout line as a training metric row.
 * Accepts pure JSON and key=value pair formats.
 * Returns null when no metric signal keys are detected.
 */
function parseMetricLine(line: string): TrainingMetricsRow | null {
  const text = line.startsWith("[stderr]") ? line.slice(8) : line;
  try {
    const obj = JSON.parse(text) as Record<string, unknown>;
    if (METRIC_SIGNAL_KEYS.some((k) => typeof obj[k] === "number")) {
      return normalizeMetricRow(obj);
    }
  } catch {}
  if (METRIC_SIGNAL_KEYS.some((k) => text.includes(k))) {
    const row: Record<string, number> = {};
    for (const [, key, val] of text.matchAll(/(\w[\w/]*)=([0-9.eE+\-]+)/g)) {
      row[key] = parseFloat(val);
    }
    if (Object.keys(row).length > 0) return normalizeMetricRow(row);
  }
  return null;
}

/**
 * Normalize Lightning CSV column aliases to the canonical TrainingMetricsRow keys.
 * Lightning logs loss as "train/rl_loss", LR as "lr-Adam", etc.
 */
function normalizeMetricRow(raw: Record<string, unknown>): TrainingMetricsRow {
  const r = { ...raw } as TrainingMetricsRow;
  // Loss normalization
  if (r.train_loss == null) {
    r.train_loss = (raw["train/rl_loss"] ?? raw["train/il_loss"]) as number | undefined;
  }
  // Validation normalization
  if (r.val_loss == null) {
    r.val_loss = (raw["val/cost"] ?? raw["val_cost"]) as number | undefined;
  }
  // LR normalization: lr-Adam, lr-SGD, lr_scheduler, etc.
  if (r.lr == null) {
    for (const key of Object.keys(raw)) {
      if (key !== "lr" && key.startsWith("lr") && typeof raw[key] === "number") {
        r.lr = raw[key] as number;
        break;
      }
    }
  }
  return r;
}

// ── Colour palette for multi-run overlay
const RUN_COLORS = [
  "#6366f1", // indigo
  "#34d399", // emerald
  "#f87171", // red
  "#fbbf24", // amber
  "#a78bfa", // violet
  "#fb923c", // orange
  "#38bdf8", // sky
  "#f472b6", // pink
];

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// ── Multi-run overlay chart (train_loss + reward per run on shared axes)
function MultiRunChart({
  runsMetrics,
  logScale,
}: {
  runsMetrics: { name: string; metrics: TrainingMetricsRow[] }[];
  logScale: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  if (runsMetrics.length === 0) return null;

  const series: object[] = [];
  const legendData: string[] = [];

  runsMetrics.forEach(({ name, metrics }, i) => {
    const color = RUN_COLORS[i % RUN_COLORS.length];
    const epochs = metrics.map((r) => r.epoch ?? r.step ?? 0);
    const scaleY = (y: number | null | undefined) =>
      y == null ? null : logScale ? Math.max(y, 1e-8) : y;
    const trainLoss = metrics.map((r, j) => [epochs[j], scaleY(r.train_loss)]);
    const valLoss = metrics.map((r, j) => [epochs[j], scaleY(r.val_loss)]);
    const reward = metrics.map((r, j) => [epochs[j], r.reward ?? null]);

    const hasReward = metrics.some((r) => r.reward != null);
    const shortName = name.length > 20 ? `…${name.slice(-18)}` : name;

    legendData.push(`${shortName} loss`, ...(hasReward ? [`${shortName} reward`] : []));

    series.push({
      name: `${shortName} loss`,
      type: "line",
      data: trainLoss,
      smooth: true,
      lineStyle: { color, width: 2 },
      itemStyle: { color },
      symbol: "none",
      encode: { x: 0, y: 1 },
    });

    if (metrics.some((r) => r.val_loss != null)) {
      series.push({
        name: `${shortName} val`,
        type: "line",
        data: valLoss,
        smooth: true,
        lineStyle: { color, width: 1.5, type: "dashed" },
        itemStyle: { color },
        symbol: "none",
        encode: { x: 0, y: 1 },
      });
    }

    if (hasReward) {
      series.push({
        name: `${shortName} reward`,
        type: "line",
        yAxisIndex: 1,
        data: reward,
        smooth: true,
        lineStyle: { color, width: 2, type: "dotted" },
        itemStyle: { color },
        symbol: "none",
        encode: { x: 0, y: 1 },
      });
    }
  });

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-canvas-muted">
          {logScale ? "Multi-run overlay (log-scale loss)" : "Multi-run overlay"}
        </p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="training-overlay"
        />
      </div>
      <ReactECharts
        ref={chartRef}
        option={{
          backgroundColor: "transparent",
          legend: {
            data: legendData,
            textStyle: { color: "#9090b0", fontSize: 10 },
            type: "scroll",
          },
          grid: { left: 50, right: 60, top: 45, bottom: 30 },
          xAxis: {
            type: "value",
            name: "Epoch",
            nameTextStyle: { color: "#9090b0" },
            axisLabel: { color: "#9090b0", fontSize: 10 },
          },
          yAxis: [
            {
              type: (logScale ? "log" : "value") as "log" | "value",
              logBase: 10,
              name: logScale ? "Loss (log)" : "Loss",
              nameTextStyle: { color: "#9090b0" },
              axisLabel: { color: "#9090b0", fontSize: 10 },
              minorSplitLine: { show: false },
            },
            {
              type: "value",
              name: "Reward",
              nameTextStyle: { color: "#9090b0" },
              axisLabel: { color: "#9090b0", fontSize: 10 },
              splitLine: { show: false },
            },
          ],
          series,
          tooltip: { trigger: "axis" },
        }}
        style={{ height: 260 }}
      />
    </div>
  );
}

// ── Shared sparkline — small ECharts for a single scalar column
function MetricSparkline({
  label,
  data,
  color,
  exportName,
  logScale = false,
}: {
  label: string;
  data: [number, number][];
  color: string;
  exportName?: string;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  if (data.length === 0) return null;
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <p className="text-xs text-canvas-muted">{label}</p>
        {exportName && (
          <ChartExportButtons
            chartRef={{ current: chartRef.current }}
            filenameStem={exportName}
            size={10}
          />
        )}
      </div>
      <ReactECharts
        ref={chartRef}
        option={{
          backgroundColor: "transparent",
          grid: { left: 40, right: 10, top: 8, bottom: 28 },
          xAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 9 } },
          yAxis: {
            type: (logScale ? "log" : "value") as "log" | "value",
            logBase: 10,
            axisLabel: { color: "#9090b0", fontSize: 9 },
            minorSplitLine: { show: false },
          },
          series: [{
            type: "line",
            data: logScale ? data.map(([x, y]) => [x, Math.max(y, 1e-8)]) : data,
            smooth: false,
            lineStyle: { color, width: 1.5 },
            areaStyle: { color: `${color}1a` },
            symbol: "none",
          }],
          tooltip: { trigger: "axis" },
        }}
        style={{ height: 80 }}
      />
    </div>
  );
}

function GradNormSparkline({
  metrics,
  logScale = false,
}: {
  metrics: TrainingMetricsRow[];
  logScale?: boolean;
}) {
  const data = metrics
    .filter((r) => r.grad_norm != null)
    .map((r): [number, number] => [r.epoch ?? r.step ?? 0, r.grad_norm!]);
  return (
    <MetricSparkline
      label={logScale ? "Gradient Norm (log)" : "Gradient Norm"}
      data={data}
      color="#f87171"
      exportName="training-grad-norm"
      logScale={logScale}
    />
  );
}

function LrSparkline({
  metrics,
  logScale = false,
}: {
  metrics: TrainingMetricsRow[];
  logScale?: boolean;
}) {
  const data = metrics
    .filter((r) => r.lr != null)
    .map((r): [number, number] => [r.step ?? r.epoch ?? 0, r.lr!]);
  return (
    <MetricSparkline
      label={logScale ? "Learning Rate (log)" : "Learning Rate"}
      data={data}
      color="#fbbf24"
      exportName="training-lr"
      logScale={logScale}
    />
  );
}

// ── Hyperparameter panel (reads hparams.yaml, renders flat key-value table)
function HparamsPanel({ runPath }: { runPath: string }) {
  const [pairs, setPairs] = useState<{ key: string; value: string }[] | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const hparamsPath = `${runPath}/hparams.yaml`;
    invoke<string>("read_text_file", { path: hparamsPath })
      .then((text) => {
        const parsed = text
          .split("\n")
          .map((line) => line.match(/^(\S[^:]*?):\s*(.*)$/))
          .filter(Boolean)
          .map((m) => ({ key: m![1].trim(), value: m![2].trim() }))
          .filter(({ key }) => !key.startsWith("#"));
        setPairs(parsed);
      })
      .catch(() => setPairs(null));
  }, [runPath]);

  if (!pairs?.length) return null;

  const visible = expanded ? pairs : pairs.slice(0, 8);

  return (
    <div className="rounded-xl border border-canvas-border overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-4 py-2.5 bg-canvas-surface hover:bg-canvas-hover text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="text-xs font-semibold text-canvas-muted uppercase tracking-wider">
          Hyperparameters
        </span>
        <span className="ml-auto text-xs text-canvas-muted">{pairs.length} params</span>
      </button>
      {expanded && (
        <div className="px-4 py-2 bg-canvas-bg border-t border-canvas-border">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5">
            {visible.map(({ key, value }) => (
              <>
                <dt key={`k-${key}`} className="text-xs text-canvas-muted font-mono py-0.5">{key}</dt>
                <dd key={`v-${key}`} className="text-xs text-gray-300 font-mono py-0.5 truncate">{value || "—"}</dd>
              </>
            ))}
          </dl>
          {pairs.length > 8 && (
            <button
              className="text-xs text-accent-primary mt-2"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded && visible.length < pairs.length
                ? `Show all ${pairs.length} params`
                : "Collapse"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ── Checkpoint browser
function CheckpointBrowser({
  runPath,
  onLoadInEvalRunner,
}: {
  runPath: string;
  onLoadInEvalRunner: (path: string) => void;
}) {
  const [checkpoints, setCheckpoints] = useState<DirEntry[] | null>(null);

  useEffect(() => {
    invoke<DirEntry[]>("list_dir", { path: `${runPath}/checkpoints` })
      .then((entries) =>
        setCheckpoints(filterCheckpointEntries(entries))
      )
      .catch(() => setCheckpoints([]));
  }, [runPath]);

  if (!checkpoints?.length) return null;

  return (
    <div className="rounded-xl border border-canvas-border overflow-hidden">
      <div className="px-4 py-2.5 bg-canvas-surface">
        <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider">
          Checkpoints ({checkpoints.length})
        </p>
      </div>
      <div className="divide-y divide-canvas-border/40 bg-canvas-bg">
        {checkpoints.map((ckpt) => (
          <div key={ckpt.path} className="flex items-center gap-3 px-4 py-2">
            <span className="text-xs font-mono text-gray-300 flex-1 truncate">{ckpt.name}</span>
            <span className="text-xs text-canvas-muted shrink-0">
              {formatBytes(ckpt.size_bytes)}
            </span>
            <button
              onClick={() => onLoadInEvalRunner(ckpt.path)}
              className="btn-ghost text-xs shrink-0"
              title="Open this checkpoint in the Evaluation Runner"
            >
              Load in Eval Runner →
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Run panel (chart + hparams + checkpoints for one or more selected runs)
function RunPanel({
  run,
  metrics,
  color,
  onLoadCheckpoint,
  logScale = false,
}: {
  run: TrainingRun;
  metrics: TrainingMetricsRow[];
  color: string;
  onLoadCheckpoint: (path: string) => void;
  logScale?: boolean;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
        <p className="text-xs font-mono text-gray-300">{run.name}</p>
        <span className="text-xs text-canvas-muted">{metrics.length} epochs</span>
      </div>
      <GradNormSparkline metrics={metrics} logScale={logScale} />
      <LrSparkline metrics={metrics} logScale={logScale} />
      {run.has_hparams && <HparamsPanel runPath={run.path} />}
      <CheckpointBrowser runPath={run.path} onLoadInEvalRunner={onLoadCheckpoint} />
    </div>
  );
}

// ── Main page
export function TrainingMonitor() {
  const { projectRoot, setMode, setPendingCheckpoint, effectiveTheme } = useAppStore();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [metricsMap, setMetricsMap] = useState<Record<string, TrainingMetricsRow[]>>({});
  const [healthMap, setHealthMap] = useState<Record<string, TrainingHealthEntry[]>>({});
  const [attentionMap, setAttentionMap] = useState<Record<string, AttentionVizEntry[]>>({});
  const [loading, setLoading] = useState(false);

  const logsPath = projectRoot ? `${projectRoot}/logs` : "";

  // ── Live training mode: watch for an active train_* or hpo_* process
  const processes = useProcessStore((s) => s.processes);
  const activeTrainId = useMemo(
    () => findActiveLiveTrainProcessId(processes),
    [processes]
  );
  const liveProcessLabel = activeTrainId
    ? liveTrainProcessLabel(activeTrainId)
    : "Live Training";
  const activeIsHpo = activeTrainId
    ? isHpoProcess(activeTrainId, processes[activeTrainId]?.command ?? "")
    : false;
  const activeTrainRunning =
    activeTrainId != null && processes[activeTrainId]?.status === "running";

  // Subscribe to stdout of the active training process and accumulate live rows
  useEffect(() => {
    if (!activeTrainId) {
      setMetricsMap((m) => {
        if (!m[LIVE_KEY]) return m;
        const next = { ...m };
        delete next[LIVE_KEY];
        return next;
      });
      setHealthMap((h) => {
        if (!h[LIVE_HEALTH_KEY]) return h;
        const next = { ...h };
        delete next[LIVE_HEALTH_KEY];
        return next;
      });
      setAttentionMap((a) => {
        if (!a[LIVE_ATTENTION_KEY]) return a;
        const next = { ...a };
        delete next[LIVE_ATTENTION_KEY];
        return next;
      });
      return;
    }
    setMetricsMap((m) => ({ ...m, [LIVE_KEY]: [] }));
    setHealthMap((h) => ({ ...h, [LIVE_HEALTH_KEY]: [] }));
    setAttentionMap((a) => ({ ...a, [LIVE_ATTENTION_KEY]: [] }));
    let unlisten: (() => void) | null = null;
    listen<StdoutLine>("process:stdout", (event) => {
      const { id, line } = event.payload;
      if (id !== activeTrainId) return;
      const row = parseMetricLine(line);
      if (row) setMetricsMap((m) => ({ ...m, [LIVE_KEY]: [...(m[LIVE_KEY] ?? []), row] }));
      const alert = parseTrainingHealthLine(line);
      if (alert) {
        setHealthMap((h) => ({
          ...h,
          [LIVE_HEALTH_KEY]: [...(h[LIVE_HEALTH_KEY] ?? []), alert],
        }));
      }
      const attention = parseAttentionVizLine(line);
      if (attention) {
        setAttentionMap((a) => ({
          ...a,
          [LIVE_ATTENTION_KEY]: [...(a[LIVE_ATTENTION_KEY] ?? []), attention],
        }));
      }
    }).then((fn) => { unlisten = fn; });
    return () => { unlisten?.(); };
  }, [activeTrainId]);

  // Auto-select the live key when a train process starts
  useEffect(() => {
    if (activeTrainId && !selected.includes(LIVE_KEY)) {
      setSelected((s) => [LIVE_KEY, ...s]);
    } else if (!activeTrainId) {
      setSelected((s) => s.filter((k) => k !== LIVE_KEY));
    }
  }, [activeTrainId]); // eslint-disable-line react-hooks/exhaustive-deps

  const discover = useCallback(async () => {
    if (!logsPath) return;
    setLoading(true);
    try {
      const found = await invoke<TrainingRun[]>("list_training_runs", { logsPath });
      setRuns(found);
    } finally {
      setLoading(false);
    }
  }, [logsPath]);

  useEffect(() => {
    if (logsPath) discover();
  }, [logsPath, discover]);

  const loadMetrics = useCallback(
    async (run: TrainingRun) => {
      if (metricsMap[run.name]) return;
      const rows = await invoke<Record<string, unknown>[]>("load_training_metrics", {
        runPath: run.path,
      });
      setMetricsMap((m) => ({ ...m, [run.name]: rows.map(normalizeMetricRow) }));
    },
    [metricsMap]
  );

  const loadHealth = useCallback(
    async (run: TrainingRun) => {
      if (healthMap[run.name]) return;
      const healthPath = `${run.path}/training_health.jsonl`;
      try {
        const entries = await invoke<TrainingHealthEntry[]>("load_training_health_log", {
          path: healthPath,
        });
        setHealthMap((h) => ({ ...h, [run.name]: entries }));
      } catch {
        setHealthMap((h) => ({ ...h, [run.name]: [] }));
      }
    },
    [healthMap]
  );

  const loadAttention = useCallback(
    async (run: TrainingRun) => {
      if (attentionMap[run.name]) return;
      const attentionPath = `${run.path}/attention_viz.jsonl`;
      try {
        const entries = await invoke<AttentionVizEntry[]>("load_attention_viz_log", {
          path: attentionPath,
        });
        setAttentionMap((a) => ({ ...a, [run.name]: entries }));
      } catch {
        setAttentionMap((a) => ({ ...a, [run.name]: [] }));
      }
    },
    [attentionMap]
  );

  const toggleRun = useCallback(
    (run: TrainingRun) => {
      setSelected((s) =>
        s.includes(run.name) ? s.filter((r) => r !== run.name) : [...s, run.name]
      );
      loadMetrics(run);
      loadHealth(run);
      loadAttention(run);
    },
    [loadMetrics, loadHealth, loadAttention]
  );

  const selectedRunObjects = useMemo(
    () => runs.filter((r) => selected.includes(r.name)),
    [runs, selected]
  );

  const displayedHealth = useMemo(() => {
    const merged: TrainingHealthEntry[] = [];
    if (activeTrainId && (healthMap[LIVE_HEALTH_KEY]?.length ?? 0) > 0) {
      merged.push(...healthMap[LIVE_HEALTH_KEY]!);
    }
    for (const run of selectedRunObjects) {
      const entries = healthMap[run.name];
      if (entries?.length) merged.push(...entries);
    }
    return merged;
  }, [activeTrainId, healthMap, selectedRunObjects]);

  const displayedAttention = useMemo(() => {
    const merged: AttentionVizEntry[] = [];
    if (activeTrainId && (attentionMap[LIVE_ATTENTION_KEY]?.length ?? 0) > 0) {
      merged.push(...attentionMap[LIVE_ATTENTION_KEY]!);
    }
    for (const run of selectedRunObjects) {
      const entries = attentionMap[run.name];
      if (entries?.length) merged.push(...entries);
    }
    return merged;
  }, [activeTrainId, attentionMap, selectedRunObjects]);

  const runsMetrics = useMemo(() => {
    const result: { name: string; metrics: TrainingMetricsRow[] }[] = [];
    // Live entry first
    if (selected.includes(LIVE_KEY) && (metricsMap[LIVE_KEY]?.length ?? 0) > 0) {
      result.push({ name: liveProcessLabel, metrics: metricsMap[LIVE_KEY] });
    }
    for (const r of selectedRunObjects) {
      const metrics = metricsMap[r.name] ?? [];
      if (metrics.length > 0) result.push({ name: r.name, metrics });
    }
    return result;
  }, [selected, selectedRunObjects, metricsMap, liveProcessLabel]);

  const handleLoadCheckpoint = useCallback(
    (checkpointPath: string) => {
      setPendingCheckpoint(checkpointPath);
      setMode("eval_runner");
    },
    [setPendingCheckpoint, setMode]
  );

  if (!projectRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-canvas-muted gap-2">
        <p className="text-sm">Project root not configured.</p>
        <p className="text-xs">Set the project root in Settings to discover training runs.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <GlobalFilterBar showLogScale />

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={discover}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? <RefreshCw size={14} className="animate-spin" /> : <FolderOpen size={14} />}
          Discover Runs
        </button>
        <span className="text-xs text-canvas-muted truncate">{logsPath}</span>
      </div>

      {runs.length === 0 && !loading && (
        <div className="card text-canvas-muted text-sm">
          No training runs found in{" "}
          <code className="font-mono text-xs">{logsPath}</code>.
        </div>
      )}

      {/* Live training indicator */}
      {activeTrainId && (
        <div className="card border-accent-success/30 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <label className="flex items-center gap-3 py-1 px-1 rounded-lg cursor-pointer flex-1 min-w-0">
              <input
                type="checkbox"
                checked={selected.includes(LIVE_KEY)}
                onChange={() =>
                  setSelected((s) =>
                    s.includes(LIVE_KEY) ? s.filter((k) => k !== LIVE_KEY) : [LIVE_KEY, ...s]
                  )
                }
                className="accent-accent-primary"
              />
              <Radio size={13} className="text-accent-success animate-pulse shrink-0" />
              <span className="text-sm text-accent-success font-mono flex-1">{liveProcessLabel}</span>
              <span className="text-xs text-canvas-muted font-mono truncate max-w-xs">{activeTrainId}</span>
              <span className="text-xs text-accent-success">
                {metricsMap[LIVE_KEY]?.length ?? 0} updates
              </span>
            </label>
            <TrainHpoNavMesh showHpoLinks={activeIsHpo} />
          </div>
          {activeTrainRunning && (
            <LiveTrainProgressBar
              processId={activeTrainId}
              fallbackValue={
                metricsMap[LIVE_KEY]?.[metricsMap[LIVE_KEY].length - 1]?.epoch
              }
            />
          )}
        </div>
      )}

      {/* Run selector */}
      {runs.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-3">
            Training Runs — select to overlay
          </p>
          <div className="space-y-1">
            {runs.map((run, i) => {
              const liveOffset = activeTrainId ? 1 : 0;
              const color = RUN_COLORS[(i + liveOffset) % RUN_COLORS.length];
              return (
                <label
                  key={run.name}
                  className="flex items-center gap-3 py-1.5 px-2 rounded-lg hover:bg-canvas-hover cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(run.name)}
                    onChange={() => toggleRun(run)}
                    className="accent-accent-primary"
                  />
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: selected.includes(run.name) ? color : "#444" }}
                  />
                  <span className="text-sm text-gray-300 font-mono flex-1 truncate">
                    {run.name}
                  </span>
                  <span className="flex gap-2 text-xs text-canvas-muted">
                    {run.has_metrics && <span className="text-accent-success">metrics</span>}
                    {run.has_hparams && <span>hparams</span>}
                  </span>
                </label>
              );
            })}
          </div>
        </div>
      )}

      {/* Training health alerts (§A.4) */}
      {(displayedHealth.length > 0 || activeTrainId) && (
        <TrainingHealthPanel entries={displayedHealth} />
      )}

      {/* Runtime attention ring-buffer (§A.2 Option A) */}
      {(displayedAttention.length > 0 || activeTrainId) && (
        <RuntimeAttentionPanel
          entries={displayedAttention}
          theme={effectiveTheme}
          logScale={logScale}
        />
      )}

      {/* Multi-run overlay chart */}
      {runsMetrics.length > 0 && <MultiRunChart runsMetrics={runsMetrics} logScale={logScale} />}

      {/* Live run detail panel */}
      {selected.includes(LIVE_KEY) && (metricsMap[LIVE_KEY]?.length ?? 0) > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full shrink-0 bg-accent-success animate-pulse" />
            <p className="text-xs font-mono text-accent-success">{liveProcessLabel}</p>
            <span className="text-xs text-canvas-muted">{metricsMap[LIVE_KEY].length} updates</span>
          </div>
          <GradNormSparkline metrics={metricsMap[LIVE_KEY]} logScale={logScale} />
          <LrSparkline metrics={metricsMap[LIVE_KEY]} logScale={logScale} />
        </div>
      )}

      {/* Per-run panels (grad norm, hparams, checkpoints) */}
      {selectedRunObjects.map((run, i) => {
        const liveOffset = activeTrainId ? 1 : 0;
        const metrics = metricsMap[run.name];
        if (!metrics) {
          return (
            <div key={run.name} className="card text-canvas-muted text-sm">
              Loading {run.name}…
            </div>
          );
        }
        return (
          <RunPanel
            key={run.name}
            run={run}
            metrics={metrics}
            color={RUN_COLORS[(i + liveOffset) % RUN_COLORS.length]}
            onLoadCheckpoint={handleLoadCheckpoint}
            logScale={logScale}
          />
        );
      })}
    </div>
  );
}

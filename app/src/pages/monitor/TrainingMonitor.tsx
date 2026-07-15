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
import { ChevronDown, ChevronRight, FolderOpen, RefreshCw } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { ProcessIdFooter } from "../../components/monitor/ProcessIdFooter";
import { TrainHpoLivePanel } from "../../components/monitor/TrainHpoLivePanel";
import { GradNormSparkline, LrSparkline } from "../../components/monitor/TrainingMetricSparklines";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { RuntimeAttentionPanel } from "../../components/analysis/RuntimeAttentionPanel";
import { TrainingHealthPanel } from "../../components/analysis/TrainingHealthPanel";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import { collectAttentionVizFromLogLines, parseAttentionVizLine } from "../../utils/attentionViz";
import { filterCheckpointEntries } from "../../utils/checkpoints";
import {
  collectTrainingHealthFromLogLines,
  parseTrainingHealthLine,
} from "../../utils/trainingHealth";
import {
  collectTrainingMetricsFromLogLines,
  normalizeTrainingMetricRow,
  parseTrainingMetricLine,
} from "../../utils/trainingMetrics";
import { outputRunPathFromLogLines } from "../../utils/outputRunPath";
import { trainingRunPathFromLogLines } from "../../utils/trainingRunPath";
import {
  findActiveLiveTrainProcessId,
  findRecentTrainOrHpoProcessId,
  isHpoProcess,
  trainHpoLivePanelTitle,
} from "../../utils/trainingProcess";
import type {
  AttentionVizEntry,
  DirEntry,
  StdoutLine,
  TrainingHealthEntry,
  TrainingRun,
  TrainingMetricsRow,
} from "../../types";

// Virtual key used for the live-process entry in metricsMap
const LIVE_KEY = "__live__";
const LIVE_HEALTH_KEY = "__live_health__";
const LIVE_ATTENTION_KEY = "__live_attention__";

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
      <GradNormSparkline
        metrics={metrics}
        logScale={logScale}
        exportName="training-monitor-grad-norm"
      />
      <LrSparkline
        metrics={metrics}
        logScale={logScale}
        exportName="training-monitor-lr"
      />
      {run.has_hparams && <HparamsPanel runPath={run.path} />}
      <CheckpointBrowser runPath={run.path} onLoadInEvalRunner={onLoadCheckpoint} />
    </div>
  );
}

// ── Main page
export function TrainingMonitor() {
  const {
    projectRoot,
    setMode,
    setPendingCheckpoint,
    effectiveTheme,
    pendingTrainingRunPath,
    setPendingTrainingRunPath,
  } = useAppStore();
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
  const activeIsHpo = activeTrainId
    ? isHpoProcess(activeTrainId, processes[activeTrainId]?.command ?? "")
    : false;
  const activeTrainRunning =
    activeTrainId != null && processes[activeTrainId]?.status === "running";

  const recentTrainId = useMemo(
    () => findRecentTrainOrHpoProcessId(processes),
    [processes]
  );
  const recentTrainProc = recentTrainId ? processes[recentTrainId] : null;
  const recentTrainDone =
    recentTrainProc != null && recentTrainProc.status !== "running";
  const recentTrainCompleted = recentTrainProc?.status === "completed";
  const recentOutputRunPath = useMemo(
    () =>
      recentTrainProc ? outputRunPathFromLogLines(recentTrainProc.logLines) : null,
    [recentTrainProc]
  );
  const recentTrainingRunPath = useMemo(
    () =>
      recentTrainProc ? trainingRunPathFromLogLines(recentTrainProc.logLines) : null,
    [recentTrainProc]
  );
  const recentTrainLogLines = recentTrainProc?.logLines ?? [];
  const liveRunLabel = useMemo(() => {
    if (activeTrainId) {
      return trainHpoLivePanelTitle({
        isRunning: activeTrainRunning,
        status: processes[activeTrainId]?.status,
        processId: activeTrainId,
        command: processes[activeTrainId]?.command,
      });
    }
    if (recentTrainId && recentTrainProc) {
      return trainHpoLivePanelTitle({
        isRunning: false,
        status: recentTrainProc.status,
        processId: recentTrainId,
        command: recentTrainProc.command,
      });
    }
    return "Live Training";
  }, [
    activeTrainId,
    activeTrainRunning,
    processes,
    recentTrainId,
    recentTrainProc,
  ]);

  const effectiveLiveMetrics = useMemo(() => {
    if (activeTrainId) return metricsMap[LIVE_KEY] ?? [];
    if (recentTrainProc) {
      return collectTrainingMetricsFromLogLines(recentTrainLogLines);
    }
    return [];
  }, [activeTrainId, metricsMap, recentTrainProc, recentTrainLogLines]);

  const effectiveLiveHealth = useMemo(() => {
    if (activeTrainId) return healthMap[LIVE_HEALTH_KEY] ?? [];
    if (recentTrainProc) {
      return collectTrainingHealthFromLogLines(recentTrainLogLines);
    }
    return [];
  }, [activeTrainId, healthMap, recentTrainProc, recentTrainLogLines]);

  const effectiveLiveAttention = useMemo(() => {
    if (activeTrainId) return attentionMap[LIVE_ATTENTION_KEY] ?? [];
    if (recentTrainProc) {
      return collectAttentionVizFromLogLines(recentTrainLogLines);
    }
    return [];
  }, [activeTrainId, attentionMap, recentTrainProc, recentTrainLogLines]);

  const latestLiveMetric = effectiveLiveMetrics[effectiveLiveMetrics.length - 1];

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
      const row = parseTrainingMetricLine(line);
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

  // Auto-select the live key when a train/HPO process starts or rehydrates after completion
  useEffect(() => {
    if (activeTrainId && !selected.includes(LIVE_KEY)) {
      setSelected((s) => [LIVE_KEY, ...s]);
    } else if (
      !activeTrainId &&
      recentTrainId &&
      effectiveLiveMetrics.length > 0 &&
      !selected.includes(LIVE_KEY)
    ) {
      setSelected((s) => [LIVE_KEY, ...s]);
    }
  }, [activeTrainId, recentTrainId, effectiveLiveMetrics.length]); // eslint-disable-line react-hooks/exhaustive-deps

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
      setMetricsMap((m) => ({ ...m, [run.name]: rows.map(normalizeTrainingMetricRow) }));
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

  useEffect(() => {
    if (!pendingTrainingRunPath) return;
    const run = runs.find((r) => r.path === pendingTrainingRunPath);
    if (run) {
      setSelected((s) => (s.includes(run.name) ? s : [...s, run.name]));
      void loadMetrics(run);
      void loadHealth(run);
      void loadAttention(run);
      setPendingTrainingRunPath(null);
      return;
    }
    if (!loading && logsPath) {
      void discover();
    }
  }, [
    pendingTrainingRunPath,
    runs,
    loading,
    logsPath,
    discover,
    loadMetrics,
    loadHealth,
    loadAttention,
    setPendingTrainingRunPath,
  ]);

  useEffect(() => {
    if (!recentTrainCompleted || !recentTrainingRunPath) return;
    const run = runs.find((r) => r.path === recentTrainingRunPath);
    if (run) {
      setSelected((s) => (s.includes(run.name) ? s : [...s, run.name]));
      void loadMetrics(run);
      void loadHealth(run);
      void loadAttention(run);
      return;
    }
    if (!loading && logsPath) {
      void discover();
    }
  }, [
    recentTrainCompleted,
    recentTrainingRunPath,
    runs,
    loading,
    logsPath,
    discover,
    loadMetrics,
    loadHealth,
    loadAttention,
  ]);

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
    if (effectiveLiveHealth.length > 0) {
      merged.push(...effectiveLiveHealth);
    }
    for (const run of selectedRunObjects) {
      const entries = healthMap[run.name];
      if (entries?.length) merged.push(...entries);
    }
    return merged;
  }, [effectiveLiveHealth, healthMap, selectedRunObjects]);

  const displayedAttention = useMemo(() => {
    const merged: AttentionVizEntry[] = [];
    if (effectiveLiveAttention.length > 0) {
      merged.push(...effectiveLiveAttention);
    }
    for (const run of selectedRunObjects) {
      const entries = attentionMap[run.name];
      if (entries?.length) merged.push(...entries);
    }
    return merged;
  }, [effectiveLiveAttention, attentionMap, selectedRunObjects]);

  const runsMetrics = useMemo(() => {
    const result: { name: string; metrics: TrainingMetricsRow[] }[] = [];
    // Live entry first
    if (selected.includes(LIVE_KEY) && effectiveLiveMetrics.length > 0) {
      result.push({ name: liveRunLabel, metrics: effectiveLiveMetrics });
    }
    for (const r of selectedRunObjects) {
      const metrics = metricsMap[r.name] ?? [];
      if (metrics.length > 0) result.push({ name: r.name, metrics });
    }
    return result;
  }, [selected, selectedRunObjects, effectiveLiveMetrics, liveRunLabel]);

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

      {/* Live / recent train-HPO indicator */}
      {recentTrainId && recentTrainProc && (
        <TrainHpoLivePanel
          cardClassName="border-accent-success/30"
          header={{
            status: activeTrainRunning
              ? "running"
              : recentTrainCompleted
                ? "completed"
                : recentTrainProc.status,
            title: trainHpoLivePanelTitle({
              isRunning: activeTrainRunning,
              status: recentTrainProc.status,
              processId: recentTrainId,
              command: recentTrainProc.command,
            }),
            metricCount: effectiveLiveMetrics.length,
            healthCount: effectiveLiveHealth.length,
            attentionCount: effectiveLiveAttention.length,
            overlaySelect:
              activeTrainId || effectiveLiveMetrics.length > 0
                ? {
                    checked: selected.includes(LIVE_KEY),
                    onChange: () =>
                      setSelected((s) =>
                        s.includes(LIVE_KEY)
                          ? s.filter((k) => k !== LIVE_KEY)
                          : [LIVE_KEY, ...s]
                      ),
                  }
                : undefined,
            navMesh: {
              showHpoLinks:
                activeIsHpo || isHpoProcess(recentTrainId, recentTrainProc.command),
              showOutputBrowser: recentTrainDone && recentTrainCompleted,
              outputRunPath: recentOutputRunPath,
              trainingRunPath: recentTrainingRunPath,
            },
          }}
          progress={
            activeTrainRunning && activeTrainId
              ? {
                  processId: activeTrainId,
                  fallbackValue: latestLiveMetric?.epoch,
                }
              : undefined
          }
          analytics={{
            metrics: effectiveLiveMetrics,
            healthEntries: effectiveLiveHealth,
            attentionEntries: effectiveLiveAttention,
            logScale,
            theme: effectiveTheme,
            exportNamePrefix: "training-monitor",
            isPostRun: recentTrainDone,
            postRunFallback:
              "Post-run shortcuts — open Output Browser or refresh metrics from the completed run",
            showHealthAttention: false,
          }}
          logLines={recentTrainLogLines}
          logTailWaiting={activeTrainRunning}
          footer={<ProcessIdFooter processId={recentTrainId} />}
        />
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
      {displayedHealth.length > 0 && (
        <TrainingHealthPanel entries={displayedHealth} />
      )}

      {/* Runtime attention ring-buffer (§A.2 Option A) */}
      {displayedAttention.length > 0 && (
        <RuntimeAttentionPanel
          entries={displayedAttention}
          theme={effectiveTheme}
          logScale={logScale}
        />
      )}

      {/* Multi-run overlay chart */}
      {runsMetrics.length > 0 && <MultiRunChart runsMetrics={runsMetrics} logScale={logScale} />}

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

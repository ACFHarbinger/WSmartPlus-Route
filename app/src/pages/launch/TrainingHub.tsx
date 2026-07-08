/**
 * Training Hub — full-featured training / HPO / eval launcher (§G.10).
 *
 * Form parameters mirror the controller justfile exactly:
 *   train:    envs, models, model.encoder.type, train.batch_size, seed
 *   hpo:      hpo.n_trials, hpo.method, hpo.num_workers
 *   eval:     eval.policy.model.load_path, eval.datasets, eval.val_size, eval.decoding.strategy
 *
 * After launch the "Live Progress" panel subscribes to process:stdout events
 * and parses JSON metric lines emitted by the Lightning CSV logger / custom
 * training scripts. Accumulates metrics into a live ECharts chart.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { Play, ChevronDown, ChevronUp, Terminal, FolderOpen, Activity, CheckCircle, XCircle } from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { useAppStore } from "../../store/app";
import { useLaunchTriggerStore } from "../../store/launchTrigger";
import { useTrainHubStore } from "../../store/launchers";
import { useSpawnProcess } from "../../hooks/useSpawnProcess";
import type { StdoutLine, StatusUpdate, ProcessStatus, TrainingMetricsRow } from "../../types";

type Mode = "train" | "hpo" | "eval";

const PROBLEMS = ["vrpp", "wcvrp", "scwcvrp"] as const;
const MODELS = ["am", "tam", "ddam", "moe"] as const;
const ENCODERS = ["gat", "gcn", "mha"] as const;
const HPO_METHODS = ["nsgaii", "tpe", "dehb", "random"] as const;
const EVAL_STRATEGIES = ["greedy", "sampling", "beam"] as const;

// Metric keys that identify a training metric line (covers Lightning column variants)
const METRIC_SIGNAL_KEYS = [
  "train_loss", "train/rl_loss", "train/il_loss",
  "val_loss", "val/cost", "val_cost",
  "reward", "grad_norm", "entropy", "epoch", "step",
];

function normalizeMetricRow(raw: Record<string, unknown>): TrainingMetricsRow {
  const r = { ...raw } as TrainingMetricsRow;
  if (r.train_loss == null) r.train_loss = (raw["train/rl_loss"] ?? raw["train/il_loss"]) as number | undefined;
  if (r.val_loss == null) r.val_loss = (raw["val/cost"] ?? raw["val_cost"]) as number | undefined;
  if (r.lr == null) {
    for (const key of Object.keys(raw)) {
      if (key !== "lr" && key.startsWith("lr") && typeof raw[key] === "number") { r.lr = raw[key] as number; break; }
    }
  }
  return r;
}

/**
 * Parse a stdout line as a Lightning / custom-logger metric row.
 * Accepts pure JSON and key=value pair formats.
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

function LiveChart({ metrics }: { metrics: TrainingMetricsRow[] }) {
  const option = useMemo(() => {
    const xs = metrics.map((_, i) => i + 1);
    const trainLoss = metrics.map((m) => m.train_loss ?? null);
    const valLoss = metrics.map((m) => m.val_loss ?? null);
    const reward = metrics.map((m) => m.reward ?? null);
    const hasReward = reward.some((v) => v !== null);

    return {
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" as const },
      legend: {
        data: ["train_loss", "val_loss", ...(hasReward ? ["reward"] : [])],
        textStyle: { color: "#9090b0", fontSize: 10 },
        top: 0,
      },
      grid: { left: 50, right: hasReward ? 55 : 10, top: 30, bottom: 30 },
      xAxis: {
        type: "category" as const,
        data: xs,
        axisLabel: { color: "#9090b0", fontSize: 9 },
        name: "update",
        nameTextStyle: { color: "#9090b0" },
      },
      yAxis: [
        {
          type: "value" as const,
          name: "Loss",
          nameTextStyle: { color: "#9090b0", fontSize: 9 },
          axisLabel: { color: "#9090b0", fontSize: 9 },
        },
        ...(hasReward
          ? [{
              type: "value" as const,
              name: "Reward",
              nameTextStyle: { color: "#9090b0", fontSize: 9 },
              axisLabel: { color: "#9090b0", fontSize: 9 },
              splitLine: { show: false },
            }]
          : []),
      ],
      series: [
        {
          name: "train_loss",
          type: "line" as const,
          smooth: true,
          symbol: "none",
          color: "#6366f1",
          data: trainLoss,
        },
        {
          name: "val_loss",
          type: "line" as const,
          smooth: true,
          symbol: "none",
          lineStyle: { type: "dashed" as const },
          color: "#34d399",
          data: valLoss,
        },
        ...(hasReward
          ? [{
              name: "reward",
              type: "line" as const,
              smooth: true,
              symbol: "none",
              yAxisIndex: 1,
              lineStyle: { type: "dotted" as const },
              color: "#fbbf24",
              data: reward,
            }]
          : []),
      ],
    };
  }, [metrics]);

  return <ReactECharts option={option} style={{ height: 200 }} />;
}

// Compact single-metric sparkline for grad_norm / entropy
function MiniSparkline({
  metrics,
  metricKey,
  label,
  color,
}: {
  metrics: TrainingMetricsRow[];
  metricKey: keyof TrainingMetricsRow;
  label: string;
  color: string;
}) {
  const data = metrics.map((m) => m[metricKey] ?? null);
  if (data.every((v) => v === null)) return null;
  return (
    <div>
      <p className="text-xs text-canvas-muted mb-1">{label}</p>
      <ReactECharts
        option={{
          backgroundColor: "transparent",
          grid: { left: 40, right: 10, top: 5, bottom: 20 },
          xAxis: {
            type: "category",
            data: metrics.map((_, i) => i + 1),
            axisLabel: { show: false },
            axisTick: { show: false },
            axisLine: { lineStyle: { color: "#3a3a4a" } },
          },
          yAxis: {
            type: "value",
            axisLabel: { color: "#9090b0", fontSize: 9 },
            splitLine: { lineStyle: { color: "#2a2a3a" } },
          },
          series: [{
            type: "line",
            smooth: true,
            symbol: "none",
            data,
            lineStyle: { color, width: 1.5 },
            areaStyle: { color: `${color}22` },
          }],
          tooltip: { trigger: "axis", axisPointer: { type: "line" } },
        }}
        style={{ height: 70 }}
      />
    </div>
  );
}

export function TrainingHub() {
  const { projectRoot, setMode } = useAppStore();
  const { spawn, launching } = useSpawnProcess();

  // Persisted form state (§D.4 session persistence)
  const {
    trainMode: mode, problem, seed, wandb, extraOverrides,
    model, encoder, batchSize, epochs,
    hpoTrials, hpoMethod, hpoWorkers,
    checkpointPath, evalDataset, evalSamples, evalStrategy,
    patch,
  } = useTrainHubStore();

  const setTrainMode = (v: Mode) => patch({ trainMode: v });
  const setProblem = (v: string) => patch({ problem: v });
  const setSeed = (v: number) => patch({ seed: v });
  const setWandb = (v: boolean) => patch({ wandb: v });
  const setExtraOverrides = (v: string) => patch({ extraOverrides: v });
  const setModel = (v: string) => patch({ model: v });
  const setEncoder = (v: string) => patch({ encoder: v });
  const setBatchSize = (v: number) => patch({ batchSize: v });
  const setEpochs = (v: number) => patch({ epochs: v });
  const setHpoTrials = (v: number) => patch({ hpoTrials: v });
  const setHpoMethod = (v: string) => patch({ hpoMethod: v });
  const setHpoWorkers = (v: number) => patch({ hpoWorkers: v });
  const setCheckpointPath = (v: string) => patch({ checkpointPath: v });
  const setEvalDataset = (v: string) => patch({ evalDataset: v });
  const setEvalSamples = (v: number) => patch({ evalSamples: v });
  const setEvalStrategy = (v: string) => patch({ evalStrategy: v });

  // Ephemeral UI state
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Live progress tracking
  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<ProcessStatus | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<TrainingMetricsRow[]>([]);

  const pickCheckpoint = async () => {
    const path = (await open({
      filters: [{ name: "Checkpoint", extensions: ["pt", "ckpt", "pth"] }],
    })) as string | null;
    if (path) setCheckpointPath(path);
  };

  const pickDataset = async () => {
    const path = (await open({
      filters: [{ name: "Dataset", extensions: ["pkl", "json", "csv"] }],
    })) as string | null;
    if (path) setEvalDataset(path);
  };

  const hydraArgs = useMemo(() => {
    const base = [`seed=${seed}`];
    if (!wandb) base.push("tracker.enabled=false");

    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);

    if (mode === "train") {
      return [
        ...base,
        `envs=${problem}`,
        `models=${model}`,
        `model.encoder.type=${encoder}`,
        `train.batch_size=${batchSize}`,
        `train.max_epochs=${epochs}`,
        `hpo.n_trials=0`,
        ...extra,
      ];
    }
    if (mode === "hpo") {
      return [
        ...base,
        `envs=${problem}`,
        `models=${model}`,
        `model.encoder.type=${encoder}`,
        `hpo.n_trials=${hpoTrials}`,
        `hpo.method=${hpoMethod}`,
        `hpo.num_workers=${hpoWorkers}`,
        ...extra,
      ];
    }
    return [
      ...base,
      ...(checkpointPath ? [`eval.policy.model.load_path=${checkpointPath}`] : []),
      ...(evalDataset ? [`eval.datasets=[${evalDataset}]`] : []),
      `eval.problem=${problem}`,
      `eval.val_size=${evalSamples}`,
      `eval.decoding.strategy=${evalStrategy}`,
      ...extra,
    ];
  }, [
    mode, seed, wandb, problem, model, encoder, batchSize, epochs,
    hpoTrials, hpoMethod, hpoWorkers,
    checkpointPath, evalDataset, evalSamples, evalStrategy,
    extraOverrides,
  ]);

  const entrypoint = mode === "train" ? "train" : mode === "hpo" ? "train" : "eval";
  const commandPreview = `python main.py ${entrypoint} \\\n  ${hydraArgs.join(" \\\n  ")}`;

  // Subscribe to live progress events for the active training run
  useEffect(() => {
    if (!liveProcessId) return;

    let unlistenOut: (() => void) | null = null;
    let unlistenStatus: (() => void) | null = null;

    listen<StdoutLine>("process:stdout", (event) => {
      const { id, line } = event.payload;
      if (id !== liveProcessId) return;
      const row = parseMetricLine(line);
      if (row) setLiveMetrics((prev) => [...prev, row]);
    }).then((fn) => { unlistenOut = fn; });

    listen<StatusUpdate>("process:status", (event) => {
      if (event.payload.id === liveProcessId) setRunStatus(event.payload.status);
    }).then((fn) => { unlistenStatus = fn; });

    return () => { unlistenOut?.(); unlistenStatus?.(); };
  }, [liveProcessId]);

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const procId = `${mode}_${Date.now()}`;
    setLiveProcessId(procId);
    setLiveMetrics([]);
    setRunStatus(null);
    await spawn({
      id: procId,
      pythonArgs: ["main.py", entrypoint, ...hydraArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, mode, entrypoint, hydraArgs, spawn]);

  const trainNonce = useLaunchTriggerStore((s) => s.trainNonce);
  useEffect(() => {
    if (trainNonce > 0) launch();
  }, [trainNonce, launch]);

  function SelectField<T extends string>({
    label, value, onChange, options,
  }: {
    label: string;
    value: T;
    onChange: (v: T) => void;
    options: readonly T[];
  }) {
    return (
      <div className="flex flex-col gap-1">
        <label className="text-xs text-canvas-muted">{label}</label>
        <select
          className="select-base w-36"
          value={value}
          onChange={(e) => onChange(e.target.value as T)}
        >
          {options.map((o) => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
      </div>
    );
  }

  const isDone = runStatus !== null && runStatus !== "running";
  const latestMetric = liveMetrics[liveMetrics.length - 1];

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Mode selector */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Mode</h2>
        <div className="flex gap-2">
          {(["train", "hpo", "eval"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setTrainMode(m)}
              className={mode === m ? "btn-primary text-sm py-1.5 px-4" : "btn-ghost text-sm py-1.5 px-4"}
            >
              {m === "train" ? "Train" : m === "hpo" ? "HPO Sweep" : "Evaluate"}
            </button>
          ))}
        </div>
      </div>

      {/* Shared: problem + seed + WandB */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Common</h2>
        <div className="flex flex-wrap gap-4 items-end">
          <SelectField label="Problem" value={problem} onChange={setProblem} options={PROBLEMS} />
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">Seed</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={seed}
              min={0}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>
          <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-300 self-end pb-2">
            <input
              type="checkbox"
              className="accent-accent-primary"
              checked={wandb}
              onChange={(e) => setWandb(e.target.checked)}
            />
            WandB logging
          </label>
        </div>
      </div>

      {/* Mode-specific form */}
      {(mode === "train" || mode === "hpo") && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">
            {mode === "train" ? "Model Architecture" : "HPO Configuration"}
          </h2>
          <div className="flex flex-wrap gap-4 items-end">
            <SelectField label="Model" value={model} onChange={setModel} options={MODELS} />
            <SelectField label="Encoder" value={encoder} onChange={setEncoder} options={ENCODERS} />
            {mode === "train" && (
              <>
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-canvas-muted">Batch Size</label>
                  <input type="number" className="input-base font-mono text-sm w-24" value={batchSize} min={1} onChange={(e) => setBatchSize(Number(e.target.value))} />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-canvas-muted">Max Epochs</label>
                  <input type="number" className="input-base font-mono text-sm w-24" value={epochs} min={1} onChange={(e) => setEpochs(Number(e.target.value))} />
                </div>
              </>
            )}
            {mode === "hpo" && (
              <>
                <SelectField label="Method" value={hpoMethod} onChange={setHpoMethod} options={HPO_METHODS} />
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-canvas-muted">Trials</label>
                  <input type="number" className="input-base font-mono text-sm w-24" value={hpoTrials} min={1} onChange={(e) => setHpoTrials(Number(e.target.value))} />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-canvas-muted">Workers</label>
                  <input type="number" className="input-base font-mono text-sm w-20" value={hpoWorkers} min={1} onChange={(e) => setHpoWorkers(Number(e.target.value))} />
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {mode === "eval" && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">Evaluation</h2>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">Checkpoint</label>
            <div className="flex gap-2">
              <input
                type="text"
                className="input-base font-mono text-xs flex-1"
                value={checkpointPath}
                onChange={(e) => setCheckpointPath(e.target.value)}
                placeholder="path/to/best.pt"
              />
              <button onClick={pickCheckpoint} className="btn-ghost text-xs flex items-center gap-1">
                <FolderOpen size={12} />
              </button>
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">Dataset</label>
            <div className="flex gap-2">
              <input
                type="text"
                className="input-base font-mono text-xs flex-1"
                value={evalDataset}
                onChange={(e) => setEvalDataset(e.target.value)}
                placeholder="path/to/dataset.pkl"
              />
              <button onClick={pickDataset} className="btn-ghost text-xs flex items-center gap-1">
                <FolderOpen size={12} />
              </button>
            </div>
          </div>
          <div className="flex flex-wrap gap-4 items-end">
            <SelectField label="Strategy" value={evalStrategy} onChange={setEvalStrategy} options={EVAL_STRATEGIES} />
            <div className="flex flex-col gap-1">
              <label className="text-xs text-canvas-muted">Val Samples</label>
              <input type="number" className="input-base font-mono text-sm w-24" value={evalSamples} min={1} onChange={(e) => setEvalSamples(Number(e.target.value))} />
            </div>
          </div>
        </div>
      )}

      {/* Advanced */}
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
            placeholder="some.param=value"
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
          disabled={launching || !projectRoot}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching ? "Launching…" : `Start ${mode}`}
        </button>
      </div>

      {/* Live progress panel */}
      {liveProcessId && (
        <div className="card space-y-3">
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
                  ? runStatus === "completed" ? "Run Complete" : `Run ${runStatus}`
                  : "Live Progress"}
              </h2>
              {liveMetrics.length > 0 && (
                <span className="text-xs text-canvas-muted">{liveMetrics.length} updates</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {isDone && runStatus === "completed" && (
                <button
                  onClick={() => setMode("output_browser")}
                  className="btn-ghost text-xs text-accent-success"
                >
                  Output Browser →
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

          {/* Latest metric snapshot */}
          {latestMetric && (
            <div className="flex flex-wrap gap-4 text-xs">
              {latestMetric.epoch != null && (
                <div>
                  <span className="text-canvas-muted">Epoch </span>
                  <span className="font-mono text-gray-200">{latestMetric.epoch}</span>
                </div>
              )}
              {latestMetric.train_loss != null && (
                <div>
                  <span className="text-canvas-muted">Train loss </span>
                  <span className="font-mono text-gray-200">{latestMetric.train_loss.toFixed(4)}</span>
                </div>
              )}
              {latestMetric.val_loss != null && (
                <div>
                  <span className="text-canvas-muted">Val loss </span>
                  <span className="font-mono text-gray-200">{latestMetric.val_loss.toFixed(4)}</span>
                </div>
              )}
              {latestMetric.reward != null && (
                <div>
                  <span className="text-canvas-muted">Reward </span>
                  <span className="font-mono text-accent-success">{latestMetric.reward.toFixed(4)}</span>
                </div>
              )}
              {latestMetric.grad_norm != null && (
                <div>
                  <span className="text-canvas-muted">‖∇‖ </span>
                  <span className="font-mono text-gray-200">{latestMetric.grad_norm.toFixed(3)}</span>
                </div>
              )}
            </div>
          )}

          {liveMetrics.length >= 2 ? (
            <LiveChart metrics={liveMetrics} />
          ) : (
            <p className="text-xs text-canvas-muted">
              {isDone
                ? liveMetrics.length === 0
                  ? "No JSON metric lines detected in stdout."
                  : "Only one metric update received."
                : "Waiting for metric JSON lines on stdout…"}
            </p>
          )}

          {/* Auxiliary sparklines: grad_norm and entropy */}
          {liveMetrics.length >= 2 && (
            <div className="grid grid-cols-2 gap-4">
              <MiniSparkline
                metrics={liveMetrics}
                metricKey="grad_norm"
                label="Gradient Norm ‖∇‖"
                color="#f87171"
              />
              <MiniSparkline
                metrics={liveMetrics}
                metricKey="entropy"
                label="Policy Entropy"
                color="#a78bfa"
              />
            </div>
          )}

          <p className="text-xs text-canvas-muted font-mono truncate">{liveProcessId}</p>
        </div>
      )}
    </div>
  );
}

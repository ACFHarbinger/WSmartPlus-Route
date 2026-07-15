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
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Play, ChevronDown, ChevronUp, Terminal, FolderOpen } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { PathHandoffButtons } from "../../components/common/PathHandoffButtons";
import { PathRunLabelChip } from "../../components/common/PathRunLabelChip";
import { parentRunBrushLabelFromCheckpointPath } from "../../utils/checkpoints";
import { EvalCheckpointLiveCard } from "../../components/monitor/EvalCheckpointLiveCard";
import { EvalResultCard } from "../../components/monitor/EvalResultCard";
import { LauncherLivePanel } from "../../components/monitor/LauncherLivePanel";
import { TrainHpoLivePanel } from "../../components/monitor/TrainHpoLivePanel";
import { ProcessIdFooter } from "../../components/monitor/ProcessIdFooter";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { open } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useLaunchTriggerStore } from "../../store/launchTrigger";
import { useTrainHubStore } from "../../store/launchers";
import { useProcessStore } from "../../store/process";
import { useRecentHandoff } from "../../hooks/useRecentHandoff";
import { useSpawnProcess } from "../../hooks/useSpawnProcess";
import { brushLogPathFromProcessLines, outputRunPathFromLogLines } from "../../utils/outputRunPath";
import { trainingRunPathFromLogLines } from "../../utils/trainingRunPath";
import { collectAttentionVizFromLogLines } from "../../utils/attentionViz";
import { collectTrainingHealthFromLogLines } from "../../utils/trainingHealth";
import { collectTrainingMetricsFromLogLines } from "../../utils/trainingMetrics";
import {
  checkpointLabelFromEvalProcess,
  collectEvalResultFromLogLines,
  evalLivePanelTitle,
  hasEvalMetrics,
  toEvalAnalyticsRows,
} from "../../utils/evalResults";
import { useProcessRunLabelBrush } from "../../hooks/useProcessRunLabelBrush";
import { findRecentLauncherProcessId } from "../../utils/launcherProcess";
import {
  findRecentHpoProcessId,
  findRecentTrainProcessId,
  trainHpoLivePanelTitle,
} from "../../utils/trainingProcess";
import type { ProcessEntry, TrainingMetricsRow } from "../../types";

type Mode = "train" | "hpo" | "eval";

function findRecentHubProcessId(
  processes: Record<string, ProcessEntry>,
  hubMode: Mode
): string | null {
  if (hubMode === "hpo") return findRecentHpoProcessId(processes);
  if (hubMode === "eval") return findRecentLauncherProcessId(processes, "eval");
  return findRecentTrainProcessId(processes);
}

const PROBLEMS = ["vrpp", "wcvrp", "scwcvrp"] as const;
const MODELS = ["am", "tam", "ddam", "moe"] as const;
const ENCODERS = ["gat", "gcn", "mha"] as const;
const HPO_METHODS = ["nsgaii", "tpe", "dehb", "random"] as const;
const EVAL_STRATEGIES = ["greedy", "sampling", "beam"] as const;

function LiveChart({
  metrics,
  logScale,
}: {
  metrics: TrainingMetricsRow[];
  logScale: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const option = useMemo(() => {
    const xs = metrics.map((_, i) => i + 1);
    const scaleLoss = (v: number | null | undefined) =>
      v == null ? null : logScale ? Math.max(v, 1e-8) : v;
    const trainLoss = metrics.map((m) => scaleLoss(m.train_loss));
    const valLoss = metrics.map((m) => scaleLoss(m.val_loss));
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
          type: (logScale ? "log" : "value") as "log" | "value",
          logBase: 10,
          name: logScale ? "Loss (log)" : "Loss",
          nameTextStyle: { color: "#9090b0", fontSize: 9 },
          axisLabel: { color: "#9090b0", fontSize: 9 },
          minorSplitLine: { show: false },
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
  }, [metrics, logScale]);

  return (
    <div className="space-y-1">
      <p className="text-[10px] text-canvas-muted">
        {logScale
          ? "Log-scale loss axis — reward stays linear on right axis"
          : "Linear loss · reward on right axis when present"}
      </p>
      <div className="flex justify-end">
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="training-live"
        />
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 200 }} />
    </div>
  );
}

export function TrainingHub() {
  const { effectiveTheme, setMode, setPendingEvalResults } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
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

  // Live progress tracking (local id set on launch; falls back to process store after navigation)
  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const processes = useProcessStore((s) => s.processes);
  const displayProcessId = useMemo(
    () => liveProcessId ?? findRecentHubProcessId(processes, mode),
    [liveProcessId, processes, mode]
  );
  const displayProc = displayProcessId ? processes[displayProcessId] : null;
  const runStatus = displayProc?.status ?? null;

  const pickCheckpoint = async () => {
    const path = (await open({
      filters: [{ name: "Checkpoint", extensions: ["pt", "ckpt", "pth"] }],
    })) as string | null;
    if (!path) return;
    handoff(path, "checkpoint", { navigate: false });
    setCheckpointPath(path);
  };

  const pickDataset = async () => {
    const path = (await open({
      filters: [{ name: "Dataset", extensions: ["pkl", "json", "csv"] }],
    })) as string | null;
    if (!path) return;
    // CSV datasets are reopenable in Data Explorer via Command Palette recents (§G.6 / §G.10).
    if (/\.csv$/i.test(path)) {
      handoff(path, "csv", { navigate: false });
    }
    setEvalDataset(path);
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

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const procId = `${mode}_${Date.now()}`;
    setLiveProcessId(procId);
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

  const liveLogLines = displayProc?.logLines ?? [];
  const liveRunLabel = useProcessRunLabelBrush(displayProcessId, liveLogLines);
  const liveMetrics = useMemo(
    () => collectTrainingMetricsFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const liveHealth = useMemo(
    () => collectTrainingHealthFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const liveAttention = useMemo(
    () => collectAttentionVizFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const outputRunPath = useMemo(
    () => outputRunPathFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const trainingRunPath = useMemo(
    () => trainingRunPathFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const liveLogPath = useMemo(() => {
    const kind = mode === "eval" ? "sim" : "train";
    return brushLogPathFromProcessLines(liveLogLines, kind);
  }, [liveLogLines, mode]);

  const isDone = runStatus !== null && runStatus !== "running";
  const latestMetric = liveMetrics[liveMetrics.length - 1];
  const showTrainingAnalytics =
    displayProcessId != null && (mode === "train" || mode === "hpo");
  const trainHpoLiveTitle = trainHpoLivePanelTitle({
    isRunning: !isDone,
    status: runStatus ?? undefined,
    processId: displayProcessId ?? undefined,
    command: displayProc?.command,
    kind: mode === "hpo" ? "hpo" : mode === "train" ? "train" : undefined,
  });

  const evalCheckpointName = useMemo(() => {
    if (!displayProcessId || !displayProc) return checkpointPath.split(/[/\\]/).pop() ?? "checkpoint";
    return checkpointLabelFromEvalProcess(displayProcessId, displayProc.command ?? "");
  }, [displayProcessId, displayProc, checkpointPath]);

  const evalResult = useMemo(() => {
    if (mode !== "eval" || !displayProc) return null;
    const result = collectEvalResultFromLogLines(liveLogLines, evalCheckpointName);
    const trimmed = checkpointPath.trim();
    if (trimmed) result.checkpointPath = trimmed;
    return result;
  }, [mode, displayProc, liveLogLines, evalCheckpointName, checkpointPath]);

  const openEvalInAnalytics = useCallback(() => {
    if (!evalResult || !hasEvalMetrics(evalResult)) return;
    setPendingEvalResults(toEvalAnalyticsRows([evalResult]));
    setMode("benchmark");
  }, [evalResult, setPendingEvalResults, setMode]);

  const evalLiveTitle = evalLivePanelTitle({
    isRunning: !isDone,
    status: runStatus ?? undefined,
  });

  const completedEvalCheckpointPath =
    mode === "eval" && isDone && runStatus === "completed" ? checkpointPath || null : null;

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
            {checkpointPath.trim() ? (
              <PathRunLabelChip
                path={checkpointPath.trim()}
                projectRoot={projectRoot}
                brushLabel={parentRunBrushLabelFromCheckpointPath(
                  checkpointPath.trim(),
                  projectRoot
                )}
                className="max-w-full"
                trailing={
                  <PathHandoffButtons
                    path={checkpointPath.trim()}
                    kind="checkpoint"
                    iconSize={11}
                  />
                }
              />
            ) : null}
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
            {evalDataset.trim() ? (
              <PathRunLabelChip
                path={evalDataset.trim()}
                projectRoot={projectRoot}
                className="max-w-full"
              />
            ) : null}
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
      {displayProcessId && (
        <GlobalFilterBar
          showLogScale
          runLabels={liveRunLabel ? [liveRunLabel] : []}
        />
      )}

      {displayProcessId && mode === "eval" && displayProc && (
        <LauncherLivePanel
          header={{
            status: isDone ? runStatus ?? "running" : "running",
            title: evalLiveTitle,
            runLabel: liveRunLabel,
            logPath: liveLogPath,
            projectRoot,
            navMesh: {
              kind: "eval",
              hideSelf: true,
              hideHub: true,
              showPostRun: isDone && runStatus === "completed",
              showOutputBrowser: isDone && runStatus === "completed",
              outputRunPath,
              checkpointPath: completedEvalCheckpointPath,
              onOpenAnalytics:
                evalResult && hasEvalMetrics(evalResult) ? openEvalInAnalytics : undefined,
            },
          }}
          footer={
            <ProcessIdFooter
              processId={displayProcessId}
              logPath={liveLogPath}
              projectRoot={projectRoot}
            />
          }
          logLines={liveLogLines}
          logTailWaiting={!isDone}
        >
          {!isDone || !evalResult || !hasEvalMetrics(evalResult) ? (
            <EvalCheckpointLiveCard
              procId={displayProcessId}
              checkpointName={evalCheckpointName}
              checkpointPath={completedEvalCheckpointPath ?? (checkpointPath.trim() || null)}
              projectRoot={projectRoot}
              status={displayProc.status}
              isRunning={!isDone}
              result={evalResult ?? undefined}
              logLines={liveLogLines}
              showLogTail={false}
            />
          ) : (
            <EvalResultCard
              result={evalResult}
              projectRoot={projectRoot}
              onOpenAnalytics={openEvalInAnalytics}
            />
          )}
        </LauncherLivePanel>
      )}

      {displayProcessId && mode !== "eval" && (
        <TrainHpoLivePanel
          header={{
            status: isDone ? runStatus : "running",
            title: trainHpoLiveTitle,
            runLabel: liveRunLabel,
            logPath: liveLogPath,
            projectRoot,
            showLiveSuffix: !isDone,
            metricCount: liveMetrics.length,
            healthCount: liveHealth.length,
            attentionCount: liveAttention.length,
            layout: "split",
            runningIcon: "activity",
            titleTone: "heading",
            navMesh: {
              hideHub: true,
              showTrainLinks: showTrainingAnalytics,
              showHpoLinks: mode === "hpo",
              showOutputBrowser: isDone && runStatus === "completed",
              outputRunPath,
              trainingRunPath,
            },
          }}
          progress={
            !isDone && displayProcessId
              ? {
                  processId: displayProcessId,
                  fallbackTotal: mode === "train" ? epochs : undefined,
                  fallbackValue: latestMetric?.epoch,
                }
              : undefined
          }
          showAnalytics={showTrainingAnalytics}
          analyticsWrapperClassName={isDone ? undefined : "pt-0"}
          analytics={{
            metrics: liveMetrics,
            healthEntries: liveHealth,
            attentionEntries: liveAttention,
            logScale,
            theme: effectiveTheme,
            exportNamePrefix: "training-hub",
            isPostRun: isDone,
            postRunFallback:
              "Post-run shortcuts — open Training Monitor or Output Browser for this run",
            middleContent: (
              <>
                {liveMetrics.length >= 2 && <GlobalFilterBar showLogScale />}
                {liveMetrics.length >= 2 ? (
                  <LiveChart metrics={liveMetrics} logScale={logScale} />
                ) : (
                  <p className="text-xs text-canvas-muted">
                    {isDone
                      ? liveMetrics.length === 0
                        ? "No JSON metric lines detected in stdout."
                        : "Only one metric update received."
                      : "Waiting for metric JSON lines on stdout…"}
                  </p>
                )}
              </>
            ),
          }}
          logLines={liveLogLines}
          logTailWaiting={!isDone}
          footer={
            <ProcessIdFooter
              processId={displayProcessId}
              logPath={liveLogPath}
              projectRoot={projectRoot}
            />
          }
        />
      )}
    </div>
  );
}

/**
 * Evaluation Runner — multi-checkpoint evaluation and comparison (§G.12).
 *
 * Spawns one `main.py eval` process per checkpoint so results can be compared
 * side-by-side in the Process Monitor. Each run is tagged with the checkpoint
 * filename so logs are easy to identify.
 *
 * Hydra args mirror the controller justfile eval recipe:
 *   eval.policy.model.load_path, eval.datasets, eval.problem,
 *   eval.val_size, eval.decoding.strategy
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import {
  BarChart3,
  ChevronDown,
  ChevronUp,
  Download,
  FolderOpen,
  Play,
  Plus,
  Terminal,
  Trash2,
} from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { EvalCheckpointLiveCard } from "../../components/monitor/EvalCheckpointLiveCard";
import { LauncherLivePanel } from "../../components/monitor/LauncherLivePanel";
import { ProcessIdFooter } from "../../components/monitor/ProcessIdFooter";
import { open } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useLaunchTriggerStore } from "../../store/launchTrigger";
import { useProcessStore } from "../../store/process";
import { useSpawnProcess } from "../../hooks/useSpawnProcess";
import type { ProcessStatus } from "../../types";
import {
  type EvalResult,
  checkpointLabelFromEvalProcess,
  collectEvalResultFromLogLines,
  evalLivePanelTitle,
  hasEvalMetrics,
  toEvalAnalyticsRows,
} from "../../utils/evalResults";
import { findRecentEvalProcessIds } from "../../utils/launcherProcess";
import { outputRunPathFromLogLines } from "../../utils/outputRunPath";

const PROBLEMS = ["vrpp", "wcvrp", "scwcvrp"] as const;
const STRATEGIES = ["greedy", "sampling", "beam"] as const;
const DEVICES = ["cpu", "cuda:0", "cuda:1"] as const;

interface CheckpointEntry {
  id: string;
  path: string;
}

const EVAL_CHART_METRICS = [
  { key: "cost", label: "Tour Cost" },
  { key: "gap", label: "Optimality Gap (%)" },
  { key: "time", label: "Time (s)" },
] as const;

const EVAL_COLORS = ["#6366f1", "#34d399", "#fbbf24", "#f87171", "#818cf8", "#a3e635"];

function ResultsGrid({
  results,
  logScale,
  onOpenAnalytics,
}: {
  results: EvalResult[];
  logScale: boolean;
  onOpenAnalytics: () => void;
}) {
  const chartRefs = useRef<Record<string, EChartsReact | null>>({});
  const allKeys = Array.from(
    new Set(results.flatMap((r) => Object.keys(r).filter((k) => k !== "checkpointName")))
  );
  const numKeys = allKeys.filter((k) =>
    results.some((r) => typeof r[k] === "number")
  );
  const checkpoints = results.map((r) => r.checkpointName);

  const makeBarOption = (metricKey: string, metricLabel: string) => ({
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 20, bottom: 55 },
    xAxis: {
      type: "category",
      data: checkpoints,
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 25 },
    },
    yAxis: {
      type: (logScale ? "log" : "value") as "log" | "value",
      logBase: 10,
      name: metricLabel,
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
      minorSplitLine: { show: false },
    },
    series: [
      {
        type: "bar",
        data: results.map((r, i) => {
          const raw = (r[metricKey] as number | undefined) ?? 0;
          return {
            value: logScale ? Math.max(raw, 0.001) : raw,
            itemStyle: { color: EVAL_COLORS[i % EVAL_COLORS.length] },
          };
        }),
      },
    ],
    tooltip: { trigger: "axis" },
  });

  const exportCsv = () => {
    const cols = ["checkpoint", ...numKeys];
    const rows = results.map((r) =>
      cols.map((c) => (c === "checkpoint" ? r.checkpointName : String(r[c] ?? ""))).join(",")
    );
    const csv = [cols.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "eval_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-3">
      <GlobalFilterBar showLogScale />

      <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-200">Results ({results.length})</h2>
        <div className="flex items-center gap-2">
          <button onClick={onOpenAnalytics} className="btn-ghost text-xs flex items-center gap-1">
            <BarChart3 size={12} />
            Open in Analytics →
          </button>
          <button onClick={exportCsv} className="btn-ghost text-xs flex items-center gap-1">
            <Download size={12} />
            Export CSV
          </button>
        </div>
      </div>
      <p className="text-[10px] text-canvas-muted">
        {logScale
          ? "Log-scale bars — tour cost · optimality gap · eval time per checkpoint"
          : "Linear bars — tour cost · optimality gap · eval time per checkpoint"}
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {EVAL_CHART_METRICS.filter(({ key }) => numKeys.includes(key)).map(({ key, label }) => (
          <div key={key} className="rounded-lg border border-canvas-border/40 p-2">
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs text-canvas-muted">{label}</p>
              <ChartExportButtons
                chartRef={{ current: chartRefs.current[key] }}
                filenameStem={`eval-runner-${key}`}
                size={10}
              />
            </div>
            <ReactECharts
              ref={(el) => {
                chartRefs.current[key] = el;
              }}
              option={makeBarOption(key, label)}
              style={{ height: 180 }}
            />
          </div>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-canvas-border">
              <th className="text-left py-1.5 pr-4 text-canvas-muted font-medium">Checkpoint</th>
              {numKeys.map((k) => (
                <th key={k} className="text-right py-1.5 px-3 text-canvas-muted font-medium">
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-canvas-border/30">
            {results.map((r) => (
              <tr key={r.checkpointName} className="hover:bg-canvas-hover/40">
                <td className="py-1.5 pr-4 font-mono text-gray-300 truncate max-w-[200px]">
                  {r.checkpointName}
                </td>
                {numKeys.map((k) => (
                  <td key={k} className="py-1.5 px-3 text-right font-mono text-gray-400">
                    {typeof r[k] === "number"
                      ? (r[k] as number).toFixed(4)
                      : "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </div>
    </div>
  );
}

function CheckpointRow({
  entry,
  onRemove,
  onPickFile,
  onChange,
}: {
  entry: CheckpointEntry;
  onRemove: () => void;
  onPickFile: () => void;
  onChange: (path: string) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="text"
        className="input-base font-mono text-xs flex-1"
        value={entry.path}
        onChange={(e) => onChange(e.target.value)}
        placeholder="path/to/checkpoint.pt"
      />
      <button
        onClick={onPickFile}
        className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-200"
        title="Browse"
      >
        <FolderOpen size={13} />
      </button>
      <button
        onClick={onRemove}
        className="btn-ghost p-1.5 text-accent-danger/60 hover:text-accent-danger"
        title="Remove"
      >
        <Trash2 size={13} />
      </button>
    </div>
  );
}

export function EvaluationRunner() {
  const { projectRoot, pendingCheckpoint, setPendingCheckpoint, setMode, setPendingEvalResults } =
    useAppStore();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const { spawn, launching } = useSpawnProcess();

  // Checkpoint list — pre-populated from Training Monitor "Load in Eval Runner" action
  const [checkpoints, setCheckpoints] = useState<CheckpointEntry[]>([
    { id: "ckpt_0", path: "" },
  ]);

  // Consume pendingCheckpoint set by TrainingMonitor checkpoint browser
  useEffect(() => {
    if (pendingCheckpoint) {
      setCheckpoints([{ id: "ckpt_pending", path: pendingCheckpoint }]);
      setPendingCheckpoint(null);
    }
  }, [pendingCheckpoint, setPendingCheckpoint]);

  // Eval params
  const [problem, setProblem] = useState("vrpp");
  const [datasetPath, setDatasetPath] = useState("");
  const [valSize, setValSize] = useState(10);
  const [strategy, setStrategy] = useState("greedy");
  const [device, setDevice] = useState("cpu");

  // Advanced
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [extraOverrides, setExtraOverrides] = useState("");

  // Live progress state (§D.2 / §G.12); falls back to process store after navigation
  const [liveProcessIds, setLiveProcessIds] = useState<string[]>([]);
  const processes = useProcessStore((s) => s.processes);
  const displayProcessIds = useMemo(
    () =>
      liveProcessIds.length > 0
        ? liveProcessIds
        : findRecentEvalProcessIds(processes),
    [liveProcessIds, processes]
  );

  const results = useMemo(() => {
    return displayProcessIds
      .map((id) => {
        const proc = processes[id];
        const checkpointName = checkpointLabelFromEvalProcess(
          id,
          proc?.command ?? ""
        );
        const result = collectEvalResultFromLogLines(
          proc?.logLines ?? [],
          checkpointName
        );
        return hasEvalMetrics(result) ? result : null;
      })
      .filter((r): r is EvalResult => r != null);
  }, [displayProcessIds, processes]);

  const validCheckpoints = useMemo(
    () => checkpoints.filter((c) => c.path.trim() !== ""),
    [checkpoints]
  );

  const liveRunSummary = useMemo(() => {
    if (displayProcessIds.length === 0) return null;
    const statuses = displayProcessIds.map((id) => processes[id]?.status ?? "running");
    const running = statuses.filter((s) => s === "running").length;
    const completed = statuses.filter((s) => s === "completed").length;
    const failed = statuses.filter((s) => s === "failed" || s === "cancelled").length;
    const allDone = running === 0;
    const aggregate: ProcessStatus | "running" = allDone
      ? failed > 0 && completed === 0
        ? "failed"
        : failed > 0
          ? "cancelled"
          : "completed"
      : "running";
    return { running, completed, failed, allDone, aggregate };
  }, [displayProcessIds, processes]);

  const completedCheckpointPath = useMemo(() => {
    if (validCheckpoints.length !== 1) return null;
    if (!liveRunSummary?.allDone || liveRunSummary.aggregate !== "completed") return null;
    return validCheckpoints[0].path;
  }, [validCheckpoints, liveRunSummary]);

  const outputRunPath = useMemo(() => {
    const lines = displayProcessIds.flatMap((id) => processes[id]?.logLines ?? []);
    return outputRunPathFromLogLines(lines);
  }, [displayProcessIds, processes]);

  const singleCheckpointEval = displayProcessIds.length === 1;
  const singleEvalProc = singleCheckpointEval ? processes[displayProcessIds[0]] : undefined;
  const singleEvalLogLines = singleEvalProc?.logLines ?? [];
  const singleEvalLogTailWaiting = singleEvalProc?.status === "running";

  const addCheckpoint = () => {
    setCheckpoints((prev) => [
      ...prev,
      { id: `ckpt_${Date.now()}`, path: "" },
    ]);
  };

  const removeCheckpoint = (id: string) => {
    setCheckpoints((prev) => prev.filter((c) => c.id !== id));
  };

  const updateCheckpoint = (id: string, path: string) => {
    setCheckpoints((prev) => prev.map((c) => (c.id === id ? { ...c, path } : c)));
  };

  const pickCheckpointFile = async (id: string) => {
    const path = (await open({
      filters: [{ name: "Checkpoint", extensions: ["pt", "ckpt", "pth"] }],
    })) as string | null;
    if (path) updateCheckpoint(id, path);
  };

  const pickDataset = async () => {
    const path = (await open({
      filters: [{ name: "Dataset", extensions: ["pkl", "json", "csv"] }],
    })) as string | null;
    if (path) setDatasetPath(path);
  };

  // Show preview for first checkpoint
  const previewArgs = useMemo(() => {
    const ckpt = validCheckpoints[0];
    const base = [
      `eval.problem=${problem}`,
      `eval.val_size=${valSize}`,
      `eval.decoding.strategy=${strategy}`,
      `eval.device=${device}`,
      ...(ckpt ? [`eval.policy.model.load_path=${ckpt.path}`] : []),
      ...(datasetPath ? [`eval.datasets=[${datasetPath}]`] : []),
    ];
    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    return [...base, ...extra];
  }, [problem, valSize, strategy, device, validCheckpoints, datasetPath, extraOverrides]);

  const commandPreview =
    validCheckpoints.length > 1
      ? `# ${validCheckpoints.length} checkpoints — one process per checkpoint\npython main.py eval \\\n  ${previewArgs.join(" \\\n  ")}`
      : `python main.py eval \\\n  ${previewArgs.join(" \\\n  ")}`;

  const openInAnalytics = useCallback(() => {
    setPendingEvalResults(toEvalAnalyticsRows(results));
    setMode("benchmark");
  }, [results, setPendingEvalResults, setMode]);

  const launch = useCallback(async () => {
    if (!projectRoot || validCheckpoints.length === 0) return;

    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);

    const launchedIds: string[] = [];
    const baseTs = Date.now();

    for (const [idx, ckpt] of validCheckpoints.entries()) {
      const ckptName = ckpt.path.split(/[/\\]/).pop() ?? ckpt.id;
      const procId = `eval_${ckptName}_${baseTs}_${idx}`;
      launchedIds.push(procId);
      const hydraArgs = [
        `eval.problem=${problem}`,
        `eval.val_size=${valSize}`,
        `eval.decoding.strategy=${strategy}`,
        `eval.device=${device}`,
        `eval.policy.model.load_path=${ckpt.path}`,
        ...(datasetPath ? [`eval.datasets=[${datasetPath}]`] : []),
        ...extra,
      ];
      await spawn({
        id: procId,
        pythonArgs: ["main.py", "eval", ...hydraArgs],
        workingDir: projectRoot,
      });
    }

    setLiveProcessIds(launchedIds);
  }, [
    projectRoot, validCheckpoints, problem, valSize, strategy, device,
    datasetPath, extraOverrides, spawn,
  ]);

  const evalNonce = useLaunchTriggerStore((s) => s.evalNonce);
  useEffect(() => {
    if (evalNonce > 0) launch();
  }, [evalNonce, launch]);

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Checkpoints */}
      <div className="card space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">Checkpoints</h2>
          <button
            onClick={addCheckpoint}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <Plus size={12} />
            Add
          </button>
        </div>

        <div className="space-y-2">
          {checkpoints.map((c) => (
            <CheckpointRow
              key={c.id}
              entry={c}
              onRemove={() => removeCheckpoint(c.id)}
              onPickFile={() => pickCheckpointFile(c.id)}
              onChange={(path) => updateCheckpoint(c.id, path)}
            />
          ))}
        </div>

        {validCheckpoints.length > 1 && (
          <p className="text-xs text-canvas-muted">
            {validCheckpoints.length} checkpoints — each will be evaluated in a separate process.
          </p>
        )}
      </div>

      {/* Eval parameters */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Evaluation</h2>

        {/* Dataset */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs text-canvas-muted">Dataset (optional — uses config default if empty)</label>
          <div className="flex gap-2">
            <input
              type="text"
              className="input-base font-mono text-xs flex-1"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              placeholder="path/to/dataset.pkl"
            />
            <button onClick={pickDataset} className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-200">
              <FolderOpen size={13} />
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-4">
          {/* Problem */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Problem</label>
            <select
              className="select-base w-36"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
            >
              {PROBLEMS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>

          {/* Strategy */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Decoding Strategy</label>
            <select
              className="select-base w-36"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
            >
              {STRATEGIES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          {/* Device */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Device</label>
            <select
              className="select-base w-28"
              value={device}
              onChange={(e) => setDevice(e.target.value)}
            >
              {DEVICES.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          {/* Val size */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Val Samples</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={valSize}
              min={1}
              onChange={(e) => setValSize(Number(e.target.value))}
            />
          </div>
        </div>
      </div>

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
            placeholder="eval.batch_size=128"
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
          {validCheckpoints.length > 0 ? commandPreview : "# Add a checkpoint path above to see the command"}
        </pre>
      </div>

      {/* Results grid */}
      {results.length > 0 && (
        <ResultsGrid results={results} logScale={logScale} onOpenAnalytics={openInAnalytics} />
      )}

      {/* Launch */}
      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">Configure Project Root in Settings first.</p>
        )}
        {validCheckpoints.length === 0 && (
          <p className="text-xs text-accent-warning">Add at least one checkpoint path.</p>
        )}
        <button
          onClick={launch}
          disabled={launching || !projectRoot || validCheckpoints.length === 0}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching
            ? "Launching…"
            : validCheckpoints.length > 1
            ? `Evaluate ${validCheckpoints.length} Checkpoints`
            : "Evaluate"}
        </button>
      </div>

      {/* Live progress panel */}
      {liveRunSummary && (
        <LauncherLivePanel
          header={{
            status: liveRunSummary.allDone
              ? liveRunSummary.aggregate
              : "running",
            title: evalLivePanelTitle({
              isRunning: !liveRunSummary.allDone,
              status: liveRunSummary.aggregate,
              total: displayProcessIds.length,
              runningCount: liveRunSummary.running,
              completedCount: liveRunSummary.completed,
            }),
            navMesh: {
              kind: "eval",
              hideSelf: true,
              showPostRun:
                liveRunSummary.allDone && liveRunSummary.aggregate === "completed",
              showOutputBrowser:
                liveRunSummary.allDone && liveRunSummary.aggregate === "completed",
              outputRunPath,
              checkpointPath: completedCheckpointPath,
              onOpenAnalytics: results.length > 0 ? openInAnalytics : undefined,
            },
          }}
          footer={<ProcessIdFooter processIds={displayProcessIds} />}
          logLines={singleCheckpointEval ? singleEvalLogLines : undefined}
          logTailWaiting={singleCheckpointEval ? singleEvalLogTailWaiting : false}
        >
          <div className="space-y-2">
            {displayProcessIds.map((procId) => {
              const proc = processes[procId];
              const ckptName = checkpointLabelFromEvalProcess(
                procId,
                proc?.command ?? ""
              );
              const isRunning = proc?.status === "running";
              const ckptResult = results.find((r) => r.checkpointName === ckptName);

              return (
                <EvalCheckpointLiveCard
                  key={procId}
                  procId={procId}
                  checkpointName={ckptName}
                  status={proc?.status}
                  isRunning={isRunning}
                  result={ckptResult}
                  logLines={proc?.logLines ?? []}
                  showLogTail={!singleCheckpointEval}
                />
              );
            })}
          </div>
        </LauncherLivePanel>
      )}
    </div>
  );
}

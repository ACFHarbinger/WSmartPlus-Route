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
import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp, Play, Plus, Terminal, Trash2, FolderOpen } from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../../store/app";
import { useSpawnProcess } from "../../hooks/useSpawnProcess";

const PROBLEMS = ["vrpp", "wcvrp", "scwcvrp"] as const;
const STRATEGIES = ["greedy", "sampling", "beam"] as const;
const DEVICES = ["cpu", "cuda:0", "cuda:1"] as const;

interface CheckpointEntry {
  id: string;
  path: string;
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
  const { projectRoot, pendingCheckpoint, setPendingCheckpoint } = useAppStore();
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

  const validCheckpoints = checkpoints.filter((c) => c.path.trim() !== "");

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

  const launch = useCallback(async () => {
    if (!projectRoot || validCheckpoints.length === 0) return;

    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);

    for (const ckpt of validCheckpoints) {
      const ckptName = ckpt.path.split(/[/\\]/).pop() ?? ckpt.id;
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
        id: `eval_${ckptName}_${Date.now()}`,
        pythonArgs: ["main.py", "eval", ...hydraArgs],
        workingDir: projectRoot,
      });
    }
  }, [
    projectRoot, validCheckpoints, problem, valSize, strategy, device,
    datasetPath, extraOverrides, spawn,
  ]);

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

      {/* Results placeholder */}
      <div className="card space-y-2">
        <h2 className="text-sm font-semibold text-gray-200">Results</h2>
        <p className="text-xs text-canvas-muted">
          Results will stream to the Process Monitor. A comparison table will appear here once structured
          result parsing is implemented (§G.12).
        </p>
      </div>

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
    </div>
  );
}

/**
 * Training Hub — full-featured training / HPO / eval launcher (§G.10).
 *
 * Form parameters mirror the controller justfile exactly:
 *   train:    envs, models, model.encoder.type, train.batch_size, seed
 *   hpo:      hpo.n_trials, hpo.method, hpo.num_workers
 *   eval:     eval.policy.model.load_path, eval.datasets, eval.val_size, eval.decoding.strategy
 *
 * Also supports WandB toggle and arbitrary extra Hydra overrides.
 * Command preview shows the exact invocation before launch.
 */
import { useCallback, useMemo, useState } from "react";
import { Play, ChevronDown, ChevronUp, Terminal, FolderOpen } from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../store/app";
import { useSpawnProcess } from "../hooks/useSpawnProcess";

type Mode = "train" | "hpo" | "eval";

const PROBLEMS = ["vrpp", "wcvrp", "scwcvrp"] as const;
const MODELS = ["am", "tam", "ddam", "moe"] as const;
const ENCODERS = ["gat", "gcn", "mha"] as const;
const HPO_METHODS = ["nsgaii", "tpe", "dehb", "random"] as const;
const EVAL_STRATEGIES = ["greedy", "sampling", "beam"] as const;

export function TrainingHub() {
  const { projectRoot } = useAppStore();
  const { spawn, launching } = useSpawnProcess();

  const [mode, setMode] = useState<Mode>("train");

  // Shared params
  const [problem, setProblem] = useState<string>("vrpp");
  const [seed, setSeed] = useState(42);
  const [wandb, setWandb] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [extraOverrides, setExtraOverrides] = useState("");

  // Train params
  const [model, setModel] = useState<string>("am");
  const [encoder, setEncoder] = useState<string>("gat");
  const [batchSize, setBatchSize] = useState(64);
  const [epochs, setEpochs] = useState(100);

  // HPO params
  const [hpoTrials, setHpoTrials] = useState(50);
  const [hpoMethod, setHpoMethod] = useState<string>("nsgaii");
  const [hpoWorkers, setHpoWorkers] = useState(1);

  // Eval params
  const [checkpointPath, setCheckpointPath] = useState("");
  const [evalDataset, setEvalDataset] = useState("");
  const [evalSamples, setEvalSamples] = useState(10);
  const [evalStrategy, setEvalStrategy] = useState<string>("greedy");

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
    // eval
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
    await spawn({
      id: `${mode}_${Date.now()}`,
      pythonArgs: ["main.py", entrypoint, ...hydraArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, mode, entrypoint, hydraArgs, spawn]);

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

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Mode selector */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Mode</h2>
        <div className="flex gap-2">
          {(["train", "hpo", "eval"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
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
    </div>
  );
}

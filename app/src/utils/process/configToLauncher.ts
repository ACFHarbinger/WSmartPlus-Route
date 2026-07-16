/**
 * Map flat Hydra config key-value rows to launcher store patches (§G.13).
 */

export type LauncherTarget = "sim_launcher" | "training_hub" | "data_gen";

export interface FlatConfigRow {
  key: string;
  value: string;
}

function parseList(value: string): string[] {
  const trimmed = value.trim();
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    return trimmed
      .slice(1, -1)
      .split(",")
      .map((s) => s.trim().replace(/^['"]|['"]$/g, ""))
      .filter(Boolean);
  }
  return trimmed.split(",").map((s) => s.trim()).filter(Boolean);
}

function parseNum(value: string): number | undefined {
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

function parseBool(value: string): boolean | undefined {
  if (value === "true") return true;
  if (value === "false") return false;
  return undefined;
}

/** Keys consumed by structured mapping — remainder goes to extraOverrides. */
const SIM_KEYS = new Set([
  "sim.policies", "sim.graph.area", "sim.graph.num_loc",
  "sim.n_samples", "sim.cpu_cores", "sim.data_distribution", "seed",
]);

const TRAIN_KEYS = new Set([
  "env.name", "problem", "model", "model.encoder.type",
  "train.max_epochs", "train.batch_size", "train.n_epochs",
  "hpo.n_trials", "hpo.method", "hpo.num_workers",
  "eval.policy.model.load_path", "eval.datasets", "eval.val_size",
  "eval.decoding.strategy", "seed", "tracker.enabled",
]);

const DATA_GEN_KEYS = new Set([
  "data.problem", "data.data_distributions", "data.dataset_type",
  "data.seed", "data.overwrite", "data.graphs.0.area",
  "data.graphs.0.num_loc", "data.graphs.0.n_samples", "data.graphs.0.n_days",
  "seed",
]);

function unmappedOverrides(rows: FlatConfigRow[], consumed: Set<string>): string {
  return rows
    .filter((r) => !consumed.has(r.key) && r.value && !r.value.startsWith("{"))
    .map((r) => `${r.key}=${r.value}`)
    .join("\n");
}

export function applyToSimLauncher(rows: FlatConfigRow[]) {
  const map = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  const patch: Record<string, unknown> = {};
  if (map["sim.policies"]) patch.selectedPolicies = parseList(map["sim.policies"]);
  if (map["sim.graph.area"]) patch.area = map["sim.graph.area"];
  if (map["sim.graph.num_loc"]) {
    const n = parseNum(map["sim.graph.num_loc"]);
    if (n != null) patch.numLoc = n;
  }
  if (map["sim.n_samples"]) {
    const n = parseNum(map["sim.n_samples"]);
    if (n != null) patch.samples = n;
  }
  if (map["sim.cpu_cores"]) {
    const n = parseNum(map["sim.cpu_cores"]);
    if (n != null) patch.nCores = n;
  }
  if (map["sim.data_distribution"]) patch.distribution = map["sim.data_distribution"];
  if (map.seed) {
    const n = parseNum(map.seed);
    if (n != null) patch.seed = n;
  }
  const extra = unmappedOverrides(rows, SIM_KEYS);
  if (extra) patch.extraOverrides = extra;
  return patch;
}

export function applyToTrainHub(rows: FlatConfigRow[]) {
  const map = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  const patch: Record<string, unknown> = {};
  const problem = map["env.name"] ?? map.problem ?? map["eval.problem"];
  if (problem) patch.problem = problem.replace(/"/g, "");
  if (map.model) patch.model = map.model.replace(/"/g, "");
  if (map["model.encoder.type"]) patch.encoder = map["model.encoder.type"];
  if (map["train.max_epochs"] ?? map["train.n_epochs"]) {
    const n = parseNum(map["train.max_epochs"] ?? map["train.n_epochs"]);
    if (n != null) patch.epochs = n;
  }
  if (map["train.batch_size"]) {
    const n = parseNum(map["train.batch_size"]);
    if (n != null) patch.batchSize = n;
  }
  if (map["hpo.n_trials"]) {
    const n = parseNum(map["hpo.n_trials"]);
    if (n != null) { patch.hpoTrials = n; patch.trainMode = "hpo"; }
  }
  if (map["hpo.method"]) { patch.hpoMethod = map["hpo.method"]; patch.trainMode = "hpo"; }
  if (map["hpo.num_workers"]) {
    const n = parseNum(map["hpo.num_workers"]);
    if (n != null) patch.hpoWorkers = n;
  }
  if (map["eval.policy.model.load_path"]) {
    patch.checkpointPath = map["eval.policy.model.load_path"];
    patch.trainMode = "eval";
  }
  if (map["eval.datasets"]) patch.evalDataset = map["eval.datasets"];
  if (map["eval.val_size"]) {
    const n = parseNum(map["eval.val_size"]);
    if (n != null) patch.evalSamples = n;
  }
  if (map["eval.decoding.strategy"]) patch.evalStrategy = map["eval.decoding.strategy"];
  if (map.seed) {
    const n = parseNum(map.seed);
    if (n != null) patch.seed = n;
  }
  if (map["tracker.enabled"]) {
    const b = parseBool(map["tracker.enabled"]);
    if (b != null) patch.wandb = b;
  }
  const extra = unmappedOverrides(rows, TRAIN_KEYS);
  if (extra) patch.extraOverrides = extra;
  return patch;
}

export function applyToDataGen(rows: FlatConfigRow[]) {
  const map = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  const patch: Record<string, unknown> = {};
  if (map["data.problem"]) patch.problem = map["data.problem"].replace(/"/g, "");
  if (map["data.data_distributions"]) patch.distributions = parseList(map["data.data_distributions"]);
  if (map["data.dataset_type"]) patch.datasetType = map["data.dataset_type"].replace(/"/g, "");
  if (map["data.seed"] ?? map.seed) {
    const n = parseNum(map["data.seed"] ?? map.seed);
    if (n != null) patch.seed = n;
  }
  if (map["data.overwrite"]) {
    const b = parseBool(map["data.overwrite"]);
    if (b != null) patch.overwrite = b;
  }
  if (map["data.graphs.0.area"]) patch.area = map["data.graphs.0.area"].replace(/"/g, "");
  if (map["data.graphs.0.num_loc"]) {
    const n = parseNum(map["data.graphs.0.num_loc"]);
    if (n != null) patch.numLoc = n;
  }
  if (map["data.graphs.0.n_samples"]) {
    const n = parseNum(map["data.graphs.0.n_samples"]);
    if (n != null) patch.nSamples = n;
  }
  if (map["data.graphs.0.n_days"]) {
    const n = parseNum(map["data.graphs.0.n_days"]);
    if (n != null) patch.nDays = n;
  }
  const extra = unmappedOverrides(rows, DATA_GEN_KEYS);
  if (extra) patch.extraOverrides = extra;
  return patch;
}

export function applyConfigToLauncher(target: LauncherTarget, rows: FlatConfigRow[]) {
  switch (target) {
    case "sim_launcher": return applyToSimLauncher(rows);
    case "training_hub": return applyToTrainHub(rows);
    case "data_gen": return applyToDataGen(rows);
  }
}

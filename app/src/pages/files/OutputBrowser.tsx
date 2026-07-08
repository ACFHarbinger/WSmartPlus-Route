/**
 * Output Browser — browse the assets/output/ directory tree (§G.14).
 *
 * Ports the PySide6 file-explorer tab. Provides:
 *   - Run list (from list_output_dirs)
 *   - File tree inside a selected run (from list_dir)
 *   - Run metadata card: auto-loads pruned_config.yaml and shows key fields
 *   - File viewer: CSV → DataExplorer-style table; YAML/text → raw view
 *   - "Open in Sim Summary" button for .jsonl log files
 */
import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save as saveDialog } from "@tauri-apps/plugin-dialog";
import {
  Folder,
  FolderOpen,
  File,
  FileText,
  RefreshCw,
  ChevronRight,
  ChevronDown,
  BarChart2,
  Save,
  Trash2,
  Package,
  Archive,
} from "lucide-react";
import { useAppStore } from "../../store/app";
import { useSessionProfilesStore } from "../../store/sessionProfiles";
import { toast } from "sonner";
import type { DirEntry, OutputDir, DayLogEntry, WsrouteBundleInfo, WsrouteExtractResult } from "../../types";
import { downloadParquetFromCsv } from "../../utils/tableExport";

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

const TEXT_EXTENSIONS = new Set(["yaml", "yml", "toml", "cfg", "ini", "txt", "log", "md", "json", "jsonl"]);
const CSV_EXTENSIONS = new Set(["csv"]);
const LOG_EXTENSIONS = new Set(["jsonl"]);

function FileIcon({ entry }: { entry: DirEntry }) {
  if (entry.is_dir) return <Folder size={13} className="text-accent-warning" />;
  if (entry.extension === "wsroute") return <File size={13} className="text-accent-primary" />;
  if (CSV_EXTENSIONS.has(entry.extension)) return <File size={13} className="text-accent-success" />;
  if (TEXT_EXTENSIONS.has(entry.extension)) return <FileText size={13} className="text-accent-secondary" />;
  return <File size={13} className="text-canvas-muted" />;
}

/** Flat YAML key-value parser (same logic as ConfigEditor). */
function parseYamlFlat(yaml: string): Array<{ key: string; value: string }> {
  const rows: Array<{ key: string; value: string }> = [];
  const stack: string[] = [];
  for (const line of yaml.split("\n")) {
    const stripped = line.replace(/\t/g, "  ");
    const trimmed = stripped.trimStart();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const indent = (stripped.length - trimmed.length) / 2;
    stack.length = indent;
    const colonIdx = trimmed.indexOf(":");
    if (colonIdx === -1) continue;
    const key = trimmed.slice(0, colonIdx).trim();
    const value = trimmed.slice(colonIdx + 1).trim();
    stack[indent] = key;
    if (value && !value.startsWith("#")) {
      rows.push({ key: stack.filter(Boolean).join("."), value });
    }
  }
  return rows;
}

const META_KEYS = [
  "task", "seed", "envs", "models", "model.encoder.type",
  "sim.graph.area", "sim.graph.num_loc", "sim.data_distribution",
  "sim.policies", "train.max_epochs", "hpo.n_trials",
];

/** Find the first .jsonl log in a run directory (top-level or hydra/). */
async function findRunJsonl(runPath: string): Promise<string | null> {
  const top = await invoke<DirEntry[]>("list_dir", { path: runPath });
  const topJsonl = top.find((f) => !f.is_dir && f.extension === "jsonl" && f.size_bytes < 20 * 1024 * 1024);
  if (topJsonl) return topJsonl.path;
  const hydra = top.find((f) => f.is_dir && f.name === "hydra");
  if (hydra) {
    const sub = await invoke<DirEntry[]>("list_dir", { path: hydra.path });
    const nested = sub.find((f) => !f.is_dir && f.extension === "jsonl" && f.size_bytes < 20 * 1024 * 1024);
    if (nested) return nested.path;
  }
  return null;
}

/** Sort: directories first, then key artefacts (config, logs), then alphabetical. */
function sortEntries(list: DirEntry[]): DirEntry[] {
  const priority = (e: DirEntry) => {
    if (e.is_dir) return 0;
    if (e.name === "pruned_config.yaml" || e.name === "config.yaml") return 1;
    if (e.extension === "jsonl") return 2;
    return 3;
  };
  return [...list].sort((a, b) => {
    const pa = priority(a);
    const pb = priority(b);
    if (pa !== pb) return pa - pb;
    return a.name.localeCompare(b.name);
  });
}

export function OutputBrowser() {
  const { projectRoot, setMode, setPendingLogPath, setPendingBenchmarkLogs } = useAppStore();
  const [runs, setRuns] = useState<OutputDir[]>([]);
  const [selectedRun, setSelectedRun] = useState<OutputDir | null>(null);
  const [compareSelection, setCompareSelection] = useState<Set<string>>(new Set());
  const [profileName, setProfileName] = useState("");
  const { profiles, saveProfile, loadProfile, deleteProfile } = useSessionProfilesStore();
  const [entries, setEntries] = useState<DirEntry[]>([]);
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  const [subEntries, setSubEntries] = useState<Record<string, DirEntry[]>>({});
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [wsrouteBundle, setWsrouteBundle] = useState<WsrouteBundleInfo | null>(null);
  const [csvRows, setCsvRows] = useState<Array<Record<string, string | number | null>> | null>(null);
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [viewingPath, setViewingPath] = useState<string | null>(null);
  const [viewingExt, setViewingExt] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [fileLoading, setFileLoading] = useState(false);
  const [bundleExporting, setBundleExporting] = useState(false);
  const [bundleExtracting, setBundleExtracting] = useState(false);
  const [parquetExporting, setParquetExporting] = useState(false);
  // Run metadata from pruned_config.yaml
  const [runMeta, setRunMeta] = useState<Array<{ key: string; value: string }> | null>(null);
  // KPI summary parsed from the first .jsonl found in the run directory
  const [runKpi, setRunKpi] = useState<Array<{ policy: string; overflows: number; kgkm: number; profit: number }> | null>(null);

  const outputPath = projectRoot ? `${projectRoot}/assets/output` : null;

  const refresh = useCallback(async () => {
    if (!outputPath) return;
    setLoading(true);
    try {
      const found = await invoke<OutputDir[]>("list_output_dirs", { outputPath });
      setRuns(found);
    } catch (err) {
      toast.error("Failed to list output directories", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [outputPath]);

  useEffect(() => {
    if (outputPath) refresh();
  }, [outputPath, refresh]);

  const selectRun = useCallback(async (run: OutputDir) => {
    setSelectedRun(run);
    setFileContent(null);
    setCsvRows(null);
    setViewingPath(null);
    setViewingExt("");
    setExpandedDirs(new Set());
    setSubEntries({});
    setRunMeta(null);
    setRunKpi(null);
    try {
      const e = sortEntries(await invoke<DirEntry[]>("list_dir", { path: run.path }));
      setEntries(e);

      // Auto-expand hydra/ subdirectory for structured tree view
      const hydraDir = e.find((f) => f.is_dir && f.name === "hydra");
      let hydraSub: DirEntry[] = [];
      if (hydraDir) {
        setExpandedDirs(new Set([hydraDir.path]));
        hydraSub = sortEntries(await invoke<DirEntry[]>("list_dir", { path: hydraDir.path }));
        setSubEntries({ [hydraDir.path]: hydraSub });
      }

      // Auto-load pruned_config.yaml for metadata panel (top-level or hydra/)
      let configEntry = e.find((f) => f.name === "pruned_config.yaml" || f.name === "config.yaml");
      if (!configEntry && hydraSub.length > 0) {
        configEntry = hydraSub.find(
          (f) => f.name === "pruned_config.yaml" || f.name === "config.yaml"
        );
      }
      if (configEntry) {
        try {
          const yaml = await invoke<string>("read_text_file", { path: configEntry.path });
          const all = parseYamlFlat(yaml);
          const meta = all.filter((r) => META_KEYS.some((k) => r.key.startsWith(k)));
          setRunMeta(meta.length > 0 ? meta : all.slice(0, 12));
        } catch { /* metadata optional */ }
      }

      // Auto-parse the first .jsonl for a KPI summary card
      const jsonlPath = await findRunJsonl(run.path);
      if (jsonlPath) {
        try {
          const text = await invoke<string>("read_text_file", { path: jsonlPath });
          const acc: Record<string, { overflows: number[]; kgkm: number[]; profit: number[] }> = {};
          for (const line of text.split("\n")) {
            if (!line.trim()) continue;
            try {
              const entry = JSON.parse(line) as DayLogEntry;
              if (!acc[entry.policy]) acc[entry.policy] = { overflows: [], kgkm: [], profit: [] };
              acc[entry.policy].overflows.push(entry.data.overflows ?? 0);
              acc[entry.policy].kgkm.push(entry.data["kg/km"] ?? 0);
              acc[entry.policy].profit.push(entry.data.profit ?? 0);
            } catch { /* skip malformed lines */ }
          }
          const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
          const kpi = Object.entries(acc).map(([policy, v]) => ({
            policy,
            overflows: avg(v.overflows),
            kgkm: avg(v.kgkm),
            profit: avg(v.profit),
          })).sort((a, b) => a.overflows - b.overflows);
          if (kpi.length > 0) setRunKpi(kpi);
        } catch { /* KPI optional */ }
      }
    } catch (err) {
      toast.error("Failed to list run directory", { description: String(err) });
    }
  }, []);

  const toggleDir = useCallback(async (entry: DirEntry) => {
    setExpandedDirs((prev) => {
      const next = new Set(prev);
      if (next.has(entry.path)) next.delete(entry.path);
      else next.add(entry.path);
      return next;
    });
    if (!subEntries[entry.path]) {
      try {
        const sub = sortEntries(await invoke<DirEntry[]>("list_dir", { path: entry.path }));
        setSubEntries((prev) => ({ ...prev, [entry.path]: sub }));
      } catch {}
    }
  }, [subEntries]);

  const openFile = useCallback(async (entry: DirEntry) => {
    if (entry.is_dir) {
      toggleDir(entry);
      return;
    }
    setViewingPath(entry.path);
    setViewingExt(entry.extension);
    setFileContent(null);
    setCsvRows(null);
    setWsrouteBundle(null);
    setFileLoading(true);

    try {
      if (entry.extension === "wsroute") {
        const info = await invoke<WsrouteBundleInfo>("inspect_wsroute_bundle", {
          path: entry.path,
        });
        setWsrouteBundle(info);
      } else if (CSV_EXTENSIONS.has(entry.extension)) {
        const csvFile = await invoke<{ headers: string[]; rows: Array<Record<string, string | number | null>> }>(
          "load_csv_file",
          { path: entry.path }
        );
        setCsvHeaders(csvFile.headers);
        setCsvRows(csvFile.rows);
      } else if (TEXT_EXTENSIONS.has(entry.extension)) {
        const text = await invoke<string>("read_text_file", { path: entry.path });
        setFileContent(text);
      }
    } catch (err) {
      toast.error("Failed to open file", { description: String(err) });
    } finally {
      setFileLoading(false);
    }
  }, [toggleDir]);

  const openInSimSummary = useCallback((path: string) => {
    setPendingLogPath(path);
    setMode("simulation_summary");
  }, [setPendingLogPath, setMode]);

  const toggleCompareRun = useCallback((runPath: string) => {
    setCompareSelection((prev) => {
      const next = new Set(prev);
      if (next.has(runPath)) next.delete(runPath);
      else next.add(runPath);
      return next;
    });
  }, []);

  const compareSelectedRuns = useCallback(async () => {
    if (compareSelection.size < 2) return;
    const refs: Array<{ path: string; label: string }> = [];
    for (const runPath of compareSelection) {
      const run = runs.find((r) => r.path === runPath);
      const jsonl = await findRunJsonl(runPath);
      if (jsonl && run) refs.push({ path: jsonl, label: run.name });
    }
    if (refs.length < 2) {
      toast.error("Need at least 2 runs with .jsonl logs to compare");
      return;
    }
    setPendingBenchmarkLogs(refs);
    setMode("benchmark");
    toast.success(`Comparing ${refs.length} runs in Benchmark Analysis`);
  }, [compareSelection, runs, setPendingBenchmarkLogs, setMode]);

  const exportRunAsBundle = useCallback(async () => {
    if (!selectedRun) return;
    const path = (await saveDialog({
      filters: [{ name: "WSmart-Route Bundle", extensions: ["wsroute"] }],
      defaultPath: `${selectedRun.name}.wsroute`,
    })) as string | null;
    if (!path) return;

    const outputPath = path.endsWith(".wsroute") ? path : `${path}.wsroute`;
    setBundleExporting(true);
    try {
      const info = await invoke<WsrouteBundleInfo>("create_wsroute_bundle", {
        sourceDir: selectedRun.path,
        outputPath,
      });
      toast.success(`Exported ${info.files.length} files`, {
        description: outputPath.split("/").pop(),
      });
    } catch (err) {
      toast.error("Failed to export bundle", { description: String(err) });
    } finally {
      setBundleExporting(false);
    }
  }, [selectedRun]);

  const extractBundleAndOpen = useCallback(async () => {
    if (!viewingPath || viewingExt !== "wsroute") return;
    const destDir = (await open({
      directory: true,
      title: "Extract bundle to…",
    })) as string | null;
    if (!destDir) return;

    setBundleExtracting(true);
    try {
      const result = await invoke<WsrouteExtractResult>("extract_wsroute_bundle", {
        path: viewingPath,
        destDir,
      });
      toast.success(`Extracted ${result.extracted_files.length} files`);
      if (result.log_path) {
        setPendingLogPath(result.log_path);
        setMode("simulation_summary");
      } else {
        toast.info("No .jsonl log found in bundle — browse extracted files manually");
      }
    } catch (err) {
      toast.error("Failed to extract bundle", { description: String(err) });
    } finally {
      setBundleExtracting(false);
    }
  }, [viewingPath, viewingExt, setPendingLogPath, setMode]);

  const pickOutputDir = useCallback(async () => {
    const path = (await open({ directory: true })) as string | null;
    if (!path) return;
    try {
      const e = await invoke<DirEntry[]>("list_dir", { path });
      const fakeRun: OutputDir = {
        name: path.split("/").pop() ?? path,
        path,
        created_at: "",
        size_bytes: 0,
      };
      setSelectedRun(fakeRun);
      setEntries(e);
      setFileContent(null);
      setCsvRows(null);
      setRunMeta(null);
    } catch (err) {
      toast.error("Failed to open directory", { description: String(err) });
    }
  }, []);

  function renderEntries(list: DirEntry[], depth = 0): React.ReactNode {
    return list.map((e) => (
      <div key={e.path}>
        <button
          onClick={() => openFile(e)}
          className={`w-full flex items-center gap-2 py-1 px-2 rounded text-xs text-left hover:bg-canvas-hover transition-colors ${
            viewingPath === e.path ? "bg-accent-primary/15 text-accent-secondary" : "text-gray-300"
          }`}
          style={{ paddingLeft: `${8 + depth * 14}px` }}
        >
          {e.is_dir ? (
            expandedDirs.has(e.path)
              ? <ChevronDown size={11} className="shrink-0 text-canvas-muted" />
              : <ChevronRight size={11} className="shrink-0 text-canvas-muted" />
          ) : (
            <span className="w-[11px] shrink-0" />
          )}
          <FileIcon entry={e} />
          <span
            className={`truncate flex-1 ${
              e.name === "pruned_config.yaml" || e.name === "config.yaml"
                ? "text-accent-secondary font-medium"
                : e.extension === "jsonl"
                ? "text-accent-success"
                : ""
            }`}
          >
            {e.name}
          </span>
          {!e.is_dir && (
            <span className="text-canvas-muted shrink-0">{formatBytes(e.size_bytes)}</span>
          )}
        </button>
        {e.is_dir && expandedDirs.has(e.path) && subEntries[e.path] && (
          <div>{renderEntries(subEntries[e.path], depth + 1)}</div>
        )}
      </div>
    ));
  }

  if (!projectRoot) {
    return (
      <div className="flex items-center justify-center h-64 text-canvas-muted text-sm">
        Set project root in settings to browse output directories.
      </div>
    );
  }

  return (
    <div className="flex gap-4 h-[calc(100vh-8rem)]">
      {/* Left: run list */}
      <div className="w-56 shrink-0 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <button
            onClick={refresh}
            disabled={loading}
            className="btn-ghost p-1.5 text-canvas-muted"
            title="Refresh"
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          </button>
          <button
            onClick={pickOutputDir}
            className="btn-ghost p-1.5 text-canvas-muted"
            title="Open directory"
          >
            <FolderOpen size={13} />
          </button>
          <span className="text-xs text-canvas-muted">{runs.length} runs</span>
        </div>

        {compareSelection.size >= 2 && (
          <button
            onClick={compareSelectedRuns}
            className="btn-primary text-xs flex items-center gap-1.5 w-full justify-center"
          >
            <BarChart2 size={12} />
            Compare {compareSelection.size} Runs →
          </button>
        )}

        <div className="card flex-1 overflow-auto p-1">
          {runs.map((r) => (
            <div
              key={r.path}
              className={`flex items-center gap-1 rounded text-xs ${
                selectedRun?.path === r.path ? "bg-accent-primary/10" : ""
              }`}
            >
              <input
                type="checkbox"
                checked={compareSelection.has(r.path)}
                onChange={() => toggleCompareRun(r.path)}
                className="ml-1.5 shrink-0 accent-accent-primary"
                title="Select for comparison"
              />
              <button
                onClick={() => selectRun(r)}
                className={`flex-1 flex items-center gap-2 py-1.5 px-1 rounded text-left hover:bg-canvas-hover transition-colors ${
                  selectedRun?.path === r.path
                    ? "text-accent-secondary"
                    : "text-gray-300"
                }`}
              >
                <Folder size={12} className="text-accent-warning shrink-0" />
                <span className="truncate">{r.name}</span>
              </button>
            </div>
          ))}
          {runs.length === 0 && !loading && (
            <p className="text-xs text-canvas-muted p-2">No run directories found.</p>
          )}
        </div>

        {/* Session profiles (§G.14 / §D.4 Option C) */}
        <div className="card p-2 space-y-2 shrink-0">
          <p className="text-xs font-semibold text-gray-300">Session Profiles</p>
          <div className="flex gap-1">
            <input
              type="text"
              className="input-base text-xs flex-1"
              value={profileName}
              onChange={(e) => setProfileName(e.target.value)}
              placeholder="Profile name…"
            />
            <button
              onClick={() => {
                saveProfile(profileName);
                setProfileName("");
                toast.success("Launcher state saved");
              }}
              className="btn-ghost p-1.5"
              title="Save current launcher forms"
            >
              <Save size={12} />
            </button>
          </div>
          {profiles.length > 0 ? (
            <div className="max-h-28 overflow-y-auto space-y-0.5">
              {profiles.map((p) => (
                <div key={p.id} className="flex items-center gap-1 text-xs">
                  <button
                    onClick={() => {
                      loadProfile(p.id);
                      toast.success(`Loaded profile "${p.name}"`);
                    }}
                    className="flex-1 text-left truncate text-gray-300 hover:text-accent-secondary py-0.5"
                  >
                    {p.name}
                  </button>
                  <button
                    onClick={() => deleteProfile(p.id)}
                    className="text-canvas-muted hover:text-accent-danger p-0.5"
                  >
                    <Trash2 size={10} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[10px] text-canvas-muted">No saved profiles yet.</p>
          )}
        </div>
      </div>

      {/* Middle: file tree + metadata */}
      <div className="w-56 shrink-0 flex flex-col gap-2">
        {selectedRun ? (
          <>
            <button
              onClick={exportRunAsBundle}
              disabled={bundleExporting}
              className="btn-ghost text-xs flex items-center gap-1.5 w-full justify-center shrink-0"
              title="Package run artefacts into a .wsroute zip bundle"
            >
              {bundleExporting ? (
                <RefreshCw size={12} className="animate-spin" />
              ) : (
                <Package size={12} />
              )}
              Export as .wsroute
            </button>
            <div className="card flex-1 overflow-auto p-1">
              <p className="text-xs text-canvas-muted px-2 py-1 font-medium truncate">
                {selectedRun.name}
              </p>
              {renderEntries(entries)}
            </div>
            {runMeta && runMeta.length > 0 && (
              <div className="card p-2 space-y-1 overflow-auto max-h-36">
                <p className="text-xs font-semibold text-gray-300 mb-1.5">Config</p>
                {runMeta.map((r) => (
                  <div key={r.key} className="flex gap-2 text-xs leading-tight">
                    <span className="text-canvas-muted font-mono truncate max-w-[50%]" title={r.key}>
                      {r.key.split(".").pop()}
                    </span>
                    <span className="text-gray-300 truncate flex-1" title={r.value}>{r.value}</span>
                  </div>
                ))}
              </div>
            )}
            {runKpi && runKpi.length > 0 && (
              <div className="card p-2 space-y-1 overflow-auto max-h-48">
                <p className="text-xs font-semibold text-gray-300 mb-1.5">KPI Summary</p>
                <div className="grid grid-cols-3 gap-x-2 text-[10px] text-canvas-muted font-medium mb-1">
                  <span>Policy</span><span className="text-right">Overflows</span><span className="text-right">kg/km</span>
                </div>
                {runKpi.map((r) => (
                  <div key={r.policy} className="grid grid-cols-3 gap-x-2 text-[10px] leading-tight">
                    <span className="text-gray-300 truncate font-mono" title={r.policy}>{r.policy}</span>
                    <span className={`text-right font-mono ${r.overflows === 0 ? "text-accent-success" : r.overflows > 20 ? "text-accent-danger" : "text-accent-warning"}`}>{r.overflows.toFixed(1)}</span>
                    <span className="text-right font-mono text-gray-400">{r.kgkm.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="card flex-1 flex items-center justify-center text-canvas-muted text-xs">
            Select a run
          </div>
        )}
      </div>

      {/* Right: file viewer */}
      <div className="flex-1 min-w-0 flex flex-col gap-2">
        {!viewingPath && !fileLoading && (
          <div className="card flex-1 flex items-center justify-center text-canvas-muted text-sm">
            Select a file to view its contents.
          </div>
        )}

        {fileLoading && (
          <div className="card flex-1 flex items-center justify-center gap-2 text-canvas-muted text-sm">
            <RefreshCw size={14} className="animate-spin" />
            Loading…
          </div>
        )}

        {!fileLoading && (fileContent !== null || csvRows !== null || wsrouteBundle !== null) && (
          <div className="flex items-center gap-3 shrink-0">
            <p className="text-xs text-canvas-muted font-mono truncate flex-1">{viewingPath}</p>
            {viewingPath && viewingExt === "wsroute" && (
              <button
                onClick={extractBundleAndOpen}
                disabled={bundleExtracting}
                className="btn-ghost text-xs flex items-center gap-1.5 text-accent-primary shrink-0"
              >
                {bundleExtracting ? (
                  <RefreshCw size={12} className="animate-spin" />
                ) : (
                  <Archive size={12} />
                )}
                Extract & Open
              </button>
            )}
            {viewingPath && CSV_EXTENSIONS.has(viewingExt) && projectRoot && (
              <button
                onClick={async () => {
                  setParquetExporting(true);
                  try {
                    const out = await downloadParquetFromCsv(
                      projectRoot,
                      viewingPath,
                      viewingPath.replace(/\.csv$/i, ".parquet")
                    );
                    if (out) toast.success("Parquet export complete", { description: out.split("/").pop() });
                  } catch (err) {
                    toast.error("Parquet export failed", { description: String(err) });
                  } finally {
                    setParquetExporting(false);
                  }
                }}
                disabled={parquetExporting}
                className="btn-ghost text-xs flex items-center gap-1.5 text-accent-primary shrink-0"
              >
                {parquetExporting ? (
                  <RefreshCw size={12} className="animate-spin" />
                ) : (
                  <Save size={12} />
                )}
                Export Parquet
              </button>
            )}
            {viewingPath && LOG_EXTENSIONS.has(viewingExt) && (
              <button
                onClick={() => openInSimSummary(viewingPath)}
                className="btn-ghost text-xs flex items-center gap-1.5 text-accent-primary shrink-0"
              >
                <BarChart2 size={12} />
                Open in Sim Summary
              </button>
            )}
          </div>
        )}

        {!fileLoading && fileContent !== null && (
          <div className="card flex-1 overflow-auto">
            <pre className="font-mono text-xs text-gray-300 whitespace-pre-wrap">{fileContent}</pre>
          </div>
        )}

        {!fileLoading && wsrouteBundle !== null && (
          <div className="card flex-1 overflow-auto space-y-3">
            <div className="flex flex-wrap gap-4 text-xs">
              <span className="text-canvas-muted">
                Version: <span className="font-mono text-gray-300">{wsrouteBundle.version ?? "—"}</span>
              </span>
              <span className="text-canvas-muted">
                Created: <span className="font-mono text-gray-300">{wsrouteBundle.created_at ?? "—"}</span>
              </span>
              <span className="text-canvas-muted">
                Files: <span className="font-mono text-gray-300">{wsrouteBundle.files.length}</span>
              </span>
            </div>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-canvas-border">
                  <th className="text-left py-2 px-3 text-canvas-muted font-medium">Path</th>
                  <th className="text-right py-2 px-3 text-canvas-muted font-medium">Size</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-canvas-border">
                {wsrouteBundle.files.map((f) => (
                  <tr key={f.path} className="hover:bg-canvas-hover">
                    <td className="py-1.5 px-3 font-mono text-gray-300">{f.path}</td>
                    <td className="py-1.5 px-3 text-right text-canvas-muted">{formatBytes(f.size_bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!fileLoading && csvRows !== null && (
          <div className="card flex-1 overflow-auto">
            <table className="w-full text-xs">
              <thead className="bg-canvas-elevated sticky top-0">
                <tr>
                  {csvHeaders.map((h) => (
                    <th key={h} className="px-3 py-2 text-left text-canvas-muted font-medium whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-canvas-border">
                {csvRows.slice(0, 200).map((row, i) => (
                  <tr key={i} className="hover:bg-canvas-hover">
                    {csvHeaders.map((h) => (
                      <td key={h} className="px-3 py-1.5 text-gray-300 whitespace-nowrap font-mono">
                        {row[h] ?? "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {csvRows.length > 200 && (
              <p className="text-xs text-canvas-muted p-3">
                Showing 200 / {csvRows.length} rows. Use Data Explorer for full pagination.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

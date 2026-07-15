/**
 * Configuration Editor — view and edit Hydra YAML configs (§G.13).
 *
 * Loads any YAML file (typically pruned_config.yaml from a run directory,
 * or a config file from logic/configs/) and presents it in three modes:
 *   Raw  — Monaco YAML editor with syntax highlighting
 *   Table — key-value pairs parsed from YAML (flat view)
 *   Diff  — compare with another YAML file
 *
 * "Copy overrides" serialises only the changed keys as Hydra override strings
 * for pasting into the Simulation Launcher or Training Hub.
 */
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, Copy, RefreshCw, FileText, Table2, GitCompare, Save, Download, Rocket, ListChecks } from "lucide-react";
import { toast } from "sonner";
import { PathRunLabelChip } from "../../components/common/PathRunLabelChip";
import { useLogPathRunLabelBrush } from "../../hooks/useLogPathRunLabelBrush";
import { useAppStore } from "../../store/app";
import { useSimLauncherStore, useTrainHubStore, useDataGenStore } from "../../store/launchers";
import { useRecentFilesStore } from "../../store/recentFiles";
import { applyConfigToLauncher, type LauncherTarget } from "../../utils/configToLauncher";
import { portfolioRunLabel } from "../../utils/arrowPipeline";

const YamlEditor = lazy(() => import("../../components/editors/YamlEditor"));

type ViewMode = "raw" | "table" | "form" | "diff";

type FieldType = "string" | "number" | "boolean";

function inferFieldType(value: string): FieldType {
  if (value === "true" || value === "false") return "boolean";
  if (value !== "" && !isNaN(Number(value)) && !value.startsWith("[")) return "number";
  return "string";
}

function rowsToYaml(rows: Array<{ key: string; value: string }>): string {
  return rows.map((r) => `${r.key}: ${r.value}`).join("\n");
}

function parseYamlFlat(yaml: string): Array<{ key: string; value: string }> {
  const rows: Array<{ key: string; value: string }> = [];
  const stack: string[] = [];
  for (const line of yaml.split("\n")) {
    const stripped = line.replace(/\t/g, "  ");
    const trimmed = stripped.trimStart();
    if (!trimmed || trimmed.startsWith("#")) continue;

    // Determine indent level (2-space based)
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

const HYDRA_TASKS = [
  { value: "test_sim", label: "test_sim" },
  { value: "train", label: "train" },
  { value: "hpo", label: "hpo" },
  { value: "eval", label: "eval" },
  { value: "gen_data", label: "gen_data" },
] as const;

const LAUNCHER_TARGETS: Array<{ value: LauncherTarget; label: string; mode: "sim_launcher" | "training_hub" | "data_gen" }> = [
  { value: "sim_launcher", label: "Simulation Launcher", mode: "sim_launcher" },
  { value: "training_hub", label: "Training Hub", mode: "training_hub" },
  { value: "data_gen", label: "Data Generation", mode: "data_gen" },
];

export function ConfigEditor() {
  const { projectRoot, pythonPath, setMode, pendingConfigPath, setPendingConfigPath } = useAppStore();
  const simPatch = useSimLauncherStore((s) => s.patch);
  const trainPatch = useTrainHubStore((s) => s.patch);
  const dataPatch = useDataGenStore((s) => s.patch);
  const pushRecent = useRecentFilesStore((s) => s.pushRecent);
  const [content, setContent] = useState("");
  const [filePath, setFilePath] = useState<string | null>(null);
  const [diffContent, setDiffContent] = useState("");
  const [diffPath, setDiffPath] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("raw");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [hydraTask, setHydraTask] = useState("test_sim");
  const [loadingHydra, setLoadingHydra] = useState(false);
  const [applyTarget, setApplyTarget] = useState<LauncherTarget>("sim_launcher");
  // Track the last-saved content to detect unsaved edits
  const savedContentRef = useRef("");
  useLogPathRunLabelBrush(filePath);
  useLogPathRunLabelBrush(diffPath);

  const loadConfigPath = useCallback(
    async (path: string, target: "primary" | "diff" = "primary") => {
      setLoading(true);
      try {
        const text = await invoke<string>("read_text_file", { path });
        if (target === "primary") {
          setContent(text);
          setFilePath(path);
          savedContentRef.current = text;
          pushRecent({
            path,
            label: portfolioRunLabel(path, undefined, projectRoot),
            kind: "config",
          });
        } else {
          setDiffContent(text);
          setDiffPath(path);
          pushRecent({
            path,
            label: portfolioRunLabel(path, undefined, projectRoot),
            kind: "config",
          });
        }
      } catch (err) {
        toast.error("Failed to read file", { description: String(err) });
      } finally {
        setLoading(false);
      }
    },
    [projectRoot, pushRecent]
  );

  // Consume pendingConfigPath set by Output Browser / Command Palette
  useEffect(() => {
    if (!pendingConfigPath) return;
    void loadConfigPath(pendingConfigPath, "primary");
    setPendingConfigPath(null);
  }, [pendingConfigPath, loadConfigPath, setPendingConfigPath]);

  const openFile = useCallback(async (target: "primary" | "diff") => {
    const path = (await open({
      filters: [{ name: "YAML / Config", extensions: ["yaml", "yml", "toml", "cfg", "ini"] }],
    })) as string | null;
    if (!path) return;
    await loadConfigPath(path, target);
  }, [loadConfigPath]);

  const saveFile = useCallback(async () => {
    if (!filePath || !content) return;
    setSaving(true);
    try {
      await invoke("write_text_file", { path: filePath, content });
      savedContentRef.current = content;
      toast.success("Saved", { description: filePath.split("/").pop() });
    } catch (err) {
      toast.error("Save failed", { description: String(err) });
    } finally {
      setSaving(false);
    }
  }, [filePath, content]);

  const copyOverrides = useCallback(async () => {
    if (!content) return;
    const rows = parseYamlFlat(content);
    // Emit only leaf scalar values as Hydra override strings
    const overrides = rows
      .filter((r) => r.value && !r.value.startsWith("{") && !r.value.startsWith("["))
      .map((r) => `${r.key}=${r.value}`)
      .join("\n");
    try {
      await navigator.clipboard.writeText(overrides);
      toast.success("Copied overrides to clipboard");
    } catch {
      toast.error("Clipboard write failed — copy manually from the Raw view");
    }
  }, [content]);

  const loadResolvedConfig = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings first");
      return;
    }
    setLoadingHydra(true);
    try {
      const yaml = await invoke<string>("dump_hydra_config", {
        task: hydraTask,
        projectRoot,
        pythonExecutable: pythonPath || null,
      });
      setContent(yaml);
      setFilePath(null);
      savedContentRef.current = yaml;
      toast.success(`Loaded resolved config for ${hydraTask}`);
    } catch (err) {
      toast.error("Hydra config dump failed", { description: String(err) });
    } finally {
      setLoadingHydra(false);
    }
  }, [projectRoot, pythonPath, hydraTask]);

  const applyToLauncher = useCallback(() => {
    if (!content.trim()) {
      toast.error("No config content to apply");
      return;
    }
    const flatRows = parseYamlFlat(content);
    const patch = applyConfigToLauncher(applyTarget, flatRows);
    const target = LAUNCHER_TARGETS.find((t) => t.value === applyTarget);
    if (applyTarget === "sim_launcher") simPatch(patch);
    else if (applyTarget === "training_hub") trainPatch(patch);
    else dataPatch(patch);
    if (target) setMode(target.mode);
    toast.success(`Applied ${Object.keys(patch).length} field(s) to ${target?.label ?? applyTarget}`);
  }, [content, applyTarget, simPatch, trainPatch, dataPatch, setMode]);

  const updateFormRow = useCallback((index: number, field: "key" | "value", newVal: string) => {
    const rows = parseYamlFlat(content);
    if (index < 0 || index >= rows.length) return;
    rows[index] = { ...rows[index], [field]: newVal };
    setContent(rowsToYaml(rows));
  }, [content]);

  const isDirty = content !== savedContentRef.current && filePath !== null;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (isDirty) void saveFile();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isDirty, saveFile]);

  const rows = parseYamlFlat(content);
  const diffRows = parseYamlFlat(diffContent);

  // Diff: entries where value differs between primary and diff
  const diffMap: Record<string, { primary: string; diff: string }> = {};
  for (const r of rows) diffMap[r.key] = { primary: r.value, diff: "" };
  for (const r of diffRows) {
    if (!diffMap[r.key]) diffMap[r.key] = { primary: "", diff: r.value };
    else diffMap[r.key].diff = r.value;
  }
  const changedKeys = Object.keys(diffMap).filter(
    (k) => diffMap[k].primary !== diffMap[k].diff
  );

  return (
    <div className="space-y-4">
      {/* Load resolved Hydra config */}
      <div className="card flex items-center gap-3 flex-wrap">
        <span className="text-xs text-canvas-muted">Load resolved config:</span>
        <select
          className="select-base w-36 text-xs"
          value={hydraTask}
          onChange={(e) => setHydraTask(e.target.value)}
        >
          {HYDRA_TASKS.map((t) => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>
        <button
          onClick={loadResolvedConfig}
          disabled={loadingHydra || !projectRoot}
          className="btn-primary flex items-center gap-2 text-sm"
        >
          {loadingHydra ? <RefreshCw size={13} className="animate-spin" /> : <Download size={13} />}
          Load via --cfg job
        </button>
        {!projectRoot && (
          <span className="text-xs text-accent-warning">Configure project root in Settings.</span>
        )}
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={() => openFile("primary")}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? <RefreshCw size={14} className="animate-spin" /> : <FolderOpen size={14} />}
          Open Config
        </button>

        {content && (
          <>
            <button
              onClick={saveFile}
              disabled={saving || !isDirty}
              className="btn-ghost flex items-center gap-2 text-sm disabled:opacity-40"
              title={isDirty ? "Save file to disk" : "No unsaved changes"}
            >
              {saving ? <RefreshCw size={13} className="animate-spin" /> : <Save size={13} />}
              {isDirty ? "Save*" : "Save"}
            </button>
            <button onClick={copyOverrides} className="btn-ghost flex items-center gap-2 text-sm">
              <Copy size={13} />
              Copy Overrides
            </button>

            <select
              className="select-base text-xs w-40"
              value={applyTarget}
              onChange={(e) => setApplyTarget(e.target.value as LauncherTarget)}
            >
              {LAUNCHER_TARGETS.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
            <button onClick={applyToLauncher} className="btn-ghost flex items-center gap-2 text-sm">
              <Rocket size={13} />
              Apply to Launcher
            </button>

            {/* View mode toggle */}
            <div className="ml-auto flex items-center gap-1 bg-canvas-elevated rounded-lg p-1">
              {(
                [
                  { mode: "raw", Icon: FileText, label: "Raw" },
                  { mode: "table", Icon: Table2, label: "Table" },
                  { mode: "form", Icon: ListChecks, label: "Form" },
                  { mode: "diff", Icon: GitCompare, label: "Diff" },
                ] as const
              ).map(({ mode, Icon, label }) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md transition-colors ${
                    viewMode === mode
                      ? "bg-accent-primary text-white"
                      : "text-canvas-muted hover:text-gray-200"
                  }`}
                >
                  <Icon size={12} />
                  {label}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {filePath && (
        <PathRunLabelChip path={filePath} projectRoot={projectRoot} className="max-w-full" />
      )}

      {!content && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Open a YAML config file (e.g. <code className="font-mono text-xs mx-1">pruned_config.yaml</code> from a run directory).
        </div>
      )}

      {/* Raw view — Monaco YAML editor (§G.13) */}
      {content && viewMode === "raw" && (
        <Suspense
          fallback={
            <div className="flex items-center justify-center h-[60vh] text-canvas-muted text-sm">
              Loading editor…
            </div>
          }
        >
          <YamlEditor value={content} onChange={setContent} />
        </Suspense>
      )}

      {/* Form view — editable typed widgets (§G.13) */}
      {content && viewMode === "form" && (
        <div className="card overflow-auto max-h-[60vh] space-y-2 p-2">
          {rows.map((r, i) => {
            const ftype = inferFieldType(r.value);
            return (
              <div key={`${r.key}-${i}`} className="flex items-center gap-3 text-xs">
                <span className="font-mono text-accent-secondary w-1/3 truncate" title={r.key}>
                  {r.key}
                </span>
                {ftype === "boolean" ? (
                  <label className="flex items-center gap-2 text-gray-300">
                    <input
                      type="checkbox"
                      className="accent-accent-primary"
                      checked={r.value === "true"}
                      onChange={(e) => updateFormRow(i, "value", e.target.checked ? "true" : "false")}
                    />
                    {r.value}
                  </label>
                ) : ftype === "number" ? (
                  <input
                    type="number"
                    className="input-base font-mono text-xs flex-1"
                    value={r.value}
                    onChange={(e) => updateFormRow(i, "value", e.target.value)}
                  />
                ) : (
                  <input
                    type="text"
                    className="input-base font-mono text-xs flex-1"
                    value={r.value}
                    onChange={(e) => updateFormRow(i, "value", e.target.value)}
                  />
                )}
              </div>
            );
          })}
          {rows.length === 0 && (
            <p className="text-canvas-muted text-sm text-center py-8">No scalar key-value pairs found.</p>
          )}
        </div>
      )}

      {/* Table view */}
      {content && viewMode === "table" && (
        <div className="card overflow-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-canvas-border">
                <th className="text-left py-2 px-3 text-canvas-muted font-medium w-1/2">Key</th>
                <th className="text-left py-2 px-3 text-canvas-muted font-medium w-1/2">Value</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-canvas-border">
              {rows.map((r, i) => (
                <tr key={i} className="hover:bg-canvas-hover">
                  <td className="py-1.5 px-3 font-mono text-accent-secondary">{r.key}</td>
                  <td className="py-1.5 px-3 text-gray-300">{r.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Diff view */}
      {content && viewMode === "diff" && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => openFile("diff")}
              className="btn-ghost flex items-center gap-2 text-sm"
            >
              <FolderOpen size={13} />
              {diffPath ? "Change comparison file…" : "Open comparison file…"}
            </button>
            {diffPath ? (
              <PathRunLabelChip path={diffPath} projectRoot={projectRoot} className="max-w-md" />
            ) : null}
          </div>

          {!diffContent && (
            <div className="card text-canvas-muted text-sm">
              Open a second YAML file to compare against the primary config.
            </div>
          )}

          {diffContent && (
            <div className="card overflow-auto">
              <p className="text-xs text-canvas-muted mb-3 flex flex-wrap items-center gap-1.5">
                <span>{changedKeys.length} difference(s) between</span>
                {filePath ? <PathRunLabelChip path={filePath} projectRoot={projectRoot} /> : null}
                <span>and</span>
                {diffPath ? <PathRunLabelChip path={diffPath} projectRoot={projectRoot} /> : null}
              </p>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-canvas-border">
                    <th className="text-left py-2 px-3 text-canvas-muted font-medium">Key</th>
                    <th className="text-left py-2 px-3 text-canvas-muted font-medium">Primary</th>
                    <th className="text-left py-2 px-3 text-canvas-muted font-medium">Comparison</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-canvas-border">
                  {changedKeys.map((k) => (
                    <tr key={k} className="bg-accent-warning/5">
                      <td className="py-1.5 px-3 font-mono text-accent-secondary">{k}</td>
                      <td className="py-1.5 px-3 text-accent-danger font-mono">
                        {diffMap[k].primary || <span className="text-canvas-muted italic">—</span>}
                      </td>
                      <td className="py-1.5 px-3 text-accent-success font-mono">
                        {diffMap[k].diff || <span className="text-canvas-muted italic">—</span>}
                      </td>
                    </tr>
                  ))}
                  {changedKeys.length === 0 && (
                    <tr>
                      <td colSpan={3} className="py-8 text-center text-accent-success">
                        No differences found — configs are identical.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

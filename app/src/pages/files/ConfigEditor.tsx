/**
 * Configuration Editor — view and edit Hydra YAML configs (§G.13).
 *
 * Loads any YAML file (typically pruned_config.yaml from a run directory,
 * or a config file from logic/configs/) and presents it in three modes:
 *   Raw  — editable textarea (Monaco planned in Phase 13 continuation)
 *   Table — key-value pairs parsed from YAML (flat view)
 *   Diff  — compare with another YAML file
 *
 * "Copy overrides" serialises only the changed keys as Hydra override strings
 * for pasting into the Simulation Launcher or Training Hub.
 */
import { useCallback, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, Copy, RefreshCw, FileText, Table2, GitCompare, Save, Download } from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";

type ViewMode = "raw" | "table" | "diff";

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

export function ConfigEditor() {
  const { projectRoot, pythonPath } = useAppStore();
  const [content, setContent] = useState("");
  const [filePath, setFilePath] = useState<string | null>(null);
  const [diffContent, setDiffContent] = useState("");
  const [diffPath, setDiffPath] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("raw");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [hydraTask, setHydraTask] = useState("test_sim");
  const [loadingHydra, setLoadingHydra] = useState(false);
  // Track the last-saved content to detect unsaved edits
  const savedContentRef = useRef("");

  const openFile = useCallback(async (target: "primary" | "diff") => {
    const path = (await open({
      filters: [{ name: "YAML / Config", extensions: ["yaml", "yml", "toml", "cfg", "ini"] }],
    })) as string | null;
    if (!path) return;

    setLoading(true);
    try {
      const text = await invoke<string>("read_text_file", { path });
      if (target === "primary") {
        setContent(text);
        setFilePath(path);
        savedContentRef.current = text;
      } else {
        setDiffContent(text);
        setDiffPath(path);
      }
    } catch (err) {
      toast.error("Failed to read file", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, []);

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

  const isDirty = content !== savedContentRef.current && filePath !== null;

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

            {/* View mode toggle */}
            <div className="ml-auto flex items-center gap-1 bg-canvas-elevated rounded-lg p-1">
              {(
                [
                  { mode: "raw", Icon: FileText, label: "Raw" },
                  { mode: "table", Icon: Table2, label: "Table" },
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
        <p className="text-xs text-canvas-muted font-mono truncate">{filePath}</p>
      )}

      {!content && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Open a YAML config file (e.g. <code className="font-mono text-xs mx-1">pruned_config.yaml</code> from a run directory).
        </div>
      )}

      {/* Raw view */}
      {content && viewMode === "raw" && (
        <textarea
          className="input-base font-mono text-xs h-[60vh] resize-y w-full"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          spellCheck={false}
        />
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
          <button
            onClick={() => openFile("diff")}
            className="btn-ghost flex items-center gap-2 text-sm"
          >
            <FolderOpen size={13} />
            {diffPath ? diffPath.split("/").pop() : "Open comparison file…"}
          </button>

          {!diffContent && (
            <div className="card text-canvas-muted text-sm">
              Open a second YAML file to compare against the primary config.
            </div>
          )}

          {diffContent && (
            <div className="card overflow-auto">
              <p className="text-xs text-canvas-muted mb-3">
                {changedKeys.length} difference(s) between{" "}
                <span className="font-mono">{filePath?.split("/").pop()}</span> and{" "}
                <span className="font-mono">{diffPath?.split("/").pop()}</span>
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

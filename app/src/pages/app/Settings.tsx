/**
 * Settings — persistent app configuration (§G.19).
 *
 * Stores all values in the Zustand persist store (Tauri Store plugin under the hood)
 * so they survive app restarts.
 *
 * Fields:
 *   Project Root    — the repo root where main.py lives; used by all launchers
 *   Python Path     — override the auto-detected Python executable (blank = auto)
 *   Theme           — dark / light
 *
 * Validation:
 *   - Project Root: Rust `validate_project_root` checks that main.py exists
 *   - Python Path:  Rust `probe_python` runs `<path> --version` and returns version string
 */
import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save as saveDialog } from "@tauri-apps/plugin-dialog";
import { CheckCircle, FolderOpen, Save, XCircle, Download, Upload, RefreshCw, Compass } from "lucide-react";
import { useStartupTiming } from "../../hooks/useStartupTiming";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import { useLayoutStore } from "../../store/layout";



type ValidationState = "idle" | "checking" | "ok" | "error";

interface FieldValidation {
  state: ValidationState;
  message: string;
}

const IDLE: FieldValidation = { state: "idle", message: "" };

function ValidationBadge({ v }: { v: FieldValidation }) {
  if (v.state === "idle" || v.state === "checking") return null;
  const ok = v.state === "ok";
  return (
    <p
      className={`flex items-center gap-1.5 text-xs mt-1 ${
        ok ? "text-accent-success" : "text-accent-danger"
      }`}
    >
      {ok ? <CheckCircle size={11} /> : <XCircle size={11} />}
      {v.message}
    </p>
  );
}

export function Settings() {
  const {
    projectRoot,
    pythonPath,
    theme,
    setProjectRoot,
    setPythonPath,
    setTheme,
  } = useAppStore();

  // Local drafts so the user can discard
  const [draftRoot, setDraftRoot] = useState(projectRoot);
  const [draftPython, setDraftPython] = useState(pythonPath);
  const [draftTheme, setDraftTheme] = useState(theme);

  const [rootValidation, setRootValidation] = useState<FieldValidation>(IDLE);
  const [pythonValidation, setPythonValidation] = useState<FieldValidation>(IDLE);
  const [appVersion, setAppVersion] = useState("…");
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const { firstPaintMs, prefetchMs, withinBudget } = useStartupTiming();
  const { setGuidedTourOpen, setGuidedTourStep } = useLayoutStore();

  useEffect(() => {
    invoke<string>("get_app_version")
      .then(setAppVersion)
      .catch(() => setAppVersion("0.1.0"));
  }, []);

  const isDirty =
    draftRoot.trim() !== projectRoot ||
    draftPython.trim() !== pythonPath ||
    draftTheme !== theme;

  const pickRoot = async () => {
    const path = (await open({ directory: true })) as string | null;
    if (path) {
      setDraftRoot(path);
      setRootValidation(IDLE);
    }
  };

  const validateRoot = async () => {
    const path = draftRoot.trim();
    if (!path) {
      setRootValidation({ state: "error", message: "Path is required" });
      return false;
    }
    setRootValidation({ state: "checking", message: "" });
    try {
      const msg = await invoke<string>("validate_project_root", { path });
      setRootValidation({ state: "ok", message: msg });
      return true;
    } catch (e) {
      setRootValidation({ state: "error", message: String(e) });
      return false;
    }
  };

  const validatePython = async () => {
    const pythonPath = draftPython.trim();
    if (!pythonPath) {
      // Empty means auto-detect — skip validation
      setPythonValidation(IDLE);
      return true;
    }
    setPythonValidation({ state: "checking", message: "" });
    try {
      const version = await invoke<string>("probe_python", { pythonPath });
      setPythonValidation({ state: "ok", message: version });
      return true;
    } catch (e) {
      setPythonValidation({ state: "error", message: String(e) });
      return false;
    }
  };

  const save = async () => {
    const rootOk = await validateRoot();
    const pythonOk = await validatePython();
    if (!rootOk || !pythonOk) {
      toast.error("Fix validation errors before saving");
      return;
    }
    setProjectRoot(draftRoot.trim());
    setPythonPath(draftPython.trim());
    setTheme(draftTheme);
    toast.success("Settings saved");
  };

  const discard = () => {
    setDraftRoot(projectRoot);
    setDraftPython(pythonPath);
    setDraftTheme(theme);
    setRootValidation(IDLE);
    setPythonValidation(IDLE);
  };

  const exportSettings = async () => {
    const path = (await saveDialog({
      filters: [{ name: "JSON", extensions: ["json"] }],
      defaultPath: "wsmart-studio-settings.json",
    })) as string | null;
    if (!path) return;
    const data = { projectRoot, pythonPath, theme };
    try {
      await invoke("write_text_file", { path, content: JSON.stringify(data, null, 2) });
      toast.success("Settings exported", { description: path.split("/").pop() });
    } catch (err) {
      toast.error("Export failed", { description: String(err) });
    }
  };

  const importSettings = async () => {
    const path = (await open({
      filters: [{ name: "JSON", extensions: ["json"] }],
    })) as string | null;
    if (!path) return;
    try {
      const text = await invoke<string>("read_text_file", { path });
      const data = JSON.parse(text) as Record<string, unknown>;
      if (typeof data.projectRoot === "string") setDraftRoot(data.projectRoot);
      if (typeof data.pythonPath === "string") setDraftPython(data.pythonPath);
      if (data.theme === "dark" || data.theme === "light") setDraftTheme(data.theme);
      setRootValidation(IDLE);
      setPythonValidation(IDLE);
      toast.success("Settings imported — review and save to apply");
    } catch (err) {
      toast.error("Import failed", { description: String(err) });
    }
  };

  return (
    <div className="max-w-xl space-y-6">
      {/* Project Root */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Project Root</h2>
        <p className="text-xs text-canvas-muted">
          The directory that contains <code className="font-mono">main.py</code>.
          All launchers use this as their working directory.
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            className="input-base flex-1 font-mono text-xs"
            value={draftRoot}
            onChange={(e) => {
              setDraftRoot(e.target.value);
              setRootValidation(IDLE);
            }}
            onBlur={validateRoot}
            placeholder="/path/to/WSmart-Route"
            spellCheck={false}
          />
          <button
            onClick={pickRoot}
            className="btn-ghost shrink-0 flex items-center gap-1.5 text-xs"
          >
            <FolderOpen size={13} />
            Browse
          </button>
        </div>
        {!draftRoot && (
          <p className="text-xs text-accent-warning">
            Required — launchers will be disabled without this.
          </p>
        )}
        <ValidationBadge v={rootValidation} />
      </div>

      {/* Python Executable */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Python Executable</h2>
        <p className="text-xs text-canvas-muted">
          Leave blank to auto-detect: Studio checks{" "}
          <code className="font-mono text-xs">&lt;project&gt;/.venv/bin/python</code>{" "}
          (uv venv) first, then <code className="font-mono text-xs">python3</code> on PATH.
        </p>
        <input
          type="text"
          className="input-base font-mono text-xs w-full"
          value={draftPython}
          onChange={(e) => {
            setDraftPython(e.target.value);
            setPythonValidation(IDLE);
          }}
          onBlur={validatePython}
          placeholder="(auto-detect)"
          spellCheck={false}
        />
        <ValidationBadge v={pythonValidation} />
      </div>

      {/* Theme */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Appearance</h2>
        <div className="flex gap-3">
          {(["dark", "light"] as const).map((t) => (
            <label
              key={t}
              className={`flex items-center gap-2 py-2 px-4 rounded-lg border cursor-pointer transition-colors ${
                draftTheme === t
                  ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                  : "border-canvas-border text-gray-400 hover:border-canvas-muted"
              }`}
            >
              <input
                type="radio"
                name="theme"
                value={t}
                checked={draftTheme === t}
                onChange={() => setDraftTheme(t)}
                className="sr-only"
              />
              <span className="text-sm capitalize">{t}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Save / Discard */}
      <div className="flex items-center gap-3">
        <button
          onClick={save}
          disabled={!isDirty}
          className="btn-primary flex items-center gap-2"
        >
          <Save size={14} />
          Save Settings
        </button>
        {isDirty && (
          <button onClick={discard} className="btn-ghost text-sm">
            Discard
          </button>
        )}
      </div>

      {/* Import / Export */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Backup & Restore</h2>
        <p className="text-xs text-canvas-muted">
          Export your settings to JSON for backup or team onboarding. Import populates the
          draft fields — review and click Save Settings to apply.
        </p>
        <div className="flex gap-3">
          <button
            onClick={exportSettings}
            className="btn-ghost flex items-center gap-2 text-sm"
          >
            <Download size={13} />
            Export Settings
          </button>
          <button
            onClick={importSettings}
            className="btn-ghost flex items-center gap-2 text-sm"
          >
            <Upload size={13} />
            Import Settings
          </button>
        </div>
      </div>

      {/* About */}
      <div className="card space-y-2 text-xs text-canvas-muted">
        <p className="font-semibold text-gray-300 text-sm">About WSmart-Route Studio</p>
        <p>Version: <span className="font-mono">{appVersion}</span></p>
        {firstPaintMs !== null && (
          <p>Startup to first paint: <span className="font-mono">{firstPaintMs} ms</span></p>
        )}
        {prefetchMs !== null && (
          <p>
            Route prefetch complete: <span className="font-mono">{prefetchMs} ms</span>
            {" · "}
            <span className={withinBudget ? "text-accent-success" : "text-accent-warning"}>
              {withinBudget ? "within 2s budget" : "over 2s budget"}
            </span>
          </p>
        )}
        <p>Runtime: Tauri 2.0 · React 19 · Rust</p>
        <button
          disabled={checkingUpdate}
          onClick={async () => {
            setCheckingUpdate(true);
            try {
              const result = await invoke<{
                available: boolean;
                current_version: string;
                latest_version: string | null;
                message: string;
              }>("check_for_updates");
              if (result.available) {
                toast.success(result.message, {
                  description: `${result.current_version} → ${result.latest_version}`,
                });
              } else {
                toast.info(result.message);
              }
            } catch (err) {
              toast.error("Update check failed", { description: String(err) });
            } finally {
              setCheckingUpdate(false);
            }
          }}
          className="btn-ghost text-xs flex items-center gap-1.5 mt-1"
        >
          <RefreshCw size={12} className={checkingUpdate ? "animate-spin" : ""} />
          Check for Updates
        </button>
        <button
          onClick={() => {
            setGuidedTourStep(0);
            setGuidedTourOpen(true);
          }}
          className="btn-ghost text-xs flex items-center gap-1.5 mt-1"
        >
          <Compass size={12} />
          Take Guided Tour
        </button>
        <p className="text-[10px]">Set <code className="font-mono">WSMART_UPDATE_URL</code> to a JSON manifest with a <code className="font-mono">version</code> field.</p>
        <p>ROADMAP: <code className="font-mono">docs/moon/ROADMAP.md §G</code></p>
      </div>
    </div>
  );
}

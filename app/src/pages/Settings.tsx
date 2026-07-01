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
import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { CheckCircle, FolderOpen, Save, XCircle } from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../store/app";

const VERSION = "0.1.0";

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

      {/* About */}
      <div className="card space-y-2 text-xs text-canvas-muted">
        <p className="font-semibold text-gray-300 text-sm">About WSmart-Route Studio</p>
        <p>Version: <span className="font-mono">{VERSION}</span></p>
        <p>Runtime: Tauri 2.0 · React 19 · Rust</p>
        <p>ROADMAP: <code className="font-mono">docs/moon/ROADMAP.md §G</code></p>
      </div>
    </div>
  );
}

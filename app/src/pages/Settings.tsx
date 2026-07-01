/**
 * Settings — persistent app configuration.
 *
 * Stores all values in the Zustand persist store (Tauri Store plugin under the hood)
 * so they survive app restarts.
 *
 * Fields:
 *   Project Root    — the repo root where main.py lives; used by all launchers
 *   Python Path     — override the auto-detected Python executable (blank = auto)
 *   Theme           — dark / light
 */
import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, Save } from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../store/app";

const VERSION = "0.1.0";

export function Settings() {
  const {
    projectRoot,
    pythonPath,
    theme,
    setProjectRoot,
    setPythonPath,
    setTheme,
  } = useAppStore();

  // Local drafts so the user can cancel
  const [draftRoot, setDraftRoot] = useState(projectRoot);
  const [draftPython, setDraftPython] = useState(pythonPath);
  const [draftTheme, setDraftTheme] = useState(theme);

  const pickRoot = async () => {
    const path = (await open({ directory: true })) as string | null;
    if (path) setDraftRoot(path);
  };

  const save = () => {
    setProjectRoot(draftRoot.trim());
    setPythonPath(draftPython.trim());
    setTheme(draftTheme);
    toast.success("Settings saved");
  };

  const isDirty =
    draftRoot !== projectRoot ||
    draftPython !== pythonPath ||
    draftTheme !== theme;

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
            onChange={(e) => setDraftRoot(e.target.value)}
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
            Required — simulation and training launchers will be disabled without this.
          </p>
        )}
      </div>

      {/* Python Executable */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Python Executable</h2>
        <p className="text-xs text-canvas-muted">
          Leave blank to auto-detect: Studio checks{" "}
          <code className="font-mono text-xs">&lt;project&gt;/.venv/bin/python</code> first,
          then <code className="font-mono text-xs">python3</code> on PATH.
        </p>
        <input
          type="text"
          className="input-base font-mono text-xs"
          value={draftPython}
          onChange={(e) => setDraftPython(e.target.value)}
          placeholder="(auto-detect)"
          spellCheck={false}
        />
        {draftPython && (
          <p className="text-xs text-canvas-muted">
            Will use <code className="font-mono">{draftPython}</code> for all subprocess spawns.
          </p>
        )}
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

      {/* Save */}
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
          <button
            onClick={() => {
              setDraftRoot(projectRoot);
              setDraftPython(pythonPath);
              setDraftTheme(theme);
            }}
            className="btn-ghost text-sm"
          >
            Discard
          </button>
        )}
      </div>

      {/* About */}
      <div className="card space-y-2 text-xs text-canvas-muted">
        <p className="font-semibold text-gray-300 text-sm">About WSmart-Route Studio</p>
        <p>Version: <span className="font-mono">{VERSION}</span></p>
        <p>Runtime: Tauri 2.0 · React 19 · Rust</p>
        <p>
          ROADMAP:{" "}
          <code className="font-mono">docs/moon/ROADMAP.md §G</code>
        </p>
      </div>
    </div>
  );
}

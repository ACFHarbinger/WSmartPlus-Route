import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, X, Zap } from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import { useLayoutStore } from "../../store/layout";

export function OnboardingDialog() {
  const { projectRoot, setProjectRoot, setMode } = useAppStore();
  const { onboardingDismissed, setOnboardingDismissed, guidedTourDismissed, setGuidedTourOpen, setGuidedTourStep } =
    useLayoutStore();
  const [draftRoot, setDraftRoot] = useState(projectRoot);
  const [validating, setValidating] = useState(false);


  const visible = !projectRoot && !onboardingDismissed;

  const pickRoot = useCallback(async () => {
    const path = (await open({ directory: true, title: "Select WSmart-Route project root" })) as string | null;
    if (path) {
      setDraftRoot(path);
    }
  }, []);

  const validateAndSave = useCallback(async () => {
    const path = draftRoot.trim();
    if (!path) {
      toast.error("Select a project root directory");
      return;
    }
    setValidating(true);
    try {
      await invoke<string>("validate_project_root", { path });
      setProjectRoot(path);
      setOnboardingDismissed(true);
      toast.success("Project root configured — launchers are now enabled");
      if (!guidedTourDismissed) {
        window.setTimeout(() => {
          setGuidedTourStep(0);
          setGuidedTourOpen(true);
        }, 600);
      }
    } catch (err) {
      toast.error("Invalid project root", { description: String(err) });
    } finally {
      setValidating(false);
    }
  }, [draftRoot, setProjectRoot, setOnboardingDismissed, guidedTourDismissed, setGuidedTourOpen, setGuidedTourStep]);

  const dismiss = useCallback(() => {
    setOnboardingDismissed(true);
  }, [setOnboardingDismissed]);

  if (!visible) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="card max-w-lg w-full space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <Zap size={18} className="text-accent-primary" />
            <h2 className="text-sm font-semibold text-gray-100">Welcome to WSmart-Route Studio</h2>
          </div>
          <button onClick={dismiss} className="btn-ghost p-1" title="Dismiss (configure later in Settings)">
            <X size={14} />
          </button>
        </div>

        <p className="text-xs text-canvas-muted leading-relaxed">
          Studio needs the repository root where <code className="font-mono">main.py</code> lives.
          This enables simulation launchers, output browsing, and Python subprocess integration.
        </p>

        <ol className="text-xs text-canvas-muted space-y-1.5 list-decimal list-inside">
          <li>Clone or locate your WSmart-Route repository</li>
          <li>Select the directory containing <code className="font-mono">main.py</code></li>
          <li>Confirm — launchers and file browsers unlock immediately</li>
        </ol>

        <div className="flex gap-2">
          <input
            type="text"
            className="input-base flex-1 font-mono text-xs"
            value={draftRoot}
            onChange={(e) => setDraftRoot(e.target.value)}
            placeholder="/path/to/WSmart-Route"
            spellCheck={false}
          />
          <button onClick={pickRoot} className="btn-ghost shrink-0 flex items-center gap-1.5 text-xs">
            <FolderOpen size={13} />
            Browse
          </button>
        </div>

        <div className="flex items-center gap-3 pt-1">
          <button
            onClick={validateAndSave}
            disabled={validating || !draftRoot.trim()}
            className="btn-primary text-sm"
          >
            {validating ? "Validating…" : "Get Started"}
          </button>
          <button
            onClick={() => {
              dismiss();
              setMode("settings");
            }}
            className="btn-ghost text-xs"
          >
            Open Settings instead
          </button>
        </div>
      </div>
    </div>
  );
}

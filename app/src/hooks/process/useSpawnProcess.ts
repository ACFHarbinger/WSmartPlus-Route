import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";

interface SpawnOptions {
  id?: string;
  pythonArgs: string[];
  workingDir: string;
}

/**
 * Wraps `spawn_python_process` with loading state and error toasts.
 *
 * Uses `pythonPath` from the app store so a user-configured executable
 * (Settings page) overrides the auto-detected one.
 *
 * The Rust command emits a `process:spawn` event that `useProcessMonitor`
 * (mounted at root) picks up — so callers do NOT need to call `addProcess` directly.
 */
export function useSpawnProcess() {
  const pythonPath = useAppStore((s) => s.pythonPath);
  const [launching, setLaunching] = useState(false);

  const spawn = useCallback(async (opts: SpawnOptions): Promise<number | null> => {
    const id = opts.id ?? `proc_${Date.now()}`;
    setLaunching(true);
    try {
      const pid = await invoke<number>("spawn_python_process", {
        id,
        pythonArgs: opts.pythonArgs,
        workingDir: opts.workingDir,
        pythonExecutable: pythonPath || null,
      });
      toast.success(`Process started (PID ${pid})`, { description: id });
      return pid;
    } catch (err) {
      toast.error("Failed to start process", {
        description: err instanceof Error ? err.message : String(err),
      });
      return null;
    } finally {
      setLaunching(false);
    }
  }, []);

  return { spawn, launching };
}

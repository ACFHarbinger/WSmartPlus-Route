import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { toast } from "sonner";
import { useProcessStore } from "../store/process";
import type { ProcessSpawned, StdoutLine, StatusUpdate } from "../types";

/**
 * Subscribes to `process:spawn`, `process:stdout`, and `process:status` Tauri events
 * and funnels them into the process store.
 *
 * Mount this once at the app root so all pages share the same process state.
 * `process:spawn` is emitted by Rust right after a process starts, so the frontend
 * always learns about a process even if it was spawned from a different page.
 */
export function useProcessMonitor() {
  const { addProcess, appendLog, updateStatus } = useProcessStore();
  const unlistenRefs = useRef<Array<() => void>>([]);

  useEffect(() => {
    const setup = async () => {
      const ulSpawn = await listen<ProcessSpawned>("process:spawn", (event) => {
        const { id, command, pid, start_time } = event.payload;
        addProcess({ id, command, pid, status: "running", startTime: start_time });
      });

      const ulStdout = await listen<StdoutLine>("process:stdout", (event) => {
        appendLog(event.payload.id, event.payload.line);
      });

      const ulStatus = await listen<StatusUpdate>("process:status", (event) => {
        const { id, status, exit_code } = event.payload;
        updateStatus(id, status, exit_code ?? undefined);

        // Show a toast for terminal states
        const label = id.split("_")[0]; // "train", "eval", "sim", etc.
        if (status === "completed") {
          toast.success(`${label} completed`, { description: id, duration: 4000 });
        } else if (status === "failed") {
          toast.error(`${label} failed`, {
            description: `${id} — exit ${exit_code ?? "?"}`,
            duration: 6000,
          });
        } else if (status === "cancelled") {
          toast.info(`${label} cancelled`, { description: id, duration: 3000 });
        }
      });

      unlistenRefs.current = [ulSpawn, ulStdout, ulStatus];
    };

    setup().catch(console.error);

    return () => {
      unlistenRefs.current.forEach((ul) => ul());
    };
  }, [addProcess, appendLog, updateStatus]);
}

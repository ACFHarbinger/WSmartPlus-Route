import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
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
        updateStatus(
          event.payload.id,
          event.payload.status,
          event.payload.exit_code ?? undefined
        );
      });

      unlistenRefs.current = [ulSpawn, ulStdout, ulStatus];
    };

    setup().catch(console.error);

    return () => {
      unlistenRefs.current.forEach((ul) => ul());
    };
  }, [addProcess, appendLog, updateStatus]);
}

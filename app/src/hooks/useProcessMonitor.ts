import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { useProcessStore } from "../store/process";
import type { StdoutLine, StatusUpdate } from "../types";

/**
 * Subscribes to `process:stdout` and `process:status` Tauri events
 * and funnels them into the process store.
 *
 * Mount this once at the app root so all pages share the same process state.
 */
export function useProcessMonitor() {
  const { appendLog, updateStatus } = useProcessStore();
  const unlistenRefs = useRef<Array<() => void>>([]);

  useEffect(() => {
    const setup = async () => {
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

      unlistenRefs.current = [ulStdout, ulStatus];
    };

    setup().catch(console.error);

    return () => {
      unlistenRefs.current.forEach((ul) => ul());
    };
  }, [appendLog, updateStatus]);
}

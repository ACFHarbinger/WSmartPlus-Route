import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from "@tauri-apps/plugin-notification";
import { toast } from "sonner";
import { useProcessStore } from "../store/process";
import { useSimStore } from "../store/sim";
import { parsePolicyVizLine } from "../utils/policyTelemetry";
import { parseSimFailureLine } from "../utils/simFailure";
import type { ProcessSpawned, StdoutLine, StatusUpdate } from "../types";

async function maybeSendOsNotification(title: string, body: string) {
  if (!document.hidden) return;
  try {
    let granted = await isPermissionGranted();
    if (!granted) {
      const result = await requestPermission();
      granted = result === "granted";
    }
    if (granted) {
      await sendNotification({ title, body });
    }
  } catch {
    // Notification plugin unavailable or denied — toast already shown
  }
}

/**
 * Subscribes to `process:spawn`, `process:stdout`, and `process:status` Tauri events
 * and funnels them into the process store.
 *
 * Mount this once at the app root so all pages share the same process state.
 * `process:spawn` is emitted by Rust right after a process starts, so the frontend
 * always learns about a process even if it was spawned from a different page.
 */
export function useProcessMonitor() {
  const { addProcess, appendLog, updateStatus, processes } = useProcessStore();
  const unlistenRefs = useRef<Array<() => void>>([]);

  useEffect(() => {
    const setup = async () => {
      const ulSpawn = await listen<ProcessSpawned>("process:spawn", (event) => {
        const { id, command, pid, start_time } = event.payload;
        addProcess({ id, command, pid, status: "running", startTime: start_time });
      });

      const ulStdout = await listen<StdoutLine>("process:stdout", (event) => {
        appendLog(event.payload.id, event.payload.line);
        const viz = parsePolicyVizLine(event.payload.line);
        if (viz) {
          useSimStore.getState().addPolicyVizEntry(viz);
        }
        const failure = parseSimFailureLine(event.payload.line);
        if (failure) {
          useSimStore.getState().addFailureEntry(failure);
        }
      });

      const ulStatus = await listen<StatusUpdate>("process:status", (event) => {
        const { id, status, exit_code } = event.payload;
        updateStatus(id, status, exit_code ?? undefined);

        const label = id.split("_")[0];
        if (status === "completed") {
          toast.success(`${label} completed`, { description: id, duration: 4000 });
          void maybeSendOsNotification(
            `${label} completed`,
            id
          );
        } else if (status === "failed") {
          toast.error(`${label} failed`, {
            description: `${id} — exit ${exit_code ?? "?"}`,
            duration: 6000,
          });
          void maybeSendOsNotification(
            `${label} failed`,
            `${id} — exit ${exit_code ?? "?"}`
          );
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

  // Global Ctrl+. shortcut — cancel the first running process (§D.7)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key !== ".") return;
      const target = e.target as HTMLElement;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target.isContentEditable
      ) return;

      const running = Object.values(processes).find((p) => p.status === "running");
      if (!running) return;

      e.preventDefault();
      invoke("cancel_process", { id: running.id }).catch(console.error);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [processes]);
}

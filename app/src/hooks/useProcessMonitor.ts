import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from "@tauri-apps/plugin-notification";
import { toast } from "sonner";
import { applyStoreRecentHandoff } from "./useRecentHandoff";
import { useProcessStore } from "../store/process";
import { useSimStore } from "../store/sim";
import { checkpointPathFromEvalCommand } from "../utils/evalResults";
import { isEvalProcess } from "../utils/launcherProcess";
import { outputRunPathFromLogLines } from "../utils/outputRunPath";
import { extractJsonlPathFromLogLines } from "../utils/policyTelemetryTrends";
import { parsePolicyVizLine } from "../utils/policyTelemetry";
import { parseSimFailureLine } from "../utils/simFailure";
import { isTrainOrHpoProcess } from "../utils/trainingProcess";
import { trainingRunPathFromLogLines } from "../utils/trainingRunPath";
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
 * Sonner action buttons for process terminal toasts when stdout yields a
 * navigable artefact (§D.8 / §G.1 / §G.14 / §G.16 / §G.17).
 */
function processToastHandoffOptions(id: string): {
  action?: { label: string; onClick: () => void };
  cancel?: { label: string; onClick: () => void };
  duration?: number;
} {
  const proc = useProcessStore.getState().processes[id];
  if (!proc) return {};

  const lines = proc.logLines;
  const jsonl = extractJsonlPathFromLogLines(lines);
  if (jsonl) {
    return {
      duration: 8000,
      action: {
        label: "Summary",
        onClick: () => {
          applyStoreRecentHandoff(jsonl, "log");
        },
      },
      cancel: {
        label: "Monitor",
        onClick: () => {
          applyStoreRecentHandoff(jsonl, "log", { mode: "simulation" });
        },
      },
    };
  }

  if (isTrainOrHpoProcess(id, proc.command)) {
    const trainPath = trainingRunPathFromLogLines(lines);
    const runPath = outputRunPathFromLogLines(lines);
    if (trainPath && runPath) {
      return {
        duration: 8000,
        action: {
          label: "Training",
          onClick: () => {
            applyStoreRecentHandoff(trainPath, "training");
          },
        },
        cancel: {
          label: "Output",
          onClick: () => {
            applyStoreRecentHandoff(runPath, "run");
          },
        },
      };
    }
    if (trainPath) {
      return {
        duration: 8000,
        action: {
          label: "Training",
          onClick: () => {
            applyStoreRecentHandoff(trainPath, "training");
          },
        },
      };
    }
    if (runPath) {
      return {
        duration: 8000,
        action: {
          label: "Output",
          onClick: () => {
            applyStoreRecentHandoff(runPath, "run");
          },
        },
      };
    }
    return {};
  }

  if (isEvalProcess(id, proc.command)) {
    const checkpoint = checkpointPathFromEvalCommand(proc.command);
    const runPath = outputRunPathFromLogLines(lines);
    if (checkpoint && runPath) {
      return {
        duration: 8000,
        action: {
          label: "Eval",
          onClick: () => {
            applyStoreRecentHandoff(checkpoint, "checkpoint");
          },
        },
        cancel: {
          label: "Output",
          onClick: () => {
            applyStoreRecentHandoff(runPath, "run");
          },
        },
      };
    }
    if (checkpoint) {
      return {
        duration: 8000,
        action: {
          label: "Eval",
          onClick: () => {
            applyStoreRecentHandoff(checkpoint, "checkpoint");
          },
        },
      };
    }
    if (runPath) {
      return {
        duration: 8000,
        action: {
          label: "Output",
          onClick: () => {
            applyStoreRecentHandoff(runPath, "run");
          },
        },
      };
    }
    return {};
  }

  const runPath = outputRunPathFromLogLines(lines);
  if (runPath) {
    return {
      duration: 8000,
      action: {
        label: "Output",
        onClick: () => {
          applyStoreRecentHandoff(runPath, "run");
        },
      },
    };
  }

  return {};
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
        const handoffOpts = processToastHandoffOptions(id);
        if (status === "completed") {
          toast.success(`${label} completed`, {
            description: id,
            duration: handoffOpts.duration ?? 4000,
            action: handoffOpts.action,
            cancel: handoffOpts.cancel,
          });
          void maybeSendOsNotification(`${label} completed`, id);
        } else if (status === "failed") {
          toast.error(`${label} failed`, {
            description: `${id} — exit ${exit_code ?? "?"}`,
            duration: handoffOpts.duration ?? 6000,
            action: handoffOpts.action,
            cancel: handoffOpts.cancel,
          });
          void maybeSendOsNotification(
            `${label} failed`,
            `${id} — exit ${exit_code ?? "?"}`
          );
        } else if (status === "cancelled") {
          toast.info(`${label} cancelled`, {
            description: id,
            duration: handoffOpts.duration ?? 3000,
            action: handoffOpts.action,
            cancel: handoffOpts.cancel,
          });
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

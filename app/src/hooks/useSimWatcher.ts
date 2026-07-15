import { useEffect } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { useSimStore } from "../store/sim";
import type { DayLogEntry, PolicyVizEntry } from "../types";

/**
 * Starts the Rust file-watcher for a simulation log file and pipes
 * new `DayLogEntry` objects into the sim store as they arrive.
 *
 * Replaces Streamlit's `stream_log_file()` + `time.sleep()` + `st.rerun()` pattern.
 * The Rust backend polls the file every 200 ms and emits `sim:day_update` events
 * only when new `GUI_DAY_LOG_START:` lines are found — zero unnecessary re-renders.
 */
export function useSimWatcher(logPath: string | null) {
  const { addEntry, addPolicyVizEntry, setWatching } = useSimStore();

  useEffect(() => {
    if (!logPath) return;

    let unlisten: (() => void) | undefined;

    (async () => {
      const ulDay = await listen<DayLogEntry>("sim:day_update", (event) => {
        addEntry(event.payload);
      });
      const ulViz = await listen<PolicyVizEntry>("sim:policy_viz_update", (event) => {
        addPolicyVizEntry(event.payload);
      });
      unlisten = () => {
        ulDay();
        ulViz();
      };

      try {
        await invoke("start_sim_watcher", { path: logPath });
        setWatching(true);
      } catch (err) {
        console.error("Failed to start sim watcher:", err);
      }
    })();

    return () => {
      unlisten?.();
      invoke("stop_sim_watcher").catch(() => {});
      setWatching(false);
    };
  }, [logPath, addEntry, addPolicyVizEntry, setWatching]);
}

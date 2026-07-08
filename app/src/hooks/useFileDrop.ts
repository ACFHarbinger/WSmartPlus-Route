import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

/** Listen for OS file drops on the Tauri window (§G.8 / §G.14). */
export function useFileDrop(onDrop: (paths: string[]) => void, enabled = true) {
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    let unlisten: (() => void) | undefined;

    void getCurrentWindow()
      .onDragDropEvent((event) => {
        if (cancelled) return;
        const { payload } = event;
        if (payload.type === "enter" || payload.type === "over") {
          setDragging(true);
        } else if (payload.type === "leave") {
          setDragging(false);
        } else if (payload.type === "drop") {
          setDragging(false);
          onDrop(payload.paths);
        }
      })
      .then((fn) => {
        if (cancelled) fn();
        else unlisten = fn;
      })
      .catch(() => {});

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [onDrop, enabled]);

  return dragging;
}

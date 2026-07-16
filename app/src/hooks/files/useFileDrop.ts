import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

/** Listen for OS file drops on the Tauri window (§G.8 / §G.14). */
export function useFileDrop(
  onDrop: (paths: string[]) => void,
  enabled = true,
  onDraggingChange?: (dragging: boolean) => void
) {
  const [dragging, setDragging] = useState(false);

  const setDrag = (value: boolean) => {
    setDragging(value);
    onDraggingChange?.(value);
  };

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    let unlisten: (() => void) | undefined;

    void getCurrentWindow()
      .onDragDropEvent((event) => {
        if (cancelled) return;
        const { payload } = event;
        if (payload.type === "enter" || payload.type === "over") {
          setDrag(true);
        } else if (payload.type === "leave") {
          setDrag(false);
        } else if (payload.type === "drop") {
          setDrag(false);
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
  }, [onDrop, enabled, onDraggingChange]);

  return dragging;
}

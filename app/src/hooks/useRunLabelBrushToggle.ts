/**
 * Toggle global ``run_label`` brush on path/file chip clicks (§G.14–§G.16 / §D.7).
 */
import { useCallback } from "react";
import { useGlobalFiltersStore } from "../store/filters";

export function useRunLabelBrushToggle() {
  const activeRunLabel = useGlobalFiltersStore((s) => s.runLabel);
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);

  const handleRunLabelClick = useCallback(
    (label: string) => {
      setRunLabel(activeRunLabel === label ? null : label);
    },
    [activeRunLabel, setRunLabel]
  );

  const isBrushActive = useCallback(
    (label: string) => Boolean(activeRunLabel && activeRunLabel === label),
    [activeRunLabel]
  );

  return { activeRunLabel, handleRunLabelClick, isBrushActive };
}

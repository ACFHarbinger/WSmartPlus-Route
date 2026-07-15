/**
 * Derive ``run_label`` from process stdout and sync the global brush (§G.9–§G.18 / §D.7).
 */
import { useEffect, useMemo } from "react";
import { useGlobalFiltersStore } from "../store/filters";
import { runLabelFromLogLines } from "../utils/policyTelemetryTrends";

export function useProcessRunLabelBrush(
  processId: string | null,
  logLines: string[] | undefined
): string | null {
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const runLabel = useMemo(
    () => (processId ? runLabelFromLogLines(logLines ?? [], processId) : null),
    [processId, logLines]
  );

  useEffect(() => {
    if (!processId || !runLabel) return;
    setRunLabel(runLabel);
  }, [processId, runLabel, setRunLabel]);

  return runLabel;
}

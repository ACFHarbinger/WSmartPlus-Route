/**
 * Derive ``run_label`` from process stdout and sync the global brush (§G.9–§G.18 / §D.7).
 */
import { useEffect, useMemo } from "react";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { runLabelFromLogLines } from "../../utils/benchmark/policyTelemetryTrends";

export function useProcessRunLabelBrush(
  processId: string | null,
  logLines: string[] | undefined
): string | null {
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const projectRoot = useAppStore((s) => s.projectRoot);
  const runLabel = useMemo(
    () =>
      processId ? runLabelFromLogLines(logLines ?? [], processId, projectRoot) : null,
    [processId, logLines, projectRoot]
  );

  useEffect(() => {
    if (!processId || !runLabel) return;
    setRunLabel(runLabel);
  }, [processId, runLabel, setRunLabel]);

  return runLabel;
}

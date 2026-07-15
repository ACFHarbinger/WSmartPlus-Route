/**
 * Derive ``run_label`` from a log/run path and sync the global brush (§G.14–§G.16 / §D.7).
 */
import { useEffect, useMemo } from "react";
import { useGlobalFiltersStore } from "../store/filters";
import { runLabelFromPath } from "../utils/policyTelemetryTrends";

export function useLogPathRunLabelBrush(logPath: string | null): string | null {
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const runLabel = useMemo(
    () => (logPath ? runLabelFromPath(logPath) : null),
    [logPath]
  );

  useEffect(() => {
    if (!logPath || !runLabel) return;
    setRunLabel(runLabel);
  }, [logPath, runLabel, setRunLabel]);

  return runLabel;
}

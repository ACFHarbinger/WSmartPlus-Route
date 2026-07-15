/**
 * Derive ``run_label`` from a log/run path and sync the global brush (§G.14–§G.16 / §D.7).
 */
import { useEffect, useMemo } from "react";
import { useAppStore } from "../store/app";
import { useGlobalFiltersStore } from "../store/filters";
import { resolveLocalProjectPath } from "../utils/outputRunPath";
import { runLabelFromPath } from "../utils/policyTelemetryTrends";

export function useLogPathRunLabelBrush(logPath: string | null): string | null {
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const projectRoot = useAppStore((s) => s.projectRoot);
  const runLabel = useMemo(() => {
    if (!logPath) return null;
    const resolved = resolveLocalProjectPath(logPath, projectRoot) ?? logPath;
    return runLabelFromPath(resolved);
  }, [logPath, projectRoot]);

  useEffect(() => {
    if (!logPath || !runLabel) return;
    setRunLabel(runLabel);
  }, [logPath, runLabel, setRunLabel]);

  return runLabel;
}

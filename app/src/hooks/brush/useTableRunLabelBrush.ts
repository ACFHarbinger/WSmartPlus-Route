/**
 * Sync global ``run_label`` brush from a DuckDB table with a single distinct label (§G.6 / §D.7).
 */
import { useEffect, useMemo } from "react";
import { useGlobalFiltersStore } from "../../store/filters";

export function useTableRunLabelBrush(
  tableName: string | null,
  tableRunLabels: string[],
  /** Skip when an ingest path already drives the brush via ``useLogPathRunLabelBrush``. */
  skip = false
): string | null {
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const singleLabel = useMemo(
    () => (tableRunLabels.length === 1 ? tableRunLabels[0]! : null),
    [tableRunLabels]
  );

  useEffect(() => {
    if (skip || !tableName || !singleLabel) return;
    setRunLabel(singleLabel);
  }, [skip, tableName, singleLabel, setRunLabel]);

  return singleLabel;
}

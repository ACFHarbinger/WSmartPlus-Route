import { useCallback, useMemo } from "react";
import { useGlobalFiltersStore } from "../store/filters";
import { resolveBrushedRunLabels, type CityRunSlice } from "../utils/cityComparison";

/**
 * Portfolio ``run_label`` brush shared across Summary / Benchmark / City views (§G.6).
 * City chart clicks expand to all runs in the group; single-run clicks set global ``runLabel``.
 */
export function usePortfolioRunBrush(runs: CityRunSlice[]) {
  const { runLabel, brushedCity, setRunLabel, setBrushedCity } = useGlobalFiltersStore();

  const runLabels = useMemo(() => runs.map((r) => r.label), [runs]);

  const brushedRunLabels = useMemo(
    () => resolveBrushedRunLabels(runLabels, runLabel, brushedCity),
    [runLabels, runLabel, brushedCity]
  );

  const handleCityClick = useCallback(
    (city: string) => {
      setRunLabel(null);
      setBrushedCity(brushedCity === city ? null : city);
    },
    [brushedCity, setRunLabel, setBrushedCity]
  );

  const handleRunLabelClick = useCallback(
    (label: string) => {
      setBrushedCity(null);
      setRunLabel(runLabel === label ? null : label);
    },
    [runLabel, setRunLabel, setBrushedCity]
  );

  return {
    runLabels,
    runLabel,
    brushedCity,
    brushedRunLabels,
    handleCityClick,
    handleRunLabelClick,
  };
}

import { useCallback, useMemo, useState } from "react";
import { useGlobalFiltersStore } from "../store/filters";
import type { CityRunSlice } from "../utils/cityComparison";

/**
 * Portfolio ``run_label`` brush shared across Summary / Benchmark / City views (§G.6).
 * City chart clicks expand to all runs in the group; single-run clicks set global ``runLabel``.
 */
export function usePortfolioRunBrush(
  runs: CityRunSlice[],
  cityGroups: Array<[string, CityRunSlice[]]>
) {
  const { runLabel, setRunLabel } = useGlobalFiltersStore();
  const [brushedCity, setBrushedCity] = useState<string | null>(null);

  const runLabels = useMemo(() => runs.map((r) => r.label), [runs]);

  const brushedRunLabels = useMemo(() => {
    if (runLabel) return [runLabel];
    if (!brushedCity) return null;
    const group = cityGroups.find(([city]) => city === brushedCity);
    if (!group) return null;
    return group[1].map((r) => r.label);
  }, [runLabel, brushedCity, cityGroups]);

  const handleCityClick = useCallback(
    (city: string) => {
      setRunLabel(null);
      setBrushedCity((current) => (current === city ? null : city));
    },
    [setRunLabel]
  );

  const handleRunLabelClick = useCallback(
    (label: string) => {
      setBrushedCity(null);
      setRunLabel(runLabel === label ? null : label);
    },
    [runLabel, setRunLabel]
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

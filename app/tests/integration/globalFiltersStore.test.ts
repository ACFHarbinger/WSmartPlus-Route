/**
 * Integration test: the global filter store drives the shared chart log-scale
 * helpers the same way GlobalFilterBar + analytics panels consume them (§G.7).
 */
import { beforeEach, describe, expect, it } from "vitest";
import { useGlobalFiltersStore } from "../../src/store/filters";
import { chartMetricYAxisType, displayBarValue } from "../../src/utils/charts/chartLogScale";
import { symlog } from "../../src/utils/charts/symlog";

describe("useGlobalFiltersStore", () => {
  beforeEach(() => {
    useGlobalFiltersStore.getState().reset();
  });

  it("starts with a clean slate", () => {
    const s = useGlobalFiltersStore.getState();
    expect(s.policy).toBeNull();
    expect(s.runLabel).toBeNull();
    expect(s.logScale).toBe(false);
  });

  it("brushes propagate through setters and reset clears them", () => {
    const s = useGlobalFiltersStore.getState();
    s.setPolicy("HGS");
    s.setRunLabel("run_42");
    s.setBrushedCity("Rio Maior");
    s.setSampleId(7);
    expect(useGlobalFiltersStore.getState()).toMatchObject({
      policy: "HGS",
      runLabel: "run_42",
      brushedCity: "Rio Maior",
      sampleId: 7,
    });
    useGlobalFiltersStore.getState().reset();
    expect(useGlobalFiltersStore.getState().policy).toBeNull();
    expect(useGlobalFiltersStore.getState().sampleId).toBeNull();
  });

  it("the logScale toggle switches chart transforms exactly like the panels do", () => {
    const { setLogScale } = useGlobalFiltersStore.getState();

    let logScale = useGlobalFiltersStore.getState().logScale;
    expect(displayBarValue(50, "overflows", logScale)).toBe(50);
    expect(chartMetricYAxisType("profit", logScale)).toBe("value");

    setLogScale(true);
    logScale = useGlobalFiltersStore.getState().logScale;
    expect(displayBarValue(50, "overflows", logScale)).toBeCloseTo(symlog(50));
    expect(chartMetricYAxisType("profit", logScale)).toBe("log");
  });
});

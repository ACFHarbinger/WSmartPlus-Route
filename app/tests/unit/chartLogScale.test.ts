import { describe, expect, it } from "vitest";
import {
  chartMetricDisplay,
  chartMetricUsesSymlog,
  chartMetricYAxisType,
  displayBarValue,
  isLogScaleMetric,
  isOverflowMetric,
  transformMatrixLogScale,
} from "../../src/utils/charts/chartLogScale";
import { symlog } from "../../src/utils/charts/symlog";

describe("metric classification", () => {
  it("detects overflow metrics", () => {
    expect(isOverflowMetric("overflows")).toBe(true);
    expect(isOverflowMetric("mean_overflow")).toBe(true);
    expect(isOverflowMetric("profit")).toBe(false);
  });

  it("detects log-scale metrics", () => {
    for (const k of ["loss", "total_cost", "profit", "km", "attention_weight"]) {
      expect(isLogScaleMetric(k)).toBe(true);
    }
    expect(isLogScaleMetric("policy_name")).toBe(false);
  });
});

describe("chartMetricDisplay", () => {
  it("passes values through when log scale is off", () => {
    expect(chartMetricDisplay(42, "profit", false)).toBe(42);
  });

  it("symlogs overflow metrics", () => {
    expect(chartMetricDisplay(100, "overflows", true)).toBeCloseTo(symlog(100));
  });

  it("clamps non-overflow log metrics to a positive floor", () => {
    expect(chartMetricDisplay(0, "profit", true)).toBe(1e-8);
    expect(chartMetricDisplay(-5, "km", true)).toBe(1e-8);
  });

  it("returns null for non-finite input", () => {
    expect(chartMetricDisplay(null, "profit", true)).toBeNull();
    expect(chartMetricDisplay(Number.NaN, "profit", true)).toBeNull();
  });
});

describe("axis helpers", () => {
  it("uses value axis + symlog transform for overflows", () => {
    expect(chartMetricYAxisType("overflows", true)).toBe("value");
    expect(chartMetricUsesSymlog("overflows", true)).toBe(true);
  });

  it("uses log axis for other log metrics", () => {
    expect(chartMetricYAxisType("profit", true)).toBe("log");
    expect(chartMetricUsesSymlog("profit", true)).toBe(false);
  });

  it("falls back to value axis when log scale is off", () => {
    expect(chartMetricYAxisType("profit", false)).toBe("value");
  });
});

describe("displayBarValue / transformMatrixLogScale", () => {
  it("bar values match the display transform", () => {
    expect(displayBarValue(7, "overflows", true)).toBeCloseTo(symlog(7));
    expect(displayBarValue(7, "name", true)).toBe(7);
    expect(displayBarValue(7, "overflows", false)).toBe(7);
  });

  it("matrix transform is a no-op when log scale is off", () => {
    const m = [[1, 2]];
    expect(transformMatrixLogScale(m, "overflows", false)).toBe(m);
  });

  it("matrix transform maps finite cells and keeps non-finite ones", () => {
    const out = transformMatrixLogScale([[100, Number.NaN]], "overflows", true);
    expect(out[0][0]).toBeCloseTo(symlog(100));
    expect(Number.isNaN(out[0][1])).toBe(true);
  });
});

import { describe, expect, it } from "vitest";
import { paretoFront, paretoStepLine, type ParetoPoint } from "../../src/utils/benchmark/pareto";

const p = (id: string, x: number, y: number): ParetoPoint => ({ id, x, y });

describe("paretoFront (maximise x, minimise y)", () => {
  it("keeps non-dominated points only", () => {
    const pts = [p("a", 1, 1), p("b", 2, 2), p("c", 3, 3), p("d", 2, 0.5)];
    const front = paretoFront(pts).map((q) => q.id);
    expect(front).toContain("c"); // best x
    expect(front).toContain("d"); // best y
    expect(front).not.toContain("a"); // dominated by d
    expect(front).not.toContain("b"); // dominated by d
  });

  it("returns all points when none dominates", () => {
    const pts = [p("a", 1, 1), p("b", 2, 2), p("c", 3, 3)];
    expect(paretoFront(pts)).toHaveLength(3);
  });

  it("keeps duplicated coordinates (neither strictly dominates)", () => {
    const pts = [p("a", 1, 1), p("b", 1, 1)];
    expect(paretoFront(pts)).toHaveLength(2);
  });

  it("handles empty input", () => {
    expect(paretoFront([])).toEqual([]);
  });
});

describe("paretoStepLine", () => {
  it("returns empty for empty front", () => {
    expect(paretoStepLine([])).toEqual([]);
  });

  it("emits horizontal-then-vertical steps sorted by x", () => {
    const steps = paretoStepLine([p("b", 3, 1), p("a", 1, 5)]);
    expect(steps).toEqual([
      [1, 5],
      [3, 5],
      [3, 1],
    ]);
  });

  it("single point yields a single coordinate", () => {
    expect(paretoStepLine([p("a", 2, 4)])).toEqual([[2, 4]]);
  });
});

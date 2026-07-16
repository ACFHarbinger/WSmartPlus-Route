import { describe, expect, it } from "vitest";
import { symexp, symlog } from "../../src/utils/charts/symlog";

describe("symlog / symexp", () => {
  it("preserves zero", () => {
    expect(symlog(0)).toBe(0);
    expect(symexp(0)).toBe(0);
  });

  it("is identity inside the linear threshold", () => {
    expect(symlog(0.5)).toBe(0.5);
    expect(symlog(-0.5)).toBe(-0.5);
    expect(symlog(1)).toBe(1);
  });

  it("compresses values above the threshold logarithmically", () => {
    expect(symlog(10)).toBeCloseTo(2); // 1 + log10(10)
    expect(symlog(100)).toBeCloseTo(3);
    expect(symlog(-100)).toBeCloseTo(-3);
  });

  it("symexp inverts symlog across the range", () => {
    for (const v of [-1234, -10, -0.7, 0, 0.3, 1, 5, 42, 1e6]) {
      expect(symexp(symlog(v))).toBeCloseTo(v, 6);
    }
  });

  it("respects a custom linthresh", () => {
    expect(symlog(5, 10)).toBe(5); // still linear
    expect(symlog(100, 10)).toBeCloseTo(10 + Math.log10(10));
    expect(symexp(symlog(100, 10), 10)).toBeCloseTo(100);
  });
});

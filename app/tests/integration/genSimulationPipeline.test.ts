/**
 * Integration test for the native §H.1 data engine: filename decoding →
 * filtering → variant aggregation → scenario/context building → Pareto front,
 * exercised together the way Report Studio drives them.
 */
import { describe, expect, it } from "vitest";
import {
  aggregate,
  buildCtx,
  detectScenarios,
  filterData,
  metricValues,
  paretoIndices,
  parseAreaDir,
  parseFilename,
  regionLabel,
  scenarioLabel,
  type SimRow,
} from "../../src/gen/data/simulation";

const baseRow: Omit<SimRow, "variant"> = {
  city: "Rio Maior",
  N: 20,
  dist: "Empirical",
  improver: "CLS",
  strategy: "LM",
  cf: "CF70",
  sl_var: "",
  acceptance: "",
  constructor: "HGS",
  overflows: 10,
  kg: 100,
  ncol: 5,
  kg_lost: 0,
  km: 50,
  kgkm: 2,
  reward: 1,
  profit: 20,
  time: 3,
  days: 30,
};

const row = (over: Partial<SimRow>): SimRow => ({ ...baseRow, variant: "", ...over });

describe("filename → metadata decoding", () => {
  it("decodes a last-minute CF70 HGS run", () => {
    expect(parseFilename("log_last_minute_cf70_hgs_cls_20N")).toEqual({
      strategy: "LM",
      cf: "CF70",
      sl_var: null,
      improver: "CLS",
      constructor: "HGS",
      acceptance: null,
    });
  });

  it("decodes acceptance tokens between constructor and improver", () => {
    expect(parseFilename("log_lookahead_alns_greedy_ftsp_100N")).toMatchObject({
      strategy: "LA",
      constructor: "ALNS",
      acceptance: "greedy",
      improver: "FTSP",
    });
  });

  it("decodes service-level variants", () => {
    expect(parseFilename("log_service_level2_bpc_cls_50N")).toMatchObject({
      strategy: "SL",
      sl_var: "SL2",
      constructor: "BPC",
    });
  });

  it("rejects stems without the log_ prefix or known strategy", () => {
    expect(parseFilename("last_minute_cf70_hgs_cls_20N")).toBeNull();
    expect(parseFilename("log_bogus_hgs_cls_20N")).toBeNull();
  });

  it("parses area directories against dir_city metadata", () => {
    expect(parseAreaDir("riomaior20")).toEqual({ city: "Rio Maior", N: 20 });
    expect(parseAreaDir("figueiradafoz148_extra")).toEqual({ city: "Figueira da Foz", N: 148 });
    expect(parseAreaDir("unknown99")).toBeNull();
  });
});

describe("filter → aggregate → context pipeline", () => {
  const rows: SimRow[] = [
    row({ cf: "CF70", overflows: 10, profit: 20 }),
    row({ cf: "CF90", overflows: 20, profit: 40 }),
    row({ strategy: "SL", sl_var: "SL1", cf: "", overflows: 5, profit: 10 }),
    row({ city: "Figueira da Foz", N: 148, constructor: "ALNS", overflows: 3, profit: 5 }),
  ];

  it("filters by scenario and policy fields", () => {
    const filtered = filterData(rows, {
      scenarios: [{ city: "Rio Maior", N: 20, dist: "Empirical" }],
      policies: { strategies: ["LM"] },
    });
    expect(filtered).toHaveLength(2);
    expect(new Set(filtered.map((r) => r.strategy))).toEqual(new Set(["LM"]));
  });

  it("aggregates CF variants into a single averaged row with variant label", () => {
    const agg = aggregate(rows.filter((r) => r.strategy === "LM" && r.city === "Rio Maior"));
    expect(agg).toHaveLength(1);
    expect(agg[0].overflows).toBe(15);
    expect(agg[0].profit).toBe(30);
    expect(agg[0].cf).toBe("");
    expect(agg[0].variant).toBe("LM");
  });

  it("labels un-aggregated variants with their CF / SL qualifiers", () => {
    const single = aggregate([rows[2]]);
    expect(single[0].variant).toBe("SL");
    // detectScenarios orders by N then city
    const scenarios = detectScenarios(rows);
    expect(scenarios.map((s) => s.N)).toEqual([20, 148]);
    expect(scenarioLabel(scenarios[0])).toBe("RM-20 / Empirical");
    expect(regionLabel("Figueira da Foz", 148)).toBe("FFZ-148");
  });

  it("builds an analysis context with metadata-ordered constructors", () => {
    const ctx = buildCtx(rows, 30);
    expect(ctx.nDays).toBe(30);
    expect(ctx.scenarios).toHaveLength(2);
    expect(ctx.strategies).toEqual(["LM", "SL"]);
    // ALNS precedes HGS in simulation_metadata.json constructor order
    expect(ctx.constructors).toEqual(["ALNS", "HGS"]);
  });

  it("computes the Pareto front over metric vectors (min overflows, max kgkm)", () => {
    const pts: SimRow[] = [
      row({ overflows: 1, kgkm: 1 }),
      row({ overflows: 2, kgkm: 5 }),
      row({ overflows: 3, kgkm: 4 }), // dominated by (2, 5)
      row({ overflows: 0.5, kgkm: 0.5 }),
    ];
    const front = paretoIndices(metricValues(pts, "overflows"), metricValues(pts, "kgkm"));
    expect(front).toEqual(new Set([0, 1, 3]));
  });
});

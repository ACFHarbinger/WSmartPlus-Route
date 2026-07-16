import { describe, expect, it } from "vitest";
import {
  computeEtaMs,
  formatDurationMs,
  getLatestProgress,
  progressPercent,
} from "../../src/utils/process/processProgress";

describe("getLatestProgress", () => {
  it("parses the newest PROGRESS marker", () => {
    const lines = [
      'PROGRESS:{"value": 1, "total": 10}',
      "some other log line",
      'PROGRESS:{"value": 4, "total": 10, "label": "epoch"}',
    ];
    expect(getLatestProgress(lines)).toEqual({ value: 4, total: 10, label: "epoch" });
  });

  it("accepts the legacy `current` field", () => {
    expect(getLatestProgress(['PROGRESS:{"current": 3, "total": 5}'])).toEqual({
      value: 3,
      total: 5,
      label: undefined,
    });
  });

  it("skips malformed markers and keeps scanning backwards", () => {
    const lines = ['PROGRESS:{"value": 2, "total": 4}', "PROGRESS:{not json}"];
    expect(getLatestProgress(lines)?.value).toBe(2);
  });

  it("only scans the last 30 lines", () => {
    const lines = ['PROGRESS:{"value": 1, "total": 2}', ...Array(40).fill("noise")];
    expect(getLatestProgress(lines)).toBeNull();
  });

  it("returns null when nothing matches", () => {
    expect(getLatestProgress(["a", "b"])).toBeNull();
  });
});

describe("progressPercent", () => {
  it("computes a clamped percentage", () => {
    expect(progressPercent({ value: 5, total: 10 })).toBe(50);
    expect(progressPercent({ value: 20, total: 10 })).toBe(100);
  });

  it("returns null without a positive total", () => {
    expect(progressPercent({ value: 5 })).toBeNull();
    expect(progressPercent({ value: 5, total: 0 })).toBeNull();
  });
});

describe("formatDurationMs", () => {
  it("formats seconds, minutes and hours", () => {
    expect(formatDurationMs(12_000)).toBe("12s");
    expect(formatDurationMs(125_000)).toBe("2m 5s");
    expect(formatDurationMs(3_720_000)).toBe("1h 2m");
  });

  it("clamps negative input to zero", () => {
    expect(formatDurationMs(-5)).toBe("0s");
  });
});

describe("computeEtaMs", () => {
  it("extrapolates remaining time linearly", () => {
    expect(computeEtaMs(2, 10, 4000)).toBe(16_000);
  });

  it("returns null for unknown or complete progress", () => {
    expect(computeEtaMs(0, 10, 1000)).toBeNull();
    expect(computeEtaMs(10, 10, 1000)).toBeNull();
    expect(computeEtaMs(5, 10, 0)).toBeNull();
  });
});

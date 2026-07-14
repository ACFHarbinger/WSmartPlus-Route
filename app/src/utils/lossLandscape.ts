/**
 * Loss landscape analysis helpers for §G.5.2 (minima sharpness, markers).
 */

export interface LossMinimaInfo {
  row: number;
  col: number;
  value: number;
  /** Finite-difference Hessian trace proxy — higher = sharper basin. */
  sharpness: number;
  label: "flat" | "moderate" | "sharp";
}

/** Locate global minimum and estimate basin sharpness via Laplacian. */
export function analyzeLossMinima(values: number[][]): LossMinimaInfo | null {
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  if (rows < 3 || cols < 3) return null;

  let minVal = Infinity;
  let minR = 0;
  let minC = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = values[r][c];
      if (Number.isFinite(v) && v < minVal) {
        minVal = v;
        minR = r;
        minC = c;
      }
    }
  }

  const center = values[minR][minC];
  const laplacian =
    (values[minR - 1]?.[minC] ?? center) +
    (values[minR + 1]?.[minC] ?? center) +
    (values[minR][minC - 1] ?? center) +
    (values[minR][minC + 1] ?? center) -
    4 * center;
  const sharpness = Math.abs(laplacian);

  let label: LossMinimaInfo["label"] = "flat";
  if (sharpness > 0.5) label = "moderate";
  if (sharpness > 2.0) label = "sharp";

  return { row: minR, col: minC, value: minVal, sharpness, label };
}

/** Map loss value to RGB (deep blue → bright red). */
export function lossToColor(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  // blue (#1e3a8a) → indigo → amber → red (#ef4444)
  if (clamped < 0.33) {
    const u = clamped / 0.33;
    return [0.12 + u * 0.27, 0.23 + u * 0.16, 0.54 + u * 0.1];
  }
  if (clamped < 0.66) {
    const u = (clamped - 0.33) / 0.33;
    return [0.39 + u * 0.35, 0.39 + u * 0.35, 0.64 - u * 0.25];
  }
  const u = (clamped - 0.66) / 0.34;
  return [0.74 + u * 0.2, 0.74 - u * 0.28, 0.39 - u * 0.15];
}

/** Normalise grid to [0, 1] for height and colour mapping. */
export function normalizeGrid(values: number[][]): { norm: number[][]; min: number; max: number } {
  const flat = values.flat().filter(Number.isFinite);
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const span = max - min || 1;
  const norm = values.map((row) => row.map((v) => (Number.isFinite(v) ? (v - min) / span : 0)));
  return { norm, min, max };
}

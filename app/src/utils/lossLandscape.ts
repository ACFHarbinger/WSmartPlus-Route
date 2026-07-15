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
  /** Empirical vs Gamma-3 generalization heuristic (§G.5.2). */
  generalizationNote: string;
}

const GENERALIZATION_NOTES: Record<LossMinimaInfo["label"], string> = {
  flat: "Flat basin — tends to generalize well across Empirical and Gamma-3 distribution shifts.",
  moderate:
    "Moderate curvature — may show mild distribution sensitivity; validate on held-out splits.",
  sharp:
    "Sharp basin — higher overfitting risk; compare Empirical vs Gamma-3 eval before deployment.",
};

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

  return {
    row: minR,
    col: minC,
    value: minVal,
    sharpness,
    label,
    generalizationNote: GENERALIZATION_NOTES[label],
  };
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

export interface LandscapeMarker {
  label: string;
  row: number;
  col: number;
  theta1?: number;
  theta2?: number;
  loss?: number;
  color?: string;
}

function nearestAxisIndex(axis: number[], value: number): number {
  if (!axis.length) return 0;
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < axis.length; i++) {
    const d = Math.abs(axis[i] - value);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

/** Map θ coordinates to loss_grid row/col using bundled axis vectors. */
export function thetaToGridCell(
  theta1Axis: number[],
  theta2Axis: number[],
  t1: number,
  t2: number
): { row: number; col: number } {
  return {
    row: nearestAxisIndex(theta1Axis, t1),
    col: nearestAxisIndex(theta2Axis, t2),
  };
}

/** Build BPC exact-solver marker from NPZ vector payloads (§G.5.2). */
export function resolveBpcMarker(
  vectors: { key: string; values: number[] }[],
  rows: number,
  cols: number
): LandscapeMarker | null {
  const t1 = vectors.find((v) => v.key === "bpc_theta1")?.values[0];
  const t2 = vectors.find((v) => v.key === "bpc_theta2")?.values[0];
  const loss = vectors.find((v) => v.key === "bpc_loss")?.values[0];
  const theta1Axis = vectors.find((v) => v.key === "theta1")?.values ?? [];
  const theta2Axis = vectors.find((v) => v.key === "theta2")?.values ?? [];

  if (t1 == null || t2 == null) return null;

  let row: number;
  let col: number;
  if (theta1Axis.length && theta2Axis.length) {
    ({ row, col } = thetaToGridCell(theta1Axis, theta2Axis, t1, t2));
  } else {
    row = Math.round(((t1 + 1) / 2) * (rows - 1));
    col = Math.round(((t2 + 1) / 2) * (cols - 1));
  }

  return {
    label: "BPC optimum",
    row: Math.max(0, Math.min(rows - 1, row)),
    col: Math.max(0, Math.min(cols - 1, col)),
    theta1: t1,
    theta2: t2,
    loss,
    color: "#f59e0b",
  };
}

/** Convert grid cell to R3F terrain coordinates (matches LossLandscape3D layout). */
export function gridCellToTerrainPosition(
  row: number,
  col: number,
  rows: number,
  cols: number,
  height: number
): [number, number, number] {
  const x = col - (cols - 1) / 2;
  const y = -(row - (rows - 1) / 2);
  return [x, y, height + 0.18];
}

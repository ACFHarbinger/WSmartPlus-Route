/**
 * Empirical vs Gamma-3 distribution labelling for §G.5.3 attention compare.
 */

export type DistributionLabel = "empirical" | "gamma3" | "unknown";

const EMP_RE = /(?:^|[_/\-])(emp|empirical)(?:[_/.\-]|$)/i;
const GAMMA_RE = /(?:^|[_/\-])(gamma3|gamma-3|g3)(?:[_/.\-]|$)/i;

/** Infer training distribution from archive path and optional NPZ keys. */
export function inferDistributionLabel(
  path: string,
  arrayKeys?: string[]
): DistributionLabel {
  const lower = path.toLowerCase();
  if (EMP_RE.test(lower)) return "empirical";
  if (GAMMA_RE.test(lower)) return "gamma3";

  if (arrayKeys?.some((k) => /^distribution$/i.test(k))) {
    // Caller may resolve scalar separately; key presence hints empirical bundle.
    if (/emp/i.test(lower)) return "empirical";
    if (/gamma/i.test(lower)) return "gamma3";
  }

  return "unknown";
}

export function distributionDisplayName(label: DistributionLabel): string {
  switch (label) {
    case "empirical":
      return "Empirical";
    case "gamma3":
      return "Gamma-3";
    default:
      return "Unknown";
  }
}

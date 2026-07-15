/**
 * Derive assets/output run directories from process stdout (§G.14 / §G.9 / §G.15).
 */
import { extractJsonlPathFromLogLines } from "./policyTelemetryTrends";

const PRUNED_CONFIG_RE = /Pruned config saved → (.+)/;
const HYDRA_SNAPSHOT_RE = /Hydra config snapshot saved → (.+)/;
const ASSETS_OUTPUT_RE = /([^\s'"]*assets\/output\/[^\s'"]+)/;

/** Resolve the Hydra run root from a simulation ``.jsonl`` log path. */
export function outputRunPathFromJsonl(jsonlPath: string): string {
  const normalized = jsonlPath.replace(/\\/g, "/");
  const hydraIdx = normalized.lastIndexOf("/hydra/");
  if (hydraIdx !== -1) {
    return normalized.slice(0, hydraIdx);
  }
  const lastSlash = normalized.lastIndexOf("/");
  if (lastSlash === -1) {
    return normalized;
  }
  return normalized.slice(0, lastSlash);
}

/** Derive the run root from a Hydra snapshot or pruned-config artefact path. */
export function outputRunPathFromHydraArtifact(artifactPath: string): string {
  const normalized = artifactPath.replace(/\\/g, "/");
  const hydraIdx = normalized.lastIndexOf("/hydra/");
  if (hydraIdx !== -1) {
    return normalized.slice(0, hydraIdx);
  }
  if (normalized.endsWith("/hydra")) {
    return normalized.slice(0, -"/hydra".length);
  }
  const lastSlash = normalized.lastIndexOf("/");
  if (lastSlash === -1) {
    return normalized;
  }
  return normalized.slice(0, lastSlash);
}

/** Scan stdout for Hydra snapshot / pruned-config / assets/output paths (newest line wins). */
export function extractOutputRunPathFromLogLines(lines: string[]): string | null {
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]!;
    const pruned = line.match(PRUNED_CONFIG_RE);
    if (pruned?.[1]) {
      return outputRunPathFromHydraArtifact(pruned[1].trim());
    }
    const hydra = line.match(HYDRA_SNAPSHOT_RE);
    if (hydra?.[1]) {
      return outputRunPathFromHydraArtifact(hydra[1].trim());
    }
    const assets = line.match(ASSETS_OUTPUT_RE);
    if (assets?.[1]) {
      const path = assets[1].replace(/\\/g, "/");
      if (path.endsWith(".jsonl")) continue;
      if (path.includes("/hydra")) {
        return outputRunPathFromHydraArtifact(path);
      }
      return path;
    }
  }
  return null;
}

/** Scan process stdout for the newest run directory under assets/output. */
export function outputRunPathFromLogLines(lines: string[]): string | null {
  const jsonl = extractJsonlPathFromLogLines(lines);
  if (jsonl) return outputRunPathFromJsonl(jsonl);
  return extractOutputRunPathFromLogLines(lines);
}

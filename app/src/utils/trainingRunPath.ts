/**
 * Derive Lightning training run directories from process stdout (§G.10 / §G.17).
 */

const SAVED_SIDECAR_RE = /Saved sidecar args\.json to (.+)/;
const METRICS_CSV_RE = /([^\s'"]*\/logs\/[^\s'"]*\/metrics\.csv)/;
const LOGS_DIR_RE = /([^\s'"]*\/logs\/[^\s'"]+)/;

function parentDir(filePath: string): string {
  const normalized = filePath.replace(/\\/g, "/");
  const lastSlash = normalized.lastIndexOf("/");
  if (lastSlash === -1) return normalized;
  return normalized.slice(0, lastSlash);
}

function isLogsRunDir(path: string): boolean {
  const normalized = path.replace(/\\/g, "/");
  return normalized.includes("/logs/") && !normalized.endsWith(".csv") && !normalized.endsWith(".json");
}

/** Scan process stdout for the newest Lightning log directory under ``logs/``. */
export function trainingRunPathFromLogLines(lines: string[]): string | null {
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]!;
    const sidecar = line.match(SAVED_SIDECAR_RE);
    if (sidecar?.[1]) {
      const dir = parentDir(sidecar[1].trim());
      if (isLogsRunDir(dir)) return dir;
    }
    const metrics = line.match(METRICS_CSV_RE);
    if (metrics?.[1]) {
      const dir = parentDir(metrics[1].trim());
      if (isLogsRunDir(dir)) return dir;
    }
  }

  for (let i = lines.length - 1; i >= 0; i--) {
    const match = lines[i]!.match(LOGS_DIR_RE);
    if (match?.[1]) {
      const path = match[1].replace(/\\/g, "/");
      if (isLogsRunDir(path)) return path;
    }
  }

  return null;
}

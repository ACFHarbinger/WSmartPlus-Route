/**
 * Derive Data Explorer / artefact paths from ``gen_data`` processes (§G.11 / §D.8 / §D.7).
 */

/** Extract ``data.sensor_file`` from a Hydra ``gen_data`` command line, if present. */
export function sensorCsvPathFromGenDataCommand(command: string): string | null {
  const match = command.match(/data\.sensor_file=([^\s]+)/);
  if (!match?.[1]) return null;
  return match[1].replace(/^['"]|['"]$/g, "");
}

/** Extract ``data.tsplib_instance`` from a Hydra ``gen_data`` command line, if present. */
export function tsplibPathFromGenDataCommand(command: string): string | null {
  const match = command.match(/data\.tsplib_instance=([^\s]+)/);
  if (!match?.[1]) return null;
  return match[1].replace(/^['"]|['"]$/g, "");
}

/**
 * Scan stdout for ``Generated <path>`` dataset artefacts (newest line wins).
 * Matches the Python generator log line from ``datasets.py``.
 */
export function generatedDatasetPathFromLogLines(lines: string[]): string | null {
  const re = /Generated\s+(.+)$/;
  for (let i = lines.length - 1; i >= 0; i--) {
    const match = lines[i]!.match(re);
    if (!match?.[1]) continue;
    const path = match[1].trim().replace(/^['"]|['"]$/g, "");
    if (path) return path;
  }
  return null;
}

/**
 * Prefer a path suitable for Data Explorer handoff from a gen_data process:
 * sensor CSV override first, then a generated ``.csv`` artefact.
 */
export function dataExplorerPathFromGenData(
  command: string,
  lines: string[] = []
): string | null {
  const sensor = sensorCsvPathFromGenDataCommand(command);
  if (sensor) return sensor;

  const generated = generatedDatasetPathFromLogLines(lines);
  if (generated && /\.csv$/i.test(generated)) return generated;

  return null;
}

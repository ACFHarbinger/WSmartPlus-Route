/** Recent stdout/stderr tail lines for live process panels (§G.12 / §G.15 / §D.7). */

/** Strip the ``[stderr]`` prefix and return the last ``maxLines`` log lines. */
export function processLogTail(logLines: string[], maxLines = 10): string[] {
  return logLines
    .slice(-maxLines)
    .map((line) => (line.startsWith("[stderr]") ? line.slice(8) : line));
}

/**
 * Derive Lightning log directories from Optuna trial user attributes (§G.18 / §D.7).
 */

const TRIAL_LOG_DIR_KEYS = ["log_dir", "run_dir", "output_dir", "log_path"] as const;

/** Extract a local log/run path from trial ``user_attrs`` when present. */
export function trialLogDirFromUserAttrs(
  attrs: Record<string, string | number | boolean | null> | undefined
): string | null {
  if (!attrs) return null;
  for (const key of TRIAL_LOG_DIR_KEYS) {
    const value = attrs[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

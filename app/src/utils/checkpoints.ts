/**
 * Shared checkpoint file detection for Training Monitor and Output Browser (§G.12 / §G.14 / §G.17).
 */
import type { DirEntry } from "../types";
import { resolveLocalProjectPath } from "./outputRunPath";
import { runLabelFromPath } from "./policyTelemetryTrends";

export const CHECKPOINT_EXTENSIONS = ["pt", "ckpt", "pth"] as const;

export type CheckpointExtension = (typeof CHECKPOINT_EXTENSIONS)[number];

export function isCheckpointExtension(ext: string): ext is CheckpointExtension {
  return (CHECKPOINT_EXTENSIONS as readonly string[]).includes(ext);
}

export function isCheckpointEntry(entry: Pick<DirEntry, "is_dir" | "extension">): boolean {
  return !entry.is_dir && isCheckpointExtension(entry.extension);
}

export function filterCheckpointEntries(entries: DirEntry[]): DirEntry[] {
  return entries.filter(isCheckpointEntry).sort((a, b) => a.name.localeCompare(b.name));
}

/** Parent Lightning / output run label for checkpoint path-chip brush (§G.12 / §G.14 / §G.17 / §D.7). */
export function parentRunBrushLabelFromCheckpointPath(
  checkpointPath: string,
  projectRoot?: string | null
): string {
  const resolved = resolveLocalProjectPath(checkpointPath, projectRoot) ?? checkpointPath;
  const parts = resolved.replace(/\\/g, "/").split("/");
  const ckptIdx = parts.lastIndexOf("checkpoints");
  if (ckptIdx > 0) {
    return runLabelFromPath(parts[ckptIdx - 1]!);
  }
  return runLabelFromPath(resolved);
}

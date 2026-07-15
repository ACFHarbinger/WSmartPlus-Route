/**
 * Shared checkpoint file detection for Training Monitor and Output Browser (§G.12 / §G.14 / §G.17).
 */
import type { DirEntry } from "../types";

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

/**
 * File-system bridge for the native report/deck generator (§H).
 *
 * Thin typed wrappers over the Rust commands used by the generation pipeline;
 * all paths are resolved against the configured project root by the callers.
 */
import { invoke } from "@tauri-apps/api/core";

export interface CsvFile {
  headers: string[];
  rows: Record<string, unknown>[];
}

export async function readTextFile(path: string): Promise<string> {
  return await invoke<string>("read_text_file", { path });
}

export async function writeTextFile(path: string, content: string): Promise<void> {
  await invoke("write_text_file", { path, content });
}

/** Write binary content from a data URL or raw base64 string. */
export async function writeBinaryFile(path: string, dataUrlOrB64: string): Promise<void> {
  const base64 = dataUrlOrB64.includes(",") ? dataUrlOrB64.slice(dataUrlOrB64.indexOf(",") + 1) : dataUrlOrB64;
  await invoke("write_binary_file", { path, base64 });
}

export async function pathExists(path: string): Promise<boolean> {
  return await invoke<boolean>("path_exists", { path });
}

export async function loadCsv(path: string): Promise<CsvFile> {
  return await invoke<CsvFile>("load_csv_file", { path });
}

export async function listFilesRecursive(
  root: string,
  opts: { prefix?: string; suffix?: string } = {}
): Promise<string[]> {
  return await invoke<string[]>("list_files_recursive", {
    root,
    prefix: opts.prefix ?? null,
    suffix: opts.suffix ?? null,
  });
}

/** Load an NPZ array of any rank as a flattened vector (§H.1 raw waste matrices). */
export async function loadNpzFlat(path: string, key: string): Promise<number[]> {
  return await invoke<number[]>("load_npz_flat", { path, key });
}

export function joinPath(root: string, rel: string): string {
  if (/^([A-Za-z]:[\\/]|\/)/.test(rel)) return rel;
  return `${root.replace(/[\\/]+$/, "")}/${rel}`;
}

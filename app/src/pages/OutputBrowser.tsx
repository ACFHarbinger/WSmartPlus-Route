/**
 * Output Browser — browse the assets/output/ directory tree (§G.14).
 *
 * Ports the PySide6 file-explorer tab. Provides:
 *   - Run list (from list_output_dirs)
 *   - File tree inside a selected run (from list_dir)
 *   - File viewer: CSV → DataExplorer-style table; YAML/text → raw view
 *   - Simulation summary: KPIs from any JSONL log in the run directory
 */
import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  Folder,
  FolderOpen,
  File,
  FileText,
  RefreshCw,
  ChevronRight,
  ChevronDown,
} from "lucide-react";
import { useAppStore } from "../store/app";
import { toast } from "sonner";
import type { DirEntry, OutputDir } from "../types";

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

const TEXT_EXTENSIONS = new Set(["yaml", "yml", "toml", "cfg", "ini", "txt", "log", "md", "json", "jsonl"]);
const CSV_EXTENSIONS = new Set(["csv"]);

function FileIcon({ entry }: { entry: DirEntry }) {
  if (entry.is_dir) return <Folder size={13} className="text-accent-warning" />;
  if (CSV_EXTENSIONS.has(entry.extension)) return <File size={13} className="text-accent-success" />;
  if (TEXT_EXTENSIONS.has(entry.extension)) return <FileText size={13} className="text-accent-secondary" />;
  return <File size={13} className="text-canvas-muted" />;
}

export function OutputBrowser() {
  const projectRoot = useAppStore((s) => s.projectRoot);
  const [runs, setRuns] = useState<OutputDir[]>([]);
  const [selectedRun, setSelectedRun] = useState<OutputDir | null>(null);
  const [entries, setEntries] = useState<DirEntry[]>([]);
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  const [subEntries, setSubEntries] = useState<Record<string, DirEntry[]>>({});
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [csvRows, setCsvRows] = useState<Array<Record<string, string | number | null>> | null>(null);
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [viewingPath, setViewingPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [fileLoading, setFileLoading] = useState(false);

  const outputPath = projectRoot ? `${projectRoot}/assets/output` : null;

  const refresh = useCallback(async () => {
    if (!outputPath) return;
    setLoading(true);
    try {
      const found = await invoke<OutputDir[]>("list_output_dirs", { outputPath });
      setRuns(found);
    } catch (err) {
      toast.error("Failed to list output directories", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [outputPath]);

  useEffect(() => {
    if (outputPath) refresh();
  }, [outputPath, refresh]);

  const selectRun = useCallback(async (run: OutputDir) => {
    setSelectedRun(run);
    setFileContent(null);
    setCsvRows(null);
    setViewingPath(null);
    setExpandedDirs(new Set());
    setSubEntries({});
    try {
      const e = await invoke<DirEntry[]>("list_dir", { path: run.path });
      setEntries(e);
    } catch (err) {
      toast.error("Failed to list run directory", { description: String(err) });
    }
  }, []);

  const toggleDir = useCallback(async (entry: DirEntry) => {
    setExpandedDirs((prev) => {
      const next = new Set(prev);
      if (next.has(entry.path)) {
        next.delete(entry.path);
      } else {
        next.add(entry.path);
      }
      return next;
    });
    if (!subEntries[entry.path]) {
      try {
        const sub = await invoke<DirEntry[]>("list_dir", { path: entry.path });
        setSubEntries((prev) => ({ ...prev, [entry.path]: sub }));
      } catch {}
    }
  }, [subEntries]);

  const openFile = useCallback(async (entry: DirEntry) => {
    if (entry.is_dir) {
      toggleDir(entry);
      return;
    }
    setViewingPath(entry.path);
    setFileContent(null);
    setCsvRows(null);
    setFileLoading(true);

    try {
      if (CSV_EXTENSIONS.has(entry.extension)) {
        const csvFile = await invoke<{ headers: string[]; rows: Array<Record<string, string | number | null>> }>(
          "load_csv_file",
          { path: entry.path }
        );
        setCsvHeaders(csvFile.headers);
        setCsvRows(csvFile.rows);
      } else if (TEXT_EXTENSIONS.has(entry.extension)) {
        const text = await invoke<string>("read_text_file", { path: entry.path });
        setFileContent(text);
      }
    } catch (err) {
      toast.error("Failed to open file", { description: String(err) });
    } finally {
      setFileLoading(false);
    }
  }, [toggleDir]);

  const pickOutputDir = useCallback(async () => {
    const path = (await open({ directory: true })) as string | null;
    if (!path) return;
    try {
      const e = await invoke<DirEntry[]>("list_dir", { path });
      const fakeRun: OutputDir = {
        name: path.split("/").pop() ?? path,
        path,
        created_at: "",
        size_bytes: 0,
      };
      setSelectedRun(fakeRun);
      setEntries(e);
      setFileContent(null);
      setCsvRows(null);
    } catch (err) {
      toast.error("Failed to open directory", { description: String(err) });
    }
  }, []);

  function renderEntries(list: DirEntry[], depth = 0): React.ReactNode {
    return list.map((e) => (
      <div key={e.path}>
        <button
          onClick={() => openFile(e)}
          className={`w-full flex items-center gap-2 py-1 px-2 rounded text-xs text-left hover:bg-canvas-hover transition-colors ${
            viewingPath === e.path ? "bg-accent-primary/15 text-accent-secondary" : "text-gray-300"
          }`}
          style={{ paddingLeft: `${8 + depth * 14}px` }}
        >
          {e.is_dir ? (
            expandedDirs.has(e.path) ? <ChevronDown size={11} className="shrink-0 text-canvas-muted" /> : <ChevronRight size={11} className="shrink-0 text-canvas-muted" />
          ) : (
            <span className="w-[11px] shrink-0" />
          )}
          <FileIcon entry={e} />
          <span className="truncate flex-1">{e.name}</span>
          {!e.is_dir && (
            <span className="text-canvas-muted shrink-0">{formatBytes(e.size_bytes)}</span>
          )}
        </button>
        {e.is_dir && expandedDirs.has(e.path) && subEntries[e.path] && (
          <div>{renderEntries(subEntries[e.path], depth + 1)}</div>
        )}
      </div>
    ));
  }

  if (!projectRoot) {
    return (
      <div className="flex items-center justify-center h-64 text-canvas-muted text-sm">
        Set project root in settings to browse output directories.
      </div>
    );
  }

  return (
    <div className="flex gap-4 h-[calc(100vh-8rem)]">
      {/* Left: run list */}
      <div className="w-56 shrink-0 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <button
            onClick={refresh}
            disabled={loading}
            className="btn-ghost p-1.5 text-canvas-muted"
            title="Refresh"
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          </button>
          <button
            onClick={pickOutputDir}
            className="btn-ghost p-1.5 text-canvas-muted"
            title="Open directory"
          >
            <FolderOpen size={13} />
          </button>
          <span className="text-xs text-canvas-muted">{runs.length} runs</span>
        </div>

        <div className="card flex-1 overflow-auto p-1">
          {runs.map((r) => (
            <button
              key={r.path}
              onClick={() => selectRun(r)}
              className={`w-full flex items-center gap-2 py-1.5 px-2 rounded text-xs text-left hover:bg-canvas-hover transition-colors ${
                selectedRun?.path === r.path
                  ? "bg-accent-primary/15 text-accent-secondary"
                  : "text-gray-300"
              }`}
            >
              <Folder size={12} className="text-accent-warning shrink-0" />
              <span className="truncate">{r.name}</span>
            </button>
          ))}
          {runs.length === 0 && !loading && (
            <p className="text-xs text-canvas-muted p-2">No run directories found.</p>
          )}
        </div>
      </div>

      {/* Middle: file tree */}
      <div className="w-52 shrink-0 flex flex-col">
        {selectedRun ? (
          <div className="card flex-1 overflow-auto p-1">
            <p className="text-xs text-canvas-muted px-2 py-1 font-medium truncate">
              {selectedRun.name}
            </p>
            {renderEntries(entries)}
          </div>
        ) : (
          <div className="card flex-1 flex items-center justify-center text-canvas-muted text-xs">
            Select a run
          </div>
        )}
      </div>

      {/* Right: file viewer */}
      <div className="flex-1 min-w-0">
        {!viewingPath && !fileLoading && (
          <div className="card h-full flex items-center justify-center text-canvas-muted text-sm">
            Select a file to view its contents.
          </div>
        )}

        {fileLoading && (
          <div className="card h-full flex items-center justify-center gap-2 text-canvas-muted text-sm">
            <RefreshCw size={14} className="animate-spin" />
            Loading…
          </div>
        )}

        {!fileLoading && fileContent !== null && (
          <div className="card h-full overflow-auto">
            <p className="text-xs text-canvas-muted mb-2 font-mono truncate">{viewingPath}</p>
            <pre className="font-mono text-xs text-gray-300 whitespace-pre-wrap">{fileContent}</pre>
          </div>
        )}

        {!fileLoading && csvRows !== null && (
          <div className="card h-full overflow-auto">
            <p className="text-xs text-canvas-muted mb-2 font-mono truncate">{viewingPath}</p>
            <table className="w-full text-xs">
              <thead className="bg-canvas-elevated sticky top-0">
                <tr>
                  {csvHeaders.map((h) => (
                    <th key={h} className="px-3 py-2 text-left text-canvas-muted font-medium whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-canvas-border">
                {csvRows.slice(0, 200).map((row, i) => (
                  <tr key={i} className="hover:bg-canvas-hover">
                    {csvHeaders.map((h) => (
                      <td key={h} className="px-3 py-1.5 text-gray-300 whitespace-nowrap font-mono">
                        {row[h] ?? "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {csvRows.length > 200 && (
              <p className="text-xs text-canvas-muted p-3">
                Showing 200 / {csvRows.length} rows. Use Data Explorer for full pagination.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

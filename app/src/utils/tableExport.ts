import { invoke } from "@tauri-apps/api/core";
import { save as saveDialog } from "@tauri-apps/plugin-dialog";

/** Escape a CSV cell value. */
function escapeCell(value: string | number | boolean | null | undefined): string {
  const s = value == null ? "" : String(value);
  if (s.includes(",") || s.includes('"') || s.includes("\n")) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

/** Trigger a browser download of tabular data as CSV (§G.7). */
export function downloadCsv(
  filename: string,
  headers: string[],
  rows: Array<Array<string | number | boolean | null | undefined>>
): void {
  const lines = [
    headers.map(escapeCell).join(","),
    ...rows.map((row) => row.map(escapeCell).join(",")),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename.endsWith(".csv") ? filename : `${filename}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

/** Export an on-disk CSV file to Parquet via Rust → Python/pyarrow (§G.7). */
export async function downloadParquetFromCsv(
  projectRoot: string,
  csvPath: string,
  defaultName?: string
): Promise<string | null> {
  const path = (await saveDialog({
    filters: [{ name: "Parquet", extensions: ["parquet"] }],
    defaultPath: defaultName ?? csvPath.replace(/\.csv$/i, ".parquet"),
  })) as string | null;
  if (!path) return null;

  const outputPath = path.endsWith(".parquet") ? path : `${path}.parquet`;
  return invoke<string>("export_csv_to_parquet", { projectRoot, csvPath, outputPath });
}

/** Export in-memory tabular data to Parquet (§G.7). */
export async function downloadParquetTable(
  projectRoot: string,
  filename: string,
  headers: string[],
  rows: Array<Array<string | number | boolean | null | undefined>>
): Promise<string | null> {
  const path = (await saveDialog({
    filters: [{ name: "Parquet", extensions: ["parquet"] }],
    defaultPath: filename.endsWith(".parquet") ? filename : `${filename}.parquet`,
  })) as string | null;
  if (!path) return null;

  const outputPath = path.endsWith(".parquet") ? path : `${path}.parquet`;
  const rowMaps = rows.map((row) =>
    Object.fromEntries(headers.map((h, i) => [h, row[i] ?? null]))
  );
  return invoke<string>("export_table_parquet", {
    projectRoot,
    headers,
    rows: rowMaps,
    outputPath,
  });
}

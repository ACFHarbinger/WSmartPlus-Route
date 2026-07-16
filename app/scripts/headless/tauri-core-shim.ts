/**
 * Node implementation of the Tauri `invoke` surface used by `src/gen/io.ts`,
 * so the native §H generation pipeline can run headless (no webview) — the
 * "generate analysis + presentation without the GUI" path.
 *
 * Aliased in place of `@tauri-apps/api/core` by scripts/headless/vite.config.ts.
 */
import fs from "node:fs";
import path from "node:path";

type Args = Record<string, unknown>;

function listRecursive(root: string, prefix: string | null, suffix: string | null): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else {
        if (prefix && !e.name.startsWith(prefix)) continue;
        if (suffix && !e.name.endsWith(suffix)) continue;
        out.push(p);
      }
    }
  };
  walk(root);
  return out.sort();
}

/** Minimal CSV parser matching the Rust `load_csv_file` output shape. */
function loadCsv(p: string): { headers: string[]; rows: Record<string, unknown>[] } {
  const text = fs.readFileSync(p, "utf8");
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (!lines.length) return { headers: [], rows: [] };
  const split = (line: string): string[] => {
    if (!line.includes('"')) return line.split(",");
    const cells: string[] = [];
    let cur = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (quoted) {
        if (ch === '"' && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else if (ch === '"') quoted = false;
        else cur += ch;
      } else if (ch === '"') quoted = true;
      else if (ch === ",") {
        cells.push(cur);
        cur = "";
      } else cur += ch;
    }
    cells.push(cur);
    return cells;
  };
  const headers = split(lines[0]);
  const rows = lines.slice(1).map((line) => {
    const cells = split(line);
    const row: Record<string, unknown> = {};
    headers.forEach((h, i) => {
      row[h] = cells[i] ?? "";
    });
    return row;
  });
  return { headers, rows };
}

export async function invoke<T>(cmd: string, args: Args = {}): Promise<T> {
  switch (cmd) {
    case "read_text_file":
      return fs.readFileSync(String(args.path), "utf8") as unknown as T;
    case "write_text_file": {
      const p = String(args.path);
      fs.mkdirSync(path.dirname(p), { recursive: true });
      fs.writeFileSync(p, String(args.content));
      return undefined as unknown as T;
    }
    case "write_binary_file": {
      const p = String(args.path);
      fs.mkdirSync(path.dirname(p), { recursive: true });
      fs.writeFileSync(p, Buffer.from(String(args.base64), "base64"));
      return undefined as unknown as T;
    }
    case "read_binary_file":
      return Array.from(fs.readFileSync(String(args.path))) as unknown as T;
    case "path_exists":
      return fs.existsSync(String(args.path)) as unknown as T;
    case "load_csv_file":
      return loadCsv(String(args.path)) as unknown as T;
    case "list_files_recursive":
      return listRecursive(
        String(args.root),
        (args.prefix as string | null) ?? null,
        (args.suffix as string | null) ?? null
      ) as unknown as T;
    default:
      throw new Error(`headless tauri shim: unimplemented command '${cmd}'`);
  }
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Search } from "lucide-react";
import { PALETTE_COMMANDS } from "../../constants/commands";
import { useWsrouteImport } from "../../hooks/useWsrouteImport";
import { useAppStore } from "../../store/app";
import { useLayoutStore } from "../../store/layout";

function matchQuery(query: string, label: string, keywords?: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const haystack = `${label} ${keywords ?? ""}`.toLowerCase();
  return q.split(/\s+/).every((token) => haystack.includes(token));
}

export function CommandPalette() {
  const { setMode, theme, setTheme } = useAppStore();
  const { commandPaletteOpen, setCommandPaletteOpen, setShortcutsOpen } = useLayoutStore();
  const importWsroute = useWsrouteImport();
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(
    () => PALETTE_COMMANDS.filter((cmd) => matchQuery(query, cmd.label, cmd.keywords)),
    [query]
  );

  const runCommand = useCallback(
    (cmd: (typeof PALETTE_COMMANDS)[number]) => {
      if (cmd.mode) setMode(cmd.mode);
      else if (cmd.action === "toggle_theme") setTheme(theme === "dark" ? "light" : "dark");
      else if (cmd.action === "shortcuts_help") setShortcutsOpen(true);
      else if (cmd.action === "import_wsroute") void importWsroute();
      setCommandPaletteOpen(false);
      setQuery("");
      setActiveIndex(0);
    },
    [setMode, setTheme, theme, setShortcutsOpen, setCommandPaletteOpen, importWsroute]
  );

  useEffect(() => {
    if (commandPaletteOpen) {
      setQuery("");
      setActiveIndex(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [commandPaletteOpen]);

  useEffect(() => {
    setActiveIndex((i) => (filtered.length ? Math.min(i, filtered.length - 1) : 0));
  }, [filtered.length, query]);

  if (!commandPaletteOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 p-4 pt-[12vh]"
      onClick={() => setCommandPaletteOpen(false)}
    >
      <div
        className="card w-full max-w-lg overflow-hidden shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 px-3 py-2 border-b border-canvas-border">
          <Search size={14} className="text-canvas-muted shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setActiveIndex((i) => (filtered.length ? (i + 1) % filtered.length : 0));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setActiveIndex((i) => (filtered.length ? (i - 1 + filtered.length) % filtered.length : 0));
              } else if (e.key === "Enter" && filtered[activeIndex]) {
                e.preventDefault();
                runCommand(filtered[activeIndex]);
              } else if (e.key === "Escape") {
                e.preventDefault();
                setCommandPaletteOpen(false);
              }
            }}
            placeholder="Search views and actions…"
            className="input-base flex-1 border-0 bg-transparent text-sm focus:ring-0"
          />
          <kbd className="text-[10px] text-canvas-muted font-mono shrink-0">Esc</kbd>
        </div>

        <ul className="max-h-72 overflow-y-auto py-1">
          {filtered.length === 0 && (
            <li className="px-3 py-4 text-xs text-canvas-muted text-center">No matching commands</li>
          )}
          {filtered.map((cmd, i) => (
            <li key={cmd.id}>
              <button
                type="button"
                onMouseEnter={() => setActiveIndex(i)}
                onClick={() => runCommand(cmd)}
                className={`w-full flex items-center justify-between px-3 py-2 text-left text-sm transition-colors ${
                  i === activeIndex
                    ? "bg-accent-primary/20 text-accent-secondary"
                    : "text-gray-300 hover:bg-canvas-hover"
                }`}
              >
                <span>{cmd.label}</span>
                <span className="text-[10px] text-canvas-muted">{cmd.section}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

import { X } from "lucide-react";
import { useLayoutStore } from "../../store/layout";

const SHORTCUTS: Array<{ keys: string; action: string }> = [
  { keys: "1 – 8", action: "Quick-nav to primary views" },
  { keys: "G", action: "Simulation monitor (geospatial)" },
  { keys: "M", action: "Simulation digital twin (map)" },
  { keys: "P", action: "Process monitor" },
  { keys: "Q", action: "HPO tracker" },
  { keys: "T", action: "Training monitor" },
  { keys: "H", action: "Training & HPO hub" },
  { keys: "E", action: "Experiment tracker" },
  { keys: "L", action: "Simulation launcher" },
  { keys: "D", action: "Data generation wizard" },
  { keys: "V", action: "Evaluation runner" },
  { keys: "B", action: "Benchmark analysis" },
  { keys: "O", action: "Output browser" },
  { keys: "Ctrl+R", action: "Launch on active launcher page" },
  { keys: "Ctrl+S", action: "Save config file (Config Editor, when dirty)" },
  { keys: "Ctrl+.", action: "Cancel first running process" },
  { keys: "Ctrl+Shift+P", action: "Process monitor" },
  { keys: "Ctrl+K", action: "Command palette — search views and actions" },
  { keys: "Ctrl+Shift+/", action: "Guided tour" },
  { keys: "Ctrl+,", action: "Settings" },
  { keys: "?", action: "Show this help overlay" },
];

export function KeyboardShortcutsHelp() {
  const { shortcutsOpen, setShortcutsOpen } = useLayoutStore();

  if (!shortcutsOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={() => setShortcutsOpen(false)}
    >
      <div
        className="card max-w-md w-full space-y-3"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">Keyboard Shortcuts</h2>
          <button onClick={() => setShortcutsOpen(false)} className="btn-ghost p-1">
            <X size={14} />
          </button>
        </div>
        <table className="w-full text-xs">
          <tbody className="divide-y divide-canvas-border">
            {SHORTCUTS.map(({ keys, action }) => (
              <tr key={keys}>
                <td className="py-1.5 pr-4 font-mono text-accent-secondary whitespace-nowrap">{keys}</td>
                <td className="py-1.5 text-canvas-muted">{action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

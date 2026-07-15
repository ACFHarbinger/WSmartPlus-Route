/**
 * Compact Summary + Simulation Monitor log handoff controls (§G.1 / §G.16 / §D.7).
 *
 * Uses the shared mode override so both targets consume ``pendingLogPath``.
 */
import { BarChart2, Map as MapIcon } from "lucide-react";
import { useRecentHandoff } from "../../hooks/useRecentHandoff";

export type LogHandoffTarget = "summary" | "monitor";

interface Props {
  path: string;
  /** Optional stored recent-file label (portfolio / run name). */
  storedLabel?: string;
  className?: string;
  /** Icon size in px (default 11 for dense rows; use 12–14 for toolbars). */
  iconSize?: number;
  /**
   * When true, render labeled text buttons instead of icon-only controls
   * (toolbar parity with Simulation Summary / Monitor).
   */
  labeled?: boolean;
  /**
   * Which destinations to expose. Default both; use a single target on pages
   * that already host the other view (e.g. Summary → Monitor only).
   */
  targets?: LogHandoffTarget[];
}

export function LogHandoffButtons({
  path,
  storedLabel,
  className = "",
  iconSize = 11,
  labeled = false,
  targets = ["summary", "monitor"],
}: Props) {
  const { handoff } = useRecentHandoff();
  const showSummary = targets.includes("summary");
  const showMonitor = targets.includes("monitor");

  if (!showSummary && !showMonitor) return null;

  if (labeled) {
    return (
      <span className={`flex items-center gap-1.5 shrink-0 ${className}`}>
        {showSummary && (
          <button
            type="button"
            title="Open in Simulation Summary"
            onClick={(e) => {
              e.stopPropagation();
              handoff(path, "log", { storedLabel });
            }}
            className="btn-ghost text-xs flex items-center gap-1.5 text-accent-primary"
          >
            <BarChart2 size={iconSize} />
            Simulation Summary →
          </button>
        )}
        {showMonitor && (
          <button
            type="button"
            title="Open in Simulation Monitor"
            onClick={(e) => {
              e.stopPropagation();
              handoff(path, "log", { storedLabel, mode: "simulation" });
            }}
            className="btn-ghost text-xs flex items-center gap-1.5 text-accent-secondary"
          >
            <MapIcon size={iconSize} />
            Simulation Monitor →
          </button>
        )}
      </span>
    );
  }

  return (
    <span className={`flex items-center gap-0.5 shrink-0 ${className}`}>
      {showSummary && (
        <button
          type="button"
          title="Open in Simulation Summary"
          onClick={(e) => {
            e.stopPropagation();
            handoff(path, "log", { storedLabel });
          }}
          className="btn-ghost p-0.5 text-accent-primary"
        >
          <BarChart2 size={iconSize} />
        </button>
      )}
      {showMonitor && (
        <button
          type="button"
          title="Open in Simulation Monitor"
          onClick={(e) => {
            e.stopPropagation();
            handoff(path, "log", { storedLabel, mode: "simulation" });
          }}
          className="btn-ghost p-0.5 text-accent-secondary"
        >
          <MapIcon size={iconSize} />
        </button>
      )}
    </span>
  );
}

/** True when ``path`` looks like a simulation day log suitable for Summary / Monitor. */
export function isSimulationLogPath(path: string): boolean {
  return path.toLowerCase().endsWith(".jsonl");
}

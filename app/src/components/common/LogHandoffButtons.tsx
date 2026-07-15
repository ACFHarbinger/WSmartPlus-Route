/**
 * Compact Summary + Simulation Monitor log handoff controls (§G.1 / §G.16 / §D.7).
 *
 * Uses the shared mode override so both targets consume ``pendingLogPath``.
 * When ``path`` is empty, buttons still navigate to the destination mode
 * (launcher nav-mesh parity without a completed ``.jsonl`` yet).
 */
import type { MouseEvent } from "react";
import { BarChart2, Map as MapIcon } from "lucide-react";
import { useRecentHandoff } from "../../hooks/useRecentHandoff";

export type LogHandoffTarget = "summary" | "monitor";

interface Props {
  /**
   * Simulation day log (``.jsonl``). When omitted/empty, clicks only switch
   * mode (no pending-path handoff / recent push).
   */
  path?: string | null;
  /** Optional stored recent-file label (portfolio / run name). */
  storedLabel?: string;
  className?: string;
  /** Icon size in px (default 11 for dense rows; use 12–14 for toolbars). */
  iconSize?: number;
  /**
   * When true, render labeled text buttons instead of icon-only controls
   * (toolbar parity with Simulation Summary / Monitor / launcher nav).
   */
  labeled?: boolean;
  /**
   * Which destinations to expose. Default both; use a single target on pages
   * that already host the other view (e.g. Summary → Monitor only), or
   * Monitor-only before a post-run Summary surface is appropriate.
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
  const { handoff, setMode } = useRecentHandoff();
  const showSummary = targets.includes("summary");
  const showMonitor = targets.includes("monitor");
  const logPath = path?.trim() ? path.trim() : null;

  if (!showSummary && !showMonitor) return null;

  const openSummary = (e: MouseEvent) => {
    e.stopPropagation();
    if (logPath) {
      handoff(logPath, "log", { storedLabel });
    } else {
      setMode("simulation_summary");
    }
  };

  const openMonitor = (e: MouseEvent) => {
    e.stopPropagation();
    if (logPath) {
      handoff(logPath, "log", { storedLabel, mode: "simulation" });
    } else {
      setMode("simulation");
    }
  };

  if (labeled) {
    return (
      <span className={`flex items-center gap-1.5 shrink-0 ${className}`}>
        {showSummary && (
          <button
            type="button"
            title={
              logPath
                ? "Open in Simulation Summary"
                : "Open Simulation Summary"
            }
            onClick={openSummary}
            className="btn-ghost text-xs flex items-center gap-1.5 text-accent-primary"
          >
            <BarChart2 size={iconSize} />
            Simulation Summary →
          </button>
        )}
        {showMonitor && (
          <button
            type="button"
            title={
              logPath
                ? "Open in Simulation Monitor"
                : "Open Simulation Monitor"
            }
            onClick={openMonitor}
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
          title={
            logPath ? "Open in Simulation Summary" : "Open Simulation Summary"
          }
          onClick={openSummary}
          className="btn-ghost p-0.5 text-accent-primary"
        >
          <BarChart2 size={iconSize} />
        </button>
      )}
      {showMonitor && (
        <button
          type="button"
          title={
            logPath ? "Open in Simulation Monitor" : "Open Simulation Monitor"
          }
          onClick={openMonitor}
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

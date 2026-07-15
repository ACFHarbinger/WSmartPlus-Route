/**
 * Path-aware handoff controls for recent-file kinds (§G.7 / §G.14 / §G.15 / §G.17 / §D.7).
 *
 * Simulation ``.jsonl`` logs keep dual Summary / Monitor icons via ``LogHandoffButtons``.
 * Other kinds expose a single icon that hands off into the default destination
 * (Training Monitor, Output Browser, Data Explorer, Eval Runner, Config Editor).
 */
import type { MouseEvent, ReactNode } from "react";
import {
  Activity,
  ClipboardList,
  Database,
  FileText,
  FolderOpen,
} from "lucide-react";
import { useRecentHandoff } from "../../hooks/useRecentHandoff";
import {
  recentKindFromPath,
  type RecentFileKind,
} from "../../store/recentFiles";
import {
  isSimulationLogPath,
  LogHandoffButtons,
} from "./LogHandoffButtons";

interface Props {
  path: string;
  /**
   * Explicit kind when known (Command Palette recents, toast handlers).
   * When omitted, ``recentKindFromPath`` classifies the path.
   */
  kind?: RecentFileKind | null;
  /** Optional stored recent-file label (portfolio / run name). */
  storedLabel?: string;
  className?: string;
  /** Icon size in px (default 11 for dense rows). */
  iconSize?: number;
  /**
   * When true, render a labeled text button instead of an icon-only control
   * (non-log kinds only; logs use ``LogHandoffButtons`` labeled mode).
   */
  labeled?: boolean;
  /** Called after a handoff (e.g. close Command Palette). */
  onAfterOpen?: () => void;
}

const KIND_META: Record<
  Exclude<RecentFileKind, "log">,
  { title: string; label: string; accent: string; icon: (size: number) => ReactNode }
> = {
  training: {
    title: "Open in Training Monitor",
    label: "Training Monitor →",
    accent: "text-accent-primary",
    icon: (size) => <Activity size={size} />,
  },
  run: {
    title: "Open in Output Browser",
    label: "Output Browser →",
    accent: "text-accent-success",
    icon: (size) => <FolderOpen size={size} />,
  },
  csv: {
    title: "Open in Data Explorer",
    label: "Data Explorer →",
    accent: "text-accent-primary",
    icon: (size) => <Database size={size} />,
  },
  checkpoint: {
    title: "Load in Eval Runner",
    label: "Eval Runner →",
    accent: "text-accent-secondary",
    icon: (size) => <ClipboardList size={size} />,
  },
  config: {
    title: "Open in Config Editor",
    label: "Config Editor →",
    accent: "text-canvas-muted",
    icon: (size) => <FileText size={size} />,
  },
};

/**
 * Icon (or labeled) handoff control for a filesystem path.
 * Returns ``null`` when the path cannot be classified into a handoff kind.
 */
export function PathHandoffButtons({
  path,
  kind,
  storedLabel,
  className = "",
  iconSize = 11,
  labeled = false,
  onAfterOpen,
}: Props) {
  const { handoff } = useRecentHandoff();
  const resolvedKind = kind ?? recentKindFromPath(path);
  if (!resolvedKind) return null;

  if (resolvedKind === "log" || isSimulationLogPath(path)) {
    return (
      <LogHandoffButtons
        path={path}
        storedLabel={storedLabel}
        className={className}
        iconSize={iconSize}
        labeled={labeled}
        onAfterOpen={onAfterOpen}
      />
    );
  }

  const meta = KIND_META[resolvedKind];
  const open = (e: MouseEvent) => {
    e.stopPropagation();
    handoff(path, resolvedKind, { storedLabel });
    onAfterOpen?.();
  };

  if (labeled) {
    return (
      <span className={`flex items-center gap-1.5 shrink-0 ${className}`}>
        <button
          type="button"
          title={meta.title}
          onClick={open}
          className={`btn-ghost text-xs flex items-center gap-1.5 ${meta.accent}`}
        >
          {meta.icon(iconSize)}
          {meta.label}
        </button>
      </span>
    );
  }

  return (
    <span className={`flex items-center gap-0.5 shrink-0 ${className}`}>
      <button
        type="button"
        title={meta.title}
        onClick={open}
        className={`btn-ghost p-0.5 ${meta.accent}`}
      >
        {meta.icon(iconSize)}
      </button>
    </span>
  );
}

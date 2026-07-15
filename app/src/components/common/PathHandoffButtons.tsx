/**
 * Path-aware handoff controls for recent-file kinds (§G.7 / §G.14 / §G.15 / §G.17 / §D.7).
 *
 * Simulation ``.jsonl`` logs keep dual Summary / Monitor icons via ``LogHandoffButtons``.
 * Other kinds expose a single icon (or labeled button) that hands off into the default
 * destination (Training Monitor, Output Browser, Data Explorer, Eval Runner, Config Editor).
 *
 * When ``path`` is empty but ``kind`` is set, buttons still navigate to the destination
 * mode (nav-mesh parity without a completed artefact yet — mirrors ``LogHandoffButtons``).
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
import { recentHandoffSpec } from "../../utils/recentHandoff";
import {
  isSimulationLogPath,
  LogHandoffButtons,
} from "./LogHandoffButtons";

interface Props {
  /**
   * Filesystem path to hand off. When omitted/empty, clicks only switch mode
   * (requires ``kind`` so the destination is known).
   */
  path?: string | null;
  /**
   * Explicit kind when known (Command Palette recents, toast handlers, nav mesh).
   * When omitted, ``recentKindFromPath`` classifies a non-empty ``path``.
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
  {
    titleWithPath: string;
    titleModeOnly: string;
    label: string;
    accent: string;
    icon: (size: number) => ReactNode;
  }
> = {
  training: {
    titleWithPath: "Open in Training Monitor",
    titleModeOnly: "Open Training Monitor",
    label: "Training Monitor →",
    accent: "text-accent-primary",
    icon: (size) => <Activity size={size} />,
  },
  run: {
    titleWithPath: "Open in Output Browser",
    titleModeOnly: "Open Output Browser",
    label: "Output Browser →",
    accent: "text-accent-success",
    icon: (size) => <FolderOpen size={size} />,
  },
  csv: {
    titleWithPath: "Open in Data Explorer",
    titleModeOnly: "Open Data Explorer",
    label: "Data Explorer →",
    accent: "text-accent-primary",
    icon: (size) => <Database size={size} />,
  },
  checkpoint: {
    titleWithPath: "Load in Eval Runner",
    titleModeOnly: "Open Evaluation Runner",
    label: "Eval Runner →",
    accent: "text-accent-secondary",
    icon: (size) => <ClipboardList size={size} />,
  },
  config: {
    titleWithPath: "Open in Config Editor",
    titleModeOnly: "Open Config Editor",
    label: "Config Editor →",
    accent: "text-canvas-muted",
    icon: (size) => <FileText size={size} />,
  },
};

/**
 * Icon (or labeled) handoff control for a filesystem path / known kind.
 * Returns ``null`` when neither path nor kind can resolve a destination.
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
  const { handoff, setMode } = useRecentHandoff();
  const trimmed = path?.trim() ? path.trim() : null;
  const resolvedKind = kind ?? (trimmed ? recentKindFromPath(trimmed) : null);
  if (!resolvedKind) return null;

  if (resolvedKind === "log" || (trimmed != null && isSimulationLogPath(trimmed))) {
    return (
      <LogHandoffButtons
        path={trimmed}
        storedLabel={storedLabel}
        className={className}
        iconSize={iconSize}
        labeled={labeled}
        onAfterOpen={onAfterOpen}
      />
    );
  }

  const meta = KIND_META[resolvedKind];
  const title = trimmed ? meta.titleWithPath : meta.titleModeOnly;
  const open = (e: MouseEvent) => {
    e.stopPropagation();
    if (trimmed) {
      handoff(trimmed, resolvedKind, { storedLabel });
    } else {
      setMode(recentHandoffSpec(resolvedKind).mode);
    }
    onAfterOpen?.();
  };

  if (labeled) {
    return (
      <span className={`flex items-center gap-1.5 shrink-0 ${className}`}>
        <button
          type="button"
          title={title}
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
        title={title}
        onClick={open}
        className={`btn-ghost p-0.5 ${meta.accent}`}
      >
        {meta.icon(iconSize)}
      </button>
    </span>
  );
}

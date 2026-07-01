import type { ProcessStatus } from "../../types";

const CONFIG: Record<ProcessStatus, { label: string; classes: string }> = {
  running: {
    label: "Running",
    classes: "bg-accent-success/20 text-accent-success border-accent-success/30",
  },
  completed: {
    label: "Completed",
    classes: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  },
  cancelled: {
    label: "Cancelled",
    classes: "bg-canvas-muted/20 text-canvas-muted border-canvas-muted/30",
  },
  failed: {
    label: "Failed",
    classes: "bg-accent-danger/20 text-accent-danger border-accent-danger/30",
  },
};

interface Props {
  status: ProcessStatus;
}

export function StatusPill({ status }: Props) {
  const { label, classes } = CONFIG[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-xs font-medium ${classes}`}
    >
      {status === "running" && (
        <span className="w-1.5 h-1.5 rounded-full bg-accent-success animate-pulse" />
      )}
      {label}
    </span>
  );
}

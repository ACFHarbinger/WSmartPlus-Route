/**
 * Shared legend for failure route-diff overlays (§A.6 Option C).
 */

interface Props {
  showOverflow?: boolean;
  showSkipped?: boolean;
  showTourDiff?: boolean;
  tourDiffLabels?: [string, string];
  className?: string;
}

export function FailureOverlayLegend({
  showOverflow = true,
  showSkipped = true,
  showTourDiff = false,
  tourDiffLabels,
  className = "",
}: Props) {
  const items: Array<{ key: string; color: string; label: string }> = [];

  if (showOverflow) {
    items.push({ key: "ovf", color: "#ef4444", label: "Overflow" });
  }
  if (showSkipped) {
    items.push({ key: "skip", color: "#fb923c", label: "Skipped high-fill" });
  }
  if (showTourDiff && tourDiffLabels) {
    items.push({ key: "diff-a", color: "#22d3ee", label: `Only ${tourDiffLabels[0]}` });
    items.push({ key: "diff-b", color: "#c084fc", label: `Only ${tourDiffLabels[1]}` });
  }

  if (items.length === 0) return null;

  return (
    <div className={`flex flex-wrap items-center gap-2 text-[10px] text-canvas-muted ${className}`.trim()}>
      {items.map((item) => (
        <span key={item.key} className="inline-flex items-center gap-1">
          <span
            className="inline-block w-2 h-2 rounded-full border border-white/30"
            style={{ backgroundColor: item.color }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

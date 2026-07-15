/**
 * Shared process-id footer row for launcher and train/HPO live panels (§G.9–§G.12 / §G.10 / §G.15 / §G.17 / §G.18 / §D.7).
 */

export interface ProcessIdFooterProps {
  processId?: string;
  processIds?: string[];
  className?: string;
}

export function ProcessIdFooter({
  processId,
  processIds,
  className = "",
}: ProcessIdFooterProps) {
  const ids =
    processIds ?? (processId != null && processId !== "" ? [processId] : []);
  if (ids.length === 0) return null;

  if (ids.length === 1) {
    return (
      <p className={`text-xs text-canvas-muted font-mono truncate ${className}`.trim()}>
        {ids[0]}
      </p>
    );
  }

  return (
    <div className={`space-y-0.5 ${className}`.trim()}>
      {ids.map((id) => (
        <p key={id} className="text-xs text-canvas-muted font-mono truncate">
          {id}
        </p>
      ))}
    </div>
  );
}

/**
 * Shared rehydration count badges for train/HPO live panels (§G.10 / §G.15 / §G.17 / §G.18).
 */
export interface TrainHpoRehydrationBadgesProps {
  metricCount?: number;
  healthCount?: number;
  attentionCount?: number;
}

export function TrainHpoRehydrationBadges({
  metricCount = 0,
  healthCount = 0,
  attentionCount = 0,
}: TrainHpoRehydrationBadgesProps) {
  if (metricCount === 0 && healthCount === 0 && attentionCount === 0) {
    return null;
  }

  return (
    <>
      {metricCount > 0 && (
        <span className="text-xs text-accent-success">
          {metricCount} metric updates
        </span>
      )}
      {healthCount > 0 && (
        <span className="text-xs text-accent-warning">
          {healthCount} health alerts
        </span>
      )}
      {attentionCount > 0 && (
        <span className="text-xs text-accent-primary">
          {attentionCount} attention snapshots
        </span>
      )}
    </>
  );
}

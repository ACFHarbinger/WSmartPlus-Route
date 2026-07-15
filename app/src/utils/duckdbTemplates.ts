/**
 * Pre-built DuckDB-Wasm query templates (§G.6).
 */

export interface SqlTemplate {
  id: string;
  label: string;
  sql: string;
}

function sqlQuoteList(values: string[]): string {
  return values.map((v) => `'${v.replace(/'/g, "''")}'`).join(", ");
}

export interface PortfolioBrushFilter {
  policies?: string[];
  runLabels?: string[];
  /** City/scale group brush — prefers ``city_scale`` column when present (§G.6). */
  cityScale?: string;
}

/** SQL that mirrors chart brushes on policy and/or ``run_label`` (§G.1 / §G.6). */
export function brushedPortfolioSql(
  tableName: string,
  filter: PortfolioBrushFilter = {}
): string {
  const t = `"${tableName}"`;
  const clauses: string[] = [];
  if (filter.policies?.length) {
    clauses.push(`policy IN (${sqlQuoteList(filter.policies)})`);
  }
  if (filter.runLabels?.length) {
    clauses.push(`run_label IN (${sqlQuoteList(filter.runLabels)})`);
  } else if (filter.cityScale) {
    clauses.push(`city_scale = '${filter.cityScale.replace(/'/g, "''")}'`);
  }
  if (!clauses.length) {
    return `SELECT * FROM ${t} ORDER BY day, policy LIMIT 500`;
  }
  return `SELECT * FROM ${t}
WHERE ${clauses.join(" AND ")}
ORDER BY day, policy`;
}

/** SQL that mirrors the Simulation Summary policy brush (§G.1). */
export function brushedPoliciesSql(
  tableName: string,
  policies: string[]
): string {
  return brushedPortfolioSql(tableName, { policies });
}

const BASE_TEMPLATES = (t: string): SqlTemplate[] => [
  {
    id: "preview",
    label: "Preview rows",
    sql: `SELECT * FROM ${t} LIMIT 100`,
  },
  {
    id: "robustness",
    label: "Robustness profile",
    sql: `SELECT policy,
  ROUND(AVG(overflows), 4) AS mean_overflows,
  ROUND(STDDEV(overflows), 4) AS std_overflows,
  COUNT(*)::INTEGER AS n
FROM ${t}
GROUP BY policy
ORDER BY mean_overflows`,
  },
  {
    id: "variance",
    label: "Variance analysis",
    sql: `SELECT policy,
  ROUND(VAR_POP(profit), 4) AS profit_var,
  ROUND(VAR_POP(kg_per_km), 4) AS kgkm_var,
  COUNT(*)::INTEGER AS n
FROM ${t}
GROUP BY policy
ORDER BY profit_var DESC`,
  },
  {
    id: "pareto",
    label: "Pareto candidates",
    sql: `SELECT policy,
  ROUND(AVG(profit), 4) AS mean_profit,
  ROUND(AVG(overflows), 4) AS mean_overflows,
  ROUND(AVG(kg_per_km), 4) AS mean_kgkm
FROM ${t}
GROUP BY policy
ORDER BY mean_profit DESC`,
  },
  {
    id: "pareto-frontier",
    label: "Pareto efficiency frontier",
    sql: `WITH agg AS (
  SELECT policy,
    ROUND(AVG(profit), 4) AS mean_profit,
    ROUND(AVG(overflows), 4) AS mean_overflows,
    ROUND(AVG(kg_per_km), 4) AS mean_kgkm
  FROM ${t}
  GROUP BY policy
)
SELECT a.policy, a.mean_profit, a.mean_overflows, a.mean_kgkm
FROM agg a
WHERE NOT EXISTS (
  SELECT 1 FROM agg b
  WHERE b.policy <> a.policy
    AND b.mean_profit >= a.mean_profit
    AND b.mean_overflows <= a.mean_overflows
    AND (b.mean_profit > a.mean_profit OR b.mean_overflows < a.mean_overflows)
)
ORDER BY a.mean_profit DESC, a.mean_overflows ASC`,
  },
];

/** Algorithm Comparison templates for single-log policy analysis (§G.6). */
export function algorithmSqlTemplates(tableName: string): SqlTemplate[] {
  const t = `"${tableName}"`;
  return [
    {
      id: "algo-ranking",
      label: "Policy ranking",
      sql: `SELECT policy,
  ROUND(AVG(profit), 4) AS mean_profit,
  ROUND(AVG(overflows), 4) AS mean_overflows,
  ROUND(AVG(kg_per_km), 4) AS mean_kgkm,
  COUNT(*)::INTEGER AS days
FROM ${t}
GROUP BY policy
ORDER BY mean_kgkm DESC`,
    },
    {
      id: "algo-worst-days",
      label: "Worst overflow days",
      sql: `SELECT day, policy, overflows, profit, kg_per_km
FROM ${t}
ORDER BY overflows DESC, day
LIMIT 20`,
    },
    {
      id: "algo-zero-overflow",
      label: "Zero-overflow rate",
      sql: `SELECT policy,
  COUNT(*) FILTER (WHERE overflows = 0)::INTEGER AS zero_days,
  COUNT(*)::INTEGER AS total_days,
  ROUND(100.0 * COUNT(*) FILTER (WHERE overflows = 0) / COUNT(*), 2) AS zero_pct
FROM ${t}
GROUP BY policy
ORDER BY zero_pct DESC`,
    },
    {
      id: "algo-profit-delta",
      label: "Day-over-day profit Δ",
      sql: `WITH ranked AS (
  SELECT policy, day, profit,
    LAG(profit) OVER (PARTITION BY policy ORDER BY day) AS prev_profit
  FROM ${t}
)
SELECT policy, day,
  ROUND(profit - prev_profit, 4) AS profit_delta
FROM ranked
WHERE prev_profit IS NOT NULL
ORDER BY policy, day`,
    },
  ];
}

/** Portfolio templates when ``run_label`` is present (§G.1.4 / §G.6). */
export function portfolioSqlTemplates(tableName: string): SqlTemplate[] {
  const t = `"${tableName}"`;
  return [
    {
      id: "portfolio-robustness",
      label: "Cross-run robustness",
      sql: `SELECT run_label, policy,
  ROUND(AVG(overflows), 4) AS mean_overflows,
  ROUND(STDDEV(overflows), 4) AS std_overflows,
  COUNT(*)::INTEGER AS n
FROM ${t}
GROUP BY run_label, policy
ORDER BY run_label, mean_overflows`,
    },
    {
      id: "portfolio-leaderboard",
      label: "Run leaderboard (kg/km)",
      sql: `SELECT run_label,
  ROUND(AVG(kg_per_km), 4) AS mean_kgkm,
  ROUND(AVG(overflows), 4) AS mean_overflows,
  ROUND(AVG(profit), 4) AS mean_profit
FROM ${t}
GROUP BY run_label
ORDER BY mean_kgkm DESC`,
    },
    {
      id: "portfolio-variance",
      label: "Run×policy variance",
      sql: `SELECT run_label, policy,
  ROUND(VAR_POP(profit), 4) AS profit_var,
  ROUND(VAR_POP(kg_per_km), 4) AS kgkm_var,
  COUNT(*)::INTEGER AS n
FROM ${t}
GROUP BY run_label, policy
ORDER BY profit_var DESC`,
    },
    {
      id: "portfolio-pareto",
      label: "Pareto by run",
      sql: `SELECT run_label, policy,
  ROUND(AVG(profit), 4) AS mean_profit,
  ROUND(AVG(overflows), 4) AS mean_overflows,
  ROUND(AVG(kg_per_km), 4) AS mean_kgkm
FROM ${t}
GROUP BY run_label, policy
ORDER BY run_label, mean_profit DESC`,
    },
    {
      id: "portfolio-city-leaderboard",
      label: "City leaderboard (kg/km)",
      sql: `SELECT city_scale,
  ROUND(AVG(kg_per_km), 4) AS mean_kgkm,
  ROUND(AVG(overflows), 4) AS mean_overflows,
  ROUND(AVG(profit), 4) AS mean_profit,
  COUNT(DISTINCT run_label)::INTEGER AS n_runs
FROM ${t}
GROUP BY city_scale
ORDER BY mean_kgkm DESC`,
    },
    {
      id: "portfolio-city-policy-matrix",
      label: "City×policy matrix (kg/km)",
      sql: `SELECT city_scale, policy,
  ROUND(AVG(kg_per_km), 4) AS mean_kgkm,
  ROUND(AVG(overflows), 4) AS mean_overflows,
  ROUND(AVG(profit), 4) AS mean_profit,
  COUNT(*)::INTEGER AS n
FROM ${t}
GROUP BY city_scale, policy
ORDER BY city_scale, mean_kgkm DESC`,
    },
    {
      id: "portfolio-run-policy-matrix",
      label: "Run×policy matrix (kg/km)",
      sql: `SELECT run_label, policy,
  ROUND(AVG(kg_per_km), 4) AS mean_kgkm,
  ROUND(AVG(overflows), 4) AS mean_overflows,
  ROUND(AVG(profit), 4) AS mean_profit,
  COUNT(*)::INTEGER AS n
FROM ${t}
GROUP BY run_label, policy
ORDER BY run_label, mean_kgkm DESC`,
    },
    {
      id: "portfolio-pareto-frontier",
      label: "Pareto efficiency frontier",
      sql: `WITH agg AS (
  SELECT run_label, policy,
    ROUND(AVG(profit), 4) AS mean_profit,
    ROUND(AVG(overflows), 4) AS mean_overflows,
    ROUND(AVG(kg_per_km), 4) AS mean_kgkm
  FROM ${t}
  GROUP BY run_label, policy
)
SELECT a.run_label, a.policy, a.mean_profit, a.mean_overflows, a.mean_kgkm
FROM agg a
WHERE NOT EXISTS (
  SELECT 1 FROM agg b
  WHERE (b.run_label <> a.run_label OR b.policy <> a.policy)
    AND b.mean_profit >= a.mean_profit
    AND b.mean_overflows <= a.mean_overflows
    AND (b.mean_profit > a.mean_profit OR b.mean_overflows < a.mean_overflows)
)
ORDER BY a.mean_profit DESC, a.mean_overflows ASC`,
    },
  ];
}

export function sqlTemplates(
  tableName: string,
  opts: { portfolio?: boolean; algorithm?: boolean } = {}
): SqlTemplate[] {
  const t = `"${tableName}"`;
  const base = BASE_TEMPLATES(t);
  const extras: SqlTemplate[] = [];
  if (opts.portfolio) extras.push(...portfolioSqlTemplates(tableName));
  if (opts.algorithm) extras.push(...algorithmSqlTemplates(tableName));
  return [...base, ...extras];
}

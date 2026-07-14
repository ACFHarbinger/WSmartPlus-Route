/**
 * Pre-built DuckDB-Wasm query templates (§G.6).
 */

export interface SqlTemplate {
  id: string;
  label: string;
  sql: string;
}

/** SQL that mirrors the Simulation Summary policy brush (§G.1). */
export function brushedPoliciesSql(
  tableName: string,
  policies: string[]
): string {
  const t = `"${tableName}"`;
  if (!policies.length) {
    return `SELECT * FROM ${t} ORDER BY day, policy LIMIT 500`;
  }
  const quoted = policies.map((p) => `'${p.replace(/'/g, "''")}'`).join(", ");
  return `SELECT * FROM ${t}
WHERE policy IN (${quoted})
ORDER BY day, policy`;
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
];

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
  ];
}

export function sqlTemplates(
  tableName: string,
  opts: { portfolio?: boolean } = {}
): SqlTemplate[] {
  const t = `"${tableName}"`;
  const base = BASE_TEMPLATES(t);
  if (!opts.portfolio) return base;
  return [...base, ...portfolioSqlTemplates(tableName)];
}

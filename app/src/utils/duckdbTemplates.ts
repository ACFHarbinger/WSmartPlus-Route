/**
 * Pre-built DuckDB-Wasm query templates (§G.6).
 */

export interface SqlTemplate {
  id: string;
  label: string;
  sql: string;
}

export function sqlTemplates(tableName: string): SqlTemplate[] {
  const t = `"${tableName}"`;
  return [
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
}

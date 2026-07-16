/**
 * Initialise DuckDB-Wasm worker on app load (§G.0 Phase 0).
 */
import { useEffect } from "react";
import { initDuckDb } from "../../utils/duckdb/duckdbClient";
import { useDuckDbStore } from "../../store/duckdb";
import { markStartup } from "../../utils/app/startupTiming";

export function useDuckDbInit() {
  const { setReady, setError } = useDuckDbStore();

  useEffect(() => {
    let cancelled = false;
    initDuckDb()
      .then(() => {
        if (!cancelled) {
          setReady(true);
          markStartup("duckdbReady");
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(String(err));
          console.warn("DuckDB-Wasm init failed:", err);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [setReady, setError]);
}

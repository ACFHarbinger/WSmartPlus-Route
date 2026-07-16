import { useEffect, useState } from "react";
import { getStartupElapsed, markStartup } from "../../utils/app/startupTiming";

export interface StartupTiming {
  firstPaintMs: number | null;
  prefetchMs: number | null;
  duckdbMs: number | null;
  withinBudget: boolean | null;
}

/** Time from module load to first React mount + prefetch milestones (§G.7). */
export function useStartupTiming(): StartupTiming {
  const [timing, setTiming] = useState<StartupTiming>({
    firstPaintMs: null,
    prefetchMs: null,
    duckdbMs: null,
    withinBudget: null,
  });

  const refresh = () => {
    const prefetchMs = getStartupElapsed("prefetchDone");
    setTiming({
      firstPaintMs: getStartupElapsed("firstPaint"),
      prefetchMs,
      duckdbMs: getStartupElapsed("duckdbReady"),
      withinBudget: prefetchMs != null ? prefetchMs <= 2000 : null,
    });
  };

  useEffect(() => {
    markStartup("firstPaint");
    refresh();
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      if (getStartupElapsed("prefetchDone") != null && getStartupElapsed("duckdbReady") != null) {
        refresh();
        window.clearInterval(id);
      }
    }, 200);
    return () => window.clearInterval(id);
  }, []);

  return timing;
}

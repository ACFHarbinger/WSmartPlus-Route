import { useEffect, useState } from "react";
import { getStartupElapsed, markStartup } from "../utils/startupTiming";

export interface StartupTiming {
  firstPaintMs: number | null;
  prefetchMs: number | null;
  withinBudget: boolean | null;
}

/** Time from module load to first React mount + prefetch milestones (§G.7). */
export function useStartupTiming(): StartupTiming {
  const [timing, setTiming] = useState<StartupTiming>({
    firstPaintMs: null,
    prefetchMs: null,
    withinBudget: null,
  });

  const refresh = () => {
    const prefetchMs = getStartupElapsed("prefetchDone");
    setTiming({
      firstPaintMs: getStartupElapsed("firstPaint"),
      prefetchMs,
      withinBudget: prefetchMs != null ? prefetchMs <= 2000 : null,
    });
  };

  useEffect(() => {
    markStartup("firstPaint");
    refresh();
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      if (getStartupElapsed("prefetchDone") != null) {
        refresh();
        window.clearInterval(id);
      }
    }, 200);
    return () => window.clearInterval(id);
  }, []);

  return timing;
}

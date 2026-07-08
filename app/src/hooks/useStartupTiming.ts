import { useEffect, useState } from "react";
import { getStartupElapsed, markStartup } from "../utils/startupTiming";

export interface StartupTiming {
  firstPaintMs: number | null;
  prefetchMs: number | null;
}

/** Time from module load to first React mount + prefetch milestones (§G.7). */
export function useStartupTiming(): StartupTiming {
  const [timing, setTiming] = useState<StartupTiming>({
    firstPaintMs: null,
    prefetchMs: null,
  });

  useEffect(() => {
    markStartup("firstPaint");
    setTiming({
      firstPaintMs: getStartupElapsed("firstPaint"),
      prefetchMs: getStartupElapsed("prefetchDone"),
    });
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      const prefetchMs = getStartupElapsed("prefetchDone");
      if (prefetchMs != null) {
        setTiming((prev) => ({ ...prev, prefetchMs }));
        window.clearInterval(id);
      }
    }, 200);
    return () => window.clearInterval(id);
  }, []);

  return timing;
}

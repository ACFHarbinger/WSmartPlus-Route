import { useEffect, useState } from "react";

const START = performance.now();

/** Time from module load to first React mount (§G.7 performance probe). */
export function useStartupTiming() {
  const [ms, setMs] = useState<number | null>(null);

  useEffect(() => {
    setMs(Math.round(performance.now() - START));
  }, []);

  return ms;
}

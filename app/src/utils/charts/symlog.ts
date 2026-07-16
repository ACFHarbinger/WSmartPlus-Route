/** Symmetric log transform — preserves zero and small values (§G.1 symlog). */

const DEFAULT_LINTHRESH = 1;

export function symlog(value: number, linthresh = DEFAULT_LINTHRESH): number {
  if (value === 0) return 0;
  const sign = Math.sign(value);
  const ax = Math.abs(value);
  if (ax <= linthresh) return value;
  return sign * (linthresh + Math.log10(ax / linthresh));
}

export function symexp(value: number, linthresh = DEFAULT_LINTHRESH): number {
  if (value === 0) return 0;
  const sign = Math.sign(value);
  const ax = Math.abs(value);
  if (ax <= linthresh) return value;
  return sign * linthresh * 10 ** (ax - linthresh);
}

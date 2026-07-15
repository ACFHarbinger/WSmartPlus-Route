/**
 * App-wide policy/sample filter controls (§G.7).
 * Options derived from entries loaded in the simulation log store.
 */
import { X } from "lucide-react";
import { useGlobalFiltersStore } from "../../store/filters";
import { useSimStore, uniquePolicies, uniqueSamples } from "../../store/sim";

interface Props {
  /** Portfolio ``run_label`` options when a multi-run table is active (§G.6). */
  runLabels?: string[];
  /** City/scale group options for portfolio city brush (§G.6). */
  cities?: string[];
}

export function GlobalFilterBar({ runLabels = [], cities = [] }: Props) {
  const { entries } = useSimStore();
  const {
    policy,
    sampleId,
    runLabel,
    brushedCity,
    setPolicy,
    setSampleId,
    setRunLabel,
    setBrushedCity,
    reset,
  } = useGlobalFiltersStore();

  const policies = uniquePolicies(entries);
  const samples = uniqueSamples(entries);

  if (
    entries.length === 0 &&
    policy == null &&
    sampleId == null &&
    runLabel == null &&
    brushedCity == null &&
    runLabels.length === 0 &&
    cities.length === 0
  ) {
    return null;
  }

  return (
    <div className="flex items-center gap-2 flex-wrap text-xs bg-canvas-elevated/50 border border-canvas-border rounded-lg px-3 py-1.5">
      <span className="text-canvas-muted shrink-0">Filters</span>

      {policies.length > 0 ? (
        <select
          className="select-base w-36 text-xs"
          value={policy ?? ""}
          onChange={(e) => setPolicy(e.target.value || null)}
        >
          <option value="">All policies</option>
          {policies.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      ) : (
        <span className="text-canvas-muted italic">Load a sim log for policy options</span>
      )}

      {samples.length > 1 && (
        <select
          className="select-base w-28 text-xs"
          value={sampleId ?? ""}
          onChange={(e) => setSampleId(e.target.value ? Number(e.target.value) : null)}
        >
          <option value="">All samples</option>
          {samples.map((s) => (
            <option key={s} value={s}>Sample {s}</option>
          ))}
        </select>
      )}

      {cities.length > 1 && (
        <select
          className="select-base w-36 text-xs"
          value={brushedCity ?? ""}
          onChange={(e) => {
            setRunLabel(null);
            setBrushedCity(e.target.value || null);
          }}
        >
          <option value="">All cities</option>
          {cities.map((city) => (
            <option key={city} value={city}>{city}</option>
          ))}
        </select>
      )}

      {runLabels.length > 0 && (
        <select
          className="select-base w-44 text-xs"
          value={runLabel ?? ""}
          onChange={(e) => {
            setBrushedCity(null);
            setRunLabel(e.target.value || null);
          }}
        >
          <option value="">All runs</option>
          {runLabels.map((label) => (
            <option key={label} value={label}>{label}</option>
          ))}
        </select>
      )}

      {(policy || sampleId != null || runLabel || brushedCity) && (
        <button
          onClick={reset}
          className="btn-ghost text-xs flex items-center gap-1 text-canvas-muted"
          title="Clear filters"
        >
          <X size={11} />
          Clear
        </button>
      )}
    </div>
  );
}

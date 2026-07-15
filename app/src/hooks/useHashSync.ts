import { useEffect, useRef } from "react";
import { useAppStore } from "../store/app";
import { useGlobalFiltersStore } from "../store/filters";
import type { AppMode } from "../types";

const VALID_MODES = new Set<string>([
  "simulation", "training", "simulation_summary", "benchmark", "city_comparison",
  "olap_explorer", "data_explorer",
  "experiment_tracker", "algorithms", "hpo_tracker", "process_monitor",
  "sim_launcher", "training_hub", "data_gen", "config_editor", "output_browser",
  "eval_runner", "settings",
]);

function encodeHash(
  mode: AppMode,
  policy: string | null,
  sampleId: number | null,
  runLabel: string | null,
  brushedCity: string | null
): string {
  const params = new URLSearchParams();
  params.set("m", mode);
  if (policy) params.set("p", policy);
  if (sampleId != null) params.set("s", String(sampleId));
  if (runLabel) params.set("r", runLabel);
  if (brushedCity) params.set("c", brushedCity);
  return `#${params.toString()}`;
}

function parseHash(hash: string): {
  mode?: AppMode;
  policy?: string;
  sampleId?: number;
  runLabel?: string;
  brushedCity?: string;
} {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  if (!raw) return {};
  const params = new URLSearchParams(raw);
  const result: {
    mode?: AppMode;
    policy?: string;
    sampleId?: number;
    runLabel?: string;
    brushedCity?: string;
  } = {};
  const m = params.get("m");
  if (m && VALID_MODES.has(m)) result.mode = m as AppMode;
  const p = params.get("p");
  if (p) result.policy = p;
  const s = params.get("s");
  if (s != null && s !== "" && !Number.isNaN(Number(s))) result.sampleId = Number(s);
  if (params.has("r")) result.runLabel = params.get("r") ?? "";
  if (params.has("c")) result.brushedCity = params.get("c") ?? "";
  return result;
}

/** Sync app mode + global filters to URL hash for bookmarkable deep-links (§G.7). */
export function useHashSync() {
  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  const policy = useGlobalFiltersStore((s) => s.policy);
  const sampleId = useGlobalFiltersStore((s) => s.sampleId);
  const runLabel = useGlobalFiltersStore((s) => s.runLabel);
  const brushedCity = useGlobalFiltersStore((s) => s.brushedCity);
  const setPolicy = useGlobalFiltersStore((s) => s.setPolicy);
  const setSampleId = useGlobalFiltersStore((s) => s.setSampleId);
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const setBrushedCity = useGlobalFiltersStore((s) => s.setBrushedCity);
  const hydrated = useRef(false);

  // Restore from hash on first mount
  useEffect(() => {
    const parsed = parseHash(window.location.hash);
    if (parsed.mode) setMode(parsed.mode);
    if (parsed.policy !== undefined) setPolicy(parsed.policy);
    if (parsed.sampleId !== undefined) setSampleId(parsed.sampleId);
    if ("runLabel" in parsed) setRunLabel(parsed.runLabel || null);
    if ("brushedCity" in parsed) setBrushedCity(parsed.brushedCity || null);
    hydrated.current = true;
  }, [setMode, setPolicy, setSampleId, setRunLabel, setBrushedCity]);

  // Write hash when state changes (after initial hydration)
  useEffect(() => {
    if (!hydrated.current) return;
    const next = encodeHash(mode, policy, sampleId, runLabel, brushedCity);
    if (window.location.hash !== next) {
      window.history.replaceState(null, "", next);
    }
  }, [mode, policy, sampleId, runLabel, brushedCity]);

  // Respond to browser back/forward
  useEffect(() => {
    const onHashChange = () => {
      const parsed = parseHash(window.location.hash);
      if (parsed.mode) setMode(parsed.mode);
      if ("policy" in parsed) setPolicy(parsed.policy ?? null);
      if ("sampleId" in parsed) setSampleId(parsed.sampleId ?? null);
      if ("runLabel" in parsed) setRunLabel(parsed.runLabel ?? null);
      if ("brushedCity" in parsed) setBrushedCity(parsed.brushedCity ?? null);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [setMode, setPolicy, setSampleId, setRunLabel, setBrushedCity]);
}

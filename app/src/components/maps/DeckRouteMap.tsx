/**
 * deck.gl route map with tile basemap (§G.16).
 * Requires lat/lng in `all_bin_coords`; lazy-loaded from SimulationMonitor.
 */
import { useMemo } from "react";
import DeckGL from "@deck.gl/react";
import { PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import MapGL from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import type { SimDayData } from "../../types";

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

function fillRgb(pct: number): [number, number, number] {
  if (pct >= 100) return [248, 113, 113];
  if (pct >= 80) return [251, 191, 36];
  return [52, 211, 153];
}

export default function DeckRouteMap({ data }: { data: SimDayData }) {
  const { all_bin_coords, tour_indices, bin_state_c } = data;

  const hasGeo = all_bin_coords?.some((b) => b.lat != null && b.lng != null);

  const { path, tourPoints, idlePoints, viewState } = useMemo(() => {
    if (!all_bin_coords?.length) {
      return {
        path: [] as [number, number][],
        tourPoints: [] as Array<{ position: [number, number]; fill: number }>,
        idlePoints: [] as Array<{ position: [number, number] }>,
        viewState: { longitude: 0, latitude: 0, zoom: 12, pitch: 0, bearing: 0 },
      };
    }

    const posById = new Map<number, [number, number]>();
    for (const b of all_bin_coords) {
      if (b.lat != null && b.lng != null) {
        posById.set(b.id, [b.lng, b.lat]);
      }
    }

    const tourSet = new Set(tour_indices ?? []);
    const pathCoords: [number, number][] = [];
    const depot = posById.get(-1);
    if (depot) pathCoords.push(depot);
    for (const id of tour_indices ?? []) {
      const p = posById.get(id);
      if (p) pathCoords.push(p);
    }
    if (depot && pathCoords.length > 1) pathCoords.push(depot);

    const tourPts = (tour_indices ?? [])
      .map((id) => {
        const position = posById.get(id);
        if (!position) return null;
        const fill = (bin_state_c?.[id] ?? 0) * 100;
        return { position, fill };
      })
      .filter(Boolean) as Array<{ position: [number, number]; fill: number }>;

    const idlePts = all_bin_coords
      .filter((b) => b.id >= 0 && !tourSet.has(b.id) && b.lat != null && b.lng != null)
      .map((b) => ({ position: [b.lng!, b.lat!] as [number, number] }));

    const lngs = [...posById.values()].map((p) => p[0]);
    const lats = [...posById.values()].map((p) => p[1]);
    const centerLng = lngs.length ? lngs.reduce((a, b) => a + b, 0) / lngs.length : 0;
    const centerLat = lats.length ? lats.reduce((a, b) => a + b, 0) / lats.length : 0;

    return {
      path: pathCoords,
      tourPoints: tourPts,
      idlePoints: idlePts,
      viewState: { longitude: centerLng, latitude: centerLat, zoom: 13, pitch: 0, bearing: 0 },
    };
  }, [all_bin_coords, tour_indices, bin_state_c]);

  const layers = useMemo(() => {
    const result = [];
    if (path.length >= 2) {
      result.push(
        new PathLayer({
          id: "route-path",
          data: [{ path }],
          getPath: (d: { path: [number, number][] }) => d.path,
          getColor: [99, 102, 241, 220],
          getWidth: 4,
          widthMinPixels: 2,
        })
      );
    }
    if (idlePoints.length) {
      result.push(
        new ScatterplotLayer({
          id: "idle-bins",
          data: idlePoints,
          getPosition: (d: { position: [number, number] }) => d.position,
          getFillColor: [75, 85, 99, 180],
          getRadius: 40,
          radiusMinPixels: 3,
          radiusMaxPixels: 8,
        })
      );
    }
    if (tourPoints.length) {
      result.push(
        new ScatterplotLayer({
          id: "tour-stops",
          data: tourPoints,
          getPosition: (d: { position: [number, number] }) => d.position,
          getFillColor: (d: { fill: number }) => [...fillRgb(d.fill), 230] as [number, number, number, number],
          getRadius: 60,
          radiusMinPixels: 5,
          radiusMaxPixels: 12,
        })
      );
    }
    return result;
  }, [path, idlePoints, tourPoints]);

  if (!hasGeo) {
    return (
      <p className="text-xs text-canvas-muted py-4 text-center">
        Tile map requires geographic lat/lng coordinates in the simulation log.
      </p>
    );
  }

  return (
    <div className="relative h-[320px] rounded-lg overflow-hidden border border-canvas-border">
      <DeckGL initialViewState={viewState} controller layers={layers}>
        <MapGL mapStyle={MAP_STYLE} />
      </DeckGL>
    </div>
  );
}

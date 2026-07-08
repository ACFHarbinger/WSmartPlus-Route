/**
 * deck.gl route map with tile basemap (§G.16).
 * Supports multi-policy route overlay; TripsLayer animation; depot marker.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { Download } from "lucide-react";
import { exportCanvasPng } from "../../utils/chartExport";
import { PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { TripsLayer } from "@deck.gl/geo-layers";
import MapGL from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import type { SimDayData } from "../../types";

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const TRIP_SEGMENT_MS = 120;

export interface MapRoute {
  id: string;
  label: string;
  data: SimDayData;
  color: [number, number, number];
}

function fillRgb(pct: number): [number, number, number] {
  if (pct >= 100) return [248, 113, 113];
  if (pct >= 80) return [251, 191, 36];
  return [52, 211, 153];
}

function buildRouteGeometry(data: SimDayData) {
  const { all_bin_coords, tour_indices, bin_state_c } = data;
  if (!all_bin_coords?.length) {
    return {
      path: [] as [number, number][],
      tripPath: [] as [number, number, number][],
      tripLength: 0,
      tourPoints: [] as Array<{ position: [number, number]; fill: number }>,
      idlePoints: [] as Array<{ position: [number, number] }>,
      depotPoint: null as { position: [number, number] } | null,
      hasGeo: false,
      center: { longitude: 0, latitude: 0 },
    };
  }

  const posById = new Map<number, [number, number]>();
  for (const b of all_bin_coords) {
    if (b.lat != null && b.lng != null) posById.set(b.id, [b.lng, b.lat]);
  }

  const tourSet = new Set(tour_indices ?? []);
  const pathCoords: [number, number][] = [];
  const tripCoords: [number, number, number][] = [];
  let t = 0;

  const pushStop = (id: number) => {
    const p = posById.get(id);
    if (!p) return;
    pathCoords.push(p);
    tripCoords.push([p[0], p[1], t]);
    t += TRIP_SEGMENT_MS;
  };

  const depot = posById.get(-1);
  if (depot) pushStop(-1);
  for (const id of tour_indices ?? []) pushStop(id);
  if (depot && pathCoords.length > 1) {
    pathCoords.push(depot);
    tripCoords.push([depot[0], depot[1], t]);
  }

  const tourPts = (tour_indices ?? [])
    .map((id) => {
      const position = posById.get(id);
      if (!position) return null;
      return { position, fill: (bin_state_c?.[id] ?? 0) * 100 };
    })
    .filter(Boolean) as Array<{ position: [number, number]; fill: number }>;

  const idlePts = all_bin_coords
    .filter((b) => b.id >= 0 && !tourSet.has(b.id) && b.lat != null && b.lng != null)
    .map((b) => ({ position: [b.lng!, b.lat!] as [number, number] }));

  const lngs = [...posById.values()].map((p) => p[0]);
  const lats = [...posById.values()].map((p) => p[1]);

  return {
    path: pathCoords,
    tripPath: tripCoords,
    tripLength: t,
    tourPoints: tourPts,
    idlePoints: idlePts,
    depotPoint: depot ? { position: depot } : null,
    hasGeo: all_bin_coords.some((b) => b.lat != null && b.lng != null),
    center: {
      longitude: lngs.length ? lngs.reduce((a, b) => a + b, 0) / lngs.length : 0,
      latitude: lats.length ? lats.reduce((a, b) => a + b, 0) / lats.length : 0,
    },
  };
}

interface Props {
  routes: MapRoute[];
  animate?: boolean;
  playbackSpeed?: number;
}

export default function DeckRouteMap({ routes, animate = false, playbackSpeed = 1 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState(0);

  const exportPng = useCallback(() => {
    const canvas = containerRef.current?.querySelector("canvas");
    exportCanvasPng(canvas, "route-map-tile.png");
  }, []);

  const geometries = useMemo(
    () => routes.map((r) => ({ route: r, ...buildRouteGeometry(r.data) })),
    [routes]
  );

  const hasGeo = geometries.some((g) => g.hasGeo);
  const maxTripLength = Math.max(...geometries.map((g) => g.tripLength), 0);

  const viewState = useMemo(() => {
    const centers = geometries.filter((g) => g.hasGeo).map((g) => g.center);
    if (!centers.length) {
      return { longitude: 0, latitude: 0, zoom: 12, pitch: 0, bearing: 0 };
    }
    const longitude = centers.reduce((a, c) => a + c.longitude, 0) / centers.length;
    const latitude = centers.reduce((a, c) => a + c.latitude, 0) / centers.length;
    return { longitude, latitude, zoom: 13, pitch: 0, bearing: 0 };
  }, [geometries]);

  const tripSignature = geometries.map((g) => g.tripPath.length).join(",");

  useEffect(() => {
    setCurrentTime(0);
  }, [tripSignature, animate]);

  useEffect(() => {
    if (!animate || maxTripLength <= 0) return;

    let frame: number;
    let last = performance.now();

    const tick = (now: number) => {
      const dt = (now - last) * playbackSpeed;
      last = now;
      setCurrentTime((t) => {
        const next = t + dt;
        return next > maxTripLength ? 0 : next;
      });
      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [animate, playbackSpeed, maxTripLength]);

  const layers = useMemo(() => {
    const result = [];
    const base = geometries[0];

    if (base?.idlePoints.length) {
      result.push(
        new ScatterplotLayer({
          id: "idle-bins",
          data: base.idlePoints,
          getPosition: (d: { position: [number, number] }) => d.position,
          getFillColor: [75, 85, 99, 120],
          getRadius: 40,
          radiusMinPixels: 3,
          radiusMaxPixels: 8,
        })
      );
    }

    if (base?.depotPoint) {
      result.push(
        new ScatterplotLayer({
          id: "depot",
          data: [base.depotPoint],
          getPosition: (d: { position: [number, number] }) => d.position,
          getFillColor: [251, 191, 36, 255],
          getLineColor: [255, 255, 255, 200],
          stroked: true,
          getRadius: 90,
          radiusMinPixels: 8,
          radiusMaxPixels: 14,
        })
      );
    }

    for (const g of geometries) {
      const rgb: [number, number, number, number] = [...g.route.color, 220];
      const id = g.route.id;

      if (animate && g.tripPath.length >= 2) {
        result.push(
          new TripsLayer({
            id: `trips-${id}`,
            data: [{ path: g.tripPath }],
            getPath: (d: { path: [number, number, number][] }) => d.path,
            getTimestamps: (d: { path: [number, number, number][] }) => d.path.map((p) => p[2]),
            getColor: rgb,
            currentTime,
            trailLength: Math.max(g.tripLength * 0.6, TRIP_SEGMENT_MS * 2),
            capRounded: true,
            fadeTrail: true,
            widthMinPixels: 3,
          })
        );
      } else if (g.path.length >= 2) {
        result.push(
          new PathLayer({
            id: `path-${id}`,
            data: [{ path: g.path }],
            getPath: (d: { path: [number, number][] }) => d.path,
            getColor: rgb,
            getWidth: routes.length > 1 ? 3 : 4,
            widthMinPixels: 2,
          })
        );
      }

      if (g.tourPoints.length) {
        result.push(
          new ScatterplotLayer({
            id: `stops-${id}`,
            data: g.tourPoints,
            getPosition: (d: { position: [number, number] }) => d.position,
            getFillColor: (d: { fill: number }) => [...fillRgb(d.fill), 230] as [number, number, number, number],
            getLineColor: [...g.route.color, 255],
            stroked: routes.length > 1,
            getRadius: routes.length > 1 ? 50 : 60,
            radiusMinPixels: 4,
            radiusMaxPixels: 12,
          })
        );
      }
    }

    return result;
  }, [geometries, animate, currentTime, routes.length]);

  if (!hasGeo) {
    return (
      <p className="text-xs text-canvas-muted py-4 text-center">
        Tile map requires geographic lat/lng coordinates in the simulation log.
      </p>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative h-[320px] rounded-lg overflow-hidden border border-canvas-border"
    >
      <DeckGL initialViewState={viewState} controller layers={layers}>
        <MapGL mapStyle={MAP_STYLE} />
      </DeckGL>
      <button
        onClick={exportPng}
        className="absolute top-2 right-2 btn-ghost text-xs flex items-center gap-1 bg-black/50 rounded px-1.5 py-0.5"
        title="Export map as PNG"
      >
        <Download size={11} />
        PNG
      </button>
      {routes.length > 1 && (
        <div className="absolute bottom-2 left-2 flex flex-wrap gap-1.5 max-w-[90%]">
          {routes.map((r) => (
            <span
              key={r.id}
              className="text-[10px] px-1.5 py-0.5 rounded bg-black/50 text-gray-200 font-mono"
              style={{ borderLeft: `3px solid rgb(${r.color.join(",")})` }}
            >
              {r.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

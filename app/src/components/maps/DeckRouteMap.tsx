/**
 * deck.gl route map with tile basemap (§G.16) or OrbitView Cartesian mode (§G.3.4).
 * Supports multi-policy route overlay; TripsLayer animation; depot marker.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { OrbitView, COORDINATE_SYSTEM } from "@deck.gl/core";
import { Download } from "lucide-react";
import { exportCanvasPng } from "../../utils/chartExport";
import { resolveBinPositions } from "../../utils/mapPositions";
import { PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { TripsLayer } from "@deck.gl/geo-layers";
import MapGL from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import type { SimDayData } from "../../types";
import { splitVehicleTourIndices, VEHICLE_COLORS_RGB } from "../../utils/vehicleTours";

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

interface VehicleGeometry {
  vehicleId: number;
  color: [number, number, number];
  path: [number, number][];
  tripPath: [number, number, number][];
  tripLength: number;
}

interface VehicleTourStop {
  position: [number, number];
  fill: number;
  vehicleId: number;
  color: [number, number, number];
}

function buildVehicleSegment(
  segment: number[],
  posById: Map<number, [number, number]>,
  vehicleId: number,
  timeOffset: number
): VehicleGeometry {
  const pathCoords: [number, number][] = [];
  const tripCoords: [number, number, number][] = [];
  let t = timeOffset;

  const pushStop = (id: number) => {
    const p = posById.get(id);
    if (!p) return;
    pathCoords.push(p);
    tripCoords.push([p[0], p[1], t]);
    t += TRIP_SEGMENT_MS;
  };

  const depot = posById.get(-1);
  if (depot) pushStop(-1);
  for (const id of segment) pushStop(id);
  if (depot && pathCoords.length > 1) {
    pathCoords.push(depot);
    tripCoords.push([depot[0], depot[1], t]);
  }

  return {
    vehicleId,
    color: VEHICLE_COLORS_RGB[vehicleId % VEHICLE_COLORS_RGB.length],
    path: pathCoords,
    tripPath: tripCoords,
    tripLength: t,
  };
}

function buildRouteGeometry(data: SimDayData) {
  const { all_bin_coords, tour_indices, bin_state_c } = data;
  if (!all_bin_coords?.length) {
    return {
      vehicles: [] as VehicleGeometry[],
      path: [] as [number, number][],
      tripPath: [] as [number, number, number][],
      tripLength: 0,
      tourPoints: [] as Array<{ position: [number, number]; fill: number }>,
      vehicleTourStops: [] as VehicleTourStop[],
      idlePoints: [] as Array<{ position: [number, number] }>,
      depotPoint: null as { position: [number, number] } | null,
      hasGeo: false,
      hasBins: false,
      center: { longitude: 0, latitude: 0 },
    };
  }

  const { posById, hasGeo } = resolveBinPositions(all_bin_coords);

  const segments = splitVehicleTourIndices(data);
  const allTourIds = new Set(segments.flat());
  const tourSet = new Set(tour_indices ?? allTourIds);

  let timeOffset = 0;
  const vehicles = segments.map((segment, i) => {
    const geom = buildVehicleSegment(segment, posById, i, timeOffset);
    timeOffset = geom.tripLength + TRIP_SEGMENT_MS;
    return geom;
  });

  const primary = vehicles[0];
  const pathCoords = primary?.path ?? [];
  const tripCoords = primary?.tripPath ?? [];
  const tripLength = vehicles.length
    ? Math.max(...vehicles.map((v) => v.tripLength), 0)
    : 0;

  const tourPts = [...tourSet]
    .map((id) => {
      const position = posById.get(id);
      if (!position) return null;
      return { position, fill: (bin_state_c?.[id] ?? 0) * 100 };
    })
    .filter(Boolean) as Array<{ position: [number, number]; fill: number }>;

  const vehicleTourStops: VehicleTourStop[] = segments.flatMap((segment, vi) =>
    segment
      .map((id) => {
        const position = posById.get(id);
        if (!position) return null;
        return {
          position,
          fill: (bin_state_c?.[id] ?? 0) * 100,
          vehicleId: vi,
          color: VEHICLE_COLORS_RGB[vi % VEHICLE_COLORS_RGB.length],
        };
      })
      .filter(Boolean) as VehicleTourStop[]
  );

  const idlePts = all_bin_coords
    .filter((b) => b.id >= 0 && !tourSet.has(b.id))
    .map((b) => {
      const position = posById.get(b.id);
      return position ? { position } : null;
    })
    .filter(Boolean) as Array<{ position: [number, number] }>;

  const depot = posById.get(-1);
  const lngs = [...posById.values()].map((p) => p[0]);
  const lats = [...posById.values()].map((p) => p[1]);

  return {
    vehicles,
    path: pathCoords,
    tripPath: tripCoords,
    tripLength,
    tourPoints: tourPts,
    vehicleTourStops,
    idlePoints: idlePts,
    depotPoint: depot ? { position: depot } : null,
    hasGeo,
    hasBins: posById.size > 0,
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
  const [pitch3d, setPitch3d] = useState(false);
  const [mapView, setMapView] = useState({
    longitude: 0,
    latitude: 0,
    zoom: 13,
    pitch: 0,
    bearing: 0,
  });
  const [orbitView, setOrbitView] = useState({
    target: [0, 0, 0] as [number, number, number],
    rotationX: 35,
    rotationOrbit: 0,
    zoom: 1.2,
  });

  const exportPng = useCallback(() => {
    const canvas = containerRef.current?.querySelector("canvas");
    exportCanvasPng(canvas, "route-map-tile.png");
  }, []);

  const geometries = useMemo(
    () => routes.map((r) => ({ route: r, ...buildRouteGeometry(r.data) })),
    [routes]
  );

  const hasGeo = geometries.some((g) => g.hasGeo);
  const hasBins = geometries.some((g) => g.hasBins);
  const cartesianMode = hasBins && !hasGeo;
  const maxTripLength = Math.max(...geometries.map((g) => g.tripLength), 0);
  const cartesianSystem = cartesianMode ? COORDINATE_SYSTEM.CARTESIAN : COORDINATE_SYSTEM.LNGLAT;

  const centerView = useMemo(() => {
    const centers = geometries.filter((g) => g.hasGeo).map((g) => g.center);
    if (!centers.length) {
      return { longitude: 0, latitude: 0, zoom: 12 };
    }
    const longitude = centers.reduce((a, c) => a + c.longitude, 0) / centers.length;
    const latitude = centers.reduce((a, c) => a + c.latitude, 0) / centers.length;
    return { longitude, latitude, zoom: 13 };
  }, [geometries]);

  useEffect(() => {
    setMapView((v) => ({
      ...v,
      ...centerView,
      pitch: pitch3d ? 45 : 0,
    }));
  }, [centerView, pitch3d]);

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

  const pos3d = (p: [number, number], z = 0): [number, number, number] => [p[0], p[1], z];

  const layers = useMemo(() => {
    const result = [];
    const base = geometries[0];

    if (base?.idlePoints.length) {
      result.push(
        new ScatterplotLayer({
          id: "idle-bins",
          data: base.idlePoints,
          coordinateSystem: cartesianSystem,
          getPosition: (d: { position: [number, number] }) =>
            cartesianMode ? pos3d(d.position, 0) : d.position,
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
          coordinateSystem: cartesianSystem,
          getPosition: (d: { position: [number, number] }) =>
            cartesianMode ? pos3d(d.position, 0.05) : d.position,
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
      const policyRgb: [number, number, number, number] = [...g.route.color, 220];
      const id = g.route.id;
      const vehicleSegments =
        g.vehicles.length > 0
          ? g.vehicles
          : [{ vehicleId: 0, color: g.route.color, path: g.path, tripPath: g.tripPath, tripLength: g.tripLength }];

      for (const vehicle of vehicleSegments) {
        const rgb: [number, number, number, number] =
          vehicleSegments.length > 1
            ? [...vehicle.color, 230]
            : policyRgb;

        if (!cartesianMode && animate && vehicle.tripPath.length >= 2) {
          result.push(
            new TripsLayer({
              id: `trips-${id}-v${vehicle.vehicleId}`,
              data: [{ path: vehicle.tripPath }],
              getPath: (d: { path: [number, number, number][] }) => d.path,
              getTimestamps: (d: { path: [number, number, number][] }) => d.path.map((p) => p[2]),
              getColor: rgb,
              currentTime,
              trailLength: Math.max(vehicle.tripLength * 0.6, TRIP_SEGMENT_MS * 2),
              capRounded: true,
              fadeTrail: true,
              widthMinPixels: 3,
            })
          );
        } else if (vehicle.path.length >= 2) {
          const pathData = cartesianMode
            ? [{ path: vehicle.path.map((p) => pos3d(p, 0.02)) }]
            : [{ path: vehicle.path }];
          result.push(
            new PathLayer({
              id: `path-${id}-v${vehicle.vehicleId}`,
              data: pathData,
              coordinateSystem: cartesianSystem,
              getPath: (d: { path: [number, number][] | [number, number, number][] }) => d.path,
              getColor: rgb,
              getWidth: vehicleSegments.length > 1 || routes.length > 1 ? 3 : 4,
              widthMinPixels: 2,
            })
          );
        }
      }

      const multiVehicleStops = g.vehicles.length > 1 && g.vehicleTourStops.length > 0;
      const stopGroups = multiVehicleStops
        ? g.vehicleTourStops.reduce<Record<number, VehicleTourStop[]>>((acc, stop) => {
            if (!acc[stop.vehicleId]) acc[stop.vehicleId] = [];
            acc[stop.vehicleId].push(stop);
            return acc;
          }, {})
        : null;

      if (stopGroups) {
        for (const [vehicleId, stops] of Object.entries(stopGroups)) {
          result.push(
            new ScatterplotLayer({
              id: `stops-${id}-v${vehicleId}`,
              data: stops,
              coordinateSystem: cartesianSystem,
              getPosition: (d: VehicleTourStop) =>
                cartesianMode ? pos3d(d.position, d.fill * 0.003) : d.position,
              getFillColor: (d: VehicleTourStop) =>
                [...fillRgb(d.fill), 230] as [number, number, number, number],
              getLineColor: (d: VehicleTourStop) => [...d.color, 255] as [number, number, number, number],
              stroked: true,
              getRadius: (d: VehicleTourStop) => {
                const base = routes.length > 1 ? 40 : 50;
                const scale = 0.5 + (Math.min(100, d.fill) / 100) * 0.5;
                return base * scale;
              },
              radiusMinPixels: 4,
              radiusMaxPixels: 14,
            })
          );
        }
      } else if (g.tourPoints.length) {
        result.push(
          new ScatterplotLayer({
            id: `stops-${id}`,
            data: g.tourPoints,
            coordinateSystem: cartesianSystem,
            getPosition: (d: { position: [number, number]; fill: number }) =>
              cartesianMode ? pos3d(d.position, d.fill * 0.003) : d.position,
            getFillColor: (d: { fill: number }) => [...fillRgb(d.fill), 230] as [number, number, number, number],
            getLineColor: [...g.route.color, 255],
            stroked: routes.length > 1,
            getRadius: (d: { fill: number }) => {
              const base = routes.length > 1 ? 40 : 50;
              const scale = 0.5 + (Math.min(100, d.fill) / 100) * 0.5;
              return base * scale;
            },
            radiusMinPixels: 4,
            radiusMaxPixels: 14,
          })
        );
      }
    }

    return result;
  }, [geometries, animate, currentTime, routes.length, cartesianMode, cartesianSystem]);

  if (!hasBins) {
    return (
      <p className="text-xs text-canvas-muted py-4 text-center">
        Load a simulation log with bin coordinates to view the route map.
      </p>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative h-[320px] rounded-lg overflow-hidden border border-canvas-border"
    >
      {cartesianMode ? (
        <DeckGL
          views={new OrbitView({ orbitAxis: "Z" })}
          viewState={orbitView}
          onViewStateChange={({ viewState }) => {
            if ("rotationOrbit" in viewState) {
              setOrbitView({
                target: (viewState.target as [number, number, number]) ?? [0, 0, 0],
                rotationX: viewState.rotationX ?? orbitView.rotationX,
                rotationOrbit: viewState.rotationOrbit ?? orbitView.rotationOrbit,
                zoom: viewState.zoom ?? orbitView.zoom,
              });
            }
          }}
          controller
          layers={layers}
        />
      ) : (
        <DeckGL
          viewState={mapView}
          onViewStateChange={({ viewState }) => {
            if ("latitude" in viewState && "longitude" in viewState) {
              setMapView({
                longitude: viewState.longitude,
                latitude: viewState.latitude,
                zoom: viewState.zoom ?? mapView.zoom,
                pitch: viewState.pitch ?? mapView.pitch,
                bearing: viewState.bearing ?? mapView.bearing,
              });
            }
          }}
          controller
          layers={layers}
        >
          <MapGL mapStyle={MAP_STYLE} />
        </DeckGL>
      )}
      <div className="absolute top-2 right-2 flex items-center gap-1">
        {cartesianMode ? (
          <span className="text-[10px] px-1.5 py-0.5 bg-black/50 rounded text-accent-secondary">
            OrbitView
          </span>
        ) : (
          <button
            onClick={() => setPitch3d((v) => !v)}
            className={`btn-ghost text-xs px-1.5 py-0.5 bg-black/50 rounded ${
              pitch3d ? "text-accent-secondary" : ""
            }`}
            title="Toggle 3D pitch"
          >
            {pitch3d ? "3D (on)" : "3D (off)"}
          </button>
        )}
        <button
          onClick={exportPng}
          className="btn-ghost text-xs flex items-center gap-1 bg-black/50 rounded px-1.5 py-0.5"
          title="Export map as PNG"
        >
          <Download size={11} />
          PNG
        </button>
      </div>
      {(routes.length > 1 || geometries.some((g) => g.vehicles.length > 1)) && (
        <div className="absolute bottom-2 left-2 flex flex-wrap gap-1.5 max-w-[90%]">
          {routes.map((r) => {
            const geom = geometries.find((g) => g.route.id === r.id);
            if (geom && geom.vehicles.length > 1) {
              return geom.vehicles.map((v) => (
                <span
                  key={`${r.id}-v${v.vehicleId}`}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-black/50 text-gray-200 font-mono"
                  style={{ borderLeft: `3px solid rgb(${v.color.join(",")})` }}
                >
                  {r.label} · V{v.vehicleId + 1}
                </span>
              ));
            }
            return (
              <span
                key={r.id}
                className="text-[10px] px-1.5 py-0.5 rounded bg-black/50 text-gray-200 font-mono"
                style={{ borderLeft: `3px solid rgb(${r.color.join(",")})` }}
              >
                {r.label}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

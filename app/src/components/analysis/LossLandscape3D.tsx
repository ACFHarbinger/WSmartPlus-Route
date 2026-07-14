/**
 * React Three Fiber 3D loss topography (§G.5.2).
 */
import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { analyzeLossMinima, lossToColor, normalizeGrid } from "../../utils/lossLandscape";

interface TerrainProps {
  values: number[][];
  minimaRow?: number;
  minimaCol?: number;
}

function TerrainMesh({ values, minimaRow, minimaCol }: TerrainProps) {
  const rows = values.length;
  const cols = values[0]?.length ?? 0;

  const { geometry, minimaPos } = useMemo(() => {
    const { norm, min, max } = normalizeGrid(values);
    const geo = new THREE.PlaneGeometry(cols - 1, rows - 1, cols - 1, rows - 1);
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);

    for (let i = 0; i < pos.count; i++) {
      const ix = i % cols;
      const iy = Math.floor(i / cols);
      const r = Math.min(iy, rows - 1);
      const c = Math.min(ix, cols - 1);
      const h = norm[r]?.[c] ?? 0;
      pos.setZ(i, h * 2.5);
      const [cr, cg, cb] = lossToColor(h);
      colors[i * 3] = cr;
      colors[i * 3 + 1] = cg;
      colors[i * 3 + 2] = cb;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    const mr = minimaRow ?? 0;
    const mc = minimaCol ?? 0;
    const x = mc - (cols - 1) / 2;
    const y = -(mr - (rows - 1) / 2);
    const z = ((norm[mr]?.[mc] ?? 0) * 2.5) + 0.15;

    return {
      geometry: geo,
      minimaPos: [x, y, z] as [number, number, number],
      range: { min, max },
    };
  }, [values, rows, cols, minimaRow, minimaCol]);

  return (
    <group>
      <mesh geometry={geometry} rotation={[-Math.PI / 2.2, 0, 0]}>
        <meshStandardMaterial vertexColors side={THREE.DoubleSide} metalness={0.1} roughness={0.65} />
      </mesh>
      {minimaRow != null && minimaCol != null && (
        <mesh position={minimaPos}>
          <sphereGeometry args={[0.12, 16, 16]} />
          <meshStandardMaterial color="#22d3ee" emissive="#0891b2" emissiveIntensity={0.6} />
        </mesh>
      )}
    </group>
  );
}

interface LossLandscape3DProps {
  values: number[][];
  height?: number;
  className?: string;
}

export function LossLandscape3D({ values, height = 280, className }: LossLandscape3DProps) {
  const minima = useMemo(() => analyzeLossMinima(values), [values]);

  if (!values.length || !values[0]?.length) {
    return (
      <div
        className={`flex items-center justify-center rounded-lg bg-canvas-elevated text-xs text-canvas-muted ${className ?? ""}`}
        style={{ height }}
      >
        No loss grid loaded
      </div>
    );
  }

  return (
    <div className={`rounded-lg overflow-hidden bg-[#0a0e1a] border border-canvas-border ${className ?? ""}`} style={{ height }}>
      <Canvas camera={{ position: [4, 4, 4], fov: 45 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.55} />
        <directionalLight position={[5, 8, 3]} intensity={0.85} />
        <directionalLight position={[-4, 2, -2]} intensity={0.35} />
        <TerrainMesh
          values={values}
          minimaRow={minima?.row}
          minimaCol={minima?.col}
        />
        <OrbitControls enableDamping dampingFactor={0.08} minDistance={2} maxDistance={12} />
      </Canvas>
      {minima && (
        <p className="text-[10px] text-canvas-muted px-2 py-1 border-t border-canvas-border">
          Global min {minima.value.toFixed(4)} at ({minima.row}, {minima.col}) · sharpness{" "}
          {minima.sharpness.toFixed(3)} ({minima.label} basin)
        </p>
      )}
    </div>
  );
}

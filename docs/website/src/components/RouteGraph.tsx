import { useEffect, useRef } from "react";

type Node = { x: number; y: number; r: number; kind: "depot" | "bin" | "hub" };
type Edge = { a: number; b: number; active: boolean };

const NODES: Node[] = [
  { x: 0.12, y: 0.55, r: 7, kind: "depot" },
  { x: 0.28, y: 0.28, r: 4.5, kind: "bin" },
  { x: 0.42, y: 0.62, r: 5, kind: "hub" },
  { x: 0.55, y: 0.22, r: 4.2, kind: "bin" },
  { x: 0.68, y: 0.48, r: 4.8, kind: "bin" },
  { x: 0.82, y: 0.32, r: 4.5, kind: "bin" },
  { x: 0.9, y: 0.68, r: 4.2, kind: "bin" },
  { x: 0.5, y: 0.82, r: 4.5, kind: "bin" },
];

const EDGES: Edge[] = [
  { a: 0, b: 1, active: true },
  { a: 1, b: 2, active: true },
  { a: 2, b: 4, active: true },
  { a: 4, b: 5, active: true },
  { a: 5, b: 6, active: true },
  { a: 6, b: 7, active: false },
  { a: 2, b: 7, active: true },
  { a: 7, b: 0, active: true },
  { a: 1, b: 3, active: false },
  { a: 3, b: 5, active: false },
  { a: 0, b: 2, active: false },
];

/** Animated canvas: depot → bins route graph for the hero card. */
export default function RouteGraph() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let start = performance.now();
    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (now: number) => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const t = reduceMotion ? 0 : (now - start) / 1000;

      ctx.clearRect(0, 0, w, h);

      // subtle grid
      ctx.strokeStyle = "rgba(255,255,255,0.04)";
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 28) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 28) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      const pts = NODES.map((n) => ({
        x: n.x * w,
        y: n.y * h,
        r: n.r,
        kind: n.kind,
      }));

      // candidate / inactive edges
      for (const e of EDGES) {
        const A = pts[e.a];
        const B = pts[e.b];
        ctx.beginPath();
        ctx.moveTo(A.x, A.y);
        ctx.lineTo(B.x, B.y);
        if (e.active) {
          ctx.strokeStyle = "rgba(76, 201, 240, 0.18)";
          ctx.lineWidth = 3.5;
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(A.x, A.y);
          ctx.lineTo(B.x, B.y);
          ctx.strokeStyle = "rgba(61, 255, 168, 0.55)";
          ctx.lineWidth = 1.4;
          ctx.setLineDash([6, 5]);
          ctx.lineDashOffset = reduceMotion ? 0 : -t * 28;
          ctx.stroke();
          ctx.setLineDash([]);
        } else {
          ctx.strokeStyle = "rgba(139, 149, 168, 0.12)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // pulse along active route
      if (!reduceMotion) {
        const active = EDGES.filter((e) => e.active);
        const seg = active[Math.floor(t * 0.7) % active.length];
        if (seg) {
          const A = pts[seg.a];
          const B = pts[seg.b];
          const u = (t * 0.7) % 1;
          const px = A.x + (B.x - A.x) * u;
          const py = A.y + (B.y - A.y) * u;
          const g = ctx.createRadialGradient(px, py, 0, px, py, 18);
          g.addColorStop(0, "rgba(61,255,168,0.55)");
          g.addColorStop(1, "rgba(61,255,168,0)");
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.arc(px, py, 18, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // nodes
      for (const p of pts) {
        const isDepot = p.kind === "depot";
        const isHub = p.kind === "hub";
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r + (isDepot ? 6 : 3), 0, Math.PI * 2);
        ctx.fillStyle = isDepot
          ? "rgba(61,255,168,0.12)"
          : isHub
            ? "rgba(240,180,41,0.12)"
            : "rgba(76,201,240,0.1)";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = isDepot
          ? "#3dffa8"
          : isHub
            ? "#f0b429"
            : "#4cc9f0";
        ctx.fill();

        if (isDepot) {
          ctx.fillStyle = "#070a0f";
          ctx.font = "600 8px DM Mono, monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("D", p.x, p.y + 0.5);
        }
      }

      // legend strip
      ctx.fillStyle = "rgba(255,255,255,0.04)";
      ctx.fillRect(12, h - 34, 210, 22);
      ctx.fillStyle = "#8b95a8";
      ctx.font = "500 9px DM Mono, monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("DEPOT  ·  HUB  ·  BINS  ·  ROUTE", 20, h - 23);

      if (!reduceMotion) raf = requestAnimationFrame(draw);
    };

    resize();
    const ro = new ResizeObserver(() => {
      resize();
      if (reduceMotion) draw(performance.now());
    });
    ro.observe(canvas);
    start = performance.now();
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="route-canvas"
      role="img"
      aria-label="Animated route graph showing depot, hubs, bins, and an optimized collection path"
    />
  );
}

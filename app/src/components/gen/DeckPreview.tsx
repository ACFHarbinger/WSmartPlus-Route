/**
 * Deck preview canvas (§H.7) — paginated in-app preview of the results deck,
 * rendered from the same slide builder the HTML/PDF exporters use, so what
 * you see is exactly what exports.
 *
 * Slides render inside a shadow root (the deck stylesheet styles `body`,
 * `.slide`, etc. and must not leak into the app) at true 1600×900 geometry,
 * scaled to fit the panel width. ←/→ paginate while the canvas is focused;
 * the notes strip below shows the active slide's speaker script.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, RefreshCw, StickyNote } from "lucide-react";
import { DECK_CSS, type HtmlSlide } from "../../gen/deck/htmlDeck";

const SLIDE_W = 1600;
const SLIDE_H = 900;

function ShadowSlide({ html }: { html: string }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const root = host.shadowRoot ?? host.attachShadow({ mode: "open" });
    root.innerHTML =
      `<style>${DECK_CSS}\n` +
      `:host { display: block; }\n` +
      `.pv-frame { width: ${SLIDE_W}px; height: ${SLIDE_H}px; display: flex; }\n` +
      `.pv-frame .slide { width: ${SLIDE_W}px; height: ${SLIDE_H}px; aspect-ratio: auto; font-size: 22px; }\n` +
      `</style><div class="pv-frame">${html}</div>`;
  }, [html]);
  return <div ref={hostRef} style={{ width: SLIDE_W, height: SLIDE_H }} />;
}

export function DeckPreview({
  slides,
  onRebuild,
  building,
}: {
  slides: HtmlSlide[];
  onRebuild: () => void;
  building: boolean;
}) {
  const [idx, setIdx] = useState(0);
  const [showNotes, setShowNotes] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [scale, setScale] = useState(0.35);

  // fit-to-width scaling (§H.7 zoom/fit)
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => setScale(Math.max(0.05, el.clientWidth / SLIDE_W));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const clamp = useCallback(
    (i: number) => Math.min(Math.max(i, 0), Math.max(slides.length - 1, 0)),
    [slides.length]
  );
  useEffect(() => {
    setIdx((i) => clamp(i));
  }, [slides, clamp]);

  const onKey = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowRight") setIdx((i) => clamp(i + 1));
      else if (e.key === "ArrowLeft") setIdx((i) => clamp(i - 1));
      else if (e.key === "Home") setIdx(0);
      else if (e.key === "End") setIdx(clamp(slides.length - 1));
    },
    [clamp, slides.length]
  );

  const slide = slides[idx];
  const notes = useMemo(() => slide?.notes ?? "", [slide]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-200">Deck Preview</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowNotes((v) => !v)}
            className={showNotes ? "btn-primary text-xs py-1 px-2 flex items-center gap-1" : "btn-ghost text-xs py-1 px-2 flex items-center gap-1"}
            title="Toggle speaker notes"
          >
            <StickyNote size={12} />
            Notes
          </button>
          <button
            onClick={onRebuild}
            disabled={building}
            className="btn-ghost text-xs py-1 px-2 flex items-center gap-1"
            title="Rebuild the preview from current settings"
          >
            <RefreshCw size={12} className={building ? "animate-spin" : ""} />
            Rebuild
          </button>
          <button
            onClick={() => setIdx((i) => clamp(i - 1))}
            disabled={idx === 0}
            className="btn-ghost text-xs py-1 px-2"
          >
            <ChevronLeft size={13} />
          </button>
          <span className="text-xs text-canvas-muted font-mono w-14 text-center">
            {slides.length ? idx + 1 : 0} / {slides.length}
          </span>
          <button
            onClick={() => setIdx((i) => clamp(i + 1))}
            disabled={idx >= slides.length - 1}
            className="btn-ghost text-xs py-1 px-2"
          >
            <ChevronRight size={13} />
          </button>
        </div>
      </div>
      <div
        ref={containerRef}
        tabIndex={0}
        onKeyDown={onKey}
        className="rounded-lg overflow-hidden border border-canvas-border outline-none focus:ring-1 focus:ring-accent-primary"
        style={{ height: SLIDE_H * scale }}
      >
        {slide ? (
          <div style={{ transform: `scale(${scale})`, transformOrigin: "top left", width: SLIDE_W, height: SLIDE_H }}>
            <ShadowSlide html={slide.html} />
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-xs text-canvas-muted">
            {building ? "Building preview…" : "No slides."}
          </div>
        )}
      </div>
      {showNotes && (
        <div className="bg-canvas-bg rounded-lg p-2 max-h-32 overflow-auto">
          <p className="text-[10px] uppercase tracking-widest text-canvas-muted mb-1">Speaker notes</p>
          <p className="text-xs text-gray-300 whitespace-pre-wrap">{notes || "—"}</p>
        </div>
      )}
      <p className="text-[10px] text-canvas-muted">
        Rendered by the same builder as the HTML/PDF exports — arrow keys paginate while the canvas is focused.
      </p>
    </div>
  );
}

"""
Generate the WSmart+ Route results PowerPoint presentation.

Builds a 16-slide deck under assets/windows/ following the agreed structure:

  1.  Cover (author + theme)
  2.  Index / agenda
  3.  The VRPP problem
  4.  Routing simulator philosophy (selection + constructor [+ acceptance] [+ improver])
  5.  Mandatory node selection strategies (LA / LM / SL)
  6.  Exact solvers (BPC + SWC-TCF)
  7.  Meta-heuristics & hyper-heuristics philosophy
  8.  Evolutionary algorithms (HGS)
  9.  Swarm intelligence (PSOMA + ACO-HH)
  10. Neighborhood search (SANS + PG-CLNS)
  11. CLS vs Fast-TSP route improvers
  12. Pareto front plot (log X preferred)
  13. Summary KPI plots (selection strategies)
  14. Overflow + kg/km heatmaps (policy configurations × scenarios)
  15. Per-scenario heatmaps (constructors × strategy·improver)
  16. Route improver bubble chart
  17. Conclusions, limitations & future work
  18. End / Q&A

Slide text lives in json/presentation_content.json; result figures are pulled
from the simulation analysis figures directory (default: the 30-day set).

Usage
-----
    uv run python logic/gen/gen_presentation.py
    uv run python logic/gen/gen_presentation.py \\
        --figures-dir public/figures/simulation/30d \\
        --out assets/windows/wsmart_route_results.pptx \\
        --author "Afonso Fernandes"
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt
from report_utils import load_json

# 16:9 slide geometry
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

ACCENT = RGBColor(0x2E, 0x74, 0xB5)
DARK = RGBColor(0x1F, 0x2D, 0x3D)
MUTED = RGBColor(0x5A, 0x6A, 0x7A)
LIGHT = RGBColor(0xF4, 0xF7, 0xFB)


def _fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    return box, tf


class DeckBuilder:
    def __init__(self, content: dict, figures_dir: Path, author: str | None):
        self.content = content
        self.figures_dir = figures_dir
        self.author = author or content["author"]
        self.prs = Presentation()
        self.prs.slide_width = SLIDE_W
        self.prs.slide_height = SLIDE_H
        self.blank = self.prs.slide_layouts[6]

    # ── Building blocks ─────────────────────────────────────────────────────────

    def _new_slide(self):
        return self.prs.slides.add_slide(self.blank)

    def _title_bar(self, slide, title: str) -> None:
        bar = slide.shapes.add_shape(1, 0, 0, SLIDE_W, Inches(1.0))  # 1 = rectangle
        _fill(bar, DARK)
        tf = bar.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.5)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    def _bullets(self, slide, bullets: list[str], top=None, height=None, size=16) -> None:
        top = top if top is not None else Inches(1.3)
        _, tf = _textbox(slide, Inches(0.6), top, SLIDE_W - Inches(1.2), height or (SLIDE_H - top - Inches(0.4)))
        for i, text in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = f"•  {text}"
            p.font.size = Pt(size)
            p.font.color.rgb = DARK
            p.space_after = Pt(10)

    def _picture_fit(self, slide, path: Path, left, top, max_w, max_h) -> None:
        from PIL import Image

        with Image.open(path) as im:
            w_px, h_px = im.size
        ratio = min(max_w / w_px, max_h / h_px)
        w, h = int(w_px * ratio), int(h_px * ratio)
        slide.shapes.add_picture(
            str(path), left + Emu(int((max_w - w) / 2)), top + Emu(int((max_h - h) / 2)), w, h
        )

    def _figure_slide(self, spec: dict) -> None:
        slide = self._new_slide()
        self._title_bar(slide, spec["title"])
        figures = spec.get("figures") or [spec.get("figure")]
        resolved = []
        for name in figures:
            p = self.figures_dir / name
            if not p.exists() and spec.get("fallback_figure"):
                p = self.figures_dir / spec["fallback_figure"]
            if p.exists():
                resolved.append(p)
            else:
                print(f"  [WARN] Figure not found: {p}")
        note_h = Inches(0.45)
        area_top = Inches(1.15)
        area_h = SLIDE_H - area_top - note_h - Inches(0.15)
        if resolved:
            w_each = (SLIDE_W - Inches(0.6)) / len(resolved)
            for i, p in enumerate(resolved):
                self._picture_fit(slide, p, Inches(0.3) + w_each * i, area_top, w_each - Inches(0.2), area_h)
        if spec.get("note"):
            _, tf = _textbox(slide, Inches(0.6), SLIDE_H - note_h - Inches(0.1), SLIDE_W - Inches(1.2), note_h)
            p = tf.paragraphs[0]
            p.text = spec["note"]
            p.font.size = Pt(12)
            p.font.italic = True
            p.font.color.rgb = MUTED

    # ── Slides ──────────────────────────────────────────────────────────────────

    def cover(self) -> None:
        slide = self._new_slide()
        bg = slide.shapes.add_shape(1, 0, 0, SLIDE_W, SLIDE_H)
        _fill(bg, DARK)
        band = slide.shapes.add_shape(1, 0, Inches(4.6), SLIDE_W, Inches(0.08))
        _fill(band, ACCENT)
        _, tf = _textbox(slide, Inches(0.9), Inches(2.2), SLIDE_W - Inches(1.8), Inches(2.2))
        p = tf.paragraphs[0]
        p.text = self.content["title"]
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p2 = tf.add_paragraph()
        p2.text = self.content["subtitle"]
        p2.font.size = Pt(20)
        p2.font.color.rgb = RGBColor(0xC9, 0xD6, 0xE4)
        _, tf2 = _textbox(slide, Inches(0.9), Inches(5.0), SLIDE_W - Inches(1.8), Inches(1.2))
        p3 = tf2.paragraphs[0]
        p3.text = self.author
        p3.font.size = Pt(20)
        p3.font.bold = True
        p3.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p4 = tf2.add_paragraph()
        p4.text = dt.date.today().strftime("%d %B %Y")
        p4.font.size = Pt(14)
        p4.font.color.rgb = RGBColor(0xC9, 0xD6, 0xE4)

    def agenda(self) -> None:
        slide = self._new_slide()
        self._title_bar(slide, "Agenda")
        items = self.content["agenda"]
        half = (len(items) + 1) // 2
        for col, chunk in enumerate([items[:half], items[half:]]):
            _, tf = _textbox(
                slide,
                Inches(0.7) + col * (SLIDE_W - Inches(1.4)) / 2,
                Inches(1.4),
                (SLIDE_W - Inches(1.6)) / 2,
                SLIDE_H - Inches(1.8),
            )
            for i, item in enumerate(chunk):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = f"{col * half + i + 1}.  {item}"
                p.font.size = Pt(16)
                p.font.color.rgb = DARK
                p.space_after = Pt(12)

    def content_slide(self, key: str) -> None:
        spec = self.content["slides"][key]
        slide = self._new_slide()
        self._title_bar(slide, spec["title"])
        self._bullets(slide, spec["bullets"])

    def qa(self) -> None:
        spec = self.content["slides"]["qa"]
        slide = self._new_slide()
        bg = slide.shapes.add_shape(1, 0, 0, SLIDE_W, SLIDE_H)
        _fill(bg, DARK)
        _, tf = _textbox(slide, Inches(0.9), Inches(2.6), SLIDE_W - Inches(1.8), Inches(2.4))
        p = tf.paragraphs[0]
        p.text = spec["title"]
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = PP_ALIGN.CENTER
        for line in spec["bullets"]:
            pp = tf.add_paragraph()
            pp.text = line
            pp.font.size = Pt(16)
            pp.font.color.rgb = RGBColor(0xC9, 0xD6, 0xE4)
            pp.alignment = PP_ALIGN.CENTER

    def build(self) -> Presentation:
        self.cover()  # 1
        self.agenda()  # 2
        self.content_slide("vrpp")  # 3
        self.content_slide("simulator")  # 4
        self.content_slide("strategies")  # 5
        self.content_slide("exact")  # 6
        self.content_slide("metaheuristics")  # 7
        self.content_slide("evolutionary")  # 8a
        self.content_slide("swarm")  # 8b
        self.content_slide("neighborhood")  # 8c
        self.content_slide("improvers")  # 9
        self._figure_slide(self.content["figure_slides"]["pareto"])  # 10
        self._figure_slide(self.content["figure_slides"]["kpi"])  # 11
        self._figure_slide(self.content["figure_slides"]["heatmaps"])  # 12
        self._figure_slide(self.content["figure_slides"]["scenario_heatmaps"])  # 13
        self._figure_slide(self.content["figure_slides"]["improver_bubble"])  # 14
        self.content_slide("conclusion")  # 15
        self.qa()  # 16
        return self.prs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures-dir", default="public/figures/simulation/30d",
                   help="Directory with the simulation analysis figures to embed")
    p.add_argument("--out", default="assets/windows/wsmart_route_results.pptx", help="Destination .pptx path")
    p.add_argument("--author", default=None, help="Author name shown on the cover slide")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    content = load_json("presentation_content.json")
    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_dir():
        raise SystemExit(f"Figures dir not found: {figures_dir} — run gen_simulation_analysis.py first")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs = DeckBuilder(content, figures_dir, args.author).build()
    prs.save(str(out))
    print(f"Written: {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()

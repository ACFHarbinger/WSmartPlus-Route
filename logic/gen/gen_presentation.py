"""
Generate the WSmart+ Route results PowerPoint presentation.

Builds an 18-slide deck under assets/windows/ following the agreed structure:

  1.  Cover (author, optional co-authors and research groups)
  2.  Index / agenda
  3.  The VRPP problem                          (equation focus)
  4.  Routing simulator philosophy              (pipeline diagram focus)
  5.  Mandatory node selection strategies       (equation focus)
  6.  Exact solvers (BPC + SWC-TCF)             (equation focus)
  7.  Meta-heuristics & hyper-heuristics        (equation focus)
  8.  Evolutionary algorithms (HGS)             (equation focus)
  9.  Swarm intelligence (PSOMA + ACO-HH)       (equation focus)
  10. Neighborhood search (SANS + PG-CLNS)      (equation focus)
  11. CLS vs Fast-TSP route improvers           (equation focus)
  12. Pareto front plot (log X, per-scenario coloured fronts)
  13. Summary KPI plots (stacked vertically, full width)
  14. Per-scenario heatmaps (30 days)
  15. Overflow + kg/km policy × scenario heatmaps (90 days)
  16. Route improver bubble chart
  17. Conclusions, limitations & future work    (radar figure focus)
  18. End / Q&A

Slide text and equations (matplotlib mathtext) live in
json/presentation_content.json; result figures are pulled from the simulation
analysis figures directories (30-day set by default; individual figure slides
may override via "figures_dir").

Usage
-----
    uv run python logic/gen/gen_presentation.py
    uv run python logic/gen/gen_presentation.py \\
        --figures-dir public/figures/simulation/30d \\
        --out assets/windows/wsmart_route_results.pptx \\
        --author "Afonso Fernandes" \\
        --coauthors "Jane Doe;John Smith" \\
        --groups "ISR Coimbra;INESC-ID Lisboa"
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt
from report_utils import load_json

# 16:9 slide geometry
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

ACCENT = RGBColor(0x2E, 0x74, 0xB5)
DARK = RGBColor(0x1F, 0x2D, 0x3D)
MUTED = RGBColor(0x5A, 0x6A, 0x7A)
LIGHT_TXT = RGBColor(0xC9, 0xD6, 0xE4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

PIPELINE_STAGES = [
    ("Mandatory bin\nselection strategy", "LA · LM · SL\nwhich bins today?"),
    ("Route\nconstructor", "exact · meta-h. · hyper-h.\nbuild the routes"),
    ("Acceptance criterion\n(optional)", "constructor-dependent\ne.g. BMC, OI"),
    ("Route improver\n(optional)", "CLS · FTSP\npost-optimisation"),
]


def _fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    return box, tf


def render_equation(lines: list[str], out_path: Path, color: str = "#1F2D3D") -> Path:
    """Render mathtext equation lines to a transparent PNG."""
    fig = plt.figure(figsize=(11, 1.1 * len(lines) + 0.3))
    fig.patch.set_alpha(0)
    for i, line in enumerate(lines):
        fig.text(
            0.5,
            1 - (i + 0.5) / len(lines),
            line,
            ha="center",
            va="center",
            fontsize=26,
            color=color,
        )
    fig.savefig(out_path, dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


class DeckBuilder:
    def __init__(self, content: dict, figures_dir: Path, author: str | None,
                 coauthors: list[str] | None = None, groups: list[str] | None = None):
        self.content = content
        self.figures_dir = figures_dir
        self.author = author or content["author"]
        self.coauthors = coauthors if coauthors is not None else content.get("coauthors", [])
        self.groups = groups if groups is not None else content.get("research_groups", [])
        self.prs = Presentation()
        self.prs.slide_width = SLIDE_W
        self.prs.slide_height = SLIDE_H
        self.blank = self.prs.slide_layouts[6]
        self._tmp = Path(tempfile.mkdtemp(prefix="wsr_pptx_"))
        self._eq_count = 0

    # ── Building blocks ─────────────────────────────────────────────────────────

    def _new_slide(self):
        return self.prs.slides.add_slide(self.blank)

    def _title_bar(self, slide, title: str) -> None:
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(1.0))
        _fill(bar, DARK)
        tf = bar.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.5)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = WHITE

    def _bullets(self, slide, bullets: list[str], left=None, top=None, width=None, height=None, size=16) -> None:
        left = left if left is not None else Inches(0.6)
        top = top if top is not None else Inches(1.3)
        width = width if width is not None else SLIDE_W - Inches(1.2)
        _, tf = _textbox(slide, left, top, width, height or (SLIDE_H - top - Inches(0.4)))
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

    def _note(self, slide, text: str) -> None:
        _, tf = _textbox(slide, Inches(0.6), SLIDE_H - Inches(0.55), SLIDE_W - Inches(1.2), Inches(0.45))
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(12)
        p.font.italic = True
        p.font.color.rgb = MUTED

    def _equation_focus(self, slide, lines: list[str]):
        """Render mathtext lines and place them as the slide's visual focus."""
        self._eq_count += 1
        path = render_equation(lines, self._tmp / f"eq_{self._eq_count}.png")
        area_h = Inches(0.75 + 0.75 * len(lines))
        band = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.25), SLIDE_W - Inches(1.2), area_h
        )
        _fill(band, RGBColor(0xF0, 0xF4, 0xFA))
        band.shadow.inherit = False
        self._picture_fit(slide, path, Inches(0.9), Inches(1.35), SLIDE_W - Inches(1.8), area_h - Inches(0.2))
        return Inches(1.25) + area_h + Inches(0.25)

    def _pipeline_diagram(self, slide) -> None:
        """Draw the policy pipeline as chevron stages with captions."""
        n = len(PIPELINE_STAGES)
        gap = Inches(0.25)
        left0 = Inches(0.6)
        total_w = SLIDE_W - Inches(1.2)
        stage_w = int((total_w - gap * (n - 1)) / n)
        top = Inches(1.7)
        h = Inches(1.5)
        for i, (name, caption) in enumerate(PIPELINE_STAGES):
            left = left0 + i * (stage_w + gap)
            shape = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, left, top, stage_w, h)
            optional = "optional" in name
            _fill(shape, ACCENT if not optional else MUTED)
            shape.shadow.inherit = False
            tf = shape.text_frame
            tf.word_wrap = True
            for li, line in enumerate(name.split("\n")):
                p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
                p.text = line
                p.font.size = Pt(15)
                p.font.bold = True
                p.font.color.rgb = WHITE
                p.alignment = PP_ALIGN.CENTER
            _, ctf = _textbox(slide, left, top + h + Inches(0.1), stage_w, Inches(0.9))
            for li, line in enumerate(caption.split("\n")):
                p = ctf.paragraphs[0] if li == 0 else ctf.add_paragraph()
                p.text = line
                p.font.size = Pt(12)
                p.font.color.rgb = MUTED
                p.alignment = PP_ALIGN.CENTER
        sim = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.1), top + h + Inches(1.2), SLIDE_W - Inches(6.2), Inches(0.8)
        )
        _fill(sim, DARK)
        sim.shadow.inherit = False
        p = sim.text_frame.paragraphs[0]
        p.text = "Multi-day simulator: region × graph size × distribution"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    def _figure_slide(self, spec: dict) -> None:
        slide = self._new_slide()
        self._title_bar(slide, spec["title"])
        fig_dir = Path(spec["figures_dir"]) if spec.get("figures_dir") else self.figures_dir
        figures = spec.get("figures") or [spec.get("figure")]
        resolved = []
        for name in figures:
            p = fig_dir / name
            if not p.exists() and spec.get("fallback_figure"):
                p = fig_dir / spec["fallback_figure"]
            if p.exists():
                resolved.append(p)
            else:
                print(f"  [WARN] Figure not found: {p}")
        area_top = Inches(1.15)
        area_h = SLIDE_H - area_top - Inches(0.75)
        if resolved:
            if spec.get("layout") == "vertical":
                h_each = int(area_h / len(resolved))
                for i, p in enumerate(resolved):
                    self._picture_fit(
                        slide, p, Inches(0.3), area_top + h_each * i, SLIDE_W - Inches(0.6), h_each - Inches(0.1)
                    )
            else:
                w_each = int((SLIDE_W - Inches(0.6)) / len(resolved))
                for i, p in enumerate(resolved):
                    self._picture_fit(slide, p, Inches(0.3) + w_each * i, area_top, w_each - Inches(0.2), area_h)
        if spec.get("note"):
            self._note(slide, spec["note"])

    # ── Slides ──────────────────────────────────────────────────────────────────

    def cover(self) -> None:
        slide = self._new_slide()
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
        _fill(bg, DARK)
        band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(4.5), SLIDE_W, Inches(0.08))
        _fill(band, ACCENT)
        _, tf = _textbox(slide, Inches(0.9), Inches(2.1), SLIDE_W - Inches(1.8), Inches(2.2))
        p = tf.paragraphs[0]
        p.text = self.content["title"]
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p2 = tf.add_paragraph()
        p2.text = self.content["subtitle"]
        p2.font.size = Pt(20)
        p2.font.color.rgb = LIGHT_TXT
        _, tf2 = _textbox(slide, Inches(0.9), Inches(4.85), SLIDE_W - Inches(1.8), Inches(2.2))
        p3 = tf2.paragraphs[0]
        p3.text = self.author
        p3.font.size = Pt(22)
        p3.font.bold = True
        p3.font.color.rgb = WHITE
        if self.coauthors:
            p4 = tf2.add_paragraph()
            p4.text = "with " + ", ".join(self.coauthors)
            p4.font.size = Pt(15)
            p4.font.color.rgb = LIGHT_TXT
        if self.groups:
            p5 = tf2.add_paragraph()
            p5.text = " · ".join(self.groups)
            p5.font.size = Pt(13)
            p5.font.italic = True
            p5.font.color.rgb = RGBColor(0x8A, 0x9B, 0xB0)

    def agenda(self) -> None:
        slide = self._new_slide()
        self._title_bar(slide, "Agenda")
        items = self.content["agenda"]
        half = (len(items) + 1) // 2
        for col, chunk in enumerate([items[:half], items[half:]]):
            _, tf = _textbox(
                slide,
                Inches(0.7) + col * int((SLIDE_W - Inches(1.4)) / 2),
                Inches(1.4),
                int((SLIDE_W - Inches(1.6)) / 2),
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
        if spec.get("equation"):
            bullets_top = self._equation_focus(slide, spec["equation"])
            self._bullets(slide, spec["bullets"], top=bullets_top, size=14)
        elif spec.get("diagram") == "pipeline":
            self._pipeline_diagram(slide)
            self._bullets(slide, spec["bullets"], top=Inches(5.6), size=14)
        elif spec.get("figure"):
            fig_path = self.figures_dir / spec["figure"]
            if fig_path.exists():
                self._picture_fit(
                    slide, fig_path, int(SLIDE_W / 2), Inches(1.2), int(SLIDE_W / 2) - Inches(0.4),
                    SLIDE_H - Inches(1.7)
                )
                self._bullets(slide, spec["bullets"], left=Inches(0.5), top=Inches(1.4),
                              width=int(SLIDE_W / 2) - Inches(0.7), size=13)
            else:
                print(f"  [WARN] Figure not found: {fig_path}")
                self._bullets(slide, spec["bullets"])
        else:
            self._bullets(slide, spec["bullets"])

    def qa(self) -> None:
        spec = self.content["slides"]["qa"]
        slide = self._new_slide()
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
        _fill(bg, DARK)
        _, tf = _textbox(slide, Inches(0.9), Inches(2.8), SLIDE_W - Inches(1.8), Inches(2.0))
        p = tf.paragraphs[0]
        p.text = spec["title"]
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        for line in spec["bullets"]:
            pp = tf.add_paragraph()
            pp.text = line
            pp.font.size = Pt(16)
            pp.font.color.rgb = LIGHT_TXT
            pp.alignment = PP_ALIGN.CENTER

    def build(self) -> Presentation:
        self.cover()  # 1
        self.agenda()  # 2
        self.content_slide("vrpp")  # 3
        self.content_slide("simulator")  # 4
        self.content_slide("strategies")  # 5
        self.content_slide("exact")  # 6
        self.content_slide("metaheuristics")  # 7
        self.content_slide("evolutionary")  # 8
        self.content_slide("swarm")  # 9
        self.content_slide("neighborhood")  # 10
        self.content_slide("improvers")  # 11
        self._figure_slide(self.content["figure_slides"]["pareto"])  # 12
        self._figure_slide(self.content["figure_slides"]["kpi"])  # 13
        self._figure_slide(self.content["figure_slides"]["scenario_heatmaps"])  # 14 (30d, first)
        self._figure_slide(self.content["figure_slides"]["heatmaps"])  # 15 (90d)
        self._figure_slide(self.content["figure_slides"]["improver_bubble"])  # 16
        self.content_slide("conclusion")  # 17
        self.qa()  # 18
        return self.prs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures-dir", default="public/figures/simulation/30d",
                   help="Directory with the simulation analysis figures to embed")
    p.add_argument("--out", default="assets/windows/wsmart_route_results.pptx", help="Destination .pptx path")
    p.add_argument("--author", default=None, help="Presenting author shown on the cover slide")
    p.add_argument("--coauthors", default=None,
                   help="Optional semicolon-separated co-authors (shown with less emphasis)")
    p.add_argument("--groups", default=None,
                   help="Optional semicolon-separated research groups of the authors")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    content = load_json("presentation_content.json")
    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_dir():
        raise SystemExit(f"Figures dir not found: {figures_dir} — run gen_simulation_analysis.py first")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    coauthors = [c.strip() for c in args.coauthors.split(";")] if args.coauthors else None
    groups = [g.strip() for g in args.groups.split(";")] if args.groups else None
    prs = DeckBuilder(content, figures_dir, args.author, coauthors, groups).build()
    prs.save(str(out))
    print(f"Written: {out} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()

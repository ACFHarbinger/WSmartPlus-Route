"""
Generate the WSmart+ Route results PowerPoint presentation.

Builds a 19-slide deck under assets/windows/ following the agreed structure:

  1.  Cover (author, optional co-authors and research groups)
  2.  Index / agenda
  3.  The VRPP problem                          (equation + caption)
  4.  Routing simulator philosophy              (pipeline diagram + caption)
  5.  Mandatory node selection strategies       (equation + caption)
  6.  Exact solvers (BPC + SWC-TCF)             (equation + caption)
  7.  Meta-heuristics & hyper-heuristics        (equation + caption)
  8.  Evolutionary algorithms (HGS)             (equation + caption)
  9.  Swarm intelligence (PSOMA + ACO-HH)       (equation + caption)
  10. Neighborhood search (SANS + PG-CLNS)      (equation + caption)
  11. CLS vs Fast-TSP route improvers           (equation + caption)
  12. Pareto front plot (log X, per-scenario coloured fronts, figure caption)
  13. Summary KPI plots (stacked vertically, full width, figure caption)
  14. Per-scenario heatmaps (30 days, figure caption)
  15. Overflow + kg/km policy × scenario heatmaps (90 days, figure caption)
  16. Route improver bubble chart (figure caption)
  17. Full results table (user-selected horizon, or all horizons — table caption)
  18. Conclusions, limitations & future work    (radar figure + caption)
  19. End / Q&A

Slide text, equations (matplotlib mathtext) and captions live in
json/presentation_content.json; result figures are pulled from the simulation
analysis figures directories (30-day set by default; individual figure slides
may override via "figures_dir"). The full results table (slide 17) is built
directly from the horizon CSV(s) referenced in
json/simulation_analysis_config.json.

A per-slide speaker script can also be generated as a .docx (see
gen_speaker_script / --speaker-script), rendered via docxtpl from a template
under logic/gen/templates/.

Usage
-----
    uv run python logic/gen/gen_presentation.py
    uv run python logic/gen/gen_presentation.py \\
        --figures-dir public/figures/simulation/30d \\
        --out assets/windows/wsmart_route_results.pptx \\
        --author "Afonso Fernandes" \\
        --coauthors "Jane Doe;John Smith" \\
        --groups "ISR Coimbra;INESC-ID Lisboa" \\
        --results-table 30d \\
        --speaker-script
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from gen_simulation_analysis import (
    _group_spans,
    aggregate,
    build_context,
    build_full_results_matrix,
    filter_data,
    load_horizon_csv,
)
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


def render_hier_table_image(
    row_keys: list[tuple],
    col_keys: list[tuple],
    cells: dict[tuple, str],
    row_labels: list[str],
    out_path: Path,
) -> Path:
    """
    Render a hierarchical (merged-header) results table as a PNG, in the style
    of a LaTeX multi-row/multi-column table: row groups (graph size > region >
    distribution) and column groups (strategy > constructor > improver, plus
    horizon if present) are drawn as merged header spans.
    """
    n_row_levels = len(row_labels)
    n_col_levels = len(col_keys[0])
    n_rows = len(row_keys)
    n_cols = len(col_keys)

    cell_w, cell_h, label_w, header_h = 1.05, 0.5, 1.3, 0.5
    fontsize = max(5.5, min(9, 260 / max(n_cols, 1)))

    x0 = label_w * n_row_levels
    y0 = header_h * n_col_levels
    fig_w = min(x0 + cell_w * n_cols, 42)
    fig_h = min(y0 + cell_h * n_rows, 32)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, x0 + cell_w * n_cols)
    ax.set_ylim(0, y0 + cell_h * n_rows)
    ax.invert_yaxis()
    ax.axis("off")

    header_colors = ["#1F2D3D", "#2E74B5", "#5A6A7A", "#8A9BB0"]
    for lvl in range(n_col_levels):
        y_top = lvl * header_h
        for start, end, label in _group_spans(col_keys, lvl):
            xs, xe = x0 + start * cell_w, x0 + end * cell_w
            ax.add_patch(mpatches.Rectangle(
                (xs, y_top), xe - xs, header_h, facecolor=header_colors[lvl % len(header_colors)],
                edgecolor="white", linewidth=0.8,
            ))
            ax.text((xs + xe) / 2, y_top + header_h / 2, str(label), ha="center", va="center",
                    fontsize=fontsize, color="white", fontweight="bold")

    for lvl in range(n_row_levels):
        x_left = lvl * label_w
        for start, end, label in _group_spans(row_keys, lvl):
            ys, ye = y0 + start * cell_h, y0 + end * cell_h
            ax.add_patch(mpatches.Rectangle(
                (x_left, ys), label_w, ye - ys, facecolor="#F0F4FA", edgecolor="#5A6A7A", linewidth=0.6,
            ))
            ax.text(x_left + label_w / 2, (ys + ye) / 2, str(label), ha="center", va="center",
                    fontsize=fontsize, color="#1F2D3D", fontweight="bold")

    for ri, rk in enumerate(row_keys):
        for ci, ck in enumerate(col_keys):
            xs, ys = x0 + ci * cell_w, y0 + ri * cell_h
            text = cells.get((rk, ck), "—").replace("<br>", "\n")
            ax.add_patch(mpatches.Rectangle(
                (xs, ys), cell_w, cell_h, fill=False, edgecolor="#CCCCCC", linewidth=0.4,
            ))
            ax.text(xs + cell_w / 2, ys + cell_h / 2, text, ha="center", va="center", fontsize=fontsize * 0.85)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def gen_speaker_script(deck_title: str, author: str, slides: list[dict], out_path: Path) -> Path:
    """Render a per-slide speaker script .docx from the docxtpl template."""
    from docxtpl import DocxTemplate

    tpl = DocxTemplate(str(TEMPLATES_DIR / "speaker_script.docx"))
    tpl.render({"deck_title": deck_title, "author": author, "slides": slides})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tpl.save(str(out_path))
    return out_path


class DeckBuilder:
    def __init__(self, content: dict, figures_dir: Path, author: str | None,
                 coauthors: list[str] | None = None, groups: list[str] | None = None,
                 results_table: str = "30d"):
        self.content = content
        self.figures_dir = figures_dir
        self.author = author or content["author"]
        self.coauthors = coauthors if coauthors is not None else content.get("coauthors", [])
        self.groups = groups if groups is not None else content.get("research_groups", [])
        self.results_table = results_table
        self._results_config: dict = {}
        self.prs = Presentation()
        self.prs.slide_width = SLIDE_W
        self.prs.slide_height = SLIDE_H
        self.blank = self.prs.slide_layouts[6]
        self._tmp = Path(tempfile.mkdtemp(prefix="wsr_pptx_"))
        self._eq_count = 0
        self._fig_count = 0
        self._tab_count = 0
        self.slide_scripts: list[dict] = []

    def _record_script(self, title: str, paragraphs: list[str]) -> None:
        """Append a speaker-script entry for the slide just built."""
        self.slide_scripts.append({
            "number": len(self.slide_scripts) + 1,
            "title": title,
            "script": "\n\n".join(p for p in paragraphs if p),
        })

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

    def _caption_box(self, slide, label: str, text: str, top=None) -> None:
        """A numbered caption ('**Figure N:** ...' / '**Equation N:** ...' / '**Table N:** ...')."""
        top = top if top is not None else SLIDE_H - Inches(0.55)
        _, tf = _textbox(slide, Inches(0.6), top, SLIDE_W - Inches(1.2), Inches(0.6))
        p = tf.paragraphs[0]
        r1 = p.add_run()
        r1.text = f"{label}: "
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = DARK
        r2 = p.add_run()
        r2.text = text
        r2.font.italic = True
        r2.font.size = Pt(11)
        r2.font.color.rgb = MUTED

    def _figure_caption(self, slide, text: str, top=None) -> None:
        self._fig_count += 1
        self._caption_box(slide, f"Figure {self._fig_count}", text, top=top)

    def _equation_caption(self, slide, text: str, top=None) -> None:
        self._caption_box(slide, f"Equation {self._eq_count}", text, top=top)

    def _table_caption(self, slide, text: str, top=None) -> None:
        self._tab_count += 1
        self._caption_box(slide, f"Table {self._tab_count}", text, top=top)

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
        caption = spec.get("caption") or spec.get("note")
        if caption:
            self._figure_caption(slide, caption)
        self._record_script(spec["title"], [caption or ""])

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
        self._record_script(
            self.content["title"],
            [f"Presented by {self.author}." + (f" With {', '.join(self.coauthors)}." if self.coauthors else "")],
        )

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
        self._record_script("Agenda", ["Today's agenda: " + "; ".join(items) + "."])

    def content_slide(self, key: str) -> None:
        spec = self.content["slides"][key]
        slide = self._new_slide()
        self._title_bar(slide, spec["title"])
        if spec.get("equation"):
            bullets_top = self._equation_focus(slide, spec["equation"])
            if spec.get("caption"):
                self._equation_caption(slide, spec["caption"], top=bullets_top)
                bullets_top += Inches(0.4)
            self._bullets(slide, spec["bullets"], top=bullets_top, size=14)
        elif spec.get("diagram") == "pipeline":
            self._pipeline_diagram(slide)
            bullets_top = Inches(5.6)
            if spec.get("caption"):
                self._figure_caption(slide, spec["caption"], top=bullets_top)
                bullets_top += Inches(0.4)
            self._bullets(slide, spec["bullets"], top=bullets_top, size=14)
        elif spec.get("figure"):
            fig_path = self.figures_dir / spec["figure"]
            if fig_path.exists():
                self._picture_fit(
                    slide, fig_path, int(SLIDE_W / 2), Inches(1.2), int(SLIDE_W / 2) - Inches(0.4),
                    SLIDE_H - Inches(1.7)
                )
                if spec.get("caption"):
                    self._figure_caption(slide, spec["caption"])
                self._bullets(slide, spec["bullets"], left=Inches(0.5), top=Inches(1.4),
                              width=int(SLIDE_W / 2) - Inches(0.7), size=13)
            else:
                print(f"  [WARN] Figure not found: {fig_path}")
                self._bullets(slide, spec["bullets"])
        else:
            self._bullets(slide, spec["bullets"])
        self._record_script(spec["title"], [spec.get("caption", "")] + list(spec["bullets"]))

    def _load_horizon(self, spec: dict) -> tuple | None:
        csv_path = Path(spec["csv"])
        if not csv_path.exists():
            return None
        df = filter_data(load_horizon_csv(csv_path), self._results_config)
        if df.empty:
            return None
        theme = {"name": "light", "fg": "#1F2D3D"}
        ctx = build_context(df, self._results_config, theme, spec["days"])
        return df, aggregate(df), ctx

    def _results_table_slide(self) -> None:
        """Full results table slide, for the horizon(s) chosen via --results-table."""
        if self.results_table == "none":
            return
        self._results_config = load_json("simulation_analysis_config.json")
        horizon_specs = sorted(self._results_config["horizons"], key=lambda h: h["days"])

        if self.results_table == "all":
            row_keys, col_keys, cells = [], [], {}
            used_days = []
            for spec in horizon_specs:
                loaded = self._load_horizon(spec)
                if loaded is None:
                    continue
                _, dfm, ctx = loaded
                rk, ck, cs = build_full_results_matrix(dfm, ctx, horizon_label=f"{spec['days']}d")
                row_keys += [r for r in rk if r not in row_keys]
                col_keys += ck
                cells.update(cs)
                used_days.append(spec["days"])
            row_keys.sort()
            if not row_keys:
                print("  [WARN] No data available for the full results table — skipping slide")
                return
            title = f"Results — Full Table (All Horizons: {', '.join(f'{d}d' for d in used_days)})"
            multi = True
        else:
            days = int(self.results_table.rstrip("d"))
            spec = next((s for s in horizon_specs if s["days"] == days), None)
            loaded = self._load_horizon(spec) if spec else None
            if loaded is None:
                print(f"  [WARN] No data for the {self.results_table} horizon — skipping full results table slide")
                return
            _, dfm, ctx = loaded
            row_keys, col_keys, cells = build_full_results_matrix(dfm, ctx)
            title = f"Results — Full Table ({days}-Day Horizon)"
            multi = False

        img_path = render_hier_table_image(
            row_keys, col_keys, cells, ["N", "Region", "Dist"], self._tmp / "full_results_table.png"
        )
        slide = self._new_slide()
        self._title_bar(slide, title)
        self._picture_fit(slide, img_path, Inches(0.3), Inches(1.15), SLIDE_W - Inches(0.6), SLIDE_H - Inches(2.1))
        col_desc = "horizon × mandatory selection strategy × route constructor × route improver" if multi \
            else "mandatory selection strategy × route constructor × route improver"
        self._table_caption(
            slide,
            f"Full results table — rows: graph size × region × data distribution; columns: {col_desc}. "
            "Each cell reports mean±std overflows and mean±std kg/km.",
        )
        self._record_script(title, [f"Full results table, columns grouped by {col_desc}."])

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
        self._record_script(spec["title"], list(spec["bullets"]))

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
        self._results_table_slide()  # 17 (full results table, if requested)
        self.content_slide("conclusion")  # 18
        self.qa()  # 19
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
    p.add_argument("--results-table", default="30d", choices=["30d", "90d", "all", "none"],
                   help="Full results table slide: a single horizon, 'all' horizons "
                        "(horizon added to the column hierarchy), or 'none' to omit the slide")
    p.add_argument("--speaker-script", action="store_true",
                   help="Also generate a per-slide speaker script .docx alongside the .pptx")
    p.add_argument("--speaker-script-out", default=None,
                   help="Destination .docx path (default: same name as --out, .docx extension)")
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
    builder = DeckBuilder(content, figures_dir, args.author, coauthors, groups, args.results_table)
    prs = builder.build()
    prs.save(str(out))
    print(f"Written: {out} ({len(prs.slides._sldIdLst)} slides)")

    if args.speaker_script:
        script_out = Path(args.speaker_script_out) if args.speaker_script_out else out.with_suffix(".docx")
        gen_speaker_script(content["title"], builder.author, builder.slide_scripts, script_out)
        print(f"Written: {script_out} ({len(builder.slide_scripts)} slide scripts)")


if __name__ == "__main__":
    main()

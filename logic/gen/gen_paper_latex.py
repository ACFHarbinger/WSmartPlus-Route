"""
Generate the LaTeX tables and figures for the MPVRPP paper.

Sibling of gen_simulation_analysis.py (markdown reports) and gen_presentation.py
(PPTX deck): same summary-CSV schema, same theme/style machinery, same Jinja
conventions -- but emitting ``.tex`` fragments that ``paper.tex`` \\input's, plus
the figures those fragments reference, written straight into the paper's own
``Images/Results/`` tree at the sizes the document uses.

The point is that no number in the paper is typed by hand. Every table cell and
every figure here is derived from
``public/global/simulation/simulation_summary.csv`` (30-day horizon) and
``simulation_summary_90d.csv`` (90-day), so the paper cannot silently drift from
the data the way a hand-maintained table does.

Two properties of that data are enforced centrally, in this module, so that the
paper, the markdown reports and the website cannot disagree about them:

**Degenerate runs are excluded from aggregates.** The simulator gives every
constructor the identical demand realisation within a scenario cell (same waste,
bin by bin, day by day -- see appendix_notes.md, "Why the comparison across
algorithms is fair"), so the total tonnage available to collect is fixed within
a cell and every constructor should collect a comparable amount of it. Runs that
collect drastically less did not lose on policy quality; they failed to complete.
``find_degenerate_runs`` flags rows whose collected tonnage falls more than
``SHORTFALL_THRESHOLD`` below their cell median. The threshold is not arbitrary:
across the 576 rows in a cell of four or more, the shortfall distribution has
median 0.0 and 99th percentile 0.057, then jumps to a handful of rows above 0.20.
The gap is the signal. Excluded rows are reported in their own table rather than
dropped silently -- they are a real and citable scale limit of the
two-commodity-flow MIP.

``drop_affected_cells`` then removes the *entire* scenario cell each degenerate
run belonged to, for every constructor. Dropping only the offending rows would
reintroduce the same selection bias this module refuses to accept in the 90-day
data: all three degenerate runs are SWC-TCF's, so removing just them would
average SWC-TCF over the scenarios where it did not fail while its rivals are
averaged over those too. Losing the cell for everyone costs rows but keeps every
constructor's mean over an identical set of scenarios, which is the only thing
that makes the marginal means comparable.

Note that the ``days`` column counts *collection* days, not elapsed days, so a
low value there is not evidence of truncation on its own: several of the best
policies collect on 15 of 30 days precisely because they bundle well. Tonnage is
the honest signal.

**The 90-day sample is conditioned on 30-day performance.** Only policies on the
30-day Pareto front were re-run at 90 days, so each constructor's 90-day rows are
the scenarios where it already did well. Cross-constructor 90-day aggregates are
therefore selection-biased and this module refuses to emit one. The horizon table
is built by ``paired_horizon_frame`` from configurations present at *both*
horizons, compared against themselves. That contrast is far less distorted than a
cross-constructor one, but it is not unbiased either: a configuration reached 90
days because its 30-day result was on the Pareto front, so the 30-day arm of
every pair is conditioned on having been high, and regression to the mean works
against the measured change. Read the paired difference as a conservative
estimate -- direction trustworthy, magnitude understated.

Non-Python content lives in sibling directories, as with the other generators:
  jinja/paper_*.tex.j2       LaTeX fragment templates
  json/paper_latex_config.json  labels, captions, defaults
  json/simulation_metadata.json colour palettes (shared)
  style/{dark,light}.mplstyle   matplotlib style sheets

Idempotent: will not overwrite existing output unless --force is passed.

Usage
-----
    uv run python logic/gen/gen_paper_latex.py --force
    uv run python logic/gen/gen_paper_latex.py \\
        --horizon 30=public/global/simulation/simulation_summary.csv \\
        --horizon 90=public/global/simulation/simulation_summary_90d.csv \\
        --tables-dir assets/papers/<paper>/Tables \\
        --figures-dir assets/papers/<paper>/Images/Results/Generated \\
        --theme light --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_utils import apply_theme, load_json, load_theme, render_template, savefig

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Collected-tonnage shortfall, relative to the scenario-cell median, above which
#: a run is treated as degenerate rather than merely bad. See the module
#: docstring for why 0.20 sits in a genuine gap in the distribution.
SHORTFALL_THRESHOLD = 0.20

#: Minimum rows in a cell before its median is trustworthy enough to judge against.
MIN_CELL_SIZE = 4

#: The columns that identify a scenario cell: everything except the constructor,
#: so that the constructors within a cell are exactly the competitors that faced
#: the identical demand realisation.
CELL_KEYS = ["horizon", "city", "N", "dist", "strategy", "cf", "sl_var", "improver"]

#: The columns that identify one policy configuration across horizons.
CONFIG_KEYS = ["city", "N", "dist", "improver", "strategy", "cf", "sl_var", "acceptance", "constructor"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


#: Columns that are populated only for the strategies that have variants (``cf``
#: for Last-Minute, ``sl_var`` for Service-Level) and are empty for Look-Ahead.
#: They are normalised to "" rather than left as NaN because pandas drops NaN
#: keys from groupby and pivot indices, which would silently delete every
#: Look-Ahead run from the aggregates.
OPTIONAL_KEYS = ["cf", "sl_var", "acceptance"]


def load_horizons(horizons: dict[int, Path]) -> pd.DataFrame:
    """Load one summary CSV per horizon into a single frame tagged with ``horizon``."""
    frames = []
    for days, csv in sorted(horizons.items()):
        if not csv.exists():
            raise SystemExit(f"Missing summary CSV for horizon {days}d: {csv}")
        df = pd.read_csv(csv)
        df["horizon"] = days
        frames.append(df)
        print(f"  Loaded {len(df):>4} runs at {days:>3}d from {csv}")
    combined = pd.concat(frames, ignore_index=True)
    for col in OPTIONAL_KEYS:
        combined[col] = combined[col].fillna("").astype(str)
    return combined


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #


def find_degenerate_runs(df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask of runs that failed to complete rather than merely performing badly.

    Within a scenario cell every constructor faces the identical demand
    realisation, so collected tonnage is comparable by construction. A run far
    below its cell median did not collect the waste that was there to collect.
    """
    grouped = df.groupby(CELL_KEYS, dropna=False)["kg"]
    cell_median = grouped.transform("median")
    cell_size = grouped.transform("size")
    shortfall = 1.0 - df["kg"] / cell_median
    return (cell_size >= MIN_CELL_SIZE) & (shortfall > SHORTFALL_THRESHOLD)


def split_degenerate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(clean, degenerate)``, both with a ``shortfall`` column added."""
    grouped = df.groupby(CELL_KEYS, dropna=False)["kg"]
    df = df.assign(shortfall=1.0 - df["kg"] / grouped.transform("median"))
    mask = find_degenerate_runs(df)
    clean, degenerate = df[~mask].copy(), df[mask].copy()
    if len(degenerate):
        print(f"  Excluded {len(degenerate)} degenerate run(s) from aggregates:")
        for _, r in degenerate.iterrows():
            print(
                f"    {r.constructor:<8} {r.horizon:>3}d {r.city} N={r.N} {r.dist} "
                f"{r.strategy}{r.cf}{r.sl_var:<4} {r.improver}: "
                f"collected {r.kg:,.0f} kg ({r.shortfall:.0%} below cell median)"
            )
    return clean, degenerate


def drop_affected_cells(df: pd.DataFrame, degenerate: pd.DataFrame) -> pd.DataFrame:
    """
    Remove every scenario cell that contained a degenerate run, for all constructors.

    Dropping only the degenerate rows would reintroduce exactly the bias this
    module refuses to accept for the 90-day sample. All three degenerate runs
    belong to SWC-TCF, so removing them alone would leave SWC-TCF averaged over
    the scenarios where it did *not* fail while every rival is averaged over
    those scenarios too -- a mean quietly computed on a favourable subset.

    Removing the whole cell costs the other constructors those scenarios as well,
    but keeps every constructor's mean over an identical set of scenarios, which
    is the only reason the marginal means are comparable in the first place.
    """
    if degenerate.empty:
        return df
    affected = set(map(tuple, degenerate[CELL_KEYS].to_numpy()))
    keep = ~df[CELL_KEYS].apply(lambda row: tuple(row) in affected, axis=1)
    dropped = int((~keep).sum())
    print(
        f"  Dropped {dropped} further run(s) so that every constructor is averaged "
        f"over the same {len(affected)} fewer scenario cells"
    )
    return df[keep].copy()


def paired_horizon_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Configurations observed at both horizons, as one row per configuration.

    This is the horizon contrast least distorted by the 90-day sampling, because
    each configuration is compared against *itself* rather than against a rival
    whose 90-day scenarios are a different, self-favouring subset.

    It is not, however, unbiased. A configuration reached 90 days precisely
    because its 30-day result put it on the Pareto front, so the 30-day arm of
    every pair is conditioned on having been high. Regression to the mean then
    works against the measured 30-to-90 change, and the paired difference should
    be read as a conservative estimate of the horizon effect -- the direction is
    trustworthy, the magnitude is understated.
    """
    wide = clean.pivot_table(index=CONFIG_KEYS, columns="horizon", values=["kgkm", "overflows", "km", "time"])
    wide = wide.dropna()
    if wide.empty:
        return wide
    return wide.reset_index()


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #


def display_name(raw: str, cfg: dict) -> str:
    """Map a CSV constructor key to the name the paper uses (ACO_HH -> ACO-HH)."""
    return cfg.get("constructor_labels", {}).get(raw, raw)


def tex_escape(s: object) -> str:
    """Escape the LaTeX specials that appear in this data (constructor names, mostly)."""
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        out = out.replace(a, b)
    return out


def fmt(value: float, decimals: int = 2, *, best: bool = False) -> str:
    """Format one table cell, bolding it when it is the best in its column."""
    if pd.isna(value):
        return "--"
    text = f"{value:,.{decimals}f}" if decimals else f"{value:,.0f}"
    return rf"\textbf{{{text}}}" if best else text


def aggregate(sub: pd.DataFrame, by: str, spec: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    """Aggregate ``sub`` by one key into the (column, aggfunc) pairs a spec asks for."""
    out = pd.DataFrame(index=sorted(sub[by].unique()))
    for col, how, _, _ in spec:
        if col == "n":
            out[(col, how)] = sub.groupby(by).size()
        else:
            out[(col, how)] = sub.groupby(by)[col].agg(how)
    return out


def build_rows(frame: pd.DataFrame, spec: list[tuple[str, str, int, str]]) -> list[dict]:
    """
    Turn an aggregated frame into template-ready rows, marking the best cell per column.

    ``spec`` is ``(column, aggfunc, decimals, direction)`` where direction is
    ``"max"`` or ``"min"``; ``"none"`` disables best-marking for columns where
    "best" is meaningless (a run count, or a descriptive quantity like the number
    of collections, which is neither good nor bad on its own).
    """
    best_of = {}
    for col, how, _, direction in spec:
        if direction in ("max", "min"):
            series = frame[(col, how)]
            best_of[(col, how)] = series.max() if direction == "max" else series.min()
    rows = []
    for label, row in frame.iterrows():
        cells = []
        for col, how, dec, _ in spec:
            key = (col, how)
            value = row[key]
            is_best = key in best_of and np.isclose(value, best_of[key])
            cells.append(fmt(value, dec, best=is_best))
        rows.append({"label": tex_escape(label), "cells": cells})
    return rows


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

#: Means are reported beside medians for the two headline metrics. A single run
#: out of ~96 moved Service-Level's mean overflow count from 9.3 to 1.4, so the
#: means here are genuinely fragile to individual scenarios and a median-based
#: reading is the honest companion, not a footnote.
CONSTRUCTOR_SPEC = [
    ("n", "size", 0, "none"),
    ("kgkm", "mean", 2, "max"),
    ("kgkm", "median", 2, "max"),
    ("overflows", "mean", 1, "min"),
    ("overflows", "median", 1, "min"),
    ("km", "mean", 0, "min"),
    ("time", "mean", 0, "min"),
]

STRATEGY_SPEC = CONSTRUCTOR_SPEC


def table_constructors(clean: pd.DataFrame, horizon: int, cfg: dict) -> str:
    """Per-constructor means and medians over the balanced grid at one horizon."""
    sub = clean[clean.horizon == horizon]
    agg = aggregate(sub, "constructor", CONSTRUCTOR_SPEC)
    agg = agg.sort_values(("kgkm", "mean"), ascending=False)
    counts = sub.groupby("constructor").size()
    agg.index = [display_name(c, cfg) for c in agg.index]
    return render_template(
        "paper_results_table.tex.j2",
        label=f"tab:constructors{horizon}",
        caption=cfg["captions"]["constructors"].format(horizon=horizon, runs=len(sub)),
        first_header=cfg["headers"]["constructor"],
        headers=cfg["headers"]["metrics"],
        column_spec="l" + "r" * len(CONSTRUCTOR_SPEC),
        rows=build_rows(agg, CONSTRUCTOR_SPEC),
        note=cfg["notes"]["constructors"].format(runs_per=int(counts.min())),
    )


def table_strategies(clean: pd.DataFrame, horizon: int, cfg: dict) -> str:
    """Per-strategy-variant means: the efficiency/service trade-off, in one table."""
    sub = clean[clean.horizon == horizon].copy()
    sub["variant"] = sub.strategy + sub.cf + sub.sl_var
    order = [v for v in cfg["variant_order"] if v in set(sub.variant)]
    agg = aggregate(sub, "variant", STRATEGY_SPEC).reindex(order)
    agg.index = [cfg["variant_labels"].get(v, v) for v in agg.index]
    return render_template(
        "paper_results_table.tex.j2",
        label=f"tab:strategies{horizon}",
        caption=cfg["captions"]["strategies"].format(horizon=horizon),
        first_header=cfg["headers"]["strategy"],
        headers=cfg["headers"]["metrics"],
        column_spec="l" + "r" * len(STRATEGY_SPEC),
        rows=build_rows(agg, STRATEGY_SPEC),
        note=cfg["notes"]["strategies"],
    )


def improver_pairs(clean: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    CLS against Fast-TSP on the configurations that differ *only* in improver.

    Pairing rather than averaging matters here: the two improvers are not run on
    identical marginal distributions of scenario, so a difference of means would
    confound improver with scenario mix. Every pair below is the same policy, the
    same scenario, the same demand realisation.
    """
    sub = clean[clean.horizon == horizon]
    keys = [k for k in CONFIG_KEYS if k != "improver"]
    wide = sub.pivot_table(index=keys, columns="improver", values=["kgkm", "km", "overflows", "time"])
    return wide.dropna()


def table_improvers(clean: pd.DataFrame, horizon: int, cfg: dict) -> str:
    pairs = improver_pairs(clean, horizon)
    rows = []
    for metric, decimals, better in cfg["improver_metrics"]:
        cls, ftsp = pairs[(metric, "CLS")], pairs[(metric, "FTSP")]
        delta = cls - ftsp
        # Ties are common on overflow counts (both improvers reorder within an
        # already-fixed bin set, so neither changes which bins were collected),
        # and a bare win count silently reads them as losses. Report all three.
        wins = int((delta > 0).sum() if better == "max" else (delta < 0).sum())
        ties = int(np.isclose(delta, 0.0).sum())
        rows.append(
            {
                "label": tex_escape(cfg["metric_labels"][metric]),
                "cells": [
                    fmt(cls.mean(), decimals),
                    fmt(ftsp.mean(), decimals),
                    fmt(delta.mean(), decimals),
                    f"{wins} / {len(pairs) - ties}",
                    f"{ties}",
                ],
            }
        )
    return render_template(
        "paper_results_table.tex.j2",
        label=f"tab:improvers{horizon}",
        caption=cfg["captions"]["improvers"].format(horizon=horizon, pairs=len(pairs)),
        first_header=cfg["headers"]["metric"],
        headers=cfg["headers"]["improver"],
        column_spec="lrrrrr",
        rows=rows,
        note=cfg["notes"]["improvers"],
    )


def table_horizon(paired: pd.DataFrame, cfg: dict) -> str:
    """Paired 30d-vs-90d comparison, per constructor, on matched configurations."""
    if paired.empty:
        return ""
    rows = []
    for constructor, grp in paired.groupby("constructor"):
        rows.append(
            {
                "label": tex_escape(display_name(constructor, cfg)),
                "cells": [
                    f"{len(grp)}",
                    fmt(grp[("kgkm", 30)].mean(), 2),
                    fmt(grp[("kgkm", 90)].mean(), 2),
                    fmt(grp[("overflows", 30)].mean(), 1),
                    fmt(grp[("overflows", 90)].mean(), 1),
                ],
            }
        )
    return render_template(
        "paper_results_table.tex.j2",
        label="tab:horizon",
        caption=cfg["captions"]["horizon"].format(pairs=len(paired)),
        first_header=cfg["headers"]["constructor"],
        headers=cfg["headers"]["horizon"],
        column_spec="lrrrrr",
        rows=rows,
        note=cfg["notes"]["horizon"],
    )


def table_excluded(degenerate: pd.DataFrame, cfg: dict) -> str:
    """The degenerate runs, reported rather than quietly dropped."""
    if degenerate.empty:
        return ""
    rows = []
    for _, r in degenerate.sort_values("shortfall", ascending=False).iterrows():
        variant = f"{r.strategy}{r.cf}{r.sl_var}"
        rows.append(
            {
                "label": tex_escape(display_name(r.constructor, cfg)),
                "cells": [
                    f"{r.horizon:.0f}",
                    tex_escape(f"{r.city} $N$={r.N}"),
                    tex_escape(f"{r.dist} / {variant} / {r.improver}"),
                    fmt(r.kg, 0),
                    f"{r.shortfall:.0%}".replace("%", r"\%"),
                    fmt(r.overflows, 0),
                ],
            }
        )
    return render_template(
        "paper_results_table.tex.j2",
        label="tab:excluded",
        caption=cfg["captions"]["excluded"],
        first_header=cfg["headers"]["constructor"],
        headers=cfg["headers"]["excluded"],
        column_spec="llllrrr",
        rows=rows,
        note=cfg["notes"]["excluded"].format(threshold=int(SHORTFALL_THRESHOLD * 100)),
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def pareto_front(points: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated set: maximise kg/km, minimise overflows."""
    ordered = points.sort_values("kgkm", ascending=False)
    best, keep = np.inf, []
    for idx, row in ordered.iterrows():
        if row.overflows < best:
            keep.append(idx)
            best = row.overflows
    return points.loc[keep].sort_values("kgkm")


def fig_pareto(clean: pd.DataFrame, horizon: int, out: Path, colors: dict, cfg: dict) -> None:
    """kg/km against overflows, per constructor, with the non-dominated front drawn."""
    sub = clean[clean.horizon == horizon]
    agg = sub.groupby("constructor").agg(kgkm=("kgkm", "mean"), overflows=("overflows", "mean"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, row in agg.iterrows():
        ax.scatter(row.overflows, row.kgkm, s=120, zorder=3,
                   color=colors.get(name, "#4e88d9"), edgecolor="none")
        # Points are annotated in place rather than via a legend: eight labels in
        # a legend box would cover the front on a figure this small.
        ax.annotate(display_name(name, cfg), (row.overflows, row.kgkm),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    front = pareto_front(agg)
    ax.plot(front.overflows, front.kgkm, drawstyle="steps-post", linewidth=1.4,
            linestyle="--", color="#888888", zorder=2)
    ax.set_xlabel(cfg["axis"]["overflows"])
    ax.set_ylabel(cfg["axis"]["kgkm"])
    ax.set_title(cfg["figure_titles"]["pareto"].format(horizon=horizon))
    ax.grid(True, alpha=0.25)
    savefig(fig, out / f"pareto_{horizon}d.png")


def fig_strategy_tradeoff(clean: pd.DataFrame, horizon: int, out: Path, cfg: dict) -> None:
    """The efficiency/service trade-off across strategy variants, as paired bars."""
    sub = clean[clean.horizon == horizon].copy()
    sub["variant"] = sub.strategy + sub.cf + sub.sl_var
    order = [v for v in cfg["variant_order"] if v in set(sub.variant)]
    agg = sub.groupby("variant")[["kgkm", "overflows"]].mean().reindex(order)
    labels = [cfg["variant_labels"].get(v, v) for v in agg.index]

    fig, ax_left = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(agg))
    ax_left.bar(x - 0.2, agg.kgkm, width=0.4, color="#4e88d9", label=cfg["axis"]["kgkm"])
    ax_left.set_ylabel(cfg["axis"]["kgkm"], color="#4e88d9")
    ax_left.tick_params(axis="y", labelcolor="#4e88d9")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)

    ax_right = ax_left.twinx()
    ax_right.bar(x + 0.2, agg.overflows, width=0.4, color="#e07830", label=cfg["axis"]["overflows"])
    ax_right.set_ylabel(cfg["axis"]["overflows"], color="#e07830")
    ax_right.tick_params(axis="y", labelcolor="#e07830")
    ax_right.grid(False)

    ax_left.set_title(cfg["figure_titles"]["strategy"].format(horizon=horizon))
    savefig(fig, out / f"strategy_tradeoff_{horizon}d.png")


def fig_improver_paired(clean: pd.DataFrame, horizon: int, out: Path, cfg: dict) -> None:
    """Per-pair CLS-minus-Fast-TSP efficiency delta, sorted -- the shape is the argument."""
    pairs = improver_pairs(clean, horizon)
    delta = (pairs[("kgkm", "CLS")] - pairs[("kgkm", "FTSP")]).sort_values().to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(np.arange(len(delta)), delta, width=1.0,
           color=np.where(delta > 0, "#5cb85c", "#c04070"))
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_xlabel(cfg["axis"]["pair_index"].format(pairs=len(delta)))
    ax.set_ylabel(cfg["axis"]["kgkm_delta"])
    ax.set_title(cfg["figure_titles"]["improver"].format(
        horizon=horizon, wins=int((delta > 0).sum()), pairs=len(delta)))
    ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, out / f"improver_delta_{horizon}d.png")


def fig_scale(clean: pd.DataFrame, horizon: int, out: Path, colors: dict, cfg: dict) -> None:
    """Runtime against instance size, per constructor -- the cost side of the trade-off."""
    sub = clean[clean.horizon == horizon]
    agg = sub.pivot_table(index="N", columns="constructor", values="time", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name in agg.columns:
        ax.plot(agg.index, agg[name], marker="o", linewidth=1.6,
                color=colors.get(name, "#4e88d9"), label=display_name(name, cfg))
    ax.set_xlabel(cfg["axis"]["n_bins"])
    ax.set_ylabel(cfg["axis"]["time"])
    ax.set_yscale("log")
    ax.set_xticks(sorted(agg.index))
    ax.set_title(cfg["figure_titles"]["scale"].format(horizon=horizon))
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    savefig(fig, out / f"runtime_scaling_{horizon}d.png")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_horizons(values: list[str] | None) -> dict[int, Path]:
    if not values:
        return {
            30: REPO_ROOT / "public/global/simulation/simulation_summary.csv",
            90: REPO_ROOT / "public/global/simulation/simulation_summary_90d.csv",
        }
    out = {}
    for item in values:
        days, _, csv = item.partition("=")
        if not csv:
            raise SystemExit(f"--horizon expects DAYS=CSV, got {item!r}")
        out[int(days)] = Path(csv)
    return out


def write(path: Path, text: str, force: bool) -> None:
    if not text:
        return
    if path.exists() and not force:
        print(f"  Skipped (exists, use --force): {path.name}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"  Wrote: {path.name}")


def main() -> None:
    cfg = load_json("paper_latex_config.json")
    default_paper = REPO_ROOT / cfg["paths"]["paper_dir"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", action="append", metavar="DAYS=CSV",
                    help="summary CSV for one horizon; repeatable (default: 30d and 90d)")
    ap.add_argument("--tables-dir", type=Path, default=default_paper / cfg["paths"]["tables_subdir"])
    ap.add_argument("--figures-dir", type=Path, default=default_paper / cfg["paths"]["figures_subdir"])
    ap.add_argument("--theme", choices=["dark", "light"], default="light",
                    help="light matches the paper's white background (default)")
    ap.add_argument("--primary-horizon", type=int, default=30,
                    help="horizon whose grid is complete and therefore leads the results")
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    theme = load_theme(args.theme)
    apply_theme(theme)
    colors = dict(load_json("simulation_metadata.json").get("constructor_colors", {}))

    print("Loading simulation summaries:")
    raw = load_horizons(parse_horizons(args.horizon))

    print("Checking data integrity:")
    clean, degenerate = split_degenerate(raw)
    clean = drop_affected_cells(clean, degenerate)
    paired = paired_horizon_frame(clean)
    print(f"  {len(clean)} runs retained; {len(paired)} configurations present at both horizons")

    horizon = args.primary_horizon

    if not args.figures_only:
        args.tables_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing LaTeX tables to {args.tables_dir}:")
        write(args.tables_dir / "results_constructors.tex", table_constructors(clean, horizon, cfg), args.force)
        write(args.tables_dir / "results_strategies.tex", table_strategies(clean, horizon, cfg), args.force)
        write(args.tables_dir / "results_improvers.tex", table_improvers(clean, horizon, cfg), args.force)
        write(args.tables_dir / "results_horizon.tex", table_horizon(paired, cfg), args.force)
        write(args.tables_dir / "results_excluded.tex", table_excluded(degenerate, cfg), args.force)

    if not args.tables_only:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing figures to {args.figures_dir}:")
        fig_pareto(clean, horizon, args.figures_dir, colors, cfg)
        fig_strategy_tradeoff(clean, horizon, args.figures_dir, cfg)
        fig_improver_paired(clean, horizon, args.figures_dir, cfg)
        fig_scale(clean, horizon, args.figures_dir, colors, cfg)

    print("Done.")


if __name__ == "__main__":
    main()

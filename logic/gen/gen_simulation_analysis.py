"""
Generate the simulation analysis markdown report and all associated figures.

The script is agnostic to the mandatory node selection strategies, route
constructors (with optional acceptance criteria), route improvers, simulation
scenarios (region, graph size, data distribution) and time horizons present in
the data: everything is auto-detected and can be narrowed via a JSON
configuration file (see logic/gen/json/simulation_analysis_config.json)
and/or CLI flags.

It also subsumes the old gen_simulation_csv.py: raw simulation output trees
(assets/output/<horizon>days/...) can be parsed directly into the summary CSV
schema, either standalone (--parse-output) or on the fly when a horizon entry
in the config specifies "output_dir" instead of (or in addition to) "csv".

For every requested horizon the same report sections are generated (ordered
from the smallest to the largest horizon); a final cross-horizon comparison
section is appended only when more than one horizon is analysed.

Non-Python content lives in sibling directories:
  jinja/simulation_analysis.md.j2   markdown template
  json/simulation_metadata.json     parsing metadata + colour palettes
  json/simulation_analysis_config.json  default analysis configuration
  json/themes.json                  dark/light theme definitions
  style/{dark,light}.mplstyle       matplotlib style sheets
  js/pareto_buttons.js              plotly button styling snippet

Usage
-----
    # Full default analysis (config-driven; 30d + 90d if both CSVs exist)
    uv run python logic/gen/gen_simulation_analysis.py --force

    # White-background charts, Pareto plots restricted to the front points
    uv run python logic/gen/gen_simulation_analysis.py \\
        --theme light --pareto-points front --force

    # Only one horizon, explicit CSV
    uv run python logic/gen/gen_simulation_analysis.py \\
        --horizon 30=public/global/simulation/simulation_summary.csv --force

    # Regenerate a summary CSV from a raw output tree (old gen_simulation_csv)
    uv run python logic/gen/gen_simulation_analysis.py --parse-output \\
        --output-dir assets/output/90days \\
        --out-csv public/global/simulation/simulation_summary_90d.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_utils import (
    PLACEHOLDER,
    finalize_markdown,
    load_js,
    load_json,
    load_theme,
    render_template,
    savefig,
)

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

META = load_json("simulation_metadata.json")

MARKER_CYCLE = ["o", "s", "D", "^", "v", "P", "X", "*"]
HATCH_CYCLE = ["", "//", "xx", ".."]
PLOTLY_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down", "pentagon", "x", "star"]


# ── Raw output tree parsing (merged from gen_simulation_csv.py) ────────────────


def _parse_area_dir(dirname: str) -> tuple[str, int] | None:
    """Map e.g. 'riomaior100_plastic' → ('Rio Maior', 100)."""
    m = re.match(r"([a-z]+?)(\d+)(?:_\w+)?$", dirname)
    if not m:
        return None
    city = META["dir_city"].get(m.group(1))
    if city is None:
        return None
    return city, int(m.group(2))


def _parse_filename(stem: str) -> dict | None:
    """
    Parse log filename stem into metadata.

    Expected form: log_{strategy_tokens}_{constructor}[_{acceptance}]_{improver}_{N}N
    """
    if not stem.startswith("log_"):
        return None
    rest = stem[4:]

    strategy = cf = sl_var = None
    for spec in META["strategy_prefixes"]:
        if rest.startswith(spec["prefix"]):
            strategy, cf, sl_var = spec["strategy"], spec["cf"], spec["sl_var"]
            rest = rest[len(spec["prefix"]) :]
            break
    if strategy is None:
        return None

    m = re.search(META["improver_pattern"], rest, re.IGNORECASE)
    if not m:
        return None
    improver = m.group(1).upper()
    middle = rest[: m.start()]

    constructor = acceptance = None
    for token, label in META["constructors"]:
        if middle.startswith(token):
            constructor = label
            acceptance = middle[len(token) :].strip("_") or None
            break
    if constructor is None:
        return None

    return {
        "strategy": strategy,
        "cf": cf,
        "sl_var": sl_var,
        "improver": improver,
        "constructor": constructor,
        "acceptance": acceptance,
    }


def parse_output_dir(output_dir: Path) -> pd.DataFrame:
    """Walk *output_dir* and return a DataFrame of simulation metrics."""
    rows: list[dict] = []
    for area_dir in sorted(output_dir.iterdir()):
        if not area_dir.is_dir():
            continue
        city_n = _parse_area_dir(area_dir.name)
        if city_n is None:
            continue
        city, N = city_n

        for dist_dir in sorted(area_dir.iterdir()):
            if not dist_dir.is_dir():
                continue
            dist = META["dist_map"].get(dist_dir.name)
            if dist is None:
                continue

            for log_file in sorted(dist_dir.rglob("log_*.json")):
                meta = _parse_filename(log_file.stem)
                if meta is None:
                    continue
                try:
                    data = json.loads(log_file.read_text())
                except Exception:
                    continue
                mean = data.get("mean", {})
                if not mean:
                    continue
                rows.append(
                    {
                        "city": city,
                        "N": N,
                        "dist": dist,
                        "improver": meta["improver"],
                        "strategy": meta["strategy"],
                        "cf": meta["cf"] or "",
                        "sl_var": meta["sl_var"] or "",
                        "acceptance": meta["acceptance"] or "",
                        "constructor": meta["constructor"],
                        "overflows": mean.get("overflows", 0),
                        "kg": mean.get("kg", 0),
                        "ncol": mean.get("ncol", 0),
                        "kg_lost": mean.get("kg_lost", 0),
                        "km": mean.get("km", 0),
                        "kgkm": mean.get("kg/km", 0),
                        "reward": mean.get("reward", 0),
                        "profit": mean.get("profit", 0),
                        "time": mean.get("time", 0),
                        "days": mean.get("days", 0),
                    }
                )
    return pd.DataFrame(rows)


# ── Labels & data preparation ──────────────────────────────────────────────────


def region_label(city: str, N: int) -> str:
    short = META["city_short"].get(city, city[:3].upper())
    return f"{short}-{N}"


def scenario_label(s: dict) -> str:
    return f"{region_label(s['city'], s['N'])} / {s['dist']}"


def variant_label(row) -> str:
    strat = row["strategy"]
    if strat == "LM" and str(row.get("cf", "")) not in ("", "nan", "None"):
        return f"LM ({row['cf']})"
    if strat == "SL" and str(row.get("sl_var", "")) not in ("", "nan", "None"):
        return f"SL ({row['sl_var']})"
    return strat


def variant_color(label: str, fallback_idx: int = 0) -> str:
    palette = META["variant_colors"]
    if label in palette:
        return palette[label]
    cycle = list(palette.values()) + META["scenario_colors"]
    return cycle[fallback_idx % len(cycle)]


def load_horizon_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("cf", "sl_var", "acceptance"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    df["variant"] = df.apply(variant_label, axis=1)
    return df


def detect_scenarios(df: pd.DataFrame) -> list[dict]:
    scen = df[["city", "N", "dist"]].drop_duplicates().to_dict("records")
    return sorted(scen, key=lambda s: (s["N"], s["city"], s["dist"]))


def filter_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Restrict *df* to the configured scenarios and policy components."""
    scenarios = config.get("scenarios")
    if scenarios:
        mask = pd.Series(False, index=df.index)
        for s in scenarios:
            mask |= (df.city == s["city"]) & (int(s["N"]) == df.N) & (df.dist == s["dist"])
        df = df[mask]
    pol = config.get("policies") or {}
    for key, col in [
        ("strategies", "strategy"),
        ("constructors", "constructor"),
        ("improvers", "improver"),
        ("acceptance", "acceptance"),
    ]:
        allowed = pol.get(key)
        if allowed:
            df = df[df[col].isin(allowed)]
    return df.reset_index(drop=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics over CF/SL variants (per scenario × strategy × constructor × improver)."""
    return (
        df.groupby(["city", "N", "dist", "improver", "strategy", "constructor"])[
            ["overflows", "kgkm", "km", "profit", "kg", "reward"]
        ]
        .mean()
        .reset_index()
    )


def _scen_sub(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    return df[(df.city == s["city"]) & (s["N"] == df.N) & (df.dist == s["dist"])]


def _panel_grid(n: int, panel_size: tuple[float, float] = (10.0, 7.0)) -> tuple[plt.Figure, list]:
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False)
    flat = [axes[i // ncols][i % ncols] for i in range(nrows * ncols)]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _norm_scales(opt, default: str = "linear") -> list[str]:
    if opt is None:
        return [default]
    if isinstance(opt, str):
        return [opt]
    return list(opt)


def pareto_indices(xs, ys) -> set[int]:
    """Positional indices of non-dominated points (min x=overflows, max y=kgkm)."""
    order = sorted(range(len(xs)), key=lambda i: (xs[i], -ys[i]))
    front, best = set(), -np.inf
    for i in order:
        if ys[i] > best:
            front.add(i)
            best = ys[i]
    return front


# ── Static figure generators ───────────────────────────────────────────────────


def gen_pareto_scatter(df: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Per-distribution panels of the overflows/kg-km space with Pareto fronts."""
    opts = ctx["charts"].get("pareto_scatter", {})
    scenarios, dists, improvers = ctx["scenarios"], ctx["dists"], ctx["improvers"]
    regions = ctx["regions"]
    variants = sorted(df["variant"].unique())
    vcolors = {v: variant_color(v, i) for i, v in enumerate(variants)}
    markers = {tuple(r): MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, r in enumerate(regions)}
    points_mode = ctx.get("pareto_points") or opts.get("points", "all")

    def _make(xscale: str) -> plt.Figure:
        fig, axes = _panel_grid(len(dists), (11, 8))
        title = f"Overflow vs Efficiency — Pareto Front ({ctx['n_days']} days)"
        if points_mode == "front":
            title += " — Front Points Only"
        if xscale != "linear":
            title += " — Log Scale"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for ax, dist in zip(axes, dists, strict=True):
            for s in [sc for sc in scenarios if sc["dist"] == dist]:
                sub = _scen_sub(df[df.dist == dist], s).reset_index(drop=True)
                if sub.empty:
                    continue
                front = pareto_indices(sub["overflows"].values, sub["kgkm"].values)
                for i, row in sub.iterrows():
                    if points_mode == "front" and i not in front:
                        continue
                    filled = row["improver"] == improvers[0]
                    color = vcolors[row["variant"]]
                    ax.scatter(
                        row["overflows"],
                        row["kgkm"],
                        marker=markers[(s["city"], s["N"])],
                        s=90 if i in front else 45,
                        facecolors=color if filled else "none",
                        edgecolors=color,
                        linewidths=1.4,
                        alpha=0.85,
                        zorder=3,
                    )
                pts = sorted((sub["overflows"][i], sub["kgkm"][i]) for i in front)
                if pts:
                    sx, sy = [pts[0][0]], [pts[0][1]]
                    for j in range(1, len(pts)):
                        sx += [pts[j][0], pts[j][0]]
                        sy += [pts[j - 1][1], pts[j][1]]
                    ax.plot(sx, sy, "--", color=ctx["theme"]["accent_line"], linewidth=1.6, alpha=0.8, zorder=2)
            if xscale != "linear":
                ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel(f"Overflows ({ctx['n_days']} days)", fontsize=10)
            ax.set_ylabel("Efficiency (kg/km)", fontsize=10)
            ax.set_title(f"{dist}", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        leg = [mpatches.Patch(color=vcolors[v], label=v) for v in variants]
        leg += [
            plt.Line2D([], [], marker=markers[tuple(r)], linestyle="", color=ctx["theme"]["muted"],
                       label=region_label(r[0], r[1]))
            for r in regions
        ]
        if len(improvers) > 1:
            leg += [
                plt.Line2D([], [], marker="o", linestyle="", color=ctx["theme"]["fg"], label=f"{improvers[0]} (filled)"),
                plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color=ctx["theme"]["fg"],
                           label=f"{improvers[-1]} (open)"),
            ]
        leg.append(plt.Line2D([0], [0], color=ctx["theme"]["accent_line"], linestyle="--", label="Pareto front"))
        fig.legend(handles=leg, loc="lower center", ncol=min(len(leg), 6), fontsize=9, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout()
        return fig

    scales = _norm_scales(opts.get("x_scale"), "linear")
    savefig(_make(scales[0]), out_dir / "pareto_scatter.png")
    if len(scales) > 1:
        savefig(_make(scales[1]), out_dir / "pareto_scatter_log.png")


def gen_kpi_bar(dfm: pd.DataFrame, metric: str, ylabel: str, fname: str, ctx: dict, out_dir: Path, opts: dict) -> None:
    """Scenario×strategy bars (mean ± min/max across constructors), improvers as paired bars."""
    scenarios, dists, strategies, improvers = ctx["scenarios"], ctx["dists"], ctx["strategies"], ctx["improvers"]

    def _make(yscale: str) -> plt.Figure:
        fig, axes = _panel_grid(len(dists), (12, 7))
        title = f"{ylabel} by Configuration (mean ± range across constructors)"
        if yscale != "linear":
            title += " — Log Scale"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        n_imp = len(improvers)
        width = 0.8 / n_imp
        for ax, dist in zip(axes, dists, strict=True):
            scen_d = [s for s in scenarios if s["dist"] == dist]
            groups = [(s, strat) for s in scen_d for strat in strategies]
            labels = [f"{region_label(s['city'], s['N'])}\n{strat}" for s, strat in groups]
            x = np.arange(len(groups), dtype=float)
            for ii, imp in enumerate(improvers):
                means, lo, hi = [], [], []
                for s, strat in groups:
                    sub = _scen_sub(dfm[(dfm.improver == imp) & (dfm.strategy == strat)], s)[metric]
                    if len(sub):
                        m = sub.mean()
                        means.append(m)
                        lo.append(m - sub.min())
                        hi.append(sub.max() - m)
                    else:
                        means.append(np.nan)
                        lo.append(np.nan)
                        hi.append(np.nan)
                xs = x + (ii - (n_imp - 1) / 2) * width
                colors = [META["strategy_colors"].get(strat, "#a0a0a0") for _, strat in groups]
                valid = ~np.isnan(means)
                ax.bar(
                    xs[valid],
                    np.array(means)[valid],
                    width * 0.92,
                    color=np.array(colors)[valid],
                    alpha=0.9 if ii == 0 else 0.6,
                    hatch=HATCH_CYCLE[ii % len(HATCH_CYCLE)],
                    edgecolor=ctx["theme"]["fg"],
                    linewidth=0.4,
                )
                for xi, m_, l_, h_ in zip(xs, means, lo, hi, strict=True):
                    if not np.isnan(m_):
                        ax.errorbar(xi, m_, yerr=[[max(l_, 0)], [max(h_, 0)]], fmt="none",
                                    color=ctx["theme"]["fg"], capsize=2, linewidth=0.9)
            if yscale != "linear":
                ax.set_yscale("symlog", linthresh=1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel(f"{ylabel} ({ctx['n_days']} days)", fontsize=10)
            ax.set_title(f"{dist}", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        patches = [mpatches.Patch(color=META["strategy_colors"].get(s, "#a0a0a0"), label=s) for s in strategies]
        patches += [
            mpatches.Patch(facecolor="#909090", hatch=HATCH_CYCLE[i % len(HATCH_CYCLE)], label=imp)
            for i, imp in enumerate(improvers)
        ]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=10, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout()
        return fig

    scales = _norm_scales(opts.get("y_scale"), "linear")
    savefig(_make(scales[0]), out_dir / f"{fname}.png")
    if len(scales) > 1:
        savefig(_make(scales[1]), out_dir / f"{fname}_log.png")


def gen_km_violin(df: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Violin plots of total km per strategy × scenario (constructors + improvers pooled)."""
    scenarios, dists, strategies = ctx["scenarios"], ctx["dists"], ctx["strategies"]
    fig, axes = _panel_grid(len(dists), (12, 7))
    fig.suptitle("Vehicle Distance Distribution by Strategy and Scenario", fontsize=14, fontweight="bold")
    for ax, dist in zip(axes, dists, strict=True):
        scen_d = [s for s in scenarios if s["dist"] == dist]
        groups, labels, strat_of = [], [], []
        for strat in strategies:
            for s in scen_d:
                grp = _scen_sub(df[df.strategy == strat], s)["km"].values
                if len(grp) == 0:
                    grp = np.array([0.0, 0.0])
                elif len(grp) == 1:
                    grp = np.array([grp[0], grp[0]])
                groups.append(grp)
                labels.append(f"{strat}\n{region_label(s['city'], s['N'])}")
                strat_of.append(strat)
        if not groups:
            continue
        parts = ax.violinplot(groups, positions=range(len(groups)), showmedians=True, showextrema=True)
        for body, strat in zip(parts["bodies"], strat_of, strict=True):
            body.set_facecolor(META["strategy_colors"].get(strat, "#a0a0a0"))
            body.set_alpha(0.7)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[key].set_color(ctx["theme"]["accent_line"] if key == "cmedians" else ctx["theme"]["guide_line"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"Total km ({ctx['n_days']} days)", fontsize=10)
        ax.set_title(f"Distance Distribution — {dist}", fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    patches = [mpatches.Patch(color=META["strategy_colors"].get(s, "#a0a0a0"), label=s) for s in strategies]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=11, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    savefig(fig, out_dir / "km_violin.png")


def _policy_rows(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    combos = df[["variant", "constructor", "improver"]].drop_duplicates().values.tolist()
    return sorted((tuple(c) for c in combos), key=lambda c: (c[0], c[1], c[2]))


def gen_policy_scenario_heatmap(df: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Heatmaps with policy configurations on the rows and scenarios on the columns."""
    scenarios = ctx["scenarios"]
    rows = _policy_rows(df)
    row_labels = [f"{v} · {c} · {i}" for v, c, i in rows]
    col_labels = [scenario_label(s) for s in scenarios]
    for metric, cmap, mlabel in [("overflows", "RdYlGn_r", "Overflow Count"), ("kgkm", "RdYlGn", "kg/km Efficiency")]:
        mat = np.full((len(rows), len(scenarios)), np.nan)
        for ri, (v, c, i) in enumerate(rows):
            sub_p = df[(df.variant == v) & (df.constructor == c) & (df.improver == i)]
            for ci, s in enumerate(scenarios):
                vals = _scen_sub(sub_p, s)[metric]
                if len(vals):
                    mat[ri, ci] = vals.mean()
        fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(scenarios)), max(8, 0.32 * len(rows)) + 1.2))
        ax.set_title(
            f"Policy × Scenario Heatmap — {mlabel} ({ctx['n_days']} days)", fontsize=14, fontweight="bold", pad=14
        )
        norm = matplotlib.colors.SymLogNorm(linthresh=10, vmin=0) if metric == "overflows" else None
        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
        plt.colorbar(im, ax=ax, shrink=0.8, label=mlabel)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        plt.tight_layout()
        savefig(fig, out_dir / f"policy_scenario_heatmap_{metric}.png")


def gen_scenario_constructor_heatmap(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """One heatmap panel per scenario: constructors on rows, strategy × improver on columns."""
    scenarios, strategies, improvers, constructors = (
        ctx["scenarios"],
        ctx["strategies"],
        ctx["improvers"],
        ctx["constructors"],
    )
    combos = [(s, i) for s in strategies for i in improvers]
    combo_labels = [f"{s}\n{i}" for s, i in combos]
    n = len(scenarios)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 12), squeeze=False)
    fig.suptitle("Per-Scenario Heatmaps — Constructors × (Strategy × Improver)", fontsize=14, fontweight="bold")
    for row_i, (metric, cmap, mlabel) in enumerate(
        [("overflows", "RdYlGn_r", "Overflows"), ("kgkm", "RdYlGn", "kg/km")]
    ):
        for col_i, s in enumerate(scenarios):
            ax = axes[row_i][col_i]
            mat = np.full((len(constructors), len(combos)), np.nan)
            sub_s = _scen_sub(dfm, s)
            for ci, con in enumerate(constructors):
                for bi, (strat, imp) in enumerate(combos):
                    vals = sub_s[(sub_s.constructor == con) & (sub_s.strategy == strat) & (sub_s.improver == imp)][
                        metric
                    ]
                    if len(vals):
                        mat[ci, bi] = vals.mean()
            norm = matplotlib.colors.SymLogNorm(linthresh=10, vmin=0) if metric == "overflows" else None
            im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
            plt.colorbar(im, ax=ax, shrink=0.75)
            ax.set_xticks(range(len(combo_labels)))
            ax.set_xticklabels(combo_labels, fontsize=6, rotation=45, ha="right")
            ax.set_yticks(range(len(constructors)))
            ax.set_yticklabels(constructors if col_i == 0 else [], fontsize=7)
            if row_i == 0:
                ax.set_title(scenario_label(s), fontsize=9, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(mlabel, fontsize=10)
    plt.tight_layout()
    savefig(fig, out_dir / "scenario_constructor_heatmap.png")


def gen_strategy_bubble(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Bubble chart per distribution: one bubble per (strategy, scenario), size ∝ N."""
    opts = ctx["charts"].get("strategy_bubble", {})
    scenarios, dists, strategies = ctx["scenarios"], ctx["dists"], ctx["strategies"]
    regions = ctx["regions"]
    markers = {tuple(r): MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, r in enumerate(regions)}

    def _make(xscale: str) -> plt.Figure:
        fig, axes = _panel_grid(len(dists), (10, 8))
        title = "Strategy Trade-off Bubble Chart" + (" (log X)" if xscale != "linear" else "")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for ax, dist in zip(axes, dists, strict=True):
            for s in [sc for sc in scenarios if sc["dist"] == dist]:
                for strat in strategies:
                    sub = _scen_sub(dfm[dfm.strategy == strat], s)
                    if sub.empty:
                        continue
                    ov, eff = sub["overflows"].mean(), sub["kgkm"].mean()
                    ax.scatter(
                        ov,
                        eff,
                        c=META["strategy_colors"].get(strat, "#a0a0a0"),
                        marker=markers[(s["city"], s["N"])],
                        s=120 + 2 * s["N"],
                        alpha=0.75,
                        edgecolors=ctx["theme"]["accent_line"],
                        linewidths=0.5,
                    )
                    ax.annotate(
                        region_label(s["city"], s["N"]),
                        (ov, eff),
                        textcoords="offset points",
                        xytext=(4, 3),
                        fontsize=7,
                    )
            if xscale != "linear":
                ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel(f"Mean overflows ({ctx['n_days']} days)", fontsize=10)
            ax.set_ylabel("Mean kg/km efficiency", fontsize=10)
            ax.set_title(f"{dist}", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        patches = [mpatches.Patch(color=META["strategy_colors"].get(s, "#a0a0a0"), label=s) for s in strategies]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=10, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout()
        return fig

    scales = _norm_scales(opts.get("x_scale"), "linear")
    savefig(_make(scales[0]), out_dir / "strategy_bubble.png")
    if len(scales) > 1:
        savefig(_make(scales[1]), out_dir / "strategy_bubble_log.png")


def gen_improver_bubble(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Bubble chart per distribution comparing route improvers (strategies pooled)."""
    opts = ctx["charts"].get("improver_bubble", {})
    scenarios, dists, improvers = ctx["scenarios"], ctx["dists"], ctx["improvers"]
    regions = ctx["regions"]
    markers = {tuple(r): MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, r in enumerate(regions)}

    def _make(xscale: str) -> plt.Figure:
        fig, axes = _panel_grid(len(dists), (10, 8))
        title = "Route Improver Trade-off Bubble Chart" + (" (log X)" if xscale != "linear" else "")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for ax, dist in zip(axes, dists, strict=True):
            for s in [sc for sc in scenarios if sc["dist"] == dist]:
                pts = {}
                for imp in improvers:
                    sub = _scen_sub(dfm[dfm.improver == imp], s)
                    if sub.empty:
                        continue
                    ov, eff = sub["overflows"].mean(), sub["kgkm"].mean()
                    pts[imp] = (ov, eff)
                    ax.scatter(
                        ov,
                        eff,
                        c=META["improver_colors"].get(imp, "#a0a0a0"),
                        marker=markers[(s["city"], s["N"])],
                        s=120 + 2 * s["N"],
                        alpha=0.8,
                        edgecolors=ctx["theme"]["accent_line"],
                        linewidths=0.5,
                    )
                if len(pts) == 2:
                    (x1, y1), (x2, y2) = pts.values()
                    ax.plot([x1, x2], [y1, y2], "-", color=ctx["theme"]["guide_line"], linewidth=0.8, alpha=0.6)
                for _imp, (ov, eff) in pts.items():
                    ax.annotate(
                        region_label(s["city"], s["N"]),
                        (ov, eff),
                        textcoords="offset points",
                        xytext=(4, 3),
                        fontsize=6,
                    )
            if xscale != "linear":
                ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel(f"Mean overflows ({ctx['n_days']} days)", fontsize=10)
            ax.set_ylabel("Mean kg/km efficiency", fontsize=10)
            ax.set_title(f"{dist}", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        patches = [mpatches.Patch(color=META["improver_colors"].get(i, "#a0a0a0"), label=i) for i in improvers]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=10, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout()
        return fig

    scales = _norm_scales(opts.get("x_scale"), "linear")
    savefig(_make(scales[0]), out_dir / "improver_bubble.png")
    if len(scales) > 1:
        savefig(_make(scales[1]), out_dir / "improver_bubble_log.png")


def gen_constructor_ranking(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Average rank of each constructor across scenarios × strategies × improvers."""
    constructors = ctx["constructors"]
    metrics = ["overflows", "kgkm", "km", "profit"]
    metric_labels = ["Overflows", "kg/km", "km", "Profit"]
    rank_asc = [True, False, True, False]
    fig, ax = plt.subplots(figsize=(max(12, 1.8 * len(constructors)), 8))
    fig.suptitle("Route Constructor Average Rankings (all scenarios, improvers pooled)", fontsize=14, fontweight="bold")
    mean_ranks: dict = {c: {m: [] for m in metrics} for c in constructors}
    for metric, asc in zip(metrics, rank_asc, strict=True):
        for _, grp in dfm.groupby(["city", "N", "dist", "strategy", "improver"]):
            gi = grp.set_index("constructor")
            r = gi[metric].rank(ascending=asc, method="average")
            for c in constructors:
                if c in r.index:
                    mean_ranks[c][metric].append(r[c])
    x = np.arange(len(constructors))
    w = 0.8 / len(metrics)
    for mi, (metric, label, color) in enumerate(zip(metrics, metric_labels, META["metric_colors"], strict=True)):
        vals = [np.mean(mean_ranks[c][metric]) if mean_ranks[c][metric] else np.nan for c in constructors]
        ax.bar(x + (mi - (len(metrics) - 1) / 2) * w, vals, w * 0.9, label=label, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(constructors, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Average rank (lower = better)", fontsize=11)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    plt.tight_layout()
    savefig(fig, out_dir / "constructor_ranking.png")


def gen_radar(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Normalised radar chart for key constructors."""
    opts = ctx["charts"].get("radar", {})
    key = [c for c in opts.get("constructors", ctx["constructors"][:4]) if c in ctx["constructors"]]
    if not key:
        key = ctx["constructors"][:4]
    metrics = ["overflows", "kgkm", "km", "profit"]
    axes_labels = ["Overflows\n(fewer ↓)", "kg/km\n(higher ↑)", "km\n(fewer ↓)", "Profit\n(higher ↑)"]
    invert = [True, False, True, False]

    scores = {}
    for c in key:
        sub = dfm[dfm.constructor == c]
        scores[c] = []
        for metric, inv in zip(metrics, invert, strict=True):
            all_vals = dfm[metric].values
            v = sub[metric].mean() if len(sub) else np.nanmean(all_vals)
            mn, mx = np.nanmin(all_vals), np.nanmax(all_vals)
            norm = (v - mn) / (mx - mn + 1e-9) if mx > mn else 0.5
            scores[c].append(1 - norm if inv else norm)

    n_axes = len(metrics)
    angles = [i / n_axes * 2 * np.pi for i in range(n_axes)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelcolor=ctx["theme"]["muted"], labelsize=8)
    for r, lbl in zip([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], strict=True):
        ax.plot(angles, [r] * (n_axes + 1), "--", color=ctx["theme"]["faint"], linewidth=0.8)
        ax.text(0, r + 0.02, lbl, ha="center", va="bottom", fontsize=8, color=ctx["theme"]["muted"])
    for c in key:
        vals = scores[c] + scores[c][:1]
        color = META["constructor_colors"].get(c, ctx["theme"]["fg"])
        ax.plot(angles, vals, "o-", color=color, linewidth=2.5, markersize=5, label=c)
        ax.fill(angles, vals, color=color, alpha=0.08)
    ax.set_title("Policy Performance Radar\n(normalised; outer = better)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)
    savefig(fig, out_dir / "policy_radar.png")


def gen_improver_delta(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Delta heatmap (last improver − first improver) per constructor × configuration."""
    improvers = ctx["improvers"]
    if len(improvers) < 2:
        return
    imp_a, imp_b = improvers[0], improvers[-1]
    scenarios, strategies, constructors = ctx["scenarios"], ctx["strategies"], ctx["constructors"]
    configs = [(s, strat) for s in scenarios for strat in strategies]
    col_labels = [f"{region_label(s['city'], s['N'])}\n{s['dist'][:3]}\n{strat}" for s, strat in configs]
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))
    fig.suptitle(f"Route Improver Delta Heatmap ({imp_b} − {imp_a})", fontsize=14, fontweight="bold")
    for ax, (metric, cmap) in zip(axes, [("overflows", "RdYlGn_r"), ("kgkm", "RdYlGn")], strict=True):
        mat = np.full((len(constructors), len(configs)), np.nan)
        for ci, c in enumerate(constructors):
            for cfi, (s, strat) in enumerate(configs):
                sub = _scen_sub(dfm[(dfm.strategy == strat) & (dfm.constructor == c)], s)
                a = sub[sub.improver == imp_a][metric]
                b = sub[sub.improver == imp_b][metric]
                if len(a) and len(b):
                    mat[ci, cfi] = b.values[0] - a.values[0]
        finite = mat[~np.isnan(mat)]
        vmax = np.nanpercentile(np.abs(finite), 95) if len(finite) else 1
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label=f"Δ {metric}")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(constructors)))
        ax.set_yticklabels(constructors, fontsize=9)
        ax.set_title(f"Δ {metric}", fontsize=11)
    plt.tight_layout()
    savefig(fig, out_dir / "improver_delta.png")


# ── Interactive (plotly) figures ───────────────────────────────────────────────


def gen_interactive_html(df: pd.DataFrame, dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> dict[str, Path]:  # noqa: C901
    if not HAS_PLOTLY:
        print("  [WARN] Plotly not available — skipping interactive HTML generation")
        return {}
    theme = ctx["theme"]
    template = theme["plotly_template"]
    scenarios, improvers = ctx["scenarios"], ctx["improvers"]
    regions = ctx["regions"]
    region_syms = {tuple(r): PLOTLY_SYMBOLS[i % len(PLOTLY_SYMBOLS)] for i, r in enumerate(regions)}
    variants = sorted(df["variant"].unique())
    vcolors = {v: variant_color(v, i) for i, v in enumerate(variants)}
    paths: dict[str, Path] = {}

    # pareto_scatter_interactive ────────────────────────────────────────────────
    fig = go.Figure()
    legend_shown: set = set()
    for s in scenarios:
        sub_s = _scen_sub(df, s)
        if sub_s.empty:
            continue
        sym_base = region_syms[(s["city"], s["N"])]
        for variant, sg in sub_s.groupby("variant"):
            color = vcolors[variant]
            key = (variant, s["dist"])
            show = key not in legend_shown
            legend_shown.add(key)
            syms = [sym_base if r.improver == improvers[0] else f"{sym_base}-open" for r in sg.itertuples()]
            fig.add_trace(
                go.Scatter(
                    x=sg["overflows"],
                    y=sg["kgkm"],
                    mode="markers",
                    name=f"{variant} ({s['dist'][:3]})",
                    legendgroup=f"{variant}_{s['dist']}",
                    showlegend=show,
                    marker=dict(symbol=syms, color=color, size=8, opacity=0.85, line=dict(width=1.5, color=color)),
                    text=[
                        f"Constructor: {r.constructor}<br>Selection: {r.variant}<br>"
                        f"Scenario: {scenario_label(s)}<br>Improver: {r.improver}<br>"
                        f"Overflows: {r.overflows:.1f}<br>kg/km: {r.kgkm:.3f}<br>"
                        f"km: {r.km:.0f}<br>Profit: {r.profit:.0f}"
                        for r in sg.itertuples()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    n_scatter_traces = len(fig.data)

    # Pareto front lines + front-only markers per scenario
    for s in scenarios:
        sub = _scen_sub(df, s).reset_index(drop=True)
        if sub.empty:
            continue
        front = pareto_indices(sub["overflows"].values, sub["kgkm"].values)
        pts = sorted((float(sub["overflows"][i]), float(sub["kgkm"][i])) for i in front)
        sx, sy = [pts[0][0]], [pts[0][1]]
        for j in range(1, len(pts)):
            sx += [pts[j][0], pts[j][0]]
            sy += [pts[j - 1][1], pts[j][1]]
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                name=f"Pareto — {scenario_label(s)}",
                line=dict(width=2, dash="dash"),
                hoverinfo="skip",
            )
        )
    n_line_traces = len(fig.data) - n_scatter_traces
    for s in scenarios:
        sub = _scen_sub(df, s).reset_index(drop=True)
        if sub.empty:
            continue
        front_rows = sub.iloc[sorted(pareto_indices(sub["overflows"].values, sub["kgkm"].values))]
        sym_base = region_syms[(s["city"], s["N"])]
        fig.add_trace(
            go.Scatter(
                x=front_rows["overflows"],
                y=front_rows["kgkm"],
                mode="markers",
                name=f"★ Pareto — {scenario_label(s)}",
                marker=dict(
                    symbol=[sym_base if r.improver == improvers[0] else f"{sym_base}-open" for r in front_rows.itertuples()],
                    color=[vcolors[r.variant] for r in front_rows.itertuples()],
                    size=13,
                    opacity=1.0,
                ),
                text=[
                    f"Constructor: {r.constructor}<br>Selection: {r.variant}<br>"
                    f"Scenario: {scenario_label(s)}<br>Improver: {r.improver}<br>"
                    f"Overflows: {r.overflows:.1f}<br>kg/km: {r.kgkm:.3f}<br><b>★ Pareto optimal</b>"
                    for r in front_rows.itertuples()
                ],
                hovertemplate="%{text}<extra></extra>",
                visible=False,
            )
        )
    n_front_traces = len(fig.data) - n_scatter_traces - n_line_traces
    all_vis = [True] * n_scatter_traces + [True] * n_line_traces + [False] * n_front_traces
    front_vis = [False] * n_scatter_traces + [True] * n_line_traces + [True] * n_front_traces
    fig.update_layout(
        title=dict(text=f"Overflow vs Efficiency — All Runs ({ctx['n_days']} days; hover for details)", x=0.5, xanchor="center"),
        xaxis_title=f"Overflows ({ctx['n_days']} days)",
        yaxis_title="kg/km",
        template=template,
        height=750,
        hovermode="closest",
        margin=dict(t=80, b=90),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.5,
                y=-0.08,
                xanchor="center",
                yanchor="top",
                buttons=[
                    dict(label="All Points", method="update", args=[{"visible": all_vis}]),
                    dict(label="Pareto Front Only", method="update", args=[{"visible": front_vis}]),
                ],
                showactive=True,
                bgcolor=theme["button_bg"],
                bordercolor=theme["button_border"],
                font=dict(color=theme["fg"], size=13),
                pad=dict(l=10, r=10, t=5, b=5),
            )
        ],
    )
    btn_js = load_js(
        "pareto_buttons.js",
        div_id="pareto-scatter",
        btn_bg=theme["button_bg"],
        btn_border=theme["button_border"],
        btn_fg=theme["fg"],
    )
    p = out_dir / "pareto_scatter_interactive.html"
    fig.write_html(str(p), include_plotlyjs="cdn", div_id="pareto-scatter", post_script=btn_js)
    paths["pareto"] = p
    print(f"  Saved: {p.name}")

    # strategy_bubble_interactive ───────────────────────────────────────────────
    agg = (
        dfm.groupby(["city", "N", "dist", "strategy", "improver"])[["overflows", "kgkm", "km", "profit"]]
        .mean()
        .reset_index()
    )
    fig2 = go.Figure()
    leg_seen: set = set()
    for strat in ctx["strategies"]:
        for s in scenarios:
            sub2 = _scen_sub(agg[agg.strategy == strat], s)
            if sub2.empty:
                continue
            sym_base = region_syms[(s["city"], s["N"])]
            key = (strat, s["dist"])
            show = key not in leg_seen
            leg_seen.add(key)
            fig2.add_trace(
                go.Scatter(
                    x=sub2["overflows"],
                    y=sub2["kgkm"],
                    mode="markers",
                    name=f"{strat} ({s['dist'][:3]})",
                    legendgroup=f"{strat}_{s['dist']}",
                    showlegend=show,
                    marker=dict(
                        color=META["strategy_colors"].get(strat, "gray"),
                        symbol=[
                            sym_base if r.improver == improvers[0] else f"{sym_base}-open" for r in sub2.itertuples()
                        ],
                        size=(sub2["N"] / 15 + 5).tolist(),
                        opacity=0.85,
                        line=dict(width=1.5, color=theme["fg"]),
                    ),
                    text=[
                        f"Strategy: {r.strategy}<br>Scenario: {scenario_label(s)}<br>"
                        f"Improver: {r.improver}<br>Mean overflows: {r.overflows:.2f}<br>"
                        f"Mean kg/km: {r.kgkm:.3f}"
                        for r in sub2.itertuples()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    fig2.update_layout(
        title=dict(
            text="Strategy Trade-off Bubble Chart (bubble size ∝ N; open marker = second improver)",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Mean overflows",
        yaxis_title="Mean kg/km",
        template=template,
        height=650,
        hovermode="closest",
        margin=dict(t=90),
    )
    p2 = out_dir / "strategy_bubble_interactive.html"
    fig2.write_html(str(p2), include_plotlyjs="cdn")
    paths["bubble"] = p2
    print(f"  Saved: {p2.name}")

    # policy_heatmap_interactive ────────────────────────────────────────────────
    rows = _policy_rows(df)
    row_labels = [f"{v} · {c} · {i}" for v, c, i in rows]
    col_labels = [scenario_label(s) for s in scenarios]
    mats = {}
    for metric in ["overflows", "kgkm"]:
        mat = np.full((len(rows), len(scenarios)), np.nan)
        for ri, (v, c, i) in enumerate(rows):
            sub_p = df[(df.variant == v) & (df.constructor == c) & (df.improver == i)]
            for ci, s in enumerate(scenarios):
                vals = _scen_sub(sub_p, s)[metric]
                if len(vals):
                    mat[ri, ci] = vals.mean()
        mats[metric] = mat
    figH = go.Figure()
    figH.add_trace(
        go.Heatmap(
            z=mats["overflows"],
            x=col_labels,
            y=row_labels,
            colorscale="RdYlGn_r" if True else "RdYlGn",
            hovertemplate="Policy: %{y}<br>Scenario: %{x}<br>Value: %{z:.2f}<extra></extra>",
        )
    )
    figH.update_layout(
        title=dict(text="Policy × Scenario Heatmap", x=0.5, xanchor="center"),
        template=template,
        height=max(700, 22 * len(rows)),
        margin=dict(b=90, l=240),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.5,
                y=-0.05,
                xanchor="center",
                yanchor="top",
                buttons=[
                    dict(
                        label="Overflows",
                        method="restyle",
                        args=[{"z": [mats["overflows"].tolist()], "colorscale": "RdYlGn_r"}],
                    ),
                    dict(
                        label="kg/km",
                        method="restyle",
                        args=[{"z": [mats["kgkm"].tolist()], "colorscale": "RdYlGn"}],
                    ),
                ],
                showactive=True,
                bgcolor=theme["button_bg"],
                bordercolor=theme["button_border"],
                font=dict(color=theme["fg"], size=12),
            )
        ],
    )
    p3 = out_dir / "policy_heatmap_interactive.html"
    figH.write_html(str(p3), include_plotlyjs="cdn")
    paths["heatmap"] = p3
    print(f"  Saved: {p3.name}")
    return paths


# ── Horizon comparison figures ─────────────────────────────────────────────────


def gen_horizon_comparison(horizons: list[dict], ctx: dict, out_dir: Path) -> None:
    """Cross-horizon comparison charts (one bar colour per horizon)."""
    scenarios, strategies = ctx["scenarios"], ctx["strategies"]
    configs = [(s, strat) for s in scenarios for strat in strategies]
    col_labels = [f"{region_label(s['city'], s['N'])}\n{s['dist'][:3]}\n{strat}" for s, strat in configs]
    hcolors = META["horizon_colors"]

    def _means(dfm: pd.DataFrame, metric: str) -> np.ndarray:
        out = []
        for s, strat in configs:
            sub = _scen_sub(dfm[dfm.strategy == strat], s)[metric]
            out.append(sub.mean() if len(sub) else np.nan)
        return np.array(out)

    for metric, fname, ylabel in [
        ("overflows", "horizon_overflow_comparison.png", "Mean overflows"),
        ("kgkm", "horizon_kgkm_comparison.png", "Mean kg/km"),
    ]:
        fig, ax = plt.subplots(figsize=(max(18, 0.9 * len(configs)), 8))
        fig.suptitle(f"Horizon Comparison: {ylabel} by Configuration", fontsize=14, fontweight="bold")
        x = np.arange(len(configs), dtype=float)
        n_h = len(horizons)
        w = 0.8 / n_h
        for hi, h in enumerate(horizons):
            vals = _means(h["dfm"], metric)
            ax.bar(
                x + (hi - (n_h - 1) / 2) * w,
                vals,
                w * 0.9,
                color=hcolors[hi % len(hcolors)],
                alpha=0.85,
                label=f"{h['days']} days",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10)
        plt.tight_layout()
        savefig(fig, out_dir / fname)

    # relative delta between the shortest and longest horizons
    first, last = horizons[0], horizons[-1]
    m_a = _means(first["dfm"], "overflows")
    m_b = _means(last["dfm"], "overflows")
    delta = np.where(m_a > 0, (m_b - m_a) / m_a * 100, np.nan)
    fig, ax = plt.subplots(figsize=(max(18, 0.9 * len(configs)), 8))
    fig.suptitle(
        f"Relative Change in Overflows: {last['days']}d vs {first['days']}d "
        f"(({last['days']}d−{first['days']}d)/{first['days']}d %)",
        fontsize=14,
        fontweight="bold",
    )
    x = np.arange(len(configs))
    valid = ~np.isnan(delta)
    colors = ["#e05c5c" if v > 0 else "#5cb85c" for v in delta[valid]]
    ax.bar(x[valid], delta[valid], color=colors, alpha=0.85)
    ax.axhline(0, color=ctx["theme"]["fg"], linewidth=0.8, linestyle="--")
    ax.set_xticks(x[valid])
    ax.set_xticklabels(np.array(col_labels)[valid], fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Δ overflows (%)", fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "horizon_overflow_delta.png")

    # constructor rankings across horizons
    all_cons = sorted(set(itertools.chain.from_iterable(h["dfm"]["constructor"].unique() for h in horizons)))
    metrics_rank = ["overflows", "kgkm", "km", "profit"]
    rank_asc = [True, False, True, False]

    def _avg_ranks(dfm: pd.DataFrame) -> dict[str, float]:
        ranks: dict[str, list] = {c: [] for c in all_cons}
        for metric, asc in zip(metrics_rank, rank_asc, strict=True):
            for _, grp in dfm.groupby(["city", "N", "dist", "strategy", "improver"]):
                gi = grp.set_index("constructor")
                r = gi[metric].rank(ascending=asc, method="average")
                for c in all_cons:
                    if c in r.index:
                        ranks[c].append(r[c])
        return {c: float(np.mean(v)) if v else np.nan for c, v in ranks.items()}

    fig, ax = plt.subplots(figsize=(max(12, 1.8 * len(all_cons)), 8))
    fig.suptitle("Constructor Average Ranking Across Horizons", fontsize=14, fontweight="bold")
    x = np.arange(len(all_cons), dtype=float)
    n_h = len(horizons)
    w = 0.8 / n_h
    for hi, h in enumerate(horizons):
        r = _avg_ranks(h["dfm"])
        ax.bar(
            x + (hi - (n_h - 1) / 2) * w,
            [r.get(c, np.nan) for c in all_cons],
            w * 0.9,
            color=hcolors[hi % len(hcolors)],
            alpha=0.85,
            label=f"{h['days']} days",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(all_cons, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Average rank (lower = better)", fontsize=11)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig(fig, out_dir / "horizon_constructor_ranking.png")


# ── Table builders ─────────────────────────────────────────────────────────────


def build_pareto_front_table(df: pd.DataFrame, ctx: dict) -> str:
    """One row per (selection variant, constructor, improver) on any scenario's Pareto front."""
    pareto_rows = []
    for s in ctx["scenarios"]:
        sub = _scen_sub(df, s).reset_index(drop=True)
        if sub.empty:
            continue
        for idx in pareto_indices(sub["overflows"].values, sub["kgkm"].values):
            row = sub.iloc[idx].copy()
            row["_scenario"] = scenario_label(s)
            pareto_rows.append(row)
    if not pareto_rows:
        return "_No Pareto-front data available._"
    pf = pd.DataFrame(pareto_rows)
    table_rows = []
    for (sel, con, imp), grp in pf.groupby(["variant", "constructor", "improver"]):
        scen = sorted(grp["_scenario"].unique())
        table_rows.append(
            {
                "sel": sel,
                "con": con,
                "imp": imp,
                "ov": grp["overflows"].mean(),
                "eff": grp["kgkm"].mean(),
                "scenarios": ", ".join(scen),
                "n": len(scen),
            }
        )
    table_rows.sort(key=lambda r: (-r["n"], r["sel"], r["con"], r["imp"]))
    lines = [
        "| Selection | Constructor | Improver | Overflows | kg/km | Pareto-Front Scenarios |",
        "|-----------|-------------|----------|----------:|------:|------------------------|",
    ]
    for r in table_rows:
        lines.append(f"| {r['sel']} | {r['con']} | {r['imp']} | {r['ov']:.1f} | {r['eff']:.3f} | {r['scenarios']} |")
    return "\n".join(lines)


def build_kpi_table(dfm: pd.DataFrame, ctx: dict, metric: str, fmt: str) -> str:
    """Combined min/max/mean table with one column group per improver."""
    improvers = ctx["improvers"]
    header = "| Config |" + "".join(f" {i} Min | {i} Max | {i} Mean |" for i in improvers)
    sep = "|--------|" + "-----|-----|------|" * len(improvers)
    lines = [header, sep]
    for s in ctx["scenarios"]:
        for strat in ctx["strategies"]:
            cells = []
            any_data = False
            for imp in improvers:
                sub = _scen_sub(dfm[(dfm.improver == imp) & (dfm.strategy == strat)], s)[metric]
                if len(sub):
                    any_data = True
                    cells.append(f" {sub.min():{fmt}} | {sub.max():{fmt}} | {sub.mean():{fmt}} |")
                else:
                    cells.append(" — | — | — |")
            if any_data:
                lines.append(f"| {scenario_label(s)} / {strat} |" + "".join(cells))
    return "\n".join(lines)


def build_strategy_best(dfm: pd.DataFrame, ctx: dict) -> str:
    """Best constructor per strategy and scenario (improvers pooled)."""
    lines = []
    for strat in ctx["strategies"]:
        lines.append(f"#### {strat} ({META['strategy_names'].get(strat, strat)})")
        lines.append("")
        for s in ctx["scenarios"]:
            sub = _scen_sub(dfm[dfm.strategy == strat], s)
            if sub.empty:
                continue
            agg = sub.groupby("constructor")[["overflows", "kgkm"]].mean()
            best_ov = agg["overflows"].idxmin()
            best_eff = agg["kgkm"].idxmax()
            lines.append(
                f"**{scenario_label(s)}:** best overflow: **{best_ov}** ({agg['overflows'].min():.1f}); "
                f"best efficiency: **{best_eff}** ({agg['kgkm'].max():.3f} kg/km)."
            )
            lines.append("")
        lines.append(PLACEHOLDER)
        lines.append("")
    return "\n".join(lines)


# ── Orchestration ──────────────────────────────────────────────────────────────


def gen_horizon_figures(df: pd.DataFrame, dfm: pd.DataFrame, ctx: dict, figures_dir: Path, private_dir: Path) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    charts = ctx["charts"]
    interactive: dict[str, Path] = {}

    def _on(name: str) -> bool:
        return charts.get(name, {}).get("enabled", True)

    if _on("pareto_scatter"):
        gen_pareto_scatter(df, ctx, figures_dir)
    if _on("overflow_bar"):
        gen_kpi_bar(dfm, "overflows", "Overflow Count", "overflow_by_config", ctx, figures_dir,
                    charts.get("overflow_bar", {}))
    if _on("kgkm_bar"):
        gen_kpi_bar(dfm, "kgkm", "kg/km Efficiency", "kgkm_by_config", ctx, figures_dir, charts.get("kgkm_bar", {}))
    if _on("km_violin"):
        gen_km_violin(df, ctx, figures_dir)
    if _on("policy_scenario_heatmap"):
        gen_policy_scenario_heatmap(df, ctx, figures_dir)
    if _on("scenario_constructor_heatmap"):
        gen_scenario_constructor_heatmap(dfm, ctx, figures_dir)
    if _on("strategy_bubble"):
        gen_strategy_bubble(dfm, ctx, figures_dir)
    if _on("improver_bubble"):
        gen_improver_bubble(dfm, ctx, figures_dir)
    if _on("constructor_ranking"):
        gen_constructor_ranking(dfm, ctx, figures_dir)
    if _on("radar"):
        gen_radar(dfm, ctx, figures_dir)
    if _on("improver_delta") and len(ctx["improvers"]) > 1:
        gen_improver_delta(dfm, ctx, figures_dir)
    if _on("interactive"):
        interactive = gen_interactive_html(df, dfm, ctx, private_dir)
    return interactive


def build_context(df: pd.DataFrame, config: dict, theme: dict, n_days: int) -> dict:
    scenarios = config.get("scenarios") or detect_scenarios(df)
    scenarios = [{"city": s["city"], "N": int(s["N"]), "dist": s["dist"]} for s in scenarios]
    scenarios = [s for s in scenarios if len(_scen_sub(df, s))]
    regions = []
    for s in scenarios:
        r = (s["city"], s["N"])
        if r not in regions:
            regions.append(r)
    pol = config.get("policies") or {}
    return {
        "scenarios": scenarios,
        "regions": regions,
        "dists": sorted({s["dist"] for s in scenarios}),
        "strategies": pol.get("strategies") or sorted(df["strategy"].unique()),
        "constructors": pol.get("constructors") or sorted(df["constructor"].unique()),
        "improvers": pol.get("improvers") or sorted(df["improver"].unique(), reverse=True),
        "charts": config.get("charts", {}),
        "theme": theme,
        "pareto_points": config.get("pareto_points"),
        "n_days": n_days,
    }


def _to_rel(p: Path) -> str:
    s = str(p)
    return s.replace("public/", "", 1) if s.startswith("public/") else s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="Analysis config JSON (defaults to json/simulation_analysis_config.json)")
    p.add_argument("--theme", choices=["dark", "light"], default=None, help="Chart theme override")
    p.add_argument("--pareto-points", choices=["all", "front"], default=None,
                   help="Plot all points or only the Pareto-front points in the static Pareto charts")
    p.add_argument("--horizon", action="append", default=None, metavar="DAYS=CSV",
                   help="Horizon spec (repeatable), e.g. --horizon 30=path/to.csv; overrides config horizons")
    p.add_argument("--scenarios", default=None,
                   help="Scenario filter: 'City:N:Dist;City:N:Dist;...' (e.g. 'Rio Maior:100:Gamma-3')")
    p.add_argument("--strategies", default=None, help="Comma-separated selection strategies to include")
    p.add_argument("--constructors", default=None, help="Comma-separated route constructors to include")
    p.add_argument("--improvers", default=None, help="Comma-separated route improvers to include")
    p.add_argument("--acceptance", default=None, help="Comma-separated acceptance criteria to include")
    p.add_argument("--out-md", default=None)
    p.add_argument("--figures-dir", default=None)
    p.add_argument("--private-dir", default=None)
    p.add_argument("--force", action="store_true", help="Overwrite existing markdown")
    p.add_argument("--figures-only", action="store_true", help="Generate figures but do not write markdown")
    # merged gen_simulation_csv mode
    p.add_argument("--parse-output", action="store_true", help="Parse a raw output tree into a summary CSV and exit")
    p.add_argument("--output-dir", default="assets/output/90days", help="Root of the raw simulation output tree")
    p.add_argument("--out-csv", default="public/global/simulation/simulation_summary_90d.csv",
                   help="Destination CSV path (with --parse-output)")
    return p.parse_args()


def run_parse_output(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    out_csv = Path(args.out_csv)
    if not output_dir.is_dir():
        raise SystemExit(f"Output dir not found: {output_dir}")
    print(f"Parsing: {output_dir}")
    df = parse_output_dir(output_dir)
    if df.empty:
        raise SystemExit("No log files found — check the output directory structure.")
    print(f"  Rows: {len(df)}")
    for col in ["city", "dist", "strategy", "improver", "constructor"]:
        print(f"  {col.capitalize()}s: {sorted(df[col].unique())}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Written: {out_csv}")


def main() -> None:  # noqa: C901
    args = parse_args()
    if args.parse_output:
        run_parse_output(args)
        return

    config = load_json("simulation_analysis_config.json")
    if args.config:
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.theme:
        config["theme"] = args.theme
    if args.pareto_points:
        config["pareto_points"] = args.pareto_points
    if args.horizon:
        config["horizons"] = []
        for spec in args.horizon:
            days, _, csv = spec.partition("=")
            config["horizons"].append({"days": int(days), "csv": csv})
    if args.scenarios:
        config["scenarios"] = []
        for part in args.scenarios.split(";"):
            city, n, dist = part.split(":")
            config["scenarios"].append({"city": city.strip(), "N": int(n), "dist": dist.strip()})
    pol = config.setdefault("policies", {})
    for key, val in [
        ("strategies", args.strategies),
        ("constructors", args.constructors),
        ("improvers", args.improvers),
        ("acceptance", args.acceptance),
    ]:
        if val:
            pol[key] = [v.strip() for v in val.split(",")]

    theme = load_theme(config.get("theme", "dark"))
    plt.style.use(theme["mplstyle_path"])

    out_md = Path(args.out_md or config["out_md"])
    figures_base = Path(args.figures_dir or config["figures_dir"])
    private_base = Path(args.private_dir or config["private_dir"])

    # ── Load all horizons (sorted smallest → largest) ───────────────────────────
    horizon_specs = sorted(config["horizons"], key=lambda h: h["days"])
    horizons: list[dict] = []
    for spec in horizon_specs:
        days = int(spec["days"])
        csv_path = Path(spec["csv"]) if spec.get("csv") else None
        if (csv_path is None or not csv_path.exists()) and spec.get("output_dir"):
            print(f"Parsing raw output for {days}d: {spec['output_dir']}")
            df = parse_output_dir(Path(spec["output_dir"]))
            if csv_path:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                print(f"  Written: {csv_path}")
            for col in ("cf", "sl_var", "acceptance"):
                df[col] = df[col].fillna("")
            df["variant"] = df.apply(variant_label, axis=1)
        elif csv_path and csv_path.exists():
            print(f"Reading {days}d CSV: {csv_path}")
            df = load_horizon_csv(csv_path)
        else:
            print(f"  [WARN] No data source for horizon {days}d — skipping")
            continue
        df = filter_data(df, config)
        if df.empty:
            print(f"  [WARN] Horizon {days}d has no rows after filtering — skipping")
            continue
        horizons.append({"days": days, "df": df, "dfm": aggregate(df), "csv": csv_path})
    if not horizons:
        raise SystemExit("No horizon data available.")

    multi = len(horizons) > 1
    suffix = lambda d: f"{d}d"  # noqa: E731

    # ── Figures ─────────────────────────────────────────────────────────────────
    for h in horizons:
        ctx = build_context(h["df"], config, theme, h["days"])
        h["ctx"] = ctx
        fig_dir = figures_base / suffix(h["days"]) if multi else figures_base
        priv_dir = private_base / suffix(h["days"]) if multi else private_base
        print(f"\nGenerating {h['days']}d figures → {fig_dir}")
        h["interactive"] = gen_horizon_figures(h["df"], h["dfm"], ctx, fig_dir, priv_dir)
        h["figures_dir"], h["private_dir"] = fig_dir, priv_dir

    cmp_dir = None
    if multi:
        cmp_dir = figures_base / "compare"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating horizon comparison figures → {cmp_dir}")
        # comparison uses the scenario/strategy space of the largest horizon's ctx
        gen_horizon_comparison(horizons, horizons[-1]["ctx"], cmp_dir)

    if args.figures_only:
        print("--figures-only: skipping markdown generation")
        return
    if out_md.exists() and not args.force:
        print(f"\n{out_md} already exists. Use --force to regenerate. Skipping markdown write.")
        return

    # ── Markdown context ────────────────────────────────────────────────────────
    ctx0 = horizons[0]["ctx"]
    all_scen = ctx0["scenarios"]
    strategies, improvers, constructors = ctx0["strategies"], ctx0["improvers"], ctx0["constructors"]
    acceptance = sorted({a for h in horizons for a in h["df"]["acceptance"].unique() if a})

    toc_items = ["1. [Experimental Setup](#1-experimental-setup)"]
    h_ctxs = []
    for i, h in enumerate(horizons, start=2):
        ctx = h["ctx"]
        anchor = f"{i}-{h['days']}-day-horizon-results"
        toc_items.append(f"{i}. [{h['days']}-Day Horizon Results](#{anchor})")
        h_ctxs.append(
            {
                "days": h["days"],
                "section_num": i,
                "n_logs": len(h["df"]),
                "constructors": ctx["constructors"],
                "figures_rel": _to_rel(h["figures_dir"]),
                "private_rel": _to_rel(h["private_dir"]),
                "pareto_table": build_pareto_front_table(h["df"], ctx),
                "overflow_table": build_kpi_table(h["dfm"], ctx, "overflows", ".1f"),
                "efficiency_table": build_kpi_table(h["dfm"], ctx, "kgkm", ".2f"),
                "km_table": build_kpi_table(h["dfm"], ctx, "km", ".0f"),
                "strategy_best": build_strategy_best(h["dfm"], ctx),
                "has_pareto_log": (h["figures_dir"] / "pareto_scatter_log.png").exists(),
                "has_overflow_log": (h["figures_dir"] / "overflow_by_config_log.png").exists(),
                "has_bubble_log": (h["figures_dir"] / "strategy_bubble_log.png").exists(),
                "has_improver_delta": (h["figures_dir"] / "improver_delta.png").exists(),
                "interactive": {k: True for k in h["interactive"]},
            }
        )
    comparison = None
    if multi:
        sec = len(horizons) + 2
        label = " vs ".join(f"{h['days']}d" for h in horizons)
        toc_items.append(f"{sec}. [Horizon Comparison ({label})](#{sec}-horizon-comparison-{label.replace(' ', '-').lower()})")
        comparison = {
            "section_num": sec,
            "label": label,
            "figures_rel": _to_rel(cmp_dir),
            "first_days": horizons[0]["days"],
            "last_days": horizons[-1]["days"],
        }

    scope = (
        f"{' and '.join(str(h['days']) + '-day' for h in horizons)} simulation runs across "
        f"{len(ctx0['regions'])} region/network configurations × {len(ctx0['dists'])} distributions × "
        f"{len(strategies)} selection strategies × {len(improvers)} route improvers × "
        f"{len(constructors)} route constructors"
    )
    md = render_template(
        "simulation_analysis.md.j2",
        scope=scope,
        horizons=h_ctxs,
        horizon_days=[h["days"] for h in horizons],
        scenario_str=", ".join(scenario_label(s) for s in all_scen),
        scenario_regions=", ".join(f"{region_label(c, n)} (N={n})" for c, n in ctx0["regions"]),
        dists=ctx0["dists"],
        strategies=strategies,
        improvers=improvers,
        constructors=constructors,
        acceptance=acceptance,
        comparison=comparison,
        toc="\n".join(toc_items),
        placeholder=PLACEHOLDER,
        figures_base_rel=_to_rel(figures_base),
        csv_list=", ".join(f"`{h['csv']}`" for h in horizons if h["csv"]),
    )
    md = finalize_markdown(md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"\nWritten: {out_md} ({len(md)} chars)")


if __name__ == "__main__":
    main()

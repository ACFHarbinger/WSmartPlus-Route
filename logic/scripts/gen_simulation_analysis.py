"""
Generate the simulation analysis markdown backbone and all associated figures.

Reads simulation results from the summary CSV, auto-detects the experiment
dimensions (cities, distributions, strategies, improvers, constructors), generates
all figures (PNG + interactive HTML), and writes a structured markdown file with
results tables and analysis placeholders ready for editing.

The output markdown is idempotent: running the script twice on the same data
produces the same file (placeholders are not overwritten if the file already
exists and user content has been added — add --force to regenerate).

Usage
-----
    uv run python logic/scripts/gen_simulation_analysis.py
    uv run python logic/scripts/gen_simulation_analysis.py \\
        --csv public/global/simulation/simulation_summary.csv \\
        --out-md public/simulation_analysis.md \\
        --figures-dir public/figures/simulation \\
        --private-dir public/private/simulation
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Optional Plotly ────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Dark matplotlib style ──────────────────────────────────────────────────────
DARK_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2d2d4e",
    "grid.alpha": 0.5,
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#e0e0e0",
}
STRATEGY_COLORS = {"LA": "#4e88d9", "LM": "#e05c5c", "SL": "#5cb85c"}
CITY_COLORS = ["#4e88d9", "#2244aa", "#e08030", "#a030e0"]

PLACEHOLDER = "<!-- [ANALYSIS: Insert your observations here] -->"
ANALYSIS_MARKER = "<!-- [ANALYSIS:"


def city_label(city: str, N: int) -> str:
    return f"FFZ-{N}" if "Figueira" in city else f"RM-{N}"


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


# ── Figure generators ──────────────────────────────────────────────────────────

def gen_overflow_bar(dfm: pd.DataFrame, panels: list, out_dir: Path) -> tuple[Path, Path]:
    """4-panel overflow bar chart (linear + log)."""
    plt.rcParams.update(DARK_STYLE)
    configs = [
        (city, N, dist, strat)
        for city, N in [("Rio Maior", 100), ("Rio Maior", 170), ("Figueira da Foz", 350)]
        for dist in ["Gamma-3", "Empirical"]
        for strat in ["LA", "LM", "SL"]
    ]
    col_labels = [f"{city_label(c,N)}\n{d[:3]}\n{s}" for c, N, d, s in configs]

    def _make(log: bool) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(22, 14))
        title = "Overflow Count by Configuration (mean ± range across constructors)"
        if log:
            title += " — Log Scale"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for idx, (dist, imp) in enumerate(panels):
            ax = axes[idx // 2][idx % 2]
            ax.set_facecolor("#16213e")
            means, lo, hi = [], [], []
            for city, N, dist_v, strat in configs:
                if dist_v != dist:
                    means.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                    continue
                sub = dfm[
                    (dfm.city == city) & (dfm.N == N) & (dfm.dist == dist_v)
                    & (dfm.improver == imp) & (dfm.strategy == strat)
                ]["overflows"]
                if len(sub):
                    m = sub.mean()
                    means.append(m)
                    lo.append(m - sub.min())
                    hi.append(sub.max() - m)
                else:
                    means.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
            x = np.arange(len(configs))
            colors = [STRATEGY_COLORS[s] for _, _, _, s in configs]
            valid = ~np.isnan(means)
            ax.bar(x[valid], np.array(means)[valid], color=np.array(colors)[valid], alpha=0.85)
            for xi, (m_, l_, h_) in enumerate(zip(means, lo, hi, strict=True)):
                if not np.isnan(m_):
                    ax.errorbar(xi, m_, yerr=[[l_], [h_]], fmt="none",
                                color="#e0e0e0", capsize=3, linewidth=1.2)
            if log:
                ax.set_yscale("symlog", linthresh=1)
            ax.set_xticks(x[valid])
            ax.set_xticklabels(np.array(col_labels)[valid], fontsize=6, rotation=45, ha="right")
            ax.set_ylabel("Mean overflows (30 days)", fontsize=10)
            ax.set_title(f"{dist} ({imp})", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        patches = [mpatches.Patch(color=STRATEGY_COLORS[s], label=s) for s in ["LA","LM","SL"]]
        fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout()
        return fig

    p1 = out_dir / "overflow_all_configs.png"
    p2 = out_dir / "overflow_all_configs_log.png"
    savefig(_make(False), p1)
    savefig(_make(True), p2)
    return p1, p2


def gen_kgkm_bar(dfm: pd.DataFrame, panels: list, out_dir: Path) -> Path:
    """4-panel kg/km efficiency bar chart."""
    plt.rcParams.update(DARK_STYLE)
    configs = [
        (city, N, dist, strat)
        for city, N in [("Rio Maior", 100), ("Rio Maior", 170), ("Figueira da Foz", 350)]
        for dist in ["Gamma-3", "Empirical"]
        for strat in ["LA", "LM", "SL"]
    ]
    col_labels = [f"{city_label(c,N)}\n{d[:3]}\n{s}" for c, N, d, s in configs]
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig.suptitle("kg/km Efficiency by Configuration (mean ± range across constructors)", fontsize=14, fontweight="bold")
    for idx, (dist, imp) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor("#16213e")
        means, lo, hi = [], [], []
        for city, N, dist_v, strat in configs:
            if dist_v != dist:
                means.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
                continue
            sub = dfm[
                (dfm.city == city) & (dfm.N == N) & (dfm.dist == dist_v)
                & (dfm.improver == imp) & (dfm.strategy == strat)
            ]["kgkm"]
            if len(sub):
                m = sub.mean()
                means.append(m)
                lo.append(m - sub.min())
                hi.append(sub.max() - m)
            else:
                means.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
        x = np.arange(len(configs))
        colors = [STRATEGY_COLORS[s] for _, _, _, s in configs]
        valid = ~np.isnan(means)
        ax.bar(x[valid], np.array(means)[valid], color=np.array(colors)[valid], alpha=0.85)
        for xi, (m_, l_, h_) in enumerate(zip(means, lo, hi, strict=True)):
            if not np.isnan(m_):
                ax.errorbar(xi, m_, yerr=[[l_], [h_]], fmt="none",
                            color="#e0e0e0", capsize=3, linewidth=1.2)
        ax.set_xticks(x[valid])
        ax.set_xticklabels(np.array(col_labels)[valid], fontsize=6, rotation=45, ha="right")
        ax.set_ylabel("Mean kg/km efficiency", fontsize=10)
        ax.set_title(f"{dist} ({imp})", fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    patches = [mpatches.Patch(color=STRATEGY_COLORS[s], label=s) for s in ["LA","LM","SL"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    p = out_dir / "kgkm_all_configs.png"
    savefig(fig, p)
    return p


def gen_km_violin(dfm: pd.DataFrame, panels: list, out_dir: Path) -> Path:
    plt.rcParams.update(DARK_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig.suptitle("Vehicle Distance Distribution by Strategy and City", fontsize=14, fontweight="bold")
    for idx, (dist, imp) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor("#16213e")
        groups, labels = [], []
        for strat in ["LA", "LM", "SL"]:
            for city, N in [("Rio Maior", 100), ("Rio Maior", 170), ("Figueira da Foz", 350)]:
                grp = dfm[
                    (dfm.dist == dist) & (dfm.improver == imp)
                    & (dfm.strategy == strat) & (dfm.city == city) & (dfm.N == N)
                ]["km"].values
                groups.append(grp if len(grp) > 1 else np.array([grp[0], grp[0]] if len(grp) else [0, 0]))
                labels.append(f"{strat}\n{city_label(city, N)}")
        strat_flat = ["LA"] * 3 + ["LM"] * 3 + ["SL"] * 3
        parts = ax.violinplot(groups, positions=range(len(groups)), showmedians=True, showextrema=True)
        for body, strat in zip(parts["bodies"], strat_flat, strict=True):
            body.set_facecolor(STRATEGY_COLORS[strat])
            body.set_alpha(0.7)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[key].set_color("#a0a0c0" if key != "cmedians" else "#ffffff")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Total km (30 days)", fontsize=10)
        ax.set_title(f"Distance Distribution — {dist} ({imp})", fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    patches = [mpatches.Patch(color=STRATEGY_COLORS[s], label=s) for s in ["LA", "LM", "SL"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    p = out_dir / "km_violin.png"
    savefig(fig, p)
    return p


def gen_policy_heatmap(dfm: pd.DataFrame, panels: list, constructors: list, out_dir: Path) -> tuple[Path, Path, Path]:  # noqa: C901
    plt.rcParams.update(DARK_STYLE)
    all_cfgs = [
        (city, N, dist, strat)
        for city, N in [("Rio Maior", 100), ("Rio Maior", 170), ("Figueira da Foz", 350)]
        for dist in ["Gamma-3", "Empirical"]
        for strat in ["LA", "LM", "SL"]
    ]
    col_labels = [f"{city_label(c,N)}\n{d[:3]}\n{s}" for c, N, d, s in all_cfgs]

    def build_mat(metric: str, imp: str) -> np.ndarray:
        mat = np.full((len(constructors), len(all_cfgs)), np.nan)
        for ci, c in enumerate(constructors):
            for cfi, (city, N, dist, strat) in enumerate(all_cfgs):
                sub = dfm[
                    (dfm.city == city) & (dfm.N == N) & (dfm.dist == dist)
                    & (dfm.improver == imp) & (dfm.strategy == strat) & (dfm.constructor == c)
                ][metric]
                if len(sub):
                    mat[ci, cfi] = sub.values[0]
        return mat

    # Combined heatmap
    fig, axes = plt.subplots(2, 2, figsize=(28, 18))
    fig.suptitle("Policy Performance Heatmap", fontsize=14, fontweight="bold")
    for idx, (metric, imp, title, cmap) in enumerate([
        ("overflows", "FTSP", "Overflow Count — FTSP", "RdYlGn_r"),
        ("overflows", "CLS",  "Overflow Count — CLS",  "RdYlGn_r"),
        ("kgkm",      "FTSP", "kg/km Efficiency — FTSP","RdYlGn"),
        ("kgkm",      "CLS",  "kg/km Efficiency — CLS", "RdYlGn"),
    ]):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor("#16213e")
        mat = build_mat(metric, imp)
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(constructors)))
        ax.set_yticklabels(constructors, fontsize=9)
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    p1 = out_dir / "policy_config_heatmap.png"
    savefig(fig, p1)

    # Split by distribution
    dist_cfgs = {
        d: [(city, N, d, s)
            for city, N in [("Rio Maior", 100), ("Rio Maior", 170), ("Figueira da Foz", 350)]
            for s in ["LA", "LM", "SL"]]
        for d in ["Gamma-3", "Empirical"]
    }
    fig2, axes2 = plt.subplots(2, 4, figsize=(36, 14))
    fig2.suptitle("Policy Heatmap — Split by Distribution & Improver", fontsize=14, fontweight="bold")
    for col_i, (dist, imp) in enumerate([
        ("Gamma-3","FTSP"),("Empirical","FTSP"),("Gamma-3","CLS"),("Empirical","CLS")
    ]):
        cfgs = dist_cfgs[dist]
        clabels = [f"{city_label(c,N)}\n{s}" for c,N,d,s in cfgs]
        for row_i, (metric, cmap) in enumerate([("overflows","RdYlGn_r"),("kgkm","RdYlGn")]):
            ax = axes2[row_i][col_i]
            ax.set_facecolor("#16213e")
            mat = np.full((len(constructors), len(cfgs)), np.nan)
            for ci, c in enumerate(constructors):
                for cfi, (city, N, d, s) in enumerate(cfgs):
                    sub = dfm[
                        (dfm.city==city)&(dfm.N==N)&(dfm.dist==d)
                        &(dfm.improver==imp)&(dfm.strategy==s)&(dfm.constructor==c)
                    ][metric]
                    if len(sub):
                        mat[ci, cfi] = sub.values[0]
            im = ax.imshow(mat, aspect="auto", cmap=cmap)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(clabels)))
            ax.set_xticklabels(clabels, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(len(constructors)))
            ax.set_yticklabels(constructors if col_i == 0 else [], fontsize=8)
            if row_i == 0:
                ax.set_title(f"{dist}\n{imp}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    p2 = out_dir / "policy_config_heatmap_by_dist.png"
    savefig(fig2, p2)

    # Split by city/graph
    city_cfgs = {
        lbl: [(city, N, d, s) for d in ["Gamma-3","Empirical"] for s in ["LA","LM","SL"]]
        for lbl, (city, N) in [
            ("RM-100", ("Rio Maior", 100)),
            ("RM-170", ("Rio Maior", 170)),
            ("FFZ-350", ("Figueira da Foz", 350)),
        ]
    }
    col_order = [
        ("RM-100","FTSP"),("RM-100","CLS"),("RM-170","FTSP"),
        ("RM-170","CLS"),("FFZ-350","FTSP"),("FFZ-350","CLS"),
    ]
    fig3, axes3 = plt.subplots(2, 6, figsize=(42, 14))
    fig3.suptitle("Policy Heatmap — Split by City/N & Improver", fontsize=14, fontweight="bold")
    for col_i, (clabel, imp) in enumerate(col_order):
        cfgs = city_cfgs[clabel]
        clabels_x = [f"{d[:3]}\n{s}" for c,N,d,s in cfgs]
        city_nm = "Rio Maior" if "RM" in clabel else "Figueira da Foz"
        N_val = int(clabel.split("-")[1])
        for row_i, (metric, cmap) in enumerate([("overflows","RdYlGn_r"),("kgkm","RdYlGn")]):
            ax = axes3[row_i][col_i]
            ax.set_facecolor("#16213e")
            mat = np.full((len(constructors), len(cfgs)), np.nan)
            for ci, c in enumerate(constructors):
                for cfi, (_, _, dist_v, strat) in enumerate(cfgs):
                    sub = dfm[
                        (dfm.city==city_nm)&(N_val == dfm.N)&(dfm.dist==dist_v)
                        &(dfm.improver==imp)&(dfm.strategy==strat)&(dfm.constructor==c)
                    ][metric]
                    if len(sub):
                        mat[ci, cfi] = sub.values[0]
            im = ax.imshow(mat, aspect="auto", cmap=cmap)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(clabels_x)))
            ax.set_xticklabels(clabels_x, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(len(constructors)))
            ax.set_yticklabels(constructors if col_i == 0 else [], fontsize=8)
            if row_i == 0:
                ax.set_title(f"{clabel}\n{imp}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    p3 = out_dir / "policy_config_heatmap_by_graph.png"
    savefig(fig3, p3)
    return p1, p2, p3


def gen_pareto_scatter(dfm: pd.DataFrame, df_raw: pd.DataFrame, panels: list, out_dir: Path) -> tuple[Path, Path]: # noqa: C901
    plt.rcParams.update(DARK_STYLE)

    def get_color(row):
        s = row["strategy"]
        if s == "LA":
            return "#4e88d9"
        if s == "LM":
            return "#e05c5c" if str(row["cf"]) == "CF70" else "#e09020"
        return "#20b2aa" if str(row["sl_var"]) == "SL1" else "#20a020"

    def get_marker(row):
        if "Figueira" in str(row["city"]):
            return "D"
        return "o" if str(row["N"]) == "100" else "s"

    def pareto_front(xs, ys):
        pts = sorted(zip(xs, ys, strict=True), key=lambda p: (p[0], -p[1]))
        front, best = [], -np.inf
        for ov, eff in pts:
            if eff > best:
                front.append((ov, eff))
        return front

    def _make(log: bool) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(22, 16))
        title = "Overflow vs Efficiency — Pareto Front (FTSP & CLS)"
        if log:
            title += " — Log Scale"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for idx, (dist, imp) in enumerate(panels):
            ax = axes[idx // 2][idx % 2]
            ax.set_facecolor("#16213e")
            sub = df_raw[(df_raw.dist == dist) & (df_raw.improver == imp)]
            for _, row in sub.iterrows():
                ax.scatter(row["overflows"], row["kgkm"],
                           c=get_color(row), marker=get_marker(row), s=40, alpha=0.75, zorder=3)
            front = pareto_front(sub["overflows"].values, sub["kgkm"].values)
            if front:
                fx, fy = zip(*front, strict=True) # pyrefly: ignore [no-matching-overload]
                step_x, step_y = [fx[0]], [fy[0]]
                for i in range(1, len(fx)):
                    step_x += [fx[i], fx[i]]
                    step_y += [fy[i-1], fy[i]]
                ax.plot(step_x, step_y, "--", color="white", linewidth=2, alpha=0.9)
            if log:
                ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel("Overflows (30 days)", fontsize=10)
            ax.set_ylabel("Efficiency (kg/km)", fontsize=10)
            ax.set_title(f"Overflow vs Efficiency — {dist} ({imp})", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        leg = [
            mpatches.Patch(color="#4e88d9", label="LA"),
            mpatches.Patch(color="#e05c5c", label="LM-CF70"),
            mpatches.Patch(color="#e09020", label="LM-CF90"),
            mpatches.Patch(color="#20b2aa", label="SL-SL1"),
            mpatches.Patch(color="#20a020", label="SL-SL2"),
            plt.Line2D([0],[0], color="white", linestyle="--", linewidth=2, label="Pareto front"),
        ]
        shape_leg = [
            plt.scatter([],[],marker="o",c="gray",s=50,label="RM N=100"),
            plt.scatter([],[],marker="s",c="gray",s=50,label="RM N=170"),
            plt.scatter([],[],marker="D",c="gray",s=50,label="FFZ N=350"),
        ]
        fig.legend(handles=leg+shape_leg, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5,-0.02))
        plt.tight_layout()
        return fig

    p1 = out_dir / "overflow_efficiency_scatter_pareto.png"
    p2 = out_dir / "overflow_efficiency_scatter_pareto_log.png"
    savefig(_make(False), p1)
    savefig(_make(True), p2)
    return p1, p2


def gen_strategy_bubble(dfm: pd.DataFrame, panels: list, out_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(DARK_STYLE)
    SIZE = {100: 200, 170: 400, 350: 800}

    def _make(log: bool) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        title = "Strategy Trade-off Bubble Chart" + (" (log X)" if log else "")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for idx, (dist, imp) in enumerate(panels):
            ax = axes[idx // 2][idx % 2]
            ax.set_facecolor("#16213e")
            sub = dfm[(dfm.dist == dist) & (dfm.improver == imp)]
            agg = sub.groupby(["city","N","strategy"])[["overflows","kgkm"]].mean().reset_index()
            for _, row in agg.iterrows():
                marker = "D" if "Figueira" in row["city"] else ("o" if row["N"] == 100 else "s")
                ax.scatter(row["overflows"], row["kgkm"],
                           c=STRATEGY_COLORS[row["strategy"]], marker=marker,
                           s=SIZE.get(row["N"], 300), alpha=0.75,
                           edgecolors="white", linewidths=0.5)
                ax.annotate(city_label(row["city"], row["N"]), (row["overflows"], row["kgkm"]),
                            textcoords="offset points", xytext=(4, 3), fontsize=7)
            if log:
                ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel("Mean overflows (30 days)", fontsize=10)
            ax.set_ylabel("Mean kg/km efficiency", fontsize=10)
            ax.set_title(f"{dist} ({imp})", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        strat_patches = [mpatches.Patch(color=STRATEGY_COLORS[s], label=s) for s in ["LA","LM","SL"]]
        fig.legend(handles=strat_patches, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5,-0.01))
        plt.tight_layout()
        return fig

    p1 = out_dir / "strategy_bubble.png"
    p2 = out_dir / "strategy_bubble_log.png"
    savefig(_make(False), p1)
    savefig(_make(True), p2)
    return p1, p2


def gen_city_comparison(dfm: pd.DataFrame, panels: list, out_dir: Path) -> tuple[Path, Path, Path]:
    plt.rcParams.update(DARK_STYLE)
    city_order = [("Rio Maior", 100), ("Rio Maior", 170), ("Figueira da Foz", 350)]
    clabels = ["RM-100", "RM-170", "FFZ-350"]
    ccolors = ["#4e88d9", "#2244aa", "#e08030"]

    def _make_bar(metric: str, ylabel: str, log: bool) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(22, 14))
        title = f"City Comparison: {ylabel} by Selection Strategy" + (" (log)" if log else "")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        width = 0.25
        for idx, (dist, imp) in enumerate(panels):
            ax = axes[idx // 2][idx % 2]
            ax.set_facecolor("#16213e")
            sub = dfm[(dfm.dist == dist) & (dfm.improver == imp)]
            x = np.arange(3)  # LA, LM, SL
            for ci, (city, N) in enumerate(city_order):
                means, lo, hi = [], [], []
                for strat in ["LA", "LM", "SL"]:
                    grp = sub[(sub.city==city)&(sub.N==N)&(sub.strategy==strat)][metric]
                    if len(grp):
                        m = grp.mean()
                        means.append(m)
                        lo.append(m-grp.min())
                        hi.append(grp.max()-m)
                    else:
                        means.append(0)
                        lo.append(0)
                        hi.append(0)
                ax.bar(x + (ci-1)*width, means, width, label=clabels[ci],
                       color=ccolors[ci], alpha=0.85)
                ax.errorbar(x + (ci-1)*width, means, yerr=[lo, hi],
                            fmt="none", color="#e0e0e0", capsize=3)
            if log:
                ax.set_yscale("symlog", linthresh=1)
            ax.set_xticks(x)
            ax.set_xticklabels(["LA","LM","SL"], fontsize=11)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{dist} ({imp})", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
            ax.legend(fontsize=9)
        plt.tight_layout()
        return fig

    p1 = out_dir / "city_comparison_overflow.png"
    p2 = out_dir / "city_comparison_overflow_log.png"
    p3 = out_dir / "city_comparison_efficiency.png"
    savefig(_make_bar("overflows", "Mean overflows (30 days)", False), p1)
    savefig(_make_bar("overflows", "Mean overflows (30 days)", True), p2)
    savefig(_make_bar("kgkm", "Mean kg/km", False), p3)
    return p1, p2, p3


def gen_city_scaling(dfm: pd.DataFrame, panels: list, out_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(DARK_STYLE)
    Ns = [100, 170, 350]
    city_map = {100: "Rio Maior", 170: "Rio Maior", 350: "Figueira da Foz"}
    xlabels = ["RM-100\n(N=100)", "RM-170\n(N=170)", "FFZ-350\n(N=350)"]

    fig1, axes1 = plt.subplots(2, 2, figsize=(22, 14))
    fig1.suptitle("City Scaling Overview: N=100 → N=170 → N=350", fontsize=14, fontweight="bold")
    for idx, (dist, imp) in enumerate(panels):
        ax = axes1[idx // 2][idx % 2]
        ax.set_facecolor("#16213e")
        sub = dfm[(dfm.dist == dist) & (dfm.improver == imp)]
        for strat, color in STRATEGY_COLORS.items():
            ov_vals = []
            for N in Ns:
                grp = sub[(sub.city == city_map[N]) & (sub.N == N) & (sub.strategy == strat)]
                ov_vals.append(grp["overflows"].mean() if len(grp) else np.nan)
            ax.plot(Ns, ov_vals, "o-", color=color, label=strat, linewidth=2, markersize=8)
        ax.axvline(x=235, color="#a0a0c0", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(f"Overflows — {dist} ({imp})", fontsize=11)
        ax.set_ylabel("Mean overflows (30 days)", fontsize=10)
        ax.set_xticks(Ns)
        ax.set_xticklabels(xlabels)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
    plt.tight_layout()
    p1 = out_dir / "city_scaling_overview.png"
    savefig(fig1, p1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
    fig2.suptitle("Network Scaling: N=100 → N=170 (Rio Maior)", fontsize=14, fontweight="bold")
    for ax_i, dist in enumerate(["Gamma-3", "Empirical"]):
        ax = axes2[ax_i]
        ax.set_facecolor("#16213e")
        for imp, ls in [("FTSP", "o-"), ("CLS", "s--")]:
            sub = dfm[(dfm.dist == dist) & (dfm.improver == imp) & (dfm.city == "Rio Maior")]
            for strat, color in STRATEGY_COLORS.items():
                ov_vals = [sub[(sub.N == N) & (sub.strategy == strat)]["overflows"].mean()
                           if len(sub[(sub.N == N) & (sub.strategy == strat)]) else np.nan
                           for N in [100, 170]]
                ax.plot([100, 170], ov_vals, ls, color=color, linewidth=1.5, markersize=7, alpha=0.8,
                        label=f"{strat} {imp}")
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel("Mean overflows", fontsize=10)
        ax.set_title(f"{dist}", fontsize=11)
        ax.set_xticks([100, 170])
        ax.legend(fontsize=8, ncol=2)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    p2 = out_dir / "scaling_chart.png"
    savefig(fig2, p2)
    return p1, p2


def gen_constructor_ranking(dfm: pd.DataFrame, constructors: list, out_dir: Path) -> Path:
    plt.rcParams.update(DARK_STYLE)
    metrics = ["overflows", "kgkm", "km", "profit"]
    metric_labels = ["Overflows", "kg/km", "km", "Profit"]
    rank_asc = [True, False, True, False]
    colors = ["#4e88d9", "#e05c5c", "#5cb85c", "#f0a030"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Route Constructor Average Rankings — FTSP vs CLS", fontsize=14, fontweight="bold")
    for ax_i, imp in enumerate(["FTSP", "CLS"]):
        ax = axes[ax_i]
        ax.set_facecolor("#16213e")
        sub = dfm[dfm.improver == imp]
        mean_ranks: dict = {c: {m: [] for m in metrics} for c in constructors}
        for metric, asc in zip(metrics, rank_asc, strict=True):
            for _, grp in sub.groupby(["city","N","dist","strategy"]):
                grp_idx = grp.set_index("constructor")
                r = grp_idx[metric].rank(ascending=asc, method="average")
                for c in constructors:
                    if c in r.index:
                        mean_ranks[c][metric].append(r[c])
        x = np.arange(len(constructors))
        w = 0.2
        for mi, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors, strict=True)):
            vals = [np.mean(mean_ranks[c][metric]) if mean_ranks[c][metric] else 4.5
                    for c in constructors]
            ax.bar(x + mi*w - 1.5*w, vals, w, label=label, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(constructors, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Average rank (lower = better)", fontsize=11)
        ax.set_title(f"Route Constructor Average Rankings\n({imp}, all configs)", fontsize=12)
        ax.set_ylim(0, 8.5)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left")
    plt.tight_layout()
    p = out_dir / "constructor_ranking.png"
    savefig(fig, p)
    return p


def gen_radar(dfm: pd.DataFrame, key_constructors: list, out_dir: Path) -> Path:
    plt.rcParams.update(DARK_STYLE)
    metrics = ["overflows", "kgkm", "km", "profit"]
    axes_labels = ["Overflows\n(fewer ↓)", "kg/km\n(higher ↑)", "km\n(fewer ↓)", "Profit\n(higher ↑)"]
    invert = [True, False, True, False]
    constructor_colors = {"ACO_HH": "#00c8a0", "HGS": "#e05c5c", "BPC": "#a060e0", "SANS": "#a0a0a0"}

    scores = {}
    for c in key_constructors:
        sub = dfm[dfm.constructor == c]
        scores[c] = []
        for metric, inv in zip(metrics, invert, strict=True):
            all_vals = dfm[metric].values
            v = sub[metric].mean() if len(sub) else np.nanmean(all_vals) # pyrefly: ignore [no-matching-overload]
            mn, mx = np.nanmin(all_vals), np.nanmax(all_vals) # pyrefly: ignore [no-matching-overload]
            norm = (v - mn) / (mx - mn + 1e-9) if mx > mn else 0.5
            scores[c].append(1 - norm if inv else norm)

    N_axes = len(metrics)
    angles = [n / N_axes * 2 * np.pi for n in range(N_axes)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor("#16213e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=11, color="#e0e0e0")
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelcolor="#a0a0c0", labelsize=8)
    circle_labels = ["25%", "50%", "75%", "100%"]
    for r, lbl in zip([0.25, 0.5, 0.75, 1.0], circle_labels, strict=True):
        ax.plot(angles, [r] * (N_axes + 1), "--", color="#3a3a5e", linewidth=0.8)
        ax.text(0, r + 0.02, lbl, ha="center", va="bottom", fontsize=8, color="#6060a0")

    for c in key_constructors:
        vals = scores[c] + scores[c][:1]
        color = constructor_colors.get(c, "#ffffff")
        ax.plot(angles, vals, "o-", color=color, linewidth=2.5, markersize=5, label=c)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_title("Policy Performance Radar\n(normalised; outer = better)", fontsize=13, fontweight="bold",
                 pad=20, color="#e0e0e0")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)
    p = out_dir / "policy_radar_combined.png"
    savefig(fig, p)
    return p


def gen_ftsp_vs_cls(dfm: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(DARK_STYLE)
    configs = [
        (city, N, dist, strat)
        for city, N in [("Rio Maior", 100), ("Rio Maior", 170), ("Figueira da Foz", 350)]
        for dist in ["Gamma-3", "Empirical"]
        for strat in ["LA", "LM", "SL"]
    ]

    # Comparison scatter
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig.suptitle("FTSP vs CLS Route Improver Comparison", fontsize=14, fontweight="bold")
    for row_i, (metric, _ylabel) in enumerate([("overflows","Overflows"),("kgkm","kg/km")]):
        for col_i, dist in enumerate(["Gamma-3","Empirical"]):
            ax = axes[row_i][col_i]
            ax.set_facecolor("#16213e")
            sub = dfm[dfm.dist == dist]
            for strat, color in STRATEGY_COLORS.items():
                ftsp = sub[(sub.improver=="FTSP")&(sub.strategy==strat)][metric].values
                cls = sub[(sub.improver=="CLS")&(sub.strategy==strat)][metric].values
                n = min(len(ftsp), len(cls))
                if n:
                    ax.scatter(ftsp[:n], cls[:n], c=color, s=40, alpha=0.7, label=strat)
            lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) if ax.get_xlim()[1] > 0 else 10
            ax.plot([0, lim], [0, lim], "--", color="#a0a0c0", linewidth=1)
            ax.set_xlabel("FTSP", fontsize=10)
            ax.set_ylabel("CLS", fontsize=10)
            ax.set_title(f"{metric} — {dist}", fontsize=11)
            ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.4)
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
    plt.tight_layout()
    p1 = out_dir / "ftsp_vs_cls_comparison.png"
    savefig(fig, p1)

    # Delta heatmap
    CONSTRUCTORS = dfm["constructor"].unique().tolist()
    col_labels = [f"{city_label(c,N)}\n{d[:3]}\n{s}" for c, N, d, s in configs]
    fig2, axes2 = plt.subplots(1, 2, figsize=(28, 10))
    fig2.suptitle("FTSP vs CLS Delta Heatmap (CLS - FTSP)", fontsize=14, fontweight="bold")
    for ax_i, (metric, cmap) in enumerate([("overflows","RdYlGn_r"),("kgkm","RdYlGn")]):
        ax = axes2[ax_i]
        ax.set_facecolor("#16213e")
        mat = np.full((len(CONSTRUCTORS), len(configs)), np.nan)
        for ci, c in enumerate(CONSTRUCTORS):
            for cfi, (city, N, dist, strat) in enumerate(configs):
                ftsp = dfm[(dfm.city==city)&(dfm.N==N)&(dfm.dist==dist)&(dfm.improver=="FTSP")
                           &(dfm.strategy==strat)&(dfm.constructor==c)][metric]
                cls = dfm[(dfm.city==city)&(dfm.N==N)&(dfm.dist==dist)&(dfm.improver=="CLS")
                          &(dfm.strategy==strat)&(dfm.constructor==c)][metric]
                if len(ftsp) and len(cls):
                    mat[ci, cfi] = cls.values[0] - ftsp.values[0]
        vmax = np.nanpercentile(np.abs(mat[~np.isnan(mat)]), 95) if not np.all(np.isnan(mat)) else 1
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label=f"Δ {metric}")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(CONSTRUCTORS)))
        ax.set_yticklabels(CONSTRUCTORS, fontsize=9)
        ax.set_title(f"Δ {metric}", fontsize=11)
    plt.tight_layout()
    p2 = out_dir / "ftsp_vs_cls_delta.png"
    savefig(fig2, p2)
    return p1, p2


def gen_interactive_html(df_raw: pd.DataFrame, dfm: pd.DataFrame, panels: list, out_dir: Path) -> dict[str, Path]: # noqa: C901
    if not HAS_PLOTLY:
        print("  [WARN] Plotly not available — skipping interactive HTML generation")
        return {}
    paths: dict[str, Path] = {}

    # pareto_scatter_interactive ─────────────────────────────────────────────
    # -- Per-point colour, label, and symbol helpers -------------------------
    def _pcol(strategy: str, cf, sl_var) -> str:
        """Return the 5-colour palette matching the static Pareto PNG."""
        if strategy == "LA":
            return "#4e88d9"
        if strategy == "LM":
            return "#e05c5c" if (pd.notna(cf) and str(cf) == "CF70") else "#e09020"
        return "#20b2aa" if (pd.notna(sl_var) and str(sl_var) == "SL1") else "#20a020"

    def _plabel(strategy: str, cf, sl_var) -> str:
        if strategy == "LA":
            return "LA"
        if strategy == "LM":
            tag = str(cf) if (pd.notna(cf) and str(cf) not in ("nan", "")) else ""
            return f"LM-{tag}" if tag else "LM"
        tag = str(sl_var) if (pd.notna(sl_var) and str(sl_var) not in ("nan", "")) else ""
        return f"SL-{tag}" if tag else "SL"

    def _psym(city: str, N, dist: str) -> str:
        """Base shape from city/N; filled=Empirical, open=Gamma-3."""
        base = "diamond" if "Figueira" in city else ("circle" if int(N) == 100 else "square")
        return base if dist == "Empirical" else f"{base}-open"

    # -- Pareto-front helpers ------------------------------------------------
    def _pareto_positions(xs: np.ndarray, ys: np.ndarray) -> set[int]:
        """Positional indices of non-dominated points (min overflows, max kgkm)."""
        indexed = sorted(enumerate(zip(xs, ys, strict=True)), key=lambda p: (p[1][0], -p[1][1]))
        idxs: list[int] = []
        best = -np.inf
        for orig_i, (_ov, eff) in indexed:
            if eff > best:
                idxs.append(orig_i)
                best = eff
        return set(idxs)

    def _step_xy(pts: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
        if not pts:
            return [], []
        sx, sy = [pts[0][0]], [pts[0][1]]
        for i in range(1, len(pts)):
            sx += [pts[i][0], pts[i][0]]
            sy += [pts[i - 1][1], pts[i][1]]
        return sx, sy

    # City/N → Pareto-front line colour (three distinct colours, dist → line dash)
    _city_line_color = {100: "#7799ff", 170: "#ffbb44", 350: "#55cc55"}
    _dist_dash = {"Empirical": "dash", "Gamma-3": "dot"}

    # ── Regular scatter traces: grouped by (dist × city/N × strategy variant)
    fig = go.Figure()
    all_dists = sorted(df_raw["dist"].unique())
    city_n_combos = sorted(df_raw[["city", "N"]].drop_duplicates().values.tolist(),
                           key=lambda r: (r[0], int(r[1])))
    legend_shown: set = set()

    for dist in all_dists:
        dist_short = "Emp" if dist == "Empirical" else "Gam"
        sub_dist = df_raw[df_raw.dist == dist]
        for city, N in city_n_combos:
            city_sub = sub_dist[(sub_dist.city == city) & (sub_dist.N == N)]
            if city_sub.empty:
                continue
            for _, sg in city_sub.groupby(["strategy", "cf", "sl_var"], dropna=False):
                if sg.empty:
                    continue
                r0 = sg.iloc[0]
                s, cf, slv = str(r0.strategy), r0.cf, r0.sl_var
                color = _pcol(s, cf, slv)
                label = _plabel(s, cf, slv)
                sym = _psym(city, N, dist)
                legend_key = (label, dist_short)
                show_leg = legend_key not in legend_shown
                if show_leg:
                    legend_shown.add(legend_key)
                fig.add_trace(go.Scatter(
                    x=sg["overflows"], y=sg["kgkm"],
                    mode="markers",
                    name=f"{label} ({dist_short})",
                    legendgroup=f"{label}_{dist_short}",
                    showlegend=show_leg,
                    marker=dict(symbol=sym, color=color, size=8, opacity=0.85,
                                line=dict(width=1.5, color=color)),
                    text=[
                        f"Constructor: {r.constructor}<br>Strategy: {_plabel(str(r.strategy), r.cf, r.sl_var)}<br>"
                        f"City: {city_label(r.city, int(r.N))}<br>"
                        f"Dist: {r.dist}<br>Improver: {r.improver}<br>"
                        f"Overflows: {r.overflows:.1f}<br>kg/km: {r.kgkm:.3f}<br>"
                        f"km: {r.km:.0f}<br>Profit: {r.profit:.0f}"
                        for r in sg.itertuples()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ))

    pareto_line_start = len(fig.data) # pyrefly: ignore [bad-argument-type]

    # ── Pareto-front step-line traces: one per (dist × city/N), always visible
    for dist in all_dists:
        dist_short = "Emp" if dist == "Empirical" else "Gam"
        for city, N in city_n_combos:
            sub = df_raw[(df_raw.dist == dist) & (df_raw.city == city) & (df_raw.N == N)]
            if sub.empty:
                continue
            front_idxs = _pareto_positions(sub["overflows"].values, sub["kgkm"].values)
            sorted_pts = sorted(
                [(float(sub.iloc[i]["overflows"]), float(sub.iloc[i]["kgkm"])) for i in front_idxs],
                key=lambda p: p[0],
            )
            sx, sy = _step_xy(sorted_pts)
            clabel = city_label(city, int(N))
            fig.add_trace(go.Scatter(
                x=sx, y=sy, mode="lines",
                name=f"Pareto — {dist_short} {clabel}",
                line=dict(color=_city_line_color.get(int(N), "#aaaaaa"), width=2,
                          dash=_dist_dash.get(dist, "dash")),
                hoverinfo="skip",
            ))

    pareto_scatter_start = len(fig.data) # pyrefly: ignore [bad-argument-type]

    # ── Pareto-only scatter traces: one per (dist × city/N), hidden by default
    # Each trace uses a single consistent symbol (correct city/N shape) so the
    # legend icon is unambiguous; per-point colour encodes the strategy variant.
    for dist in all_dists:
        dist_short = "Emp" if dist == "Empirical" else "Gam"
        for city, N in city_n_combos:
            sub = df_raw[(df_raw.dist == dist) & (df_raw.city == city) & (df_raw.N == N)]
            if sub.empty:
                continue
            front_idxs = _pareto_positions(sub["overflows"].values, sub["kgkm"].values)
            front_rows = sub.iloc[sorted(front_idxs, key=lambda i: sub.iloc[i]["overflows"])]
            sym = _psym(city, N, dist)
            colors = [_pcol(str(r.strategy), r.cf, r.sl_var) for r in front_rows.itertuples()]
            texts = [
                f"Constructor: {r.constructor}<br>Strategy: {_plabel(str(r.strategy), r.cf, r.sl_var)}<br>"
                f"City: {city_label(r.city, int(r.N))}<br>"
                f"Dist: {r.dist}<br>Improver: {r.improver}<br>"
                f"Overflows: {r.overflows:.1f}<br>kg/km: {r.kgkm:.3f}<br>"
                f"km: {r.km:.0f}<br>Profit: {r.profit:.0f}<br><b>★ Pareto optimal</b>"
                for r in front_rows.itertuples()
            ]
            clabel = city_label(city, int(N))
            fig.add_trace(go.Scatter(
                x=front_rows["overflows"].tolist(),
                y=front_rows["kgkm"].tolist(),
                mode="markers",
                name=f"★ Pareto — {dist_short} {clabel}",
                marker=dict(symbol=sym, color=colors, size=13, opacity=1.0,
                            line=dict(width=2, color=colors)),
                text=texts,
                hovertemplate="%{text}<extra></extra>",
                visible=False,
            ))

    n_lines   = pareto_scatter_start - pareto_line_start
    n_scatter = len(fig.data) - pareto_scatter_start # pyrefly: ignore [bad-argument-type]
    _all_vis   = [True]  * pareto_line_start + [True]  * n_lines + [False] * n_scatter
    _front_vis = [False] * pareto_line_start + [True]  * n_lines + [True]  * n_scatter

    fig.update_layout(
        title=dict(
            text=(
                "Overflow vs Efficiency — All Runs (hover for details)<br>"
                "<sup>Filled marker = Empirical · Open marker = Gamma-3 · "
                "Circle = RM-100 · Square = RM-170 · Diamond = FFZ-350</sup>"
            ),
            x=0.5, xanchor="center",
        ),
        xaxis_title="Overflows (30 days)", yaxis_title="kg/km",
        template="plotly_dark", height=750, hovermode="closest",
        margin=dict(t=80, b=90),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.5, y=-0.08, xanchor="center", yanchor="top",
            buttons=[
                dict(label="All Points", method="update",
                     args=[{"visible": _all_vis}]),
                dict(label="Pareto Front Only", method="update",
                     args=[{"visible": _front_vis}]),
            ],
            showactive=True,
            bgcolor="#1a2255",
            bordercolor="#4455aa",
            font=dict(color="white", size=13),
            pad=dict(l=10, r=10, t=5, b=5),
        )],
    )

    # Inject CSS (highest priority) to permanently override Plotly's white active-button fill.
    # CSS fill property with !important beats SVG presentation attributes Plotly sets on re-render.
    _btn_js = """
(function() {
    /* --- Persistent CSS override ------------------------------------------ */
    var s = document.createElement('style');
    s.textContent = [
        /* Force ALL buttons dark — overrides Plotly's white active-button fill */
        'g.updatemenu-button rect { fill: #1a2255 !important; stroke: #4455aa !important; }',
        'g.updatemenu-button text { fill: #ffffff !important; }',
        /* Active button gets bright blue */
        'g.updatemenu-button.ps-active rect { fill: #3355cc !important; }',
        /* Hover */
        'g.updatemenu-button.ps-hover rect { fill: #5577ee !important; }',
    ].join('\\n');
    document.head.appendChild(s);

    /* --- Active-button tracker -------------------------------------------- */
    var activeIdx = 0;  /* 0 = "All Points" is the default active button */

    function applyActive() {
        var gd = document.getElementById('pareto-scatter');
        if (!gd) return;
        var btns = Array.from(gd.querySelectorAll('g.updatemenu-button'));
        if (!btns.length) return;
        btns.forEach(function(btn, i) {
            btn.classList.toggle('ps-active', i === activeIdx);
            if (!btn._psHooked) {
                btn._psHooked = true;
                btn.addEventListener('mouseenter', function() { btn.classList.add('ps-hover'); });
                btn.addEventListener('mouseleave', function() { btn.classList.remove('ps-hover'); });
            }
        });
    }

    /* Wire Plotly button-click event to update active index */
    var _wire = function() {
        var gd = document.getElementById('pareto-scatter');
        if (!gd || !gd.on) { setTimeout(_wire, 300); return; }
        gd.on('plotly_buttonclicked', function(data) {
            /* data.active = the new active button index within the clicked menu */
            activeIdx = (typeof data.active === 'number') ? data.active : 0;
            setTimeout(applyActive, 60);
        });
        applyActive();
    };

    /* Run on load and after short delay (Plotly renders asynchronously) */
    setTimeout(_wire, 400);
    window.addEventListener('load', function() { setTimeout(_wire, 100); });
})();
"""
    p = out_dir / "pareto_scatter_interactive.html"
    fig.write_html(str(p), include_plotlyjs="cdn",
                   div_id="pareto-scatter", post_script=_btn_js)
    paths["pareto"] = p

    # strategy_bubble_interactive ─────────────────────────────────────────────
    # Shape: circle=RM-100, square=RM-170, diamond=FFZ-350
    # Fill:  Empirical=solid, Gamma-3=open marker
    # Color: LA=#4e88d9, LM=#e09020, SL=#20b2aa
    # Improver: FTSP=bright/thick white border; CLS=dim/thin border
    agg = dfm.groupby(["city","N","dist","strategy","improver"])[
        ["overflows","kgkm","km","profit"]].mean().reset_index()
    fig2 = go.Figure()
    _bub_col = {"LA":"#4e88d9","LM":"#e09020","SL":"#20b2aa"}
    _leg2_seen: set = set()
    for strat in sorted(agg["strategy"].unique()):
        for dist in sorted(agg["dist"].unique()):
            dist_short = "Emp" if dist == "Empirical" else "Gam"
            color = _bub_col.get(strat, "gray")
            for imp in sorted(agg["improver"].unique()):
                sub2 = agg[(agg.strategy == strat) & (agg.dist == dist) & (agg.improver == imp)]
                if sub2.empty:
                    continue
                syms = []
                for r in sub2.itertuples():
                    base = "diamond" if "Figueira" in r.city else ("circle" if int(r.N) == 100 else "square")
                    syms.append(base if dist == "Empirical" else f"{base}-open")
                leg_key = (strat, dist_short)
                show_leg = leg_key not in _leg2_seen
                if show_leg:
                    _leg2_seen.add(leg_key)
                fig2.add_trace(go.Scatter(
                    x=sub2["overflows"], y=sub2["kgkm"],
                    mode="markers",
                    name=f"{strat} ({dist_short})",
                    legendgroup=f"{strat}_{dist_short}",
                    showlegend=show_leg,
                    marker=dict(
                        color=color,
                        symbol=syms,
                        size=(sub2["N"] / 15 + 5).tolist(),
                        opacity=0.9 if imp == "FTSP" else 0.5,
                        line=dict(
                            width=2.5 if imp == "FTSP" else 0.5,
                            color="white" if imp == "FTSP" else color,
                        ),
                    ),
                    text=[
                        f"Strategy: {r.strategy}<br>City: {city_label(r.city, int(r.N))}<br>"
                        f"Dist: {r.dist}<br>Improver: {r.improver}<br>"
                        f"Mean overflows: {r.overflows:.2f}<br>Mean kg/km: {r.kgkm:.3f}<br>N: {r.N}"
                        for r in sub2.itertuples()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ))
    fig2.update_layout(
        title=dict(
            text=(
                "Strategy Trade-off Bubble Chart (bubble size ∝ N)<br>"
                "<sup>Filled=Empirical · Open=Gamma-3 · "
                "Circle=RM-100 · Square=RM-170 · Diamond=FFZ-350 · "
                "Bright/thick border=FTSP · Dim/thin border=CLS</sup>"
            ),
            x=0.5, xanchor="center",
        ),
        xaxis_title="Mean overflows", yaxis_title="Mean kg/km",
        template="plotly_dark", height=650, hovermode="closest",
        margin=dict(t=90),
    )
    p2 = out_dir / "strategy_bubble_interactive.html"
    fig2.write_html(str(p2), include_plotlyjs="cdn")
    paths["bubble"] = p2

    # policy_heatmap_interactive ──────────────────────────────────────────────
    # Toggle views: All (18 cols), By Distribution (2), By Region (3), By Strategy (3), Dist×Region (6)
    _h_constructors = sorted(dfm["constructor"].unique())
    _h_dists = sorted(dfm["dist"].unique())
    _h_strats = sorted(dfm["strategy"].unique())
    _h_city_n = [("Rio Maior",100),("Rio Maior",170),("Figueira da Foz",350)]

    def _build_hm(view: str):
        """Return {(metric, imp): (matrix, x_labels)} for the given grouping view."""
        if view == "all":
            cfgs = [(c,N,d,s) for c,N in _h_city_n for d in _h_dists for s in _h_strats]
            labels = [f"{city_label(c,N)} {d[:3]} {s}" for c,N,d,s in cfgs]
        elif view == "by_dist":
            cfgs = _h_dists
            labels = list(_h_dists)
        elif view == "by_region":
            cfgs = _h_city_n
            labels = [city_label(c,N) for c,N in _h_city_n]
        elif view == "by_strategy":
            cfgs = _h_strats
            labels = list(_h_strats)
        elif view == "by_dist_region":
            cfgs = [(c,N,d) for c,N in _h_city_n for d in _h_dists]
            labels = [f"{city_label(c,N)} {d[:3]}" for c,N,d in cfgs]
        else:
            cfgs, labels = [], []

        result = {}
        for metric in ["overflows","kgkm"]:
            for imp in ["FTSP","CLS"]:
                sub_imp = dfm[dfm.improver == imp]
                mat = np.full((len(_h_constructors), len(cfgs)), np.nan)
                for ci, con in enumerate(_h_constructors):
                    for bi, cfg in enumerate(cfgs):
                        if view == "all":
                            c,N,d,s = cfg # pyrefly: ignore [bad-unpacking]
                            rows = sub_imp[(sub_imp.city==c)&(sub_imp.N==N)&(sub_imp.dist==d)&
                                          (sub_imp.strategy==s)&(sub_imp.constructor==con)][metric]
                        elif view == "by_dist":
                            rows = sub_imp[(sub_imp.dist==cfg)&(sub_imp.constructor==con)][metric]
                        elif view == "by_region":
                            c,N = cfg # pyrefly: ignore [bad-unpacking]
                            rows = sub_imp[(sub_imp.city==c)&(sub_imp.N==N)&(sub_imp.constructor==con)][metric]
                        elif view == "by_strategy":
                            rows = sub_imp[(sub_imp.strategy==cfg)&(sub_imp.constructor==con)][metric]
                        elif view == "by_dist_region":
                            c,N,d = cfg # pyrefly: ignore [bad-unpacking]
                            rows = sub_imp[(sub_imp.city==c)&(sub_imp.N==N)&(sub_imp.dist==d)&
                                          (sub_imp.constructor==con)][metric]
                        else:
                            rows = pd.Series([], dtype=float)
                        if len(rows):
                            mat[ci, bi] = rows.mean() # pyrefly: ignore [no-matching-overload]
                result[(metric, imp)] = (mat, labels)
        return result

    _hm_views = ["all","by_dist","by_region","by_strategy","by_dist_region"]
    _hm_view_labels = ["All Configs (18)","By Distribution (2)","By Region (3)","By Strategy (3)","Dist × Region (6)"]
    _hm_data = {v: _build_hm(v) for v in _hm_views}

    figH = make_subplots(rows=2, cols=2,
        subplot_titles=["Overflows — FTSP","Overflows — CLS","kg/km — FTSP","kg/km — CLS"])
    _hm_specs = [("overflows","FTSP"),("overflows","CLS"),("kgkm","FTSP"),("kgkm","CLS")]
    _hm_cs = ["RdYlGn_r","RdYlGn_r","RdYlGn","RdYlGn"]
    # Add traces for the default view (all)
    for si, ((metric, imp), cs) in enumerate(zip(_hm_specs, _hm_cs)):
        row, col = si//2+1, si%2+1
        mat, labels = _hm_data["all"][(metric, imp)]
        figH.add_trace(go.Heatmap(
            z=mat, x=labels, y=_h_constructors,
            colorscale=cs, name=f"{metric} {imp}",
            hovertemplate="Constructor: %{y}<br>Config: %{x}<br>Value: %{z:.2f}<extra></extra>",
        ), row=row, col=col)
    # Build restyle buttons that swap z and x for all 4 traces simultaneously
    _hm_buttons = []
    for view, vlabel in zip(_hm_views, _hm_view_labels):
        new_z, new_x = [], []
        for metric, imp in [("overflows","FTSP"),("overflows","CLS"),("kgkm","FTSP"),("kgkm","CLS")]:
            mat, lbl = _hm_data[view][(metric, imp)]
            new_z.append(mat.tolist())
            new_x.append(lbl)
        _hm_buttons.append(dict(
            label=vlabel, method="restyle",
            args=[{"z": new_z, "x": new_x}, [0,1,2,3]],
        ))
    figH.update_layout(
        title=dict(text="Policy Configuration Heatmap", x=0.5, xanchor="center"),
        template="plotly_dark", height=900,
        margin=dict(b=90),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.5, y=-0.05, xanchor="center", yanchor="top",
            buttons=_hm_buttons,
            showactive=True,
            bgcolor="#1a2255", bordercolor="#4455aa",
            font=dict(color="white", size=12),
        )],
    )
    p3 = out_dir / "policy_heatmap_interactive.html"
    figH.write_html(str(p3), include_plotlyjs="cdn")
    paths["heatmap"] = p3

    # constructor_comparison_interactive ──────────────────────────────────────
    # 3 toggle views:
    #   Grouped     — 2 bars per constructor (FTSP vs CLS, avg over all dists + regions)
    #   By Dist     — 4 bars per constructor (Emp-FTSP, Emp-CLS, Gam-FTSP, Gam-CLS)
    #   By Region   — 6 bars per constructor (RM-100/RM-170/FFZ-350 × FTSP/CLS)
    _cc_constructors = sorted(dfm["constructor"].unique())
    _cc_dists = sorted(dfm["dist"].unique())
    _cc_city_n = [("Rio Maior",100),("Rio Maior",170),("Figueira da Foz",350)]
    _cc_metrics = ["overflows","kgkm","km","profit"]

    # Color palette
    _imp_col = {"FTSP":"#5588dd","CLS":"#dd8844"}
    _dist_imp_col = {
        ("FTSP","Empirical"):"#4477cc",("FTSP","Gamma-3"):"#88aaff",
        ("CLS","Empirical"):"#cc6600",("CLS","Gamma-3"):"#ffaa66",
    }
    _reg_imp_col = {
        (100,"FTSP"):"#4e88d9",(100,"CLS"):"#88bbf0",
        (170,"FTSP"):"#e09020",(170,"CLS"):"#f5c878",
        (350,"FTSP"):"#20a020",(350,"CLS"):"#66cc66",
    }

    def _con_mean(metric, imp, dist=None, city=None, N=None):
        sub = dfm[dfm.improver == imp]
        if dist is not None: sub = sub[sub.dist == dist]
        if city is not None: sub = sub[(sub.city == city) & (sub.N == N)]
        return sub.groupby("constructor")[metric].mean().reindex(_cc_constructors).tolist()

    # Build trace specs: (group_mode, trace_name, color, legendgroup, fn_per_metric)
    _cc_trace_specs = []
    # Grouped
    for imp in ["FTSP","CLS"]:
        _cc_trace_specs.append(("grouped", f"{imp}", _imp_col[imp], f"grp_{imp}",
                                 lambda m,i=imp: _con_mean(m, i)))
    # By dist
    for imp in ["FTSP","CLS"]:
        for d in _cc_dists:
            ds = "Emp" if d == "Empirical" else "Gam"
            _cc_trace_specs.append(("by_dist", f"{imp} ({ds})", _dist_imp_col[(imp,d)],
                                    f"dst_{imp}_{ds}", lambda m,i=imp,dd=d: _con_mean(m,i,dist=dd)))
    # By region
    for c,N in _cc_city_n:
        for imp in ["FTSP","CLS"]:
            rlbl = city_label(c, N)
            # pyrefly: ignore [bad-argument-type]
            _cc_trace_specs.append(("by_region", f"{rlbl} {imp}", _reg_imp_col[(N,imp)],
                                    f"reg_{N}_{imp}", lambda m,cc=c,nn=N,i=imp: _con_mean(m,i,city=cc,N=nn)))

    figC = make_subplots(rows=2, cols=2,
        subplot_titles=["Overflows","kg/km","km","Profit"],
        shared_xaxes=True)
    _cc_trace_modes = []
    for mi, metric in enumerate(_cc_metrics):
        row, col = mi//2+1, mi%2+1
        for group, name, color, lgname, fn in _cc_trace_specs:
            vals = fn(metric)
            visible = (group == "grouped")
            figC.add_trace(go.Bar(
                name=name,
                x=_cc_constructors, y=vals,
                legendgroup=lgname,
                showlegend=(mi == 0),
                marker_color=color,
                visible=visible,
                hovertemplate=f"Constructor: %{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>",
            ), row=row, col=col)
            if mi == 0:
                _cc_trace_modes.append(group)

    # Each subplot has len(_cc_trace_specs) traces; total = 4 subplots × len(specs)
    def _ccvis(mode):
        return [t == mode for t in _cc_trace_modes] * len(_cc_metrics)

    figC.update_layout(
        title=dict(
            text=(
                "Constructor Comparison<br>"
                "<sup>FTSP = Fast-TSP route improver · CLS = Clarke-Wright Savings route improver</sup>"
            ),
            x=0.5, xanchor="center",
        ),
        template="plotly_dark", height=850, barmode="group",
        margin=dict(b=90),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.5, y=-0.06, xanchor="center", yanchor="top",
            buttons=[
                dict(label="FTSP vs CLS (Grouped)", method="update",
                     args=[{"visible": _ccvis("grouped")}]),
                dict(label="Split by Distribution", method="update",
                     args=[{"visible": _ccvis("by_dist")}]),
                dict(label="Split by Region", method="update",
                     args=[{"visible": _ccvis("by_region")}]),
            ],
            showactive=True,
            bgcolor="#1a2255", bordercolor="#4455aa",
            font=dict(color="white", size=12),
        )],
    )
    p4 = out_dir / "constructor_comparison_interactive.html"
    figC.write_html(str(p4), include_plotlyjs="cdn")
    paths["constructor"] = p4

    # city_comparison_interactive ─────────────────────────────────────────────
    # Toggle: Grouped (avg over dists) vs Split by Distribution (Emp / Gamma-3)
    figCC = make_subplots(rows=1, cols=2, subplot_titles=["Overflows","kg/km"])
    _cc2_strat_col = {"LA":"#4e88d9","LM":"#e09020","SL":"#20b2aa"}
    _cc2_trace_modes: list[str] = []

    # Mode A: Grouped (avg over distributions) — 6 traces (3 strats × 2 imps)
    for col_i, metric in enumerate(["overflows","kgkm"], 1):
        sub_g = dfm.groupby(["city","N","strategy","improver"])[metric].mean().reset_index()
        sub_g["lbl"] = sub_g.apply(lambda r: city_label(r.city, int(r.N)), axis=1)
        for imp in ["FTSP","CLS"]:
            for strat in ["LA","LM","SL"]:
                s = sub_g[(sub_g.improver==imp)&(sub_g.strategy==strat)]
                color = _cc2_strat_col.get(strat, "gray")
                figCC.add_trace(go.Bar(
                    name=f"{strat} {imp}",
                    x=s["lbl"], y=s[metric],
                    legendgroup=f"{strat}_{imp}_grp",
                    showlegend=(col_i == 1),
                    marker=dict(color=color, opacity=0.9 if imp=="FTSP" else 0.55),
                    hovertemplate=f"City: %{{x}}<br>{metric}: %{{y:.2f}}<br>(avg over dists)<extra></extra>",
                ), row=1, col=col_i)
                if col_i == 1:
                    _cc2_trace_modes.append("grouped")

    # Mode B: Split by Distribution — 12 traces (3 strats × 2 imps × 2 dists)
    for col_i, metric in enumerate(["overflows","kgkm"], 1):
        sub_d = dfm.groupby(["city","N","dist","strategy","improver"])[metric].mean().reset_index()
        sub_d["lbl"] = sub_d.apply(lambda r: city_label(r.city, int(r.N)), axis=1)
        for imp in ["FTSP","CLS"]:
            for strat in ["LA","LM","SL"]:
                for dist in sorted(dfm["dist"].unique()):
                    s = sub_d[(sub_d.improver==imp)&(sub_d.strategy==strat)&(sub_d.dist==dist)]
                    ds = "Emp" if dist == "Empirical" else "Gam"
                    base_col = _cc2_strat_col.get(strat, "gray")
                    # Lighten the colour for Gamma-3
                    if dist != "Empirical":
                        r_h=int(base_col[1:3],16); g_h=int(base_col[3:5],16); b_h=int(base_col[5:7],16)
                        base_col = f"#{min(255,r_h+60):02x}{min(255,g_h+60):02x}{min(255,b_h+60):02x}"
                    figCC.add_trace(go.Bar(
                        name=f"{strat} {imp} ({ds})",
                        x=s["lbl"], y=s[metric],
                        legendgroup=f"{strat}_{imp}_{ds}_split",
                        showlegend=(col_i == 1),
                        marker=dict(color=base_col, opacity=0.9 if imp=="FTSP" else 0.55),
                        visible=False,
                        hovertemplate=f"City: %{{x}}<br>{metric}: %{{y:.2f}}<br>Dist: {dist}<extra></extra>",
                    ), row=1, col=col_i)
                    if col_i == 1:
                        _cc2_trace_modes.append("split")

    def _cc2vis(mode):
        return [t == mode for t in _cc2_trace_modes] * 2  # 2 subplots

    figCC.update_layout(
        title=dict(text="City Comparison", x=0.5, xanchor="center"),
        template="plotly_dark", height=650, barmode="group",
        margin=dict(b=90),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.5, y=-0.09, xanchor="center", yanchor="top",
            buttons=[
                dict(label="All Distributions (Grouped)", method="update",
                     args=[{"visible": _cc2vis("grouped")}]),
                dict(label="Split by Distribution", method="update",
                     args=[{"visible": _cc2vis("split")}]),
            ],
            showactive=True,
            bgcolor="#1a2255", bordercolor="#4455aa",
            font=dict(color="white", size=12),
        )],
    )
    p5 = out_dir / "city_comparison_interactive.html"
    figCC.write_html(str(p5), include_plotlyjs="cdn")
    paths["city"] = p5

    return paths


# ── Markdown builder ───────────────────────────────────────────────────────────

def _apply_figure_table_numbers(md: str) -> str:
    """Add sequential Figure N and Table N labels to generated markdown."""
    import re as _re

    fig_n = [0]
    tab_n = [0]

    # Figure numbering: <figure>...</figure> followed by *italic caption*
    # → adds "**Figure N:**" prefix to the italic caption
    def _fig_num(m):
        fig_n[0] += 1
        return f'{m.group(1)}\n\n**Figure {fig_n[0]}:** {m.group(2)}\n'

    md = _re.sub(
        r'(<figure\b[^>]*>.*?</figure>)\n+(\*[^*\n][^\n]*\*)\n',
        _fig_num,
        md,
        flags=_re.DOTALL,
    )

    # Table numbering: _TABCAP_: description → **Table N:** *description*
    def _tab_num(m):
        tab_n[0] += 1
        return f'**Table {tab_n[0]}:** *{m.group(1).strip()}*'

    md = _re.sub(r'_TABCAP_: ([^\n]+)', _tab_num, md)
    return md


def build_pareto_front_table(df: pd.DataFrame) -> str:
    """
    Build the Pareto-front policy catalogue table for section 2.

    One row per unique (mandatory selection variant, constructor, improver) that
    appears on the Pareto front of at least one panel.  The 'Pareto-Front Scenarios'
    column lists every (region/N / distribution) combination where that configuration
    reached the front.  Rows are sorted by descending scenario count, then by
    selection label and constructor name.
    """

    def _pareto_idxs(xs, ys):
        pts = sorted(zip(xs, ys, range(len(xs)), strict=True), key=lambda p: (p[0], -p[1]))
        front, best = [], -np.inf
        for _ov, eff, idx in pts:
            if eff > best:
                front.append(idx)
                best = eff
        return set(front)

    def _selection_label(row):
        s = row["strategy"]
        if s == "LM":
            cf = str(row["cf"]) if pd.notna(row["cf"]) else ""
            return f"LM ({cf})" if cf else "LM"
        if s == "SL":
            sl = str(row["sl_var"]) if pd.notna(row["sl_var"]) else ""
            return f"SL ({sl})" if sl else "SL"
        return s

    dists = sorted(df["dist"].unique())
    improvers = sorted(df["improver"].unique())
    panels = [(d, i) for d in dists for i in improvers]

    pareto_rows = []
    for dist, imp in panels:
        sub = df[(df.dist == dist) & (df.improver == imp)].copy().reset_index(drop=True)
        for idx in _pareto_idxs(sub["overflows"].values, sub["kgkm"].values):
            row = sub.iloc[idx].copy()
            row["_dist"] = dist
            row["_imp"] = imp
            pareto_rows.append(row)

    if not pareto_rows:
        return "_No Pareto-front data available._"

    pf = pd.DataFrame(pareto_rows)
    pf["selection"] = pf.apply(_selection_label, axis=1)
    pf["scenario"] = pf.apply(
        lambda r: f"{city_label(r['city'], int(r['N']))} / {r['_dist']}", axis=1
    )

    table_rows = []
    for (sel, con, imp), grp in pf.groupby(["selection", "constructor", "improver"]):
        scenarios = sorted(grp["scenario"].unique())
        table_rows.append({
            "sel": sel,
            "con": con,
            "imp": imp,
            "ov": grp["overflows"].mean(),
            "eff": grp["kgkm"].mean(),
            "scenarios": ", ".join(scenarios),
            "n": len(scenarios),
        })

    table_rows.sort(key=lambda r: (-r["n"], r["sel"], r["con"], r["imp"]))

    lines = [
        "| Selection | Constructor | Improver | Overflows | kg/km | Pareto-Front Scenarios |",
        "|-----------|-------------|----------|----------:|------:|------------------------|",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['sel']} | {r['con']} | {r['imp']} "
            f"| {r['ov']:.1f} | {r['eff']:.3f} | {r['scenarios']} |"
        )
    return "\n".join(lines)


def build_overflow_table(dfm: pd.DataFrame, imp: str) -> str:
    rows = []
    for city, N in [("Rio Maior",100),("Rio Maior",170),("Figueira da Foz",350)]:
        for dist in ["Gamma-3","Empirical"]:
            for strat in ["LA","LM","SL"]:
                sub = dfm[(dfm.city==city)&(dfm.N==N)&(dfm.dist==dist)&
                          (dfm.improver==imp)&(dfm.strategy==strat)]["overflows"]
                if len(sub):
                    rows.append(f"| {city_label(city,N)} / {dist} / {strat} | "
                                f"{sub.min():.0f} | {sub.max():.0f} | {sub.mean():.1f} |")
    header = "| Config | Min | Max | Mean |\n|--------|-----|-----|------|"
    return header + "\n" + "\n".join(rows)


def build_efficiency_table(dfm: pd.DataFrame, imp: str) -> str:
    rows = []
    for city, N in [("Rio Maior",100),("Rio Maior",170),("Figueira da Foz",350)]:
        for dist in ["Gamma-3","Empirical"]:
            for strat in ["LA","LM","SL"]:
                sub = dfm[(dfm.city==city)&(dfm.N==N)&(dfm.dist==dist)&
                          (dfm.improver==imp)&(dfm.strategy==strat)]["kgkm"]
                if len(sub):
                    rows.append(f"| {city_label(city,N)} / {dist} / {strat} | "
                                f"{sub.min():.2f} | {sub.max():.2f} | {sub.mean():.2f} |")
    header = "| Config | Min | Max | Mean |\n|--------|-----|-----|------|"
    return header + "\n" + "\n".join(rows)


def build_km_table(dfm: pd.DataFrame, imp: str) -> str:
    rows = []
    for city, N in [("Rio Maior",100),("Rio Maior",170),("Figueira da Foz",350)]:
        for dist in ["Gamma-3","Empirical"]:
            for strat in ["LA","LM","SL"]:
                sub = dfm[(dfm.city==city)&(dfm.N==N)&(dfm.dist==dist)&
                          (dfm.improver==imp)&(dfm.strategy==strat)]["km"]
                if len(sub):
                    rows.append(f"| {city_label(city,N)} / {dist} / {strat} | "
                                f"{sub.min():.0f} | {sub.max():.0f} | {sub.mean():.0f} |")
    header = "| Config | Min km | Max km | Mean km |\n|--------|--------|--------|---------|"
    return header + "\n" + "\n".join(rows)


def generate_markdown(df: pd.DataFrame, dfm: pd.DataFrame, panels: list,
                      cities: list, dists: list, strategies: list,
                      improvers: list, constructors: list,
                      figures_rel: str, private_rel: str) -> str:
    total = len(df)
    city_str = ", ".join(f"{city_label(c,N)} (N={N})" for c,N in cities)
    has_both_imp = len(improvers) == 2

    toc_items = [
        "1. [Experimental Setup](#1-experimental-setup)",
        "2. [Analytics Comparison — Pareto View](#2-analytics-comparison--pareto-view)",
        "3. [Summary KPI Analysis](#3-summary-kpi-analysis)",
        "   - 3.1 [Overflow Performance](#31-overflow-performance)",
        "   - 3.2 [Route Efficiency (kg/km)](#32-route-efficiency-kgkm)",
        "   - 3.3 [Distance Driven (km)](#33-distance-driven-km)",
        "   - 3.4 [Policy Ranking Heatmaps](#34-policy-ranking-heatmaps)",
        "4. [Selection Strategy Comparison](#4-selection-strategy-comparison)",
        "5. [Distribution Comparison](#5-distribution-comparison)",
        "6. [Network Size Comparison](#6-network-size-comparison)",
        "7. [Daily Output Analysis](#7-daily-output-analysis)",
    ]
    if has_both_imp:
        toc_items.append("8. [FTSP vs CLS Route Improver Comparison](#8-ftsp-vs-cls-route-improver-comparison)")
    for i, (city, N) in enumerate(cities[2:], start=9 if has_both_imp else 8):
        anchor = city.lower().replace(" ","-")
        toc_items.append(f"{i}. [{city} — City Analysis (N={N})](#{i}-{anchor}--city-analysis-n{N})")
    toc_items.append(f"{len(toc_items)+1}. [City Comparison](#city-comparison-across-all-cities)")
    toc_items.append(f"{len(toc_items)+1}. [Key Findings & Recommendations](#key-findings--recommendations)")
    toc = "\n".join(toc_items)

    config_rows = "\n".join([
        f"| **Cities / N** | {city_str} |",
        f"| **Waste distribution** | {', '.join(dists)} |",
        f"| **Selection strategy** | {', '.join(strategies)} |",
        f"| **Route constructors** | {', '.join(constructors)} |",
        f"| **Route improvers** | {', '.join(improvers)} |",
        "| **Simulation days** | 30 |",
    ])

    strat_sec_lines = []
    for strat in strategies:
        strat_sec_lines.append(f"### {strat}")
        for imp in improvers:
            strat_sec_lines.append(f"#### {strat}+{imp}")
            for city, N in cities:
                for dist in dists:
                    strat_sec_lines.append(f"**{city_label(city,N)}, {dist}:**")
                    sub = dfm[(dfm.city==city)&(dfm.N==N)&(dfm.dist==dist)&
                              (dfm.improver==imp)&(dfm.strategy==strat)]
                    if len(sub):
                        best_ov_c = sub.loc[sub.overflows.idxmin(), "constructor"]
                        best_eff_c = sub.loc[sub.kgkm.idxmax(), "constructor"]
                        strat_sec_lines.append(
                            f"Best overflow: **{best_ov_c}** ({sub.overflows.min():.1f}); "
                            f"Best efficiency: **{best_eff_c}** ({sub.kgkm.max():.3f} kg/km)."
                        )
                    strat_sec_lines.append("")
            strat_sec_lines.append(PLACEHOLDER)
            strat_sec_lines.append("")

    dist_sec_lines = []
    for dist in dists:
        dist_sec_lines.append(f"### {dist}")
        for imp in improvers:
            dist_sec_lines.append(f"#### {dist} — {imp}")
            dist_sec_lines.append(PLACEHOLDER)
            dist_sec_lines.append("")

    city_sections = []
    for _ci, (city, N) in enumerate(cities):
        if city == "Rio Maior":
            continue
        anchor_num = 9 if has_both_imp else 8
        city_sections.append(f"## {anchor_num}. {city} — New City Analysis (N={N})")
        city_sections.append("")
        for imp in improvers:
            city_sections.append(f"### {imp}")
            sub = dfm[(dfm.city==city)&(dfm.N==N)&(dfm.improver==imp)]
            if len(sub):
                best_row = sub.loc[sub.kgkm.idxmax()]
                city_sections.append(
                    f"Best efficiency: **{best_row.constructor}** {best_row.strategy} "
                    f"({best_row.kgkm:.3f} kg/km, {best_row.overflows:.1f} overflows)."
                )
            city_sections.append(PLACEHOLDER)
            city_sections.append("")
        anchor_num += 1

    recommendations_rows = "\n".join([
        "| Use Case | Strategy | Constructor | Route Improver | Notes |",
        "|----------|:--------:|:-----------:|:--------------:|-------|",
        "| Overflow prevention | SL | <!-- best_ov_constructor --> | <!-- improver --> | <!-- notes --> |",
        "| Maximum efficiency | LM | <!-- best_eff_constructor --> | <!-- improver --> | <!-- notes --> |",
        "| Balanced trade-off | LA | <!-- constructor --> | <!-- improver --> | <!-- notes --> |",
    ])

    # Pre-compute conditional sections (avoid nested f-strings — not supported in Python <3.12)
    if has_both_imp:
        section8 = (
            "\n## 8. FTSP vs CLS Route Improver Comparison\n\n"
            f"![FTSP vs CLS Comparison]({figures_rel}/ftsp_vs_cls_comparison.png)\n\n"
            "*FTSP vs CLS scatter per metric. Points above the diagonal = CLS > FTSP; below = FTSP > CLS.*\n\n"
            f"![FTSP vs CLS Delta Heatmap]({figures_rel}/ftsp_vs_cls_delta.png)\n\n"
            "*Delta heatmap (CLS - FTSP) per constructor x configuration. Red = FTSP better, green = CLS better.*\n\n"
            f"{PLACEHOLDER}\n\n"
            "---\n"
        )
    else:
        section8 = ""

    md = f"""\
# WSmart+ Route — Simulation Analysis Report

> **Scope:** 30-day simulation runs across {len(cities)} city/network configurations × {len(dists)} distributions × {len(strategies)} selection strategies × {len(improvers)} route improvers × {len(constructors)} route constructors
> **Total logs analysed:** {total}
> **Horizon:** 30 days
> **Cities:** {city_str}
> **Generated:** <!-- date -->

---

## Table of Contents

{toc}

---

## 1. Experimental Setup

### Configuration Space

_TABCAP_: Configuration space — experimental dimensions and the values tested in this study.

| Dimension | Values |
|-----------|--------|
{config_rows}

### Policy Naming Convention

Each log file encodes the full pipeline as:
`{{mandatory_selection}}_{{route_constructor}}[_{{engine}}]_{{route_improver}}`

For Last-Minute (LM), two critical fill threshold variants are tested: **CF70** (70% fill triggers mandatory collection) and **CF90** (90% threshold). Service-Level (SL) tests two service level targets: **SL1** and **SL2**. Results in this report aggregate CF70 and CF90 under **LM**, and SL1/SL2 under **SL**, unless otherwise specified.

### Metrics Tracked

_TABCAP_: Metrics tracked per simulation run, their optimisation direction, and a brief description.

| Metric | Direction | Description |
|--------|-----------|-------------|
| `overflows` | ↓ lower better | Bins exceeding 100% capacity during simulation |
| `kg` | ↑ higher better | Total waste collected (kg) over 30 days |
| `km` | ↓ lower better | Total vehicle distance driven (km) |
| `kg/km` | ↑ higher better | Route efficiency (waste per unit distance) |
| `ncol` | contextual | Number of collection events |
| `kg_lost` | ↓ lower better | Waste that overflowed and was not collected |
| `profit` | ↑ higher better | Revenue from collection minus operational cost |
| `days` | contextual | Active collection days in the 30-day horizon |

---

## 2. Analytics Comparison — Pareto View

![Overflow vs Efficiency Scatter — Pareto Front]({figures_rel}/overflow_efficiency_scatter_pareto.png)

*Scatter of all simulation runs in the overflows–kg/km space, coloured by selection strategy and CF/SL variant. Four panels: Gamma-3/FTSP, Empirical/FTSP, Gamma-3/CLS, Empirical/CLS. Shape encodes city/N. Dashed white line = Pareto front.*

![Overflow vs Efficiency Scatter — Pareto Front (log scale)]({figures_rel}/overflow_efficiency_scatter_pareto_log.png)

*Same four-panel chart with symlog X-axis — spreads the densely clustered low-overflow region.*

**[Interactive version]({private_rel}/pareto_scatter_interactive.html)**

### LA+FTSP (Lookahead + Fast-TSP)

{PLACEHOLDER}

### LM+FTSP (Last-Minute + Fast-TSP)

{PLACEHOLDER}

### SL+FTSP (Service-Level + Fast-TSP)

{PLACEHOLDER}

### Pareto-Front Policy Catalogue

_TABCAP_: Pareto-optimal policy configurations — each unique (selection variant, constructor, improver) that appeared on the Pareto front of at least one scenario, sorted by scenario count; metrics averaged across those scenarios.

{build_pareto_front_table(df)}

---

## 3. Summary KPI Analysis

### 3.1 Overflow Performance

![Overflow Count by Configuration]({figures_rel}/overflow_all_configs.png)

*Mean overflow count for all 18 configurations, shown for both FTSP and CLS (2×2 layout). Whiskers span the min–max range across all 8 route constructors.*

![Overflow Count by Configuration (log scale)]({figures_rel}/overflow_all_configs_log.png)

*Same chart with symlog Y axis — reveals structure in the RM configurations compressed in the linear scale.*

_TABCAP_: Overflow counts by configuration — mean ± min/max range across all route constructors (FTSP route improver).

{build_overflow_table(dfm, "FTSP")}

_TABCAP_: Overflow counts by configuration — mean ± min/max range across all route constructors (CLS route improver).

{build_overflow_table(dfm, "CLS")}

{PLACEHOLDER}

### 3.2 Route Efficiency (kg/km)

![kg/km Efficiency by Configuration]({figures_rel}/kgkm_all_configs.png)

*Mean kg/km efficiency for all 18 configurations, with min–max range whiskers, for both FTSP and CLS.*

_TABCAP_: Route efficiency (kg/km) by configuration — mean ± min/max range across all route constructors (FTSP route improver).

{build_efficiency_table(dfm, "FTSP")}

_TABCAP_: Route efficiency (kg/km) by configuration — mean ± min/max range across all route constructors (CLS route improver).

{build_efficiency_table(dfm, "CLS")}

{PLACEHOLDER}

### 3.3 Distance Driven (km)

![Vehicle Distance by Strategy]({figures_rel}/km_violin.png)

*Distribution of total vehicle distance (km over 30 days) per selection strategy and city, for both FTSP and CLS.*

_TABCAP_: Vehicle distance driven (km) by configuration — mean ± min/max range across all route constructors (FTSP route improver).

{build_km_table(dfm, "FTSP")}

{PLACEHOLDER}

### 3.4 Policy Ranking Heatmaps

![Policy × Configuration Performance Heatmap]({figures_rel}/policy_config_heatmap.png)

*Four panels: Overflow FTSP | Overflow CLS | Efficiency FTSP | Efficiency CLS. Rows = constructors, columns = all 18 configurations.*

![Policy Heatmap — Split by Distribution]({figures_rel}/policy_config_heatmap_by_dist.png)

*Heatmaps split into Gamma-3 and Empirical panels for each improver.*

![Policy Heatmap — Split by City/N]({figures_rel}/policy_config_heatmap_by_graph.png)

*Heatmaps for each city/N separately, for each improver.*

**[Interactive heatmap]({private_rel}/policy_heatmap_interactive.html)**

{PLACEHOLDER}

---

## 4. Selection Strategy Comparison ({" vs ".join(strategies)})

![Strategy Trade-off Bubble Chart]({figures_rel}/strategy_bubble.png)

*Four panels (Gamma-3/FTSP, Empirical/FTSP, Gamma-3/CLS, Empirical/CLS). Each bubble = one (strategy, city/N) combination.*

![Strategy Trade-off Bubble Chart (log X scale)]({figures_rel}/strategy_bubble_log.png)

*Same chart with symlog X axis.*

**[Interactive bubble chart]({private_rel}/strategy_bubble_interactive.html)**

{chr(10).join(strat_sec_lines)}

---

## 5. Distribution Comparison ({" vs ".join(dists)})

{chr(10).join(dist_sec_lines)}

---

## 6. Network Size Comparison

![Network Scaling]({figures_rel}/scaling_chart.png)

*Scaling chart for Rio Maior (N=100 → N=170) for both FTSP and CLS.*

{PLACEHOLDER}

---

## 7. Daily Output Analysis

### 7.1 Collection Calendar Patterns

{PLACEHOLDER}

### 7.2 Day-by-Day Metric Trajectories

{PLACEHOLDER}

---
{section8}

{chr(10).join(city_sections)}

## City Comparison Across All Cities

![City Comparison — Overflow]({figures_rel}/city_comparison_overflow.png)

*Mean overflow counts for each selection strategy across all city/N configurations, for both FTSP and CLS.*

![City Comparison: Overflow Counts (log scale)]({figures_rel}/city_comparison_overflow_log.png)

*Log-scale version of the overflow comparison.*

**[Interactive city comparison]({private_rel}/city_comparison_interactive.html)**

![City Comparison — Efficiency]({figures_rel}/city_comparison_efficiency.png)

*Mean kg/km efficiency across cities.*

![City Scaling Overview]({figures_rel}/city_scaling_overview.png)

*Scaling chart from N=100 → N=350, for both FTSP and CLS.*

{PLACEHOLDER}

---

## Key Findings & Recommendations

### Policy Performance Radar

![Policy Performance Radar — Combined]({figures_rel}/policy_radar_combined.png)

*Overlaid radar chart for key constructors (ACO_HH, HGS, BPC, SANS). Outer = better on all axes.*

### Constructor Average Ranking

![Route Constructor Average Rank]({figures_rel}/constructor_ranking.png)

*Average rank of each route constructor across all configurations, for FTSP and CLS. Bars grow upward — shorter = better.*

**[Interactive constructor comparison]({private_rel}/constructor_comparison_interactive.html)**

{PLACEHOLDER}

### Deployment Recommendations

_TABCAP_: Deployment recommendations — suggested policy configuration per operational use case.

{recommendations_rows}

---

*All figures stored in `{figures_rel}/`.*
*Raw simulation data: `public/global/simulation/simulation_summary.csv`.*

## Interactive Charts

- [Overflow vs Efficiency — Pareto View]({private_rel}/pareto_scatter_interactive.html)
- [Strategy Trade-off Bubble Chart]({private_rel}/strategy_bubble_interactive.html)
- [Policy Configuration Heatmap]({private_rel}/policy_heatmap_interactive.html)
- [Constructor Comparison]({private_rel}/constructor_comparison_interactive.html)
- [City Comparison]({private_rel}/city_comparison_interactive.html)
    """
    import re as _re

    # Convert all ![alt](path) markdown images to full-width HTML.
    md = _re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: (
            f'<figure style="display:block;width:100%;margin:0.8em 0;padding:0;">'
            f'<img src="{m.group(2)}" alt="{m.group(1)}" width="100%"'
            f' style="width:100% !important;max-width:100% !important;'
            f'height:auto !important;display:block !important;margin:0;" />'
            f'</figure>'
        ),
        md,
    )

    # Add sequential Figure N and Table N captions.
    md = _apply_figure_table_numbers(md)
    return md


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="public/global/simulation/simulation_summary.csv")
    p.add_argument("--out-md", default="public/simulation_analysis.md")
    p.add_argument("--figures-dir", default="public/figures/simulation")
    p.add_argument("--private-dir", default="public/private/simulation")
    p.add_argument("--force", action="store_true", help="Overwrite existing markdown (default: skip if exists)")
    p.add_argument("--figures-only", action="store_true", help="Generate figures but do not write markdown")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_md = Path(args.out_md)
    figures_dir = Path(args.figures_dir)
    private_dir = Path(args.private_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Average over CF/sl_var variants per constructor
    dfm = df.groupby(["city","N","dist","improver","strategy","constructor"])[
        ["overflows","kgkm","km","profit","kg","reward"]
    ].mean().reset_index()

    # Detect dimensions
    cities = sorted(df[["city","N"]].drop_duplicates().values.tolist(), key=lambda x: x[1])
    dists = sorted(df["dist"].unique())
    strategies = sorted(df["strategy"].unique())
    improvers = sorted(df["improver"].unique())
    constructors = sorted(df["constructor"].unique())

    panels = [(d, i) for d in dists for i in improvers]  # 4 panels
    print(f"  Cities: {cities}")
    print(f"  Distributions: {dists}")
    print(f"  Strategies: {strategies}")
    print(f"  Improvers: {improvers}")
    print(f"  Constructors: {constructors}")
    print(f"  Panels: {panels}")

    # ── Generate figures ────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    gen_overflow_bar(dfm, panels, figures_dir)
    gen_kgkm_bar(dfm, panels, figures_dir)
    gen_km_violin(dfm, panels, figures_dir)
    gen_policy_heatmap(dfm, panels, constructors, figures_dir)
    gen_pareto_scatter(dfm, df, panels, figures_dir)
    gen_strategy_bubble(dfm, panels, figures_dir)
    gen_city_comparison(dfm, panels, figures_dir)
    gen_city_scaling(dfm, panels, figures_dir)
    gen_constructor_ranking(dfm, constructors, figures_dir)
    gen_radar(dfm, ["ACO_HH","HGS","BPC","SANS"], figures_dir)
    if len(improvers) > 1:
        gen_ftsp_vs_cls(dfm, figures_dir)
    html_paths = gen_interactive_html(df, dfm, panels, private_dir)
    print(f"  Generated {len(html_paths)} interactive HTML files")

    if args.figures_only:
        print("--figures-only: skipping markdown generation")
        return

    # ── Write markdown ──────────────────────────────────────────────────────────
    str(figures_dir).replace("public/", "figures/") if "public/" in str(figures_dir) else str(figures_dir)
    str(private_dir).replace("public/", "private/") if "public/" in str(private_dir) else str(private_dir)

    if out_md.exists() and not args.force:
        print(f"\n{out_md} already exists. Use --force to regenerate. Skipping markdown write.")
        return

    print(f"\nGenerating markdown: {out_md}")
    md = generate_markdown(
        df, dfm, panels, cities, dists, strategies, improvers, constructors,
        figures_rel="figures/simulation",
        private_rel="private/simulation",
    )
    out_md.write_text(md, encoding="utf-8")
    print(f"  Written: {out_md} ({len(md)} chars)")


if __name__ == "__main__":
    main()

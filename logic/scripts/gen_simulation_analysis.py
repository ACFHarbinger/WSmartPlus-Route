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
import os
import sys
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
                    means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    continue
                sub = dfm[
                    (dfm.city == city) & (dfm.N == N) & (dfm.dist == dist_v)
                    & (dfm.improver == imp) & (dfm.strategy == strat)
                ]["overflows"]
                if len(sub):
                    m = sub.mean()
                    means.append(m); lo.append(m - sub.min()); hi.append(sub.max() - m)
                else:
                    means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            x = np.arange(len(configs))
            colors = [STRATEGY_COLORS[s] for _, _, _, s in configs]
            valid = ~np.isnan(means)
            ax.bar(x[valid], np.array(means)[valid], color=np.array(colors)[valid], alpha=0.85)
            for xi, (m_, l_, h_) in enumerate(zip(means, lo, hi)):
                if not np.isnan(m_):
                    ax.errorbar(xi, m_, yerr=[[l_], [h_]], fmt="none",
                                color="#e0e0e0", capsize=3, linewidth=1.2)
            if log:
                ax.set_yscale("symlog", linthresh=1)
            ax.set_xticks(x[valid])
            ax.set_xticklabels(np.array(col_labels)[valid], fontsize=6, rotation=45, ha="right")
            ax.set_ylabel("Mean overflows (30 days)", fontsize=10)
            ax.set_title(f"{dist} ({imp})", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
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
                means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            sub = dfm[
                (dfm.city == city) & (dfm.N == N) & (dfm.dist == dist_v)
                & (dfm.improver == imp) & (dfm.strategy == strat)
            ]["kgkm"]
            if len(sub):
                m = sub.mean()
                means.append(m); lo.append(m - sub.min()); hi.append(sub.max() - m)
            else:
                means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        x = np.arange(len(configs))
        colors = [STRATEGY_COLORS[s] for _, _, _, s in configs]
        valid = ~np.isnan(means)
        ax.bar(x[valid], np.array(means)[valid], color=np.array(colors)[valid], alpha=0.85)
        for xi, (m_, l_, h_) in enumerate(zip(means, lo, hi)):
            if not np.isnan(m_):
                ax.errorbar(xi, m_, yerr=[[l_], [h_]], fmt="none",
                            color="#e0e0e0", capsize=3, linewidth=1.2)
        ax.set_xticks(x[valid])
        ax.set_xticklabels(np.array(col_labels)[valid], fontsize=6, rotation=45, ha="right")
        ax.set_ylabel("Mean kg/km efficiency", fontsize=10)
        ax.set_title(f"{dist} ({imp})", fontsize=11)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
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
        for body, strat in zip(parts["bodies"], strat_flat):
            body.set_facecolor(STRATEGY_COLORS[strat]); body.set_alpha(0.7)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[key].set_color("#a0a0c0" if key != "cmedians" else "#ffffff")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Total km (30 days)", fontsize=10)
        ax.set_title(f"Distance Distribution — {dist} ({imp})", fontsize=11)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    patches = [mpatches.Patch(color=STRATEGY_COLORS[s], label=s) for s in ["LA", "LM", "SL"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    p = out_dir / "km_violin.png"
    savefig(fig, p)
    return p


def gen_policy_heatmap(dfm: pd.DataFrame, panels: list, constructors: list, out_dir: Path) -> tuple[Path, Path, Path]:
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
                        (dfm.city==city_nm)&(dfm.N==N_val)&(dfm.dist==dist_v)
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


def gen_pareto_scatter(dfm: pd.DataFrame, df_raw: pd.DataFrame, panels: list, out_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(DARK_STYLE)

    def get_color(row):
        s = row["strategy"]
        if s == "LA": return "#4e88d9"
        if s == "LM": return "#e05c5c" if row.get("cf") == "CF70" else "#e09020"
        return "#5cb85c" if row.get("sl_var") == "SL1" else "#20a020"

    def get_marker(row):
        if "Figueira" in str(row["city"]): return "D"
        return "o" if row["N"] == 100 else "s"

    def pareto_front(xs, ys):
        pts = sorted(zip(xs, ys), key=lambda p: (p[0], -p[1]))
        front, best = [], -np.inf
        for ov, eff in pts:
            if eff > best:
                front.append((ov, eff)); best = eff
        return front

    def _make(log: bool) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(22, 16))
        title = "Overflow vs Efficiency — Pareto Front (FTSP & CLS)"
        if log: title += " — Log Scale"
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
                fx, fy = zip(*front)
                step_x, step_y = [fx[0]], [fy[0]]
                for i in range(1, len(fx)):
                    step_x += [fx[i], fx[i]]; step_y += [fy[i-1], fy[i]]
                ax.plot(step_x, step_y, "--", color="white", linewidth=2, alpha=0.9)
            if log:
                ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel("Overflows (30 days)", fontsize=10)
            ax.set_ylabel("Efficiency (kg/km)", fontsize=10)
            ax.set_title(f"Overflow vs Efficiency — {dist} ({imp})", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4); ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        leg = [
            mpatches.Patch(color="#4e88d9", label="LA"),
            mpatches.Patch(color="#e05c5c", label="LM-CF70"),
            mpatches.Patch(color="#e09020", label="LM-CF90"),
            mpatches.Patch(color="#5cb85c", label="SL-SL1"),
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
            ax.yaxis.grid(True, alpha=0.4); ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
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
                        means.append(m); lo.append(m-grp.min()); hi.append(grp.max()-m)
                    else:
                        means.append(0); lo.append(0); hi.append(0)
                ax.bar(x + (ci-1)*width, means, width, label=clabels[ci],
                       color=ccolors[ci], alpha=0.85)
                ax.errorbar(x + (ci-1)*width, means, yerr=[lo, hi],
                            fmt="none", color="#e0e0e0", capsize=3)
            if log:
                ax.set_yscale("symlog", linthresh=1)
            ax.set_xticks(x); ax.set_xticklabels(["LA","LM","SL"], fontsize=11)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{dist} ({imp})", fontsize=11)
            ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
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
        ax.set_xticks(Ns); ax.set_xticklabels(xlabels)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True); ax.legend(fontsize=9)
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
        ax.set_xlabel("Network size N", fontsize=10); ax.set_ylabel("Mean overflows", fontsize=10)
        ax.set_title(f"{dist}", fontsize=11)
        ax.set_xticks([100, 170]); ax.legend(fontsize=8, ncol=2)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
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
        for metric, asc in zip(metrics, rank_asc):
            for _, grp in sub.groupby(["city","N","dist","strategy"]):
                grp_idx = grp.set_index("constructor")
                r = grp_idx[metric].rank(ascending=asc, method="average")
                for c in constructors:
                    if c in r.index:
                        mean_ranks[c][metric].append(r[c])
        x = np.arange(len(constructors))
        w = 0.2
        for mi, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            vals = [np.mean(mean_ranks[c][metric]) if mean_ranks[c][metric] else 4.5
                    for c in constructors]
            ax.bar(x + mi*w - 1.5*w, vals, w, label=label, color=color, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(constructors, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Average rank (lower = better)", fontsize=11)
        ax.set_title(f"Route Constructor Average Rankings\n({imp}, all configs)", fontsize=12)
        ax.set_ylim(0, 8.5)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True); ax.legend(loc="upper left")
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
        for metric, inv in zip(metrics, invert):
            all_vals = dfm[metric].values
            v = sub[metric].mean() if len(sub) else np.nanmean(all_vals)
            mn, mx = np.nanmin(all_vals), np.nanmax(all_vals)
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
    for r, lbl in zip([0.25, 0.5, 0.75, 1.0], circle_labels):
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
    for row_i, (metric, ylabel) in enumerate([("overflows","Overflows"),("kgkm","kg/km")]):
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
            ax.set_xlabel("FTSP", fontsize=10); ax.set_ylabel("CLS", fontsize=10)
            ax.set_title(f"{metric} — {dist}", fontsize=11); ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.4); ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    p1 = out_dir / "ftsp_vs_cls_comparison.png"
    savefig(fig, p1)

    # Delta heatmap
    CONSTRUCTORS = dfm["constructor"].unique().tolist()
    col_labels = [f"{city_label(c,N)}\n{d[:3]}\n{s}" for c, N, d, s in configs]
    fig2, axes2 = plt.subplots(1, 2, figsize=(28, 10))
    fig2.suptitle("FTSP vs CLS Delta Heatmap (CLS − FTSP)", fontsize=14, fontweight="bold")
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


def gen_interactive_html(df_raw: pd.DataFrame, dfm: pd.DataFrame, panels: list, out_dir: Path) -> dict[str, Path]:
    if not HAS_PLOTLY:
        print("  [WARN] Plotly not available — skipping interactive HTML generation")
        return {}
    paths: dict[str, Path] = {}

    # pareto_scatter_interactive
    fig = go.Figure()
    for dist, imp in panels:
        sub = df_raw[(df_raw.dist == dist) & (df_raw.improver == imp)]
        for strategy in sub["strategy"].unique():
            for city, N in sub[["city","N"]].drop_duplicates().values:
                s = sub[(sub.strategy==strategy)&(sub.city==city)&(sub.N==N)]
                if not len(s):
                    continue
                marker_sym = "diamond" if "Figueira" in city else ("circle" if N==100 else "square")
                color_map = {"LA":"#4e88d9","LM":"#e05c5c","SL":"#5cb85c"}
                fig.add_trace(go.Scatter(
                    x=s["overflows"], y=s["kgkm"],
                    mode="markers",
                    name=f"{strategy} {city_label(city,N)} {dist} {imp}",
                    marker=dict(symbol=marker_sym, color=color_map.get(strategy,"gray"),
                                size=8, opacity=0.8, line=dict(width=1,color="white")),
                    text=[
                        f"Constructor: {row.constructor}<br>Strategy: {row.strategy}<br>"
                        f"CF: {row.cf}<br>City: {city_label(row.city,row.N)}<br>"
                        f"Dist: {row.dist}<br>Improver: {row.improver}<br>"
                        f"Overflows: {row.overflows:.1f}<br>kg/km: {row.kgkm:.3f}<br>"
                        f"km: {row.km:.0f}<br>Profit: {row.profit:.0f}"
                        for row in s.itertuples()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ))
    fig.update_layout(title="Overflow vs Efficiency — All Runs (hover for details)",
                      xaxis_title="Overflows (30 days)", yaxis_title="kg/km",
                      template="plotly_dark", height=700, hovermode="closest")
    p = out_dir / "pareto_scatter_interactive.html"
    fig.write_html(str(p), include_plotlyjs="cdn"); paths["pareto"] = p

    # strategy_bubble_interactive
    agg = dfm.groupby(["city","N","dist","strategy","improver"])[["overflows","kgkm","km","profit"]].mean().reset_index()
    fig2 = go.Figure()
    color_map = {"LA":"#4e88d9","LM":"#e05c5c","SL":"#5cb85c"}
    for strat in agg["strategy"].unique():
        sub = agg[agg.strategy==strat]
        fig2.add_trace(go.Scatter(
            x=sub["overflows"], y=sub["kgkm"],
            mode="markers",
            name=strat,
            marker=dict(color=color_map.get(strat,"gray"), size=(sub["N"]/15+5).tolist(), opacity=0.8),
            text=[
                f"Strategy: {row.strategy}<br>City: {city_label(row.city,row.N)}<br>"
                f"Dist: {row.dist}<br>Improver: {row.improver}<br>"
                f"Mean overflows: {row.overflows:.2f}<br>Mean kg/km: {row.kgkm:.3f}<br>"
                f"N: {row.N}"
                for row in sub.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig2.update_layout(title="Strategy Trade-off Bubble Chart (bubble size ∝ N)",
                       xaxis_title="Mean overflows", yaxis_title="Mean kg/km",
                       template="plotly_dark", height=600, hovermode="closest")
    p2 = out_dir / "strategy_bubble_interactive.html"
    fig2.write_html(str(p2), include_plotlyjs="cdn"); paths["bubble"] = p2

    # policy_heatmap_interactive
    constructors = sorted(dfm["constructor"].unique())
    configs = [
        (city, N, dist, strat)
        for city, N in [("Rio Maior",100),("Rio Maior",170),("Figueira da Foz",350)]
        for dist in ["Gamma-3","Empirical"]
        for strat in ["LA","LM","SL"]
    ]
    col_labels = [f"{city_label(c,N)} {d[:3]} {s}" for c,N,d,s in configs]
    mats_ov_ftsp = np.full((len(constructors), len(configs)), np.nan)
    mats_ov_cls = np.full_like(mats_ov_ftsp, np.nan)
    mats_eff_ftsp = np.full_like(mats_ov_ftsp, np.nan)
    mats_eff_cls = np.full_like(mats_ov_ftsp, np.nan)
    for ci, c in enumerate(constructors):
        for cfi, (city, N, dist, strat) in enumerate(configs):
            for imp, mat_ov, mat_eff in [("FTSP",mats_ov_ftsp,mats_eff_ftsp),
                                          ("CLS",mats_ov_cls,mats_eff_cls)]:
                sub = dfm[(dfm.city==city)&(dfm.N==N)&(dfm.dist==dist)&
                          (dfm.improver==imp)&(dfm.strategy==strat)&(dfm.constructor==c)]
                if len(sub):
                    mat_ov[ci,cfi] = sub["overflows"].values[0]
                    mat_eff[ci,cfi] = sub["kgkm"].values[0]
    figH = make_subplots(rows=2, cols=2,
        subplot_titles=["Overflows — FTSP","Overflows — CLS","kg/km — FTSP","kg/km — CLS"])
    for ri, (mat, colorscale, title) in enumerate([
        (mats_ov_ftsp,"RdYlGn_r","Overflows FTSP"),
        (mats_ov_cls,"RdYlGn_r","Overflows CLS"),
        (mats_eff_ftsp,"RdYlGn","kg/km FTSP"),
        (mats_eff_cls,"RdYlGn","kg/km CLS"),
    ]):
        row, col = ri//2+1, ri%2+1
        figH.add_trace(go.Heatmap(
            z=mat, x=col_labels, y=constructors,
            colorscale=colorscale, name=title,
            hovertemplate="Constructor: %{y}<br>Config: %{x}<br>Value: %{z:.2f}<extra></extra>",
        ), row=row, col=col)
    figH.update_layout(title="Policy Configuration Heatmap", template="plotly_dark", height=900)
    p3 = out_dir / "policy_heatmap_interactive.html"
    figH.write_html(str(p3), include_plotlyjs="cdn"); paths["heatmap"] = p3

    # constructor_comparison_interactive
    figC = make_subplots(rows=2, cols=2,
        subplot_titles=["Overflows","kg/km","km","Profit"],
        shared_xaxes=True)
    for mi, metric in enumerate(["overflows","kgkm","km","profit"]):
        row, col = mi//2+1, mi%2+1
        for imp in ["FTSP","CLS"]:
            sub = dfm[dfm.improver==imp].groupby("constructor")[metric].mean().reset_index()
            figC.add_trace(go.Bar(
                name=f"{metric} {imp}",
                x=sub["constructor"], y=sub[metric],
                legendgroup=imp,
                hovertemplate=f"Constructor: %{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>",
            ), row=row, col=col)
    figC.update_layout(title="Constructor Comparison", template="plotly_dark",
                       height=800, barmode="group")
    p4 = out_dir / "constructor_comparison_interactive.html"
    figC.write_html(str(p4), include_plotlyjs="cdn"); paths["constructor"] = p4

    # city_comparison_interactive
    figCC = make_subplots(rows=1, cols=2, subplot_titles=["Overflows","kg/km"])
    for col_i, metric in enumerate(["overflows","kgkm"], 1):
        sub = dfm.groupby(["city","N","strategy","improver"])[metric].mean().reset_index()
        sub["lbl"] = sub.apply(lambda r: city_label(r.city, r.N), axis=1)
        for imp in ["FTSP","CLS"]:
            for strat in ["LA","LM","SL"]:
                s = sub[(sub.improver==imp)&(sub.strategy==strat)]
                figCC.add_trace(go.Bar(
                    name=f"{strat} {imp}", x=s["lbl"], y=s[metric],
                    legendgroup=f"{strat}{imp}",
                    hovertemplate=f"City: %{{x}}<br>{metric}: %{{y:.2f}}<extra></extra>",
                ), row=1, col=col_i)
    figCC.update_layout(title="City Comparison", template="plotly_dark",
                        height=600, barmode="group")
    p5 = out_dir / "city_comparison_interactive.html"
    figCC.write_html(str(p5), include_plotlyjs="cdn"); paths["city"] = p5

    return paths


# ── Markdown builder ───────────────────────────────────────────────────────────

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
    for ci, (city, N) in enumerate(cities):
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

    md = textwrap.dedent(f"""\
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

    | Dimension | Values |
    |-----------|--------|
    {config_rows}

    ### Policy Naming Convention

    Each log file encodes the full pipeline as:
    `{{mandatory_selection}}_{{route_constructor}}[_{{engine}}]_{{route_improver}}`

    For Last-Minute (LM), two critical fill threshold variants are tested: **CF70** (70% fill triggers mandatory collection) and **CF90** (90% threshold). Service-Level (SL) tests two service level targets: **SL1** and **SL2**. Results in this report aggregate CF70 and CF90 under **LM**, and SL1/SL2 under **SL**, unless otherwise specified.

    ### Metrics Tracked

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

    ---

    ## 3. Summary KPI Analysis

    ### 3.1 Overflow Performance

    ![Overflow Count by Configuration]({figures_rel}/overflow_all_configs.png)

    *Mean overflow count for all 18 configurations, shown for both FTSP and CLS (2×2 layout). Whiskers span the min–max range across all 8 route constructors.*

    ![Overflow Count by Configuration (log scale)]({figures_rel}/overflow_all_configs_log.png)

    *Same chart with symlog Y axis — reveals structure in the RM configurations compressed in the linear scale.*

    > **Overflow counts by configuration (mean ± range across constructors, FTSP)**

    {build_overflow_table(dfm, "FTSP")}

    > **Overflow counts by configuration (mean ± range across constructors, CLS)**

    {build_overflow_table(dfm, "CLS")}

    {PLACEHOLDER}

    ### 3.2 Route Efficiency (kg/km)

    ![kg/km Efficiency by Configuration]({figures_rel}/kgkm_all_configs.png)

    *Mean kg/km efficiency for all 18 configurations, with min–max range whiskers, for both FTSP and CLS.*

    > **Efficiency by configuration (mean ± range across constructors, FTSP)**

    {build_efficiency_table(dfm, "FTSP")}

    > **Efficiency by configuration (mean ± range across constructors, CLS)**

    {build_efficiency_table(dfm, "CLS")}

    {PLACEHOLDER}

    ### 3.3 Distance Driven (km)

    ![Vehicle Distance by Strategy]({figures_rel}/km_violin.png)

    *Distribution of total vehicle distance (km over 30 days) per selection strategy and city, for both FTSP and CLS.*

    > **Distance by configuration (mean ± range across constructors, FTSP)**

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
    {"" if not has_both_imp else f"""
    ## 8. FTSP vs CLS Route Improver Comparison

    ![FTSP vs CLS Comparison]({figures_rel}/ftsp_vs_cls_comparison.png)

    *FTSP vs CLS scatter per metric. Points above the diagonal = CLS > FTSP; below = FTSP > CLS.*

    ![FTSP vs CLS Delta Heatmap]({figures_rel}/ftsp_vs_cls_delta.png)

    *Delta heatmap (CLS − FTSP) per constructor × configuration. Red = FTSP better, green = CLS better.*

    {PLACEHOLDER}

    ---
    """}

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
    """)
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
    figures_rel = str(figures_dir).replace("public/", "figures/") if "public/" in str(figures_dir) else str(figures_dir)
    private_rel = str(private_dir).replace("public/", "private/") if "public/" in str(private_dir) else str(private_dir)

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

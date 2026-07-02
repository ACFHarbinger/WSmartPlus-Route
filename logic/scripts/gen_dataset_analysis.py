"""
Generate the dataset analysis markdown backbone and all associated figures.

Reads NPZ and TD statistics CSVs, auto-detects available cities and distributions,
generates all dataset figures (PNG + interactive HTML), and writes a structured
markdown file with results tables and analysis placeholders ready for editing.

Idempotent: will not overwrite an existing markdown unless --force is passed.

Usage
-----
    uv run python logic/scripts/gen_dataset_analysis.py
    uv run python logic/scripts/gen_dataset_analysis.py \\
        --npz-csv public/global/datasets/npz_stats.csv \\
        --td-csv public/global/datasets/td_stats.csv \\
        --out-md public/dataset_analysis.md \\
        --figures-dir public/figures/datasets \\
        --private-dir public/private/datasets
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

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

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

CITY_LABELS = {"figueiradafoz": "Figueira da Foz", "riomaior": "Rio Maior"}
DIST_LABELS = {"emp": "Empirical", "gamma3": "Gamma-3"}
CITY_COLORS = {"figueiradafoz": "#e08030", "riomaior": "#4e88d9"}
DIST_COLORS = {"emp": "#5cb85c", "gamma3": "#e05c5c"}
PLACEHOLDER = "<!-- [ANALYSIS: Insert your observations here] -->"


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Figure generators ──────────────────────────────────────────────────────────

def gen_npz_stats_bar(npz: pd.DataFrame, out_dir: Path) -> Path:
    plt.rcParams.update(DARK_STYLE)
    sub30 = npz[npz["horizon"] == 30]
    cities = sorted(sub30["city"].unique())
    metrics = [
        ("mean_kg", "Mean Waste (kg/bin/day)"),
        ("std_kg", "Std Waste (kg/bin/day)"),
        ("max_kg", "Max Waste (kg)"),
        ("overflow_pct", "Overflow % (30-day)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("NPZ Dataset Statistics by City and Distribution", fontsize=14, fontweight="bold")
    for ax_i, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_i // 2][ax_i % 2]
        ax.set_facecolor("#16213e")
        for dist in ["emp", "gamma3"]:
            sub_d = sub30[sub30["dist"] == dist]
            for ci, city in enumerate(cities):
                val = sub_d[sub_d["city"] == city][metric].values
                if len(val):
                    ax.bar(
                        ci + (0.2 if dist == "gamma3" else -0.2), val[0], 0.35,
                        color=DIST_COLORS[dist], alpha=0.85,
                        label=DIST_LABELS.get(dist, dist) if ci == 0 else "",
                    )
        ax.set_xticks(range(len(cities)))
        ax.set_xticklabels([CITY_LABELS.get(c, c) for c in cities], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        if ax_i == 0:
            ax.legend(fontsize=9)
    plt.tight_layout()
    p = out_dir / "npz_stats_bar.png"
    savefig(fig, p)
    return p


def gen_npz_size_scaling(npz: pd.DataFrame, out_dir: Path) -> Path:
    plt.rcParams.update(DARK_STYLE)
    sub30 = npz[npz["horizon"] == 30].copy()
    rm = sub30[sub30["city"] == "riomaior"].sort_values("N")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("NPZ Statistics vs Network Size", fontsize=14, fontweight="bold")
    for ax_i, (metric, ylabel) in enumerate([
        ("mean_kg", "Mean Waste (kg)"),
        ("std_kg", "Std Waste (kg)"),
        ("skewness", "Skewness"),
    ]):
        ax = axes[ax_i]
        ax.set_facecolor("#16213e")
        for dist in ["emp", "gamma3"]:
            sub_d = rm[rm["dist"] == dist]
            ax.plot(sub_d["N"].values, sub_d[metric].values, "o-",
                    color=DIST_COLORS[dist], linewidth=2, markersize=8,
                    label=DIST_LABELS.get(dist, dist))
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{metric} vs N", fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    p = out_dir / "npz_size_scaling.png"
    savefig(fig, p)
    return p


def gen_npz_city_comparison(npz: pd.DataFrame, out_dir: Path) -> Path:
    plt.rcParams.update(DARK_STYLE)
    sub30 = npz[npz["horizon"] == 30]
    cities = sorted(sub30["city"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("City Comparison Overview — NPZ Datasets", fontsize=14, fontweight="bold")
    for ax_i, (metric, ylabel) in enumerate([
        ("mean_kg", "Mean Waste (kg)"),
        ("std_kg", "Std (kg)"),
        ("overflow_pct", "Overflow %"),
        ("skewness", "Skewness"),
    ]):
        ax = axes[ax_i // 2][ax_i % 2]
        ax.set_facecolor("#16213e")
        for dist in ["emp", "gamma3"]:
            sub_d = sub30[sub30["dist"] == dist]
            for city in cities:
                sub_cd = sub_d[sub_d["city"] == city].sort_values("N")
                for _, row in sub_cd.iterrows():
                    marker = "o" if city == "riomaior" else "D"
                    ec = "white" if dist == "gamma3" else "none"
                    ax.scatter(row["N"], row[metric],
                               c=CITY_COLORS.get(city, "#a0a0a0"),
                               marker=marker, s=150, alpha=0.85,
                               edgecolors=ec, linewidths=1.5)
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    city_patches = [mpatches.Patch(color=CITY_COLORS.get(c, "gray"), label=CITY_LABELS.get(c, c))
                    for c in cities]
    dist_patches = [
        plt.scatter([], [], c="gray", s=60, marker="o", edgecolors="none", label="Empirical"),
        plt.scatter([], [], c="gray", s=60, marker="o", edgecolors="white", linewidths=1.5, label="Gamma-3"),
    ]
    fig.legend(handles=city_patches + dist_patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    p = out_dir / "npz_city_comparison.png"
    savefig(fig, p)
    return p


def gen_npz_horizon_comparison(npz: pd.DataFrame, out_dir: Path) -> Path:
    plt.rcParams.update(DARK_STYLE)
    horizons = sorted(npz["horizon"].unique())
    if len(horizons) < 2:
        return out_dir / "npz_horizon_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("30-day vs 90-day Horizon Comparison", fontsize=14, fontweight="bold")
    rm = npz[npz["city"] == "riomaior"].sort_values(["N", "dist", "horizon"])
    for ax_i, (metric, ylabel) in enumerate([
        ("mean_kg", "Mean Waste (kg)"),
        ("std_kg", "Std Waste (kg)"),
        ("skewness", "Skewness"),
    ]):
        ax = axes[ax_i]
        ax.set_facecolor("#16213e")
        for dist in ["emp", "gamma3"]:
            for horizon, ls in zip(horizons, ["-", "--"], strict=True): # pyrefly: ignore [no-matching-overload]
                sub = rm[(rm["dist"] == dist) & (rm["horizon"] == horizon)].sort_values("N")
                label = f"{DIST_LABELS.get(dist,dist)} {horizon}d"
                ax.plot(sub["N"].values, sub[metric].values,
                        ls, color=DIST_COLORS.get(dist, "gray"),
                        linewidth=2 if horizon == 30 else 1.5,
                        alpha=1.0 if horizon == 30 else 0.7,
                        label=label, markersize=7, marker="o")
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{metric}", fontsize=11)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    p = out_dir / "npz_horizon_comparison.png"
    savefig(fig, p)
    return p


def gen_npz_td_alignment(npz: pd.DataFrame, td: pd.DataFrame, out_dir: Path) -> Path:
    plt.rcParams.update(DARK_STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#16213e")
    ax.set_title("Training (TD) vs Simulator (NPZ) Mean Waste Alignment", fontsize=13, fontweight="bold")
    sub30 = npz[npz["horizon"] == 30]
    rm_npz = sub30[sub30["city"] == "riomaior"]
    for dist in ["emp", "gamma3"]:
        rm_sub = rm_npz[rm_npz["dist"] == dist].sort_values("N")
        td_sub = td[td["dist"] == dist].sort_values("N")
        color = DIST_COLORS.get(dist, "gray")
        dist_label = DIST_LABELS.get(dist, dist)
        ax.plot(rm_sub["N"].values, rm_sub["mean_kg"].values, "o-",
                color=color, linewidth=2, markersize=8, label=f"NPZ {dist_label}")
        if len(td_sub):
            ax.plot(td_sub["N"].values, td_sub["waste_mean"].values * 100, "s--",
                    color=color, linewidth=1.5, markersize=6, alpha=0.7,
                    label=f"TD {dist_label} (×100)")
    ax.set_xlabel("Network size N", fontsize=11)
    ax.set_ylabel("Mean waste (kg/bin/day)", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    p = out_dir / "npz_td_alignment.png"
    savefig(fig, p)
    return p


def gen_td_stats(td: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(DARK_STYLE)
    # stats comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Training Data (TD) Statistics Comparison", fontsize=14, fontweight="bold")
    for ax_i, (metric, ylabel) in enumerate([
        ("waste_mean", "Mean waste fraction"),
        ("waste_std", "Std waste fraction"),
        ("waste_skew", "Skewness"),
    ]):
        ax = axes[ax_i]
        ax.set_facecolor("#16213e")
        for dist in ["emp", "gamma3"]:
            sub = td[td["dist"] == dist].sort_values("N")
            ax.plot(sub["N"].values, sub[metric].values, "o-",
                    color=DIST_COLORS.get(dist, "gray"), linewidth=2, markersize=8,
                    label=DIST_LABELS.get(dist, dist))
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    p1 = out_dir / "td_stats_comparison.png"
    savefig(fig, p1)

    # distributions bar
    ns = sorted(td["N"].unique())
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("Training Data Waste Distributions", fontsize=14, fontweight="bold")
    for ax_i, metric in enumerate(["waste_mean", "waste_std"]):
        ax = axes2[ax_i]
        ax.set_facecolor("#16213e")
        for dist in ["emp", "gamma3"]:
            sub = td[td["dist"] == dist].sort_values("N")
            offset = 0.2 if dist == "gamma3" else -0.2
            ax.bar(np.arange(len(sub)) + offset, sub[metric].values, 0.35,
                   color=DIST_COLORS.get(dist, "gray"), alpha=0.85,
                   label=DIST_LABELS.get(dist, dist))
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"N={n}" for n in ns], fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    p2 = out_dir / "td_waste_distributions.png"
    savefig(fig2, p2)
    return p1, p2


def gen_dataset_interactive_html(npz: pd.DataFrame, td: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    if not HAS_PLOTLY:
        print("  [WARN] Plotly not available — skipping interactive HTML")
        return {}
    paths: dict[str, Path] = {}

    sub30 = npz[npz["horizon"] == 30].copy()
    sub30["city_label"] = sub30["city"].map(CITY_LABELS).fillna(sub30["city"])
    sub30["dist_label"] = sub30["dist"].map(DIST_LABELS).fillna(sub30["dist"])

    # npz_stats_interactive
    fig = go.Figure()
    for city in sub30["city_label"].unique():
        for dist in sub30["dist_label"].unique():
            sub = sub30[(sub30["city_label"] == city) & (sub30["dist_label"] == dist)]
            if not len(sub):
                continue
            sym = "circle" if dist == "Empirical" else "square"
            color = "#e08030" if "Figueira" in city else "#4e88d9"
            hover = (
                sub["city_label"] + "<br>N=" + sub["N"].astype(str)
                + "<br>Dist: " + sub["dist_label"]
                + "<br>Horizon: " + sub["horizon"].astype(str) + " days"
                + "<br>Mean kg: " + sub["mean_kg"].round(3).astype(str)
                + "<br>Std kg: " + sub["std_kg"].round(3).astype(str)
                + "<br>Max kg: " + sub["max_kg"].round(1).astype(str)
                + "<br>Overflow %: " + sub["overflow_pct"].round(3).astype(str)
                + "<br>Skewness: " + sub["skewness"].round(3).astype(str)
            )
            fig.add_trace(go.Scatter(
                x=sub["mean_kg"], y=sub["std_kg"],
                mode="markers",
                name=f"{city} — {dist}",
                marker=dict(
                    color=color, symbol=sym,
                    size=(sub["N"] / 20 + 8).tolist(),
                    line=dict(width=1, color="white"), opacity=0.85,
                ),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ))
    fig.update_layout(
        title="NPZ Dataset Statistics — Mean vs Std Waste (30-day horizon)",
        xaxis_title="Mean waste (kg/bin/day)",
        yaxis_title="Std waste (kg/bin/day)",
        template="plotly_dark", height=600, hovermode="closest",
        legend_title="City — Distribution",
    )
    p = out_dir / "npz_stats_interactive.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    paths["npz_stats"] = p
    print(f"  Saved: {p.name}")

    # waste_distribution_interactive
    cities = sorted(sub30["city"].unique())
    fig2 = make_subplots(rows=1, cols=len(cities),
                         subplot_titles=[CITY_LABELS.get(c, c) for c in cities],
                         shared_yaxes=True)
    for col_i, city in enumerate(cities, 1):
        sub = sub30[sub30["city"] == city].sort_values("N")
        for dist in ["emp", "gamma3"]:
            dist_l = DIST_LABELS.get(dist, dist)
            sub_d = sub[sub["dist"] == dist]
            color = DIST_COLORS.get(dist, "gray")
            hover = (
                sub_d["city"].map(CITY_LABELS).fillna(sub_d["city"])
                + ", N=" + sub_d["N"].astype(str) + ", " + sub_d["dist"].map(DIST_LABELS).fillna(sub_d["dist"])
                + "<br>Mean: " + sub_d["mean_kg"].round(3).astype(str) + " kg"
                + "<br>Std: " + sub_d["std_kg"].round(3).astype(str) + " kg"
                + "<br>Max: " + sub_d["max_kg"].round(1).astype(str) + " kg"
                + "<br>Skewness: " + sub_d["skewness"].round(3).astype(str)
            )
            fig2.add_trace(go.Bar(
                name=dist_l,
                x=[f"N={n}" for n in sub_d["N"]],
                y=sub_d["mean_kg"],
                error_y=dict(type="data", array=sub_d["std_kg"].tolist(), visible=True),
                marker_color=color, marker_opacity=0.85,
                text=hover, hovertemplate="%{text}<extra></extra>",
                legendgroup=dist_l, showlegend=(col_i == 1),
            ), row=1, col=col_i)
    fig2.update_layout(
        title="Waste Distribution Statistics per City and Network Size",
        yaxis_title="Mean waste (kg/bin/day)",
        template="plotly_dark", barmode="group", height=550,
        legend_title="Distribution",
    )
    p2 = out_dir / "waste_distribution_interactive.html"
    fig2.write_html(str(p2), include_plotlyjs="cdn")
    paths["waste_dist"] = p2
    print(f"  Saved: {p2.name}")

    # city_network_comparison_interactive
    metrics = ["mean_kg", "std_kg", "skewness", "max_kg"]
    fig3 = make_subplots(rows=2, cols=2,
        subplot_titles=["Mean Waste (kg/bin)", "Std Waste (kg/bin)", "Skewness", "Max Waste (kg)"])
    for mi, metric in enumerate(metrics):
        row, col = mi // 2 + 1, mi % 2 + 1
        for city in cities:
            city_l = CITY_LABELS.get(city, city)
            for dist in ["emp", "gamma3"]:
                dist_l = DIST_LABELS.get(dist, dist)
                sub = sub30[(sub30["city"] == city) & (sub30["dist"] == dist)].sort_values("N")
                if not len(sub):
                    continue
                color = CITY_COLORS.get(city, "gray")
                opacity = 0.85 if dist == "emp" else 0.55
                hover = (
                    city_l + "<br>N=" + sub["N"].astype(str)
                    + "<br>Dist: " + dist_l
                    + "<br>" + metric + ": " + sub[metric].round(3).astype(str)
                )
                fig3.add_trace(go.Bar(
                    name=f"{city_l} {dist_l}",
                    x=[f"{city_l[:3]} N={n}" for n in sub["N"]],
                    y=sub[metric],
                    marker_color=color, marker_opacity=opacity,
                    text=hover, hovertemplate="%{text}<extra></extra>",
                    legendgroup=f"{city}{dist}", showlegend=(mi == 0),
                ), row=row, col=col)
    fig3.update_layout(
        title="City & Network Comparison — NPZ Dataset Statistics",
        template="plotly_dark", height=800, barmode="group",
        legend_title="City & Distribution",
    )
    p3 = out_dir / "city_network_comparison_interactive.html"
    fig3.write_html(str(p3), include_plotlyjs="cdn")
    paths["city_net"] = p3
    print(f"  Saved: {p3.name}")
    return paths


# ── Table helpers ──────────────────────────────────────────────────────────────

def build_npz_table(npz: pd.DataFrame, horizon: int = 30) -> str:
    sub = npz[npz["horizon"] == horizon].copy()
    sub["city_label"] = sub["city"].map(CITY_LABELS).fillna(sub["city"])
    sub["dist_label"] = sub["dist"].map(DIST_LABELS).fillna(sub["dist"])
    rows = ["| City | N | Distribution | Mean kg | Std kg | Max kg | Overflow % | Skewness |",
            "|------|---|-------------|---------|--------|--------|------------|---------|"]
    for _, row in sub.sort_values(["city","N","dist"]).iterrows():
        rows.append(
            f"| {row.city_label} | {row.N} | {row.dist_label} | "
            f"{row.mean_kg:.2f} | {row.std_kg:.2f} | {row.max_kg:.1f} | "
            f"{row.overflow_pct:.3f} | {row.skewness:.3f} |"
        )
    return "\n".join(rows)


def build_td_table(td: pd.DataFrame) -> str:
    rows = ["| N | Distribution | Instances | Mean Waste | Std Waste | Skewness |",
            "|---|-------------|-----------|------------|-----------|---------|"]
    for _, row in td.sort_values(["N","dist"]).iterrows():
        dist_l = DIST_LABELS.get(row["dist"], row["dist"])
        rows.append(
            f"| {row.N} | {dist_l} | {int(row.instances):,} | "
            f"{row.waste_mean:.4f} | {row.waste_std:.4f} | {row.waste_skew:.3f} |"
        )
    return "\n".join(rows)


# ── Markdown builder ───────────────────────────────────────────────────────────

def generate_markdown(npz: pd.DataFrame, td: pd.DataFrame,
                      figures_rel: str, private_rel: str) -> str:
    cities = sorted(npz["city"].unique())
    city_labels = [CITY_LABELS.get(c, c) for c in cities]
    dists = sorted(npz["dist"].unique())
    horizons = sorted(npz["horizon"].unique())
    total_npz = len(npz)
    total_td = len(td) if td is not None else 0

    toc_items = [
        "1. [Training Data (TD)](#1-training-data-td)" if total_td else None,
    ]
    sec_num = 2 if total_td else 1
    for city in cities:
        city_l = CITY_LABELS.get(city, city)
        anchor = city_l.lower().replace(" ", "-")
        toc_items.append(f"{sec_num}. [{city_l} NPZ Datasets](#{sec_num}-{anchor}-npz-datasets)")
        sec_num += 1
    if len(cities) > 1:
        toc_items.append(f"{sec_num}. [City Comparison](#{sec_num}-city-comparison)")
        sec_num += 1
    if total_td:
        toc_items.append(f"{sec_num}. [TD vs NPZ Alignment](#{sec_num}-td-vs-npz-alignment)")
    toc = "\n".join(t for t in toc_items if t)

    city_sections = []
    for city_idx, city in enumerate(cities, start=2 if total_td else 1):
        city_l = CITY_LABELS.get(city, city)
        npz_sub = npz[npz["city"] == city]
        Ns = sorted(npz_sub["N"].unique())
        city_sections.append(f"## {city_idx}. {city_l} NPZ Datasets\n")
        city_sections.append(f"**Network sizes:** N = {', '.join(str(n) for n in Ns)}  ")
        city_sections.append(f"**Distributions:** {', '.join(DIST_LABELS.get(d, d) for d in dists)}  ")
        city_sections.append(f"**Horizons:** {', '.join(str(h) + ' days' for h in horizons)}\n")
        city_sections.append(f"![NPZ Statistics Bar Chart]({figures_rel}/npz_stats_bar.png)\n")
        city_sections.append("*Mean, std, max waste and overflow percentage per city and distribution.*\n")
        city_sections.append(f"![Statistics vs Network Size]({figures_rel}/npz_size_scaling.png)\n")
        city_sections.append("*How mean waste, std, and skewness vary with network size.*\n")
        if len(horizons) > 1:
            city_sections.append(f"![30-day vs 90-day Horizon Comparison]({figures_rel}/npz_horizon_comparison.png)\n")
            city_sections.append("*Comparison of 30-day and 90-day horizon statistics.*\n")
        city_sections.append(f"\n### Statistics Summary — {city_l} (30-day horizon)\n")
        city_sections.append(build_npz_table(npz_sub))
        city_sections.append(f"\n{PLACEHOLDER}\n")

    # Build conditional blocks as plain strings (avoids triple-quote nesting, Python 3.10 compat)
    td_section = (
        "\n## 1. Training Data (TD)\n\n"
        "Training data used for supervised learning models (stored as TensorDict `.td` files).\n"
        "Each entry contains normalised waste values in [0, 1] (divide by 100 to convert to kg/kg).\n\n"
        "![Waste Statistics Comparison](figures/datasets/td_stats_comparison.png)\n\n"
        "*Mean, std, and skewness of training waste values per network size and distribution.*\n\n"
        "![Training Data Waste Distributions](figures/datasets/td_waste_distributions.png)\n\n"
        "*Bar chart of mean and std waste fractions per network size.*\n\n"
        "### TD Statistics Summary\n\n"
        + build_td_table(td)
        + "\n\n"
        + PLACEHOLDER
        + "\n\n---"
    ) if total_td else "---"

    city_cmp_section = (
        "\n## " + str(len(cities) + (2 if total_td else 1)) + ". City Comparison\n\n"
        "![City Comparison Overview](figures/datasets/npz_city_comparison.png)\n\n"
        "*Key statistics across cities and distributions.*\n\n"
        "### Statistics Summary — All Cities (30-day horizon)\n\n"
        + build_npz_table(npz)
        + "\n\n"
        + PLACEHOLDER
        + "\n\n---"
    ) if len(cities) > 1 else ""

    alignment_section = (
        "\n## " + str(len(cities) + (3 if total_td else 2)) + ". TD vs NPZ Alignment\n\n"
        "![Training (TD) vs Simulator (NPZ) Mean Waste Alignment]"
        "(figures/datasets/npz_td_alignment.png)\n\n"
        "*Comparison of mean waste levels between TD training data (normalised × 100) and NPZ simulator\n"
        "data. Close alignment validates that training distribution matches simulation.*\n\n"
        + PLACEHOLDER
        + "\n\n---"
    ) if total_td else "---"

    md = textwrap.dedent(f"""\
    # WSmart+ Route — Dataset Analysis Report

    > **Scope:** NPZ simulator datasets and TensorDict training datasets
    > **Cities:** {', '.join(city_labels)}
    > **Distributions:** {', '.join(DIST_LABELS.get(d, d) for d in dists)}
    > **Horizons analysed:** {', '.join(str(h) + ' days' for h in horizons)}
    > **Total NPZ dataset entries:** {total_npz}
    > **Generated:** <!-- date -->

    ---

    ## Table of Contents

    {toc}

    ---

    {td_section}

    {"".join(city_sections)}

    {city_cmp_section}

    {alignment_section}

    *Figures are stored in `{figures_rel}/`.*
    *Raw statistics: `public/global/datasets/td_stats.csv` and `public/global/datasets/npz_stats.csv`.*

    ## Interactive Charts

    - [NPZ Statistics — Mean vs Std Scatter]({private_rel}/npz_stats_interactive.html)
    - [Waste Distribution by City and Network Size]({private_rel}/waste_distribution_interactive.html)
    - [City & Network Comparison]({private_rel}/city_network_comparison_interactive.html)
    """)
    return md


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-csv", default="public/global/datasets/npz_stats.csv")
    p.add_argument("--td-csv", default="public/global/datasets/td_stats.csv")
    p.add_argument("--out-md", default="public/dataset_analysis.md")
    p.add_argument("--figures-dir", default="public/figures/datasets")
    p.add_argument("--private-dir", default="public/private/datasets")
    p.add_argument("--force", action="store_true", help="Overwrite existing markdown")
    p.add_argument("--figures-only", action="store_true", help="Generate figures only, skip markdown")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz_csv)
    td_path = Path(args.td_csv)
    out_md = Path(args.out_md)
    figures_dir = Path(args.figures_dir)
    private_dir = Path(args.private_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {npz_path}")
    npz = pd.read_csv(npz_path)
    td: pd.DataFrame | None = None
    if td_path.exists():
        print(f"Reading: {td_path}")
        td = pd.read_csv(td_path)
    else:
        print(f"  [WARN] TD CSV not found: {td_path}")
        td = pd.DataFrame()

    print(f"  NPZ cities: {sorted(npz['city'].unique())}")
    print(f"  NPZ distributions: {sorted(npz['dist'].unique())}")
    print(f"  NPZ horizons: {sorted(npz['horizon'].unique())}")

    # Generate figures
    print("\nGenerating figures...")
    gen_npz_stats_bar(npz, figures_dir)
    gen_npz_size_scaling(npz, figures_dir)
    gen_npz_city_comparison(npz, figures_dir)
    gen_npz_horizon_comparison(npz, figures_dir)
    if td is not None and len(td):
        gen_td_stats(td, figures_dir)
        gen_npz_td_alignment(npz, td, figures_dir)
    html_paths = gen_dataset_interactive_html(npz, td, private_dir)
    print(f"  Generated {len(html_paths)} interactive HTML files")

    if args.figures_only:
        print("--figures-only: skipping markdown generation")
        return

    if out_md.exists() and not args.force:
        print(f"\n{out_md} already exists. Use --force to regenerate. Skipping.")
        return

    print(f"\nGenerating markdown: {out_md}")
    md = generate_markdown(npz, td, figures_rel="figures/datasets", private_rel="private/datasets")
    out_md.write_text(md, encoding="utf-8")
    print(f"  Written: {out_md} ({len(md)} chars)")


if __name__ == "__main__":
    main()

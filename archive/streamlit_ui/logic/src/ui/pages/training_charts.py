"""Chart-heavy section renderers for the Training Monitor page.

This module provides specialized visualization components for analyzing
Deep Reinforcement Learning training dynamics. It includes renderers for
learning rate schedules, summary KPI cards, epoch timing analysis, and
multi-run performance comparison tables.

Example:
    import pandas as pd
    df = pd.DataFrame({"epoch": [1, 2], "loss": [0.5, 0.4]})
    _render_training_kpis(df)

Attributes:
    _render_lr_schedule: Plots learning rate dynamics over time.
    _render_training_kpis: Displays high-level status cards for training.
    _render_epoch_timing: Visualizes training velocity and hardware efficiency.
    _render_run_comparison: Renders a side-by-side performance matrix.
    _render_all_metrics_table: Provides a raw data explorer for metrics.
"""

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from logic.src.constants.dashboard import PLOTLY_LAYOUT_DEFAULTS
from logic.src.ui.services.data_loader import load_hparams
from logic.src.ui.styles.kpi import create_kpi_row


def _render_lr_schedule(runs_data: Dict[str, pd.DataFrame]) -> None:
    """Plots the learning rate schedule over time if data is available.

    Args:
        runs_data: Mapping from run name to its metrics DataFrame.
    """
    has_lr = False
    fig = go.Figure()

    for run_name, df in runs_data.items():
        if df.empty:
            continue

        # Find LR columns (Lightning logs as lr-Adam, lr-SGD, etc.)
        lr_cols = [col for col in df.columns if col.startswith("lr")]
        if not lr_cols:
            continue

        for lr_col in lr_cols:
            lr_data = df[["step", lr_col]].dropna() if "step" in df.columns else df[[lr_col]].dropna()
            if lr_data.empty:
                continue

            has_lr = True
            x_data = lr_data["step"] if "step" in lr_data.columns else list(range(len(lr_data)))

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=lr_data[lr_col],
                    mode="lines",
                    name=f"{run_name} - {lr_col}",
                    hovertemplate=f"LR: %{{y:.2e}}<extra>{run_name}</extra>",
                )
            )

    if not has_lr:
        return

    st.subheader("Learning Rate Schedule")
    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Learning Rate",
        yaxis_type="log",
        height=300,
        hovermode="x unified",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, width="stretch")


def _render_training_kpis(runs_data: Dict[str, pd.DataFrame]) -> None:
    """Displays summary KPI cards for each selected experiment run.

    Calculates and renders metrics like epoch count, step count, latest loss,
    best loss, and validation performance using standardized styles.

    Args:
        runs_data: Mapping from run name to its metrics DataFrame.
    """
    for run_name, df in runs_data.items():
        if df.empty:
            continue

        metrics: Dict[str, Any] = {}

        # Epoch count
        if "epoch" in df.columns:
            metrics["Epochs"] = int(df["epoch"].dropna().max()) + 1 if not df["epoch"].dropna().empty else 0

        # Step count
        if "step" in df.columns:
            metrics["Steps"] = int(df["step"].dropna().max()) if not df["step"].dropna().empty else 0

        # Latest and best training loss (try common column names)
        for loss_col in ["train/rl_loss", "train/il_loss", "train_loss"]:
            if loss_col in df.columns:
                valid = df[loss_col].dropna()
                if not valid.empty:
                    metrics["Latest Loss"] = round(float(valid.iloc[-1]), 4)
                    metrics["Best Loss"] = round(float(valid.min()), 4)
                break

        # Latest and best validation cost
        for val_col in ["val/cost", "val_cost", "val_loss"]:
            if val_col in df.columns:
                valid = df[val_col].dropna()
                if not valid.empty:
                    metrics["Latest Val"] = round(float(valid.iloc[-1]), 4)
                    metrics["Best Val"] = round(float(valid.min()), 4)
                break

        # Time per epoch
        if "time/epoch_s" in df.columns:
            valid = df["time/epoch_s"].dropna()
            if not valid.empty:
                metrics["Time/Epoch (s)"] = round(float(valid.iloc[-1]), 1)

        if metrics:
            if len(runs_data) > 1:
                st.caption(run_name)
            st.markdown(create_kpi_row(metrics), unsafe_allow_html=True)
            st.markdown("")  # spacing


def _render_epoch_timing(runs_data: Dict[str, pd.DataFrame]) -> None:
    """Displays epoch timing analysis and hardware efficiency metrics.

    Args:
        runs_data: Mapping from run name to its metrics DataFrame.
    """
    has_timing = any(
        "time/epoch_s" in df.columns and not df["time/epoch_s"].dropna().empty for df in runs_data.values()
    )
    if not has_timing:
        return

    st.subheader("Epoch Timing")
    fig = go.Figure()

    for run_name, df in runs_data.items():
        if "time/epoch_s" not in df.columns:
            continue
        timing_df = df[["step", "time/epoch_s"]].dropna()
        if timing_df.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=list(range(len(timing_df))),
                y=timing_df["time/epoch_s"],
                mode="lines+markers",
                name=run_name,
                marker=dict(size=4),
                hovertemplate="Time: %{y:.1f}s<extra>%{x}</extra>",
            )
        )

        # Summary stats
        times = timing_df["time/epoch_s"]
        total = times.sum()
        st.markdown(
            f"**{run_name}**: mean={times.mean():.1f}s, "
            f"min={times.min():.1f}s, max={times.max():.1f}s, "
            f"total={total:.0f}s ({total / 60:.1f}min)"
        )

    fig.update_layout(
        xaxis_title="Logged Step",
        yaxis_title="Time (seconds)",
        height=350,
        hovermode="x unified",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, width="stretch")


def _render_run_comparison(
    selected_runs: List[str],
    runs_data: Dict[str, pd.DataFrame],
) -> None:
    """Renders a side-by-side run comparison table for multiple experiments.

    Args:
        selected_runs: List of selected run names to compare.
        runs_data: Mapping from run name to its metrics DataFrame.
    """
    if len(selected_runs) < 2:
        return

    st.subheader("Run Comparison")
    rows: List[Dict[str, Any]] = []

    for run_name in selected_runs:
        df = runs_data.get(run_name)
        if df is None or df.empty:
            continue

        row: Dict[str, Any] = {"Run": run_name}

        # Epochs
        if "epoch" in df.columns and not df["epoch"].dropna().empty:
            row["Epochs"] = int(df["epoch"].dropna().max()) + 1

        # Final and best for each numeric column
        for col in df.columns:
            if col in ("epoch", "step"):
                continue
            valid = df[col].dropna()
            if valid.empty:
                continue
            row[f"{col} (final)"] = round(float(valid.iloc[-1]), 4)
            row[f"{col} (best)"] = round(float(valid.min()), 4)

        # Hparams highlights
        hparams = load_hparams(run_name)
        if hparams:
            row["optimizer"] = hparams.get("optimizer", "")
            opt_kwargs = hparams.get("optimizer_kwargs", {})
            if isinstance(opt_kwargs, dict):
                row["lr"] = opt_kwargs.get("lr", "")
            row["batch_size"] = hparams.get("batch_size", "")
            row["baseline"] = hparams.get("baseline", "")

        rows.append(row)

    if rows:
        comparison_df = pd.DataFrame(rows).set_index("Run").T
        # Fix Arrow conversion error by casting to string (mixed floats/strings)
        st.dataframe(comparison_df.astype(str), width="stretch", height=400)


def _render_all_metrics_table(runs_data: Dict[str, pd.DataFrame]) -> None:
    """Renders a full data explorer with filtering and download capabilities.

    Args:
        runs_data: Mapping from run name to its metrics DataFrame.
    """
    with st.expander("Full Metrics Table"):
        for run_name, df in runs_data.items():
            if df.empty:
                continue

            st.markdown(f"**{run_name}** ({len(df)} rows)")

            # Column filter
            all_cols = list(df.columns)
            selected_cols = st.multiselect(
                "Columns",
                options=all_cols,
                default=all_cols,
                key=f"cols_{run_name}",
            )

            if not selected_cols:
                selected_cols = all_cols

            # Row limit (Robust constraints)
            max_r = len(df)
            min_r = min(10, max_r)
            n_rows = st.number_input(
                "Max rows",
                min_value=min_r,
                max_value=max_r,
                value=min(max_r, 100),
                step=max(1, min(10, max_r // 10)),
                key=f"rows_{run_name}",
            )

            st.dataframe(
                df[selected_cols].tail(n_rows),
                width="stretch",
                height=400,
            )

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                f"Download {run_name} CSV",
                csv,
                file_name=f"{run_name}_metrics.csv",
                mime="text/csv",
                key=f"download_{run_name}",
            )

            if len(runs_data) > 1:
                st.markdown("---")

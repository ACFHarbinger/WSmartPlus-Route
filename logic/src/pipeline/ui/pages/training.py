"""
Training Monitor mode for the Streamlit dashboard.
"""

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from logic.src.pipeline.ui.components.charts import (
    PLOTLY_LAYOUT_DEFAULTS,
    create_training_loss_chart,
)
from logic.src.pipeline.ui.components.sidebar import (
    render_training_controls,
)
from logic.src.pipeline.ui.services.data_loader import (
    discover_training_runs,
    load_hparams,
    load_multiple_training_runs,
)
from logic.src.pipeline.ui.styles.styling import create_kpi_row

# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_run_overview(selected_runs: List[str]) -> None:
    """Display hyperparameters and run configuration for selected runs."""
    with st.expander("Run Configuration", expanded=len(selected_runs) == 1):
        for run_name in selected_runs:
            hparams = load_hparams(run_name)
            if not hparams:
                st.markdown(f"**{run_name}**: No `hparams.yaml` found.")
                continue

            st.markdown(f"**{run_name}**")

            # Key params as metric cards
            key_params: Dict[str, Any] = {}
            for key in ["env_name", "optimizer", "baseline", "loss_fn", "batch_size", "train_data_size"]:
                if key in hparams:
                    key_params[key] = hparams[key]

            # Extract lr from optimizer_kwargs
            opt_kwargs = hparams.get("optimizer_kwargs", {})
            if isinstance(opt_kwargs, dict) and "lr" in opt_kwargs:
                key_params["lr"] = opt_kwargs["lr"]

            if key_params:
                cols = st.columns(len(key_params))
                for i, (k, v) in enumerate(key_params.items()):
                    cols[i].metric(k, str(v))

            st.json(hparams)

            if len(selected_runs) > 1:
                st.markdown("---")


def _render_training_kpis(runs_data: Dict[str, pd.DataFrame]) -> None:
    """Display summary KPI cards for each selected run."""
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
    """Display epoch timing analysis."""
    has_timing = any(
        "time/epoch_s" in df.columns and not df["time/epoch_s"].dropna().empty for df in runs_data.values()
    )
    if not has_timing:
        return

    with st.expander("Epoch Timing"):
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
        st.plotly_chart(fig, use_container_width=True)


def _render_run_comparison(
    selected_runs: List[str],
    runs_data: Dict[str, pd.DataFrame],
) -> None:
    """Side-by-side run comparison table (only for 2+ runs)."""
    if len(selected_runs) < 2:
        return

    with st.expander("Run Comparison"):
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
            st.dataframe(comparison_df, width="stretch", height=400)


def _render_all_metrics_table(runs_data: Dict[str, pd.DataFrame]) -> None:
    """Full data explorer replacing the old 'last 20 rows' view."""
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

            # Row limit
            n_rows = st.number_input(
                "Max rows",
                min_value=10,
                max_value=len(df),
                value=min(len(df), 100),
                step=10,
                key=f"rows_{run_name}",
            )

            st.dataframe(
                df[selected_cols].tail(n_rows),
                width="stretch",
                height=400,
            )

            if len(runs_data) > 1:
                st.markdown("---")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def render_training_monitor() -> None:
    """Render the Training Monitor mode."""
    st.title("Training Monitor")

    # Discover available runs
    runs = discover_training_runs()

    if not runs:
        st.info(
            "No training runs found in `logs/output/`.\n\nStart a training run with PyTorch Lightning to see metrics here."
        )
        return

    run_names = [name for name, _ in runs]

    # Subtitle with quick stats
    st.markdown(f"Monitor and compare PyTorch Lightning training runs. **{len(run_names)} run(s)** available.")

    # Get available metrics from first run
    first_run_data = load_multiple_training_runs([run_names[0]])
    available_metrics: List[str] = []
    if run_names[0] in first_run_data and not first_run_data[run_names[0]].empty:
        available_metrics = [col for col in first_run_data[run_names[0]].columns if col not in ["epoch", "step"]]

    if not available_metrics:
        available_metrics = ["train_loss", "val_loss", "val_cost"]

    # Render sidebar controls
    controls = render_training_controls(run_names, available_metrics)

    if not controls["selected_runs"]:
        st.info("Select at least one run from the sidebar to view metrics.")
        return

    selected_runs: List[str] = controls["selected_runs"]

    # Load selected runs
    with st.spinner(f"Loading {len(selected_runs)} training run(s)..."):
        runs_data = load_multiple_training_runs(selected_runs)

    # 1. Run Overview / Hyperparameters
    _render_run_overview(selected_runs)

    # 2. Training KPIs
    st.subheader("Training Summary")
    _render_training_kpis(runs_data)

    # 3. Loss Curves (existing)
    st.subheader("Loss Curves")
    fig = create_training_loss_chart(
        runs_data=runs_data,
        metric_y1=controls["primary_metric"],
        metric_y2=controls["secondary_metric"],
        x_axis=controls["x_axis"],
        smoothing=controls["smoothing"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Epoch Timing
    _render_epoch_timing(runs_data)

    # 5. Run Comparison (2+ runs)
    _render_run_comparison(selected_runs, runs_data)

    # 6. Full Metrics Table
    _render_all_metrics_table(runs_data)

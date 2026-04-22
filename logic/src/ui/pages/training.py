"""
Training Monitor mode for the Streamlit dashboard.
"""

import os
from typing import Any, Dict, List, Optional

import jinja2
import pandas as pd
import streamlit as st

from logic.src.ui.components.charts import (
    create_training_loss_chart,
)
from logic.src.ui.components.sidebar import (
    render_training_controls,
)
from logic.src.ui.pages.training_charts import (
    _render_all_metrics_table,
    _render_epoch_timing,
    _render_lr_schedule,
    _render_run_comparison,
    _render_training_kpis,
)
from logic.src.ui.services.data_loader import (
    discover_training_runs,
    load_hparams,
    load_multiple_training_runs,
)

# Set up template loader
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
STATUS_TEMPLATE = jinja_env.get_template("status_pill.html")

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


def _render_training_progress(runs_data: Dict[str, pd.DataFrame], selected_runs: List[str]) -> None:
    """Display training progress bars based on current vs total epochs."""
    for run_name in selected_runs:
        hparams = load_hparams(run_name)
        total_epochs = hparams.get("n_epochs")
        if total_epochs is None:
            continue

        df = runs_data.get(run_name)
        if df is None or df.empty or "epoch" not in df.columns:
            continue

        epoch_vals = df["epoch"].dropna()
        if epoch_vals.empty:
            continue

        current = int(epoch_vals.max()) + 1
        total = int(total_epochs)
        if total <= 0:
            continue

        pct = min(current / total, 1.0)
        label = f"{run_name}: {current}/{total} epochs ({pct:.0%})"

        if current >= total:
            st.progress(1.0, text=f"{label} -- Complete")
        else:
            st.progress(pct, text=label)


def _render_convergence_status(runs_data: Dict[str, pd.DataFrame]) -> None:
    """Show convergence status for each run (plateau detection)."""
    window = 10
    threshold = 0.001  # 0.1% relative improvement

    for run_name, df in runs_data.items():
        if df.empty:
            continue

        # Find the loss column
        loss_col: Optional[str] = None
        for col in ["train/rl_loss", "train/il_loss", "train_loss"]:
            if col in df.columns:
                loss_col = col
                break

        if loss_col is None:
            continue

        valid = df[loss_col].dropna()
        if len(valid) < window + 1:
            continue

        recent = valid.iloc[-window:]
        earlier = valid.iloc[-(window + 1)]

        best_recent = float(recent.min())
        rel_improvement = abs(earlier - best_recent) / (abs(earlier) + 1e-8)

        if rel_improvement < threshold:
            plateau_epochs = window
            for i in range(window + 1, min(len(valid), 50)):
                val = float(valid.iloc[-i])
                if abs(val - best_recent) / (abs(val) + 1e-8) > threshold:
                    break
                plateau_epochs = i

            html_msg = STATUS_TEMPLATE.render(
                status="warning", message=f"{run_name}: Loss plateaued for ~{plateau_epochs} epochs"
            )
            st.markdown(html_msg, unsafe_allow_html=True)
        else:
            html_msg = STATUS_TEMPLATE.render(status="good", message=f"{run_name}: Converging (improving)")
            st.markdown(html_msg, unsafe_allow_html=True)


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

    # 3. Training Progress
    _render_training_progress(runs_data, selected_runs)

    # 4. Convergence Status
    _render_convergence_status(runs_data)

    # 5. Loss Curves
    st.subheader("Loss Curves")
    fig = create_training_loss_chart(
        runs_data=runs_data,
        metric_y1=controls["primary_metric"],
        metric_y2=controls["secondary_metric"],
        x_axis=controls["x_axis"],
        smoothing=controls["smoothing"],
    )
    st.plotly_chart(fig, width="stretch")

    # 6. Learning Rate Schedule
    _render_lr_schedule(runs_data)

    # 7. Epoch Timing (promoted from expander to section)
    _render_epoch_timing(runs_data)

    # 8. Run Comparison (promoted from expander, 2+ runs)
    _render_run_comparison(selected_runs, runs_data)

    # 9. Full Metrics Table (with download)
    _render_all_metrics_table(runs_data)

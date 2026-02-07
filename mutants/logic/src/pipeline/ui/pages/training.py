"""
Training Monitor mode for the Streamlit dashboard.
"""

from typing import List

import streamlit as st
from logic.src.pipeline.ui.components.charts import (
    create_training_loss_chart,
)
from logic.src.pipeline.ui.components.sidebar import (
    render_training_controls,
)
from logic.src.pipeline.ui.services.data_loader import (
    discover_training_runs,
    load_multiple_training_runs,
)


def render_training_monitor() -> None:
    """Render the Training Monitor mode."""
    st.title("📈 Training Monitor")
    st.markdown("Monitor and compare PyTorch Lightning training runs.")

    # Discover available runs
    runs = discover_training_runs()

    if not runs:
        st.warning(
            "⚠️ No training runs found in `logs/`.\n\nStart a training run with PyTorch Lightning to see metrics here."
        )
        return

    run_names = [name for name, _ in runs]

    # Get available metrics from first run
    first_run_data = load_multiple_training_runs([run_names[0]])
    available_metrics: List[str] = []
    if run_names[0] in first_run_data and not first_run_data[run_names[0]].empty:
        available_metrics = [col for col in first_run_data[run_names[0]].columns if col not in ["epoch", "step"]]

    if not available_metrics:
        available_metrics = ["train_loss", "val_loss", "val_cost"]

    # Render controls
    controls = render_training_controls(run_names, available_metrics)

    if not controls["selected_runs"]:
        st.info("👆 Select at least one run from the sidebar to view metrics.")
        return

    # Load selected runs
    runs_data = load_multiple_training_runs(controls["selected_runs"])

    # Create and display chart
    st.subheader("Loss Curves")

    fig = create_training_loss_chart(
        runs_data=runs_data,
        metric_y1=controls["primary_metric"],
        metric_y2=controls["secondary_metric"],
        x_axis=controls["x_axis"],
        smoothing=controls["smoothing"],
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show data tables in expander
    with st.expander("📊 View Raw Data"):
        for run_name, df in runs_data.items():
            st.markdown(f"**{run_name}**")
            st.dataframe(df.tail(20), use_container_width=True)

"""
Benchmark Analysis mode for the Streamlit dashboard.
"""

import pandas as pd
import streamlit as st

from logic.src.pipeline.ui.components.benchmark_charts import (
    create_benchmark_comparison_chart,
    create_latency_throughput_scatter,
)
from logic.src.pipeline.ui.services.benchmark_loader import (
    get_unique_benchmarks,
    load_benchmark_data,
)


def render_benchmark_analysis() -> None:
    """Render the Benchmark Analysis mode."""
    st.title("📊 Benchmark Analysis")
    st.markdown("Analyze performance metrics, latency, and throughput across different solvers and models.")

    # Load benchmark data
    df = load_benchmark_data()

    if df.empty:
        st.warning(
            "⚠️ No benchmark data found in `logs/benchmarks/benchmarks.jsonl`.\n\n"
            "Run `just benchmark` to generate some benchmarks."
        )
        return

    # Benchmark categories
    benchmarks = get_unique_benchmarks(df)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Benchmark Filters")
    selected_bench = st.sidebar.multiselect(
        "Select Benchmarks",
        options=benchmarks,
        default=benchmarks[:1] if benchmarks else [],
    )

    if not selected_bench:
        st.info("👆 Select at least one benchmark type from the sidebar.")
        return

    # Filter data
    filtered_df = df[df["benchmark"].isin(selected_bench)]

    # Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["🏆 Performance Table", "📈 Comparison Charts", "⏱️ Latency & Throughput"])

    with tab1:
        st.subheader("Benchmark Results")
        st.dataframe(filtered_df, use_container_width=True)

    with tab2:
        st.subheader("Metric Comparison")
        metrics = [
            col for col in filtered_df.columns if col not in ["timestamp", "time_str", "benchmark", "device", "problem"]
        ]
        if metrics:
            selected_metric: str = st.selectbox("Select Metric to Visualize", options=metrics)
            if selected_metric:
                fig = create_benchmark_comparison_chart(
                    pd.DataFrame(filtered_df),
                    metric=selected_metric,
                    title=f"{selected_metric.capitalize()} Comparison",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric metrics available for the selected benchmarks.")

    with tab3:
        st.subheader("Inference Efficiency")
        if "latency" in filtered_df.columns and "throughput" in filtered_df.columns:
            fig = create_latency_throughput_scatter(pd.DataFrame(filtered_df))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Latency and throughput data only available for neural benchmarks.")

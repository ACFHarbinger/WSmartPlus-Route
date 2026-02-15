"""
Benchmark Analysis mode for the Streamlit dashboard.
"""

from typing import Any, Dict, List

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
from logic.src.pipeline.ui.styles.kpi import create_kpi_row

# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_benchmark_kpis(df: pd.DataFrame) -> None:
    """Display summary KPI cards for the benchmark dataset."""
    metrics: Dict[str, Any] = {"Total Runs": len(df)}

    # Unique policies/models
    for col in ["policy", "model"]:
        if col in df.columns:
            unique = df[col].dropna().nunique()
            if unique > 0:
                metrics[f"Unique {col.capitalize()}s"] = unique

    # Best latency
    if "latency" in df.columns:
        valid = df["latency"].dropna()
        if not valid.empty:
            metrics["Best Latency (s)"] = round(float(valid.min()), 4)

    # Best throughput
    if "throughput" in df.columns:
        valid = df["throughput"].dropna()
        if not valid.empty:
            metrics["Best Throughput"] = round(float(valid.max()), 1)

    # Unique benchmarks
    if "benchmark" in df.columns:
        metrics["Benchmark Types"] = df["benchmark"].nunique()

    if metrics:
        st.markdown(create_kpi_row(metrics), unsafe_allow_html=True)
        st.markdown("")


def _render_benchmark_metadata(df: pd.DataFrame) -> None:
    """Display benchmark environment and configuration info."""
    has_metadata = any(col in df.columns for col in ["device", "num_nodes", "batch_size"])
    if not has_metadata:
        return

    with st.expander("Benchmark Environment"):
        info: Dict[str, Any] = {}

        if "device" in df.columns:
            info["Devices"] = sorted(df["device"].dropna().unique().tolist())

        if "num_nodes" in df.columns:
            valid = df["num_nodes"].dropna()
            if not valid.empty:
                info["Problem Sizes (num_nodes)"] = sorted(valid.unique().tolist())

        if "batch_size" in df.columns:
            valid = df["batch_size"].dropna()
            if not valid.empty:
                info["Batch Sizes"] = sorted(valid.unique().tolist())

        if "model" in df.columns:
            valid = df["model"].dropna()
            if not valid.empty:
                info["Models"] = sorted(valid.unique().tolist())

        if "policy" in df.columns:
            valid = df["policy"].dropna()
            if not valid.empty:
                info["Policies"] = sorted(valid.unique().tolist())

        if "timestamp" in df.columns:
            valid = df["timestamp"].dropna()
            if not valid.empty:
                info["Time Range"] = f"{valid.min()} to {valid.max()}"

        st.json(info)


def _render_performance_table(filtered_df: pd.DataFrame) -> None:
    """Enhanced performance table with column selector and formatting."""
    st.subheader("Benchmark Results")
    st.caption(f"{len(filtered_df)} entries")

    all_cols = list(filtered_df.columns)

    # Exclude timestamp-like columns from default view
    default_cols = [c for c in all_cols if c not in ("timestamp", "time_str", "level", "message")]

    selected_cols = st.multiselect(
        "Columns to display",
        options=all_cols,
        default=default_cols if default_cols else all_cols,
        key="bench_perf_cols",
    )

    if not selected_cols:
        selected_cols = all_cols

    st.dataframe(filtered_df[selected_cols], width="stretch", height=400)


def _render_comparison_charts(filtered_df: pd.DataFrame) -> None:
    """Enhanced comparison charts with grouping selector."""
    st.subheader("Metric Comparison")

    # Identify numeric metrics
    non_metric_cols = {"timestamp", "time_str", "benchmark", "device", "problem", "level", "message"}
    metrics: List[str] = [col for col in filtered_df.columns if col not in non_metric_cols]

    if not metrics:
        st.info("No numeric metrics available for the selected benchmarks.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_metric: str = st.selectbox("Metric", options=metrics, key="bench_metric")
    with col2:
        # Grouping options
        group_options = [c for c in ["policy", "model", "benchmark", "num_nodes"] if c in filtered_df.columns]
        group_by = st.selectbox(
            "Group by",
            options=group_options if group_options else ["benchmark"],
            key="bench_group",
        )

    if selected_metric:
        fig = create_benchmark_comparison_chart(
            pd.DataFrame(filtered_df),
            metric=selected_metric,
            title=f"{selected_metric.replace('_', ' ').capitalize()} Comparison",
            x_axis=group_by,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_latency_throughput(filtered_df: pd.DataFrame) -> None:
    """Latency vs throughput scatter plot."""
    st.subheader("Inference Efficiency")
    if "latency" in filtered_df.columns and "throughput" in filtered_df.columns:
        fig = create_latency_throughput_scatter(pd.DataFrame(filtered_df))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Latency and throughput data only available for neural benchmarks.")


def _render_per_run_details(filtered_df: pd.DataFrame) -> None:
    """Drill-down into a single benchmark entry."""
    st.subheader("Per-Run Details")

    if filtered_df.empty:
        st.info("No entries to inspect.")
        return

    # Build display labels for the selectbox
    labels = []
    for i, row in filtered_df.iterrows():
        parts = []
        if "benchmark" in row and pd.notna(row["benchmark"]):
            parts.append(str(row["benchmark"]))
        if "policy" in row and pd.notna(row["policy"]):
            parts.append(str(row["policy"]))
        elif "model" in row and pd.notna(row["model"]):
            parts.append(str(row["model"]))
        if "num_nodes" in row and pd.notna(row["num_nodes"]):
            parts.append(f"n={int(row['num_nodes'])}")
        label = " | ".join(parts) if parts else f"Entry {i}"
        labels.append((label, i))

    selected_label = st.selectbox(
        "Select entry",
        options=[label for label, _ in labels],
        key="bench_detail_select",
    )

    if selected_label:
        idx = next(i for label, i in labels if label == selected_label)
        entry = filtered_df.loc[idx]
        st.json(entry.dropna().to_dict())


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def render_benchmark_analysis() -> None:
    """Render the Benchmark Analysis mode."""
    st.title("Benchmark Analysis")
    st.markdown("Analyze performance metrics, latency, and throughput across different solvers and models.")

    # Load benchmark data
    df = load_benchmark_data()

    if df.empty:
        st.warning(
            "No benchmark data found in `logs/benchmarks/benchmarks.jsonl`.\n\n"
            "Run `just benchmark` to generate some benchmarks."
        )
        return

    # Benchmark categories
    benchmarks = get_unique_benchmarks(df)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Benchmark Filters")
    selected_bench = st.sidebar.multiselect(
        "Select Benchmarks",
        options=benchmarks,
        default=benchmarks[:1] if benchmarks else [],
    )

    if not selected_bench:
        st.info("Select at least one benchmark type from the sidebar.")
        return

    # Filter data
    filtered_df = pd.DataFrame(df[df["benchmark"].isin(selected_bench)])

    # 1. KPI Summary
    _render_benchmark_kpis(filtered_df)

    # 2. Benchmark Environment
    _render_benchmark_metadata(filtered_df)

    # 3. Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Performance Table", "Comparison Charts", "Latency & Throughput", "Per-Run Details"]
    )

    with tab1:
        _render_performance_table(filtered_df)

    with tab2:
        _render_comparison_charts(filtered_df)

    with tab3:
        _render_latency_throughput(filtered_df)

    with tab4:
        _render_per_run_details(filtered_df)

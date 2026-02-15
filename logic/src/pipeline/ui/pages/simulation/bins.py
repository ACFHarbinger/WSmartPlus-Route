from typing import Any

import pandas as pd
import streamlit as st

from logic.src.pipeline.ui.components.charts import create_stacked_bar_chart
from logic.src.pipeline.ui.pages.simulation.map import render_bin_heatmap


def style_bin_table(df: pd.DataFrame) -> Any:
    """Apply conditional formatting to the bin state table."""
    return (
        df.style.background_gradient(
            subset=["Fill Before (%)"],
            cmap="RdYlGn_r",
            vmin=0,
            vmax=120,
        )
        .map(
            lambda v: "background-color: #ffebee; font-weight: bold" if v else "",
            subset=["Overflow"],
        )
        .map(
            lambda v: "color: #2e7d32; font-weight: bold" if v else "",
            subset=["Collected"],
        )
        .format(
            {
                "Fill Before (%)": "{:.1f}",
                "Fill After (%)": "{:.1f}",
                "Collected (kg)": "{:.2f}",
            }
        )
    )


def render_bin_state_inspector(display_entry: Any) -> None:
    """Render detailed bin state inspection table."""
    st.subheader("Bin State Inspector")
    data = display_entry.data
    bin_states_before = data.get("bin_state_c", [])
    bin_states_after = data.get("bins_state_real_c_after", [])
    bin_collected = data.get("bin_state_collected", [])
    must_go = data.get("must_go")

    if not bin_states_before:
        st.info("No bin state data available.")
        return

    n_bins = len(bin_states_before)
    must_go_set = set(must_go) if must_go else set()

    rows = []
    for i in range(n_bins):
        before = bin_states_before[i] if i < len(bin_states_before) else 0
        after = bin_states_after[i] if i < len(bin_states_after) else 0
        collected_amount = bin_collected[i] if i < len(bin_collected) else 0
        was_selected = (i + 1) in must_go_set
        was_collected = collected_amount > 0
        is_overflow = before > 100

        rows.append(
            {
                "Bin ID": i + 1,
                "Fill Before (%)": round(before, 1),
                "Fill After (%)": round(after, 1),
                "Collected (kg)": round(collected_amount, 2),
                "Selected (must_go)": was_selected,
                "Collected": was_collected,
                "Overflow": is_overflow,
            }
        )

    df = pd.DataFrame(rows)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bins", n_bins)
    with col2:
        st.metric("Bins Selected", len(must_go_set))
    with col3:
        st.metric("Bins Collected", sum(1 for r in rows if r["Collected"]))
    with col4:
        st.metric("Bins Overflowing", sum(1 for r in rows if r["Overflow"]))

    filter_opt = st.radio(
        "Filter bins",
        ["All", "Selected (must_go)", "Collected", "Overflowing"],
        horizontal=True,
        key="bin_filter_radio",
    )

    filtered_df = df
    if filter_opt == "Selected (must_go)":
        filtered_df = df[df["Selected (must_go)"]]
    elif filter_opt == "Collected":
        filtered_df = df[df["Collected"]]
    elif filter_opt == "Overflowing":
        filtered_df = df[df["Overflow"]]

    styled = style_bin_table(pd.DataFrame(filtered_df))
    st.dataframe(styled, height=300, use_container_width=True)


def render_collection_details(display_entry: Any) -> None:
    """Render collection details for the day."""
    st.subheader("Collection Details")
    data = display_entry.data
    bin_collected = data.get("bin_state_collected", [])
    bin_states_before = data.get("bin_state_c", [])
    must_go = data.get("must_go")

    if not bin_collected:
        st.info("No collection data available.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Collections", data.get("ncol", 0))
    with col2:
        st.metric("Total Collected (kg)", f"{sum(c for c in bin_collected if c > 0):.2f}")
    with col3:
        st.metric("Bins in must_go", len(must_go) if must_go else 0)

    must_go_set = set(must_go) if must_go else set()
    collected_bins = []
    for i, amount in enumerate(bin_collected):
        if amount > 0:
            fill_before = bin_states_before[i] if i < len(bin_states_before) else 0
            collected_bins.append(
                {
                    "Bin ID": i + 1,
                    "Fill Before (%)": round(fill_before, 1),
                    "Amount Collected (kg)": round(amount, 2),
                    "Was in must_go": (i + 1) in must_go_set,
                }
            )

    if collected_bins:
        st.markdown("**Per-Bin Collection Breakdown:**")
        st.dataframe(pd.DataFrame(collected_bins), use_container_width=True)
        categories = [str(b["Bin ID"]) for b in collected_bins]
        fig = create_stacked_bar_chart(
            categories=categories,
            series={
                "Collected (kg)": [b["Amount Collected (kg)"] for b in collected_bins],
                "Remaining (%)": [
                    float(max(0, b["Fill Before (%)"] - b["Amount Collected (kg)"])) for b in collected_bins
                ],
            },
            title="Collection Breakdown per Bin",
            x_label="Bin ID",
            y_label="Amount",
            colors=["#43a047", "#e8eaed"],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bins were collected on this day.")


def render_bin_tab(display_entry: Any) -> None:
    """Render the Bins tab content."""
    st.subheader("Bin Fill Level Heatmap")
    render_bin_heatmap(display_entry)
    render_bin_state_inspector(display_entry)
    render_collection_details(display_entry)

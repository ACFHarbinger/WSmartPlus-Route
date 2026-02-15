import contextlib
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from logic.src.pipeline.ui.components.charts import create_sparkline_svg
from logic.src.pipeline.ui.services.data_loader import (
    compute_cumulative_stats,
    compute_day_deltas,
    get_metric_history,
)
from logic.src.pipeline.ui.styles.kpi import create_kpi_row, create_kpi_row_with_deltas

# Mapping from data keys to display labels for KPI deltas
_PRIMARY_KPI_MAP = {
    "profit": "Profit",
    "km": "Distance (km)",
    "kg": "Waste (kg)",
    "overflows": "Overflows",
}

_SECONDARY_KPI_MAP = {
    "ncol": "Collections",
    "kg_lost": "Waste Lost (kg)",
    "kg/km": "Efficiency (kg/km)",
    "cost": "Cost",
}


def render_kpi_dashboard(
    display_entry: Any,
    entries: List[Any],
    controls: Dict[str, Any],
) -> None:
    """Render the Key Performance Indicators section with deltas and sparklines."""
    st.subheader("Key Metrics")

    data = display_entry.data
    current_day = display_entry.day

    # Compute day-over-day deltas
    deltas = compute_day_deltas(
        entries,
        current_day=current_day,
        policy=controls["selected_policy"],
        sample_id=controls["selected_sample"],
    )

    # Build sparklines from metric history
    sparklines: Dict[str, str] = {}
    for data_key, label in {**_PRIMARY_KPI_MAP, **_SECONDARY_KPI_MAP}.items():
        history = get_metric_history(
            entries,
            metric=data_key,
            policy=controls["selected_policy"],
            sample_id=controls["selected_sample"],
            last_n_days=7,
        )
        svg = create_sparkline_svg(history)
        if svg:
            sparklines[label] = svg

    # Row 1: Primary metrics with deltas
    primary_kpis: Dict[str, Tuple[Any, Optional[float]]] = {
        "Day": (display_entry.day, None),
        "Profit": (data.get("profit", 0), deltas.get("profit")),
        "Distance (km)": (data.get("km", 0), deltas.get("km")),
        "Waste (kg)": (data.get("kg", 0), deltas.get("kg")),
        "Overflows": (data.get("overflows", 0), deltas.get("overflows")),
    }
    st.markdown(create_kpi_row_with_deltas(primary_kpis, sparklines), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Secondary/efficiency metrics with deltas
    secondary_kpis: Dict[str, Tuple[Any, Optional[float]]] = {
        "Collections": (data.get("ncol", 0), deltas.get("ncol")),
        "Waste Lost (kg)": (data.get("kg_lost", 0), deltas.get("kg_lost")),
        "Efficiency (kg/km)": (data.get("kg/km", 0), deltas.get("kg/km")),
        "Cost": (data.get("cost", 0), deltas.get("cost")),
    }
    st.markdown(create_kpi_row_with_deltas(secondary_kpis, sparklines), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def render_cumulative_summary(entries: List[Any], controls: Dict[str, Any]) -> None:
    """Render cumulative/aggregate statistics across all days."""
    cumulative = compute_cumulative_stats(
        entries,
        policy=controls["selected_policy"],
        sample_id=controls["selected_sample"],
    )
    if not cumulative:
        return

    with st.expander("Cumulative Summary (All Days)", expanded=False):
        st.markdown(create_kpi_row(cumulative), unsafe_allow_html=True)


def render_policy_info(display_entry: Any) -> None:
    """Render policy configuration details parsed from the policy string."""
    with st.expander("Policy Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Policy", display_entry.policy)
        with col2:
            st.metric("Sample ID", display_entry.sample_id)
        with col3:
            st.metric("Day", display_entry.day)

        policy_str = display_entry.policy
        parts = policy_str.split("_")
        details: Dict[str, Any] = {"Full Policy String": policy_str}

        known_policies = ["hgs", "alns", "gurobi", "tsp", "neural", "am", "ddam", "bcp"]
        known_selections = ["regular", "last_minute", "lookahead", "revenue", "service_level"]
        known_engines = ["gurobi", "pyvrp", "ortools"]

        detected_policy = next((p for p in known_policies if p in parts), None)
        detected_selection = next((s for s in known_selections if any(s in part for part in parts)), None)
        detected_engine = next((e for e in known_engines if e in parts), None)

        if detected_policy:
            details["Routing Policy"] = detected_policy
        if detected_selection:
            details["Selection Strategy"] = detected_selection
        if detected_engine:
            details["Solver Engine"] = detected_engine

        for part in parts:
            if part.startswith("lvl"):
                with contextlib.suppress(ValueError):
                    details["Selection Threshold"] = float(part[3:])
            elif part.startswith("gamma"):
                with contextlib.suppress(ValueError):
                    details["Gamma Parameter"] = float(part[5:])
            elif part.startswith("temp"):
                with contextlib.suppress(ValueError):
                    details["Temperature"] = float(part[4:])

        st.json(details)

from typing import Any

import pandas as pd
import streamlit as st


def render_tour_details(display_entry: Any) -> None:
    """Render tour sequence and leg details."""
    st.subheader("Tour Details")
    data = display_entry.data
    tour = data.get("tour", [])

    if not tour or len(tour) <= 1:
        st.info("No tour executed on this day (empty or depot-only tour).")
        return

    n_stops = sum(1 for p in tour if int(p.get("id", 0)) != 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stops", n_stops)
    with col2:
        st.metric("Tour Length (nodes)", len(tour))
    with col3:
        st.metric("Distance (km)", f"{data.get('km', 0):.2f}")

    st.markdown("**Tour Sequence:**")
    tour_rows = []
    for i, point in enumerate(tour):
        point_id = point.get("id", "?")
        is_depot = str(point_id) == "0"
        tour_rows.append(
            {
                "Step": i,
                "Node ID": point_id,
                "Type": "depot" if is_depot else "bin",
                "Latitude": round(point["lat"], 6) if "lat" in point else "N/A",
                "Longitude": round(point["lng"], 6) if "lng" in point else "N/A",
            }
        )
    st.dataframe(pd.DataFrame(tour_rows), use_container_width=True)

    must_go = data.get("must_go", [])
    if must_go:
        st.markdown(f"**must_go Selection** ({len(must_go)} bins): `{must_go}`")


def render_raw_data_view(display_entry: Any) -> None:
    """Render raw data view for debugging."""
    st.subheader("Raw Data (JSON)")
    st.markdown(
        f"**Policy**: `{display_entry.policy}` | "
        f"**Sample**: `{display_entry.sample_id}` | "
        f"**Day**: `{display_entry.day}`"
    )
    st.json(display_entry.data)

"""
Algorithms Registry Page.

Displays all registered policies, models, and components with advanced filtering
by tags, paradigms, and pipeline phases.
"""

from typing import Any, Dict, Set

import streamlit as st

from logic.src.enums.environment_tags import EnvironmentTag
from logic.src.enums.model_tags import ModelTag
from logic.src.enums.operator_tags import OperatorTag
from logic.src.enums.policy_tags import PolicyTag
from logic.src.enums.registry import AnyTag, GlobalRegistry
from logic.src.enums.trainer_tags import TrainerTag


def get_tag_display_name(tag: AnyTag) -> str:
    """Returns a readable name for a tag including its category."""
    return f"{tag.__class__.__name__.replace('Tag', '')}: {tag.name}"


def get_tag_color(tag: AnyTag) -> str:
    """Returns a CSS color based on the tag type."""
    if isinstance(tag, PolicyTag):
        return "#28B463"  # Green
    if isinstance(tag, ModelTag):
        return "#AF7AC5"  # Purple
    if isinstance(tag, EnvironmentTag):
        return "#F39C12"  # Orange
    if isinstance(tag, OperatorTag):
        return "#E74C3C"  # Red
    if isinstance(tag, TrainerTag):
        return "#5D6D7E"  # Greyish
    return "#2E86C1"  # Default Blue


def render_algo_card(item: Dict[str, Any]) -> None:
    """Renders a single algorithm as a card-like expander."""
    with st.expander(f"**{item['name']}**", expanded=False):
        # Short description
        st.markdown(f"_{item['doc']}_")

        # Tags row
        tag_html = ""
        # Sort tags by category then name
        sorted_tags = sorted(list(item["tags"]), key=lambda x: (x.__class__.__name__, x.name))
        for t in sorted_tags:
            color = get_tag_color(t)
            tag_html += (
                f'<span style="background-color:{color}; color:white; '
                f"padding: 2px 10px; border-radius: 12px; margin-right: 6px; "
                f"margin-bottom: 6px; font-size: 0.75rem; display: inline-block; "
                f'font-weight: 500; font-family: sans-serif;">{t.name}</span>'
            )

        st.markdown(tag_html, unsafe_allow_html=True)

        # Technical details
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.caption("**Registry Name**")
            st.text(item["name"])
        with col2:
            st.caption("**Implementation Path**")
            try:
                path = f"{item['obj'].__module__}.{item['obj'].__name__}"
            except AttributeError:
                path = str(item["obj"])
            st.code(path, language="python")


def render_algorithms() -> None:  # noqa: C901
    """Renders the Algorithms registry page in the Streamlit UI."""

    st.title("🧩 Policy Algorithms Registry")
    st.markdown(
        "Explore the global registry of optimization policies, neural architectures, "
        "and algorithmic components powering the WSmart-Route framework."
    )

    # 1. Fetch data from registry
    registry_data = GlobalRegistry.get_all()

    # 2. Extract and sort unique tags for filters
    all_tags: Set[AnyTag] = set()
    for tags in registry_data.values():
        all_tags.update(tags)

    sorted_tags_list = sorted(list(all_tags), key=lambda x: (x.__class__.__name__, x.name))
    tag_options = {get_tag_display_name(t): t for t in sorted_tags_list}

    # 3. Sidebar Filtering Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Registry Filters")

    search_query = st.sidebar.text_input("Search Name or Description", placeholder="e.g. ALNS, HGS...")

    include_tag_names = st.sidebar.multiselect(
        "Include Tags",
        options=list(tag_options.keys()),
        help="Show algorithms matching these tags. Behavior depends on the logic below.",
    )

    logic_mode = st.sidebar.radio(
        "Inclusion Logic",
        options=["Intersection (AND)", "Union (OR)"],
        index=1,  # Default to Union (OR)
        horizontal=True,
        help="Intersection: Match ALL selected tags. Union: Match AT LEAST ONE selected tag.",
    )

    exclude_tag_names = st.sidebar.multiselect(
        "Exclude Tags",
        options=list(tag_options.keys()),
        help="Hide any algorithm that contains ANY of these selected tags.",
    )

    # 4. Filter Logic Implementation
    include_tags = [tag_options[n] for n in include_tag_names]
    exclude_tags = [tag_options[n] for n in exclude_tag_names]

    filtered_results = []

    for obj, obj_tags in registry_data.items():
        name = GlobalRegistry.get_name(obj)
        doc = (getattr(obj, "__doc__", "") or "No documentation provided.").split("\n")[0]

        # A. Search Filter
        if search_query:
            q = search_query.lower()
            if q not in name.lower() and q not in doc.lower():
                continue

        # B. Inclusion Filter
        if include_tags:
            if logic_mode == "Intersection (AND)":
                if not set(include_tags).issubset(obj_tags):
                    continue
            else:  # Union (OR)
                if set(include_tags).isdisjoint(obj_tags):
                    continue

        # C. Exclusion Filter
        if exclude_tags and not set(exclude_tags).isdisjoint(obj_tags):
            continue

        filtered_results.append({"obj": obj, "name": name, "tags": obj_tags, "doc": doc.strip()})

    # 5. Pipeline Phase Mapping for Tabs
    phase_mapping = {
        "Mandatory Selection": PolicyTag.SELECTION,
        "Route Construction": PolicyTag.CONSTRUCTION,
        "Acceptance Criteria": PolicyTag.ACCEPTANCE,
        "Route Improvement": PolicyTag.IMPROVEMENT,
        "Selection and Construction": PolicyTag.JOINT,
        "Operators": PolicyTag.OPERATOR,
    }

    # 6. Render Results
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        st.metric("Total Algorithms", len(registry_data))
    with kpi_cols[1]:
        st.metric("Found", len(filtered_results))
    with kpi_cols[2]:
        active_filters = len(include_tag_names) + len(exclude_tag_names) + (1 if search_query else 0)
        st.metric("Filters Active", active_filters)

    if not filtered_results:
        st.warning("No algorithms found matching your current filter criteria.")
        if st.button("Clear All Filters"):
            st.rerun()
        return

    # Tabs based on requested categories
    tab_list = list(phase_mapping.keys()) + ["All Matches"]
    tabs = st.tabs(tab_list)

    for i, label in enumerate(tab_list):
        with tabs[i]:
            if label == "All Matches":
                target_items = filtered_results
            else:
                target_tag = phase_mapping[label]
                target_items = [item for item in filtered_results if target_tag in item["tags"]]

            if not target_items:
                st.info(f"No algorithms in the '{label}' phase match your current filters.")
            else:
                # Sort alphabetically by name
                sorted_items = sorted(target_items, key=lambda x: x["name"])
                for item in sorted_items:
                    render_algo_card(item)

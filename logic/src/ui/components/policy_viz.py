"""
Policy Visualization Dispatcher.

Single entry-point Streamlit component that renders the telemetry
captured by :class:`~logic.src.models.policies.viz_mixin.PolicyVizMixin`
for any registered policy type.

Usage::

    from logic.src.ui.components.policy_viz import render_policy_viz

    # After running a policy that carries PolicyVizMixin:
    render_policy_viz(policy.get_viz_data())

The dispatcher inspects the keys present in *viz_data* and automatically
routes to the correct sub-renderer:

======================  ===========================================
Key signature           Renderer
======================  ===========================================
``d_idx`` present       :func:`_render_alns`  (ALNS)
``generation`` present  :func:`_render_hgs`   (HGS)
``tau_mean`` present    :func:`_render_aco`   (ACO)
``perturb_mode`` present:func:`_render_ils`   (ILS)
``n_selected`` present  :func:`_render_selector` (any selector)
fallback                :func:`_render_rls`   (RLS / generic)
======================  ===========================================
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def render_policy_viz(
    viz_data: Dict[str, List[Any]],
    height: int = 400,
    title: Optional[str] = None,
    smooth_window: int = 1,
) -> None:
    """
    Render policy telemetry captured by :class:`PolicyVizMixin`.

    Args:
        viz_data:      Dict returned by ``policy.get_viz_data()``.
        height:        Chart height in pixels.
        title:         Optional section heading.  Defaults to the
                       auto-detected policy type name.
        smooth_window: Exponential-moving-average window for noisy
                       time-series (1 = no smoothing).
    """
    if not viz_data:
        st.info("No policy telemetry recorded yet.  Run the policy first.")
        return

    if title:
        st.subheader(title)

    if "d_idx" in viz_data:
        _render_alns(viz_data, height, smooth_window)
    elif "generation" in viz_data:
        _render_hgs(viz_data, height, smooth_window)
    elif "tau_mean" in viz_data:
        _render_aco(viz_data, height, smooth_window)
    elif "perturb_mode" in viz_data:
        _render_ils(viz_data, height, smooth_window)
    elif "n_selected" in viz_data:
        _render_selector(viz_data, height)
    else:
        _render_rls(viz_data, height, smooth_window)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ema(values: List[float], window: int) -> List[float]:
    """Apply exponential moving average smoothing."""
    if window <= 1 or not values:
        return values
    alpha = 2.0 / (window + 1)
    smoothed: List[float] = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def _bar_counts(values: List[Any]) -> Dict[Any, int]:
    counts: Dict[Any, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# ALNS renderer
# ---------------------------------------------------------------------------

_ALNS_OP_NAMES = {
    "d": ["Random Removal", "Worst Removal", "Cluster Removal"],
    "r": ["Greedy Insertion", "Regret-k Insertion"],
}


def _render_alns(data: Dict[str, List[Any]], height: int, smooth: int) -> None:
    st.markdown("**ALNS – Iteration Telemetry**")

    iterations = data.get("iteration", list(range(len(data.get("best_cost", [])))))

    col1, col2 = st.columns(2)

    # Best cost trajectory
    with col1:
        best_costs = data.get("best_cost", [])
        if best_costs:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=_ema([float(c) for c in best_costs], smooth),
                    mode="lines",
                    name="Best Cost",
                    line={"color": "#2196F3", "width": 2},
                )
            )
            current_costs = data.get("current_cost", [])
            if current_costs:
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=_ema([float(c) for c in current_costs], smooth),
                        mode="lines",
                        name="Current Cost",
                        line={"color": "#FF9800", "width": 1, "dash": "dot"},
                    )
                )
            fig.update_layout(
                title="Cost Trajectory",
                xaxis_title="Iteration",
                yaxis_title="Cost",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    # Temperature decay
    with col2:
        temps = data.get("temperature", [])
        if temps:
            fig = go.Figure(
                go.Scatter(
                    x=iterations,
                    y=[float(t) for t in temps],
                    mode="lines",
                    line={"color": "#F44336", "width": 2},
                )
            )
            fig.update_layout(
                title="SA Temperature",
                xaxis_title="Iteration",
                yaxis_title="T",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    # Operator usage histogram
    d_indices = data.get("d_idx", [])
    r_indices = data.get("r_idx", [])
    if d_indices or r_indices:
        col3, col4 = st.columns(2)
        with col3:
            counts = _bar_counts(d_indices)
            labels = [_ALNS_OP_NAMES["d"][k] if k < len(_ALNS_OP_NAMES["d"]) else f"D{k}" for k in counts]
            fig = go.Figure(go.Bar(x=labels, y=list(counts.values()), marker_color="#4CAF50"))
            fig.update_layout(
                title="Destroy Operator Usage",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            counts = _bar_counts(r_indices)
            labels = [_ALNS_OP_NAMES["r"][k] if k < len(_ALNS_OP_NAMES["r"]) else f"R{k}" for k in counts]
            fig = go.Figure(go.Bar(x=labels, y=list(counts.values()), marker_color="#9C27B0"))
            fig.update_layout(
                title="Repair Operator Usage",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    # Acceptance info
    n_accepted = data.get("n_accepted", [])
    n_improved = data.get("n_improved", [])
    if n_accepted or n_improved:
        col5, col6, col7 = st.columns(3)
        col5.metric("Total Iterations", len(iterations))
        if n_improved:
            col6.metric("Improvements", sum(int(v) for v in n_improved))
        if n_accepted:
            col7.metric("SA Acceptances", sum(int(v) for v in n_accepted))


# ---------------------------------------------------------------------------
# HGS renderer
# ---------------------------------------------------------------------------


def _render_hgs(data: Dict[str, List[Any]], height: int, smooth: int) -> None:
    st.markdown("**HGS – Generation Telemetry**")

    generations = data.get("generation", list(range(len(data.get("best_cost", [])))))

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for key, name, color in [
            ("best_cost", "Best", "#2196F3"),
            ("mean_cost", "Mean", "#FF9800"),
            ("worst_cost", "Worst", "#F44336"),
        ]:
            vals = data.get(key, [])
            if vals:
                fig.add_trace(
                    go.Scatter(
                        x=generations,
                        y=_ema([float(v) for v in vals], smooth),
                        mode="lines",
                        name=name,
                        line={"color": color, "width": 2},
                    )
                )
        fig.update_layout(
            title="Population Fitness",
            xaxis_title="Generation",
            yaxis_title="Cost",
            height=height,
            margin={"t": 40, "b": 30, "l": 50, "r": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        no_improv = data.get("no_improv", [])
        if no_improv:
            restarted = data.get("restarted", [False] * len(generations))
            restart_gens = [g for g, r in zip(generations, restarted) if r]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=[int(v) for v in no_improv],
                    mode="lines",
                    name="No-Improv Streak",
                    line={"color": "#607D8B", "width": 2},
                )
            )
            for rg in restart_gens:
                fig.add_vline(x=rg, line_dash="dash", line_color="#E91E63", annotation_text="restart")
            fig.update_layout(
                title="Stagnation Counter",
                xaxis_title="Generation",
                yaxis_title="Gens w/o Improvement",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    best_costs = data.get("best_cost", [])
    col3, col4, col5 = st.columns(3)
    col3.metric("Generations", len(generations))
    if best_costs:
        col4.metric("Best Cost", f"{float(best_costs[-1]):.4f}")
        col5.metric("Initial Cost", f"{float(best_costs[0]):.4f}")


# ---------------------------------------------------------------------------
# ACO renderer
# ---------------------------------------------------------------------------


def _render_aco(data: Dict[str, List[Any]], height: int, smooth: int) -> None:
    st.markdown("**ACO – Pheromone & Cost Telemetry**")

    iterations = data.get("iteration", list(range(len(data.get("global_best_cost", [])))))

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for key, name, color in [
            ("global_best_cost", "Global Best", "#2196F3"),
            ("iter_best_cost", "Iter Best", "#FF9800"),
        ]:
            vals = data.get(key, [])
            if vals:
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=_ema([float(v) for v in vals], smooth),
                        mode="lines",
                        name=name,
                        line={"color": color, "width": 2},
                    )
                )
        fig.update_layout(
            title="Cost Convergence",
            xaxis_title="Iteration",
            yaxis_title="Cost",
            height=height,
            margin={"t": 40, "b": 30, "l": 50, "r": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tau_mean = data.get("tau_mean", [])
        tau_max = data.get("tau_max", [])
        if tau_mean or tau_max:
            fig = go.Figure()
            if tau_mean:
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=[float(v) for v in tau_mean],
                        mode="lines",
                        name="τ mean",
                        line={"color": "#4CAF50", "width": 2},
                    )
                )
            if tau_max:
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=[float(v) for v in tau_max],
                        mode="lines",
                        name="τ max",
                        line={"color": "#9C27B0", "width": 2},
                    )
                )
            fig.update_layout(
                title="Pheromone Statistics",
                xaxis_title="Iteration",
                yaxis_title="Pheromone",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    global_best = data.get("global_best_cost", [])
    col3, col4 = st.columns(2)
    col3.metric("Iterations", len(iterations))
    if global_best:
        col4.metric("Global Best Cost", f"{float(global_best[-1]):.4f}")


# ---------------------------------------------------------------------------
# ILS renderer
# ---------------------------------------------------------------------------

_PERTURB_COLORS = {"double_bridge": "#2196F3", "shuffle": "#FF9800", "random_swap": "#4CAF50"}


def _render_ils(data: Dict[str, List[Any]], height: int, smooth: int) -> None:
    st.markdown("**ILS – Restart & Perturbation Telemetry**")

    restarts = data.get("restart", list(range(len(data.get("best_cost", [])))))

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        best_cost = data.get("best_cost", [])
        candidate_cost = data.get("candidate_cost", [])
        if best_cost:
            fig.add_trace(
                go.Scatter(
                    x=restarts,
                    y=_ema([float(v) for v in best_cost], smooth),
                    mode="lines",
                    name="Best Cost",
                    line={"color": "#2196F3", "width": 2},
                )
            )
        if candidate_cost:
            fig.add_trace(
                go.Scatter(
                    x=restarts,
                    y=_ema([float(v) for v in candidate_cost], smooth),
                    mode="lines",
                    name="Candidate Cost",
                    line={"color": "#FF9800", "width": 1, "dash": "dot"},
                )
            )
        fig.update_layout(
            title="Cost per Restart",
            xaxis_title="Restart",
            yaxis_title="Cost",
            height=height,
            margin={"t": 40, "b": 30, "l": 50, "r": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        perturb_modes = data.get("perturb_mode", [])
        if perturb_modes:
            counts = _bar_counts(perturb_modes)
            colors = [_PERTURB_COLORS.get(str(k), "#9E9E9E") for k in counts]
            fig = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color=colors))
            fig.update_layout(
                title="Perturbation Operator Usage",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    n_improved = data.get("n_improved", [])
    col3, col4 = st.columns(2)
    col3.metric("Total Restarts", len(restarts))
    if n_improved:
        col4.metric("Total Improvements", sum(int(v) for v in n_improved))


# ---------------------------------------------------------------------------
# Selector renderer
# ---------------------------------------------------------------------------


def _render_selector(data: Dict[str, List[Any]], height: int) -> None:
    st.markdown("**Selection Strategy – Call Telemetry**")

    calls = list(range(len(data.get("n_selected", []))))

    col1, col2 = st.columns(2)

    with col1:
        n_selected = data.get("n_selected", [])
        if n_selected:
            fig = go.Figure(
                go.Bar(
                    x=calls,
                    y=[int(v) for v in n_selected],
                    marker_color="#2196F3",
                )
            )
            fig.update_layout(
                title="Bins Selected per Call",
                xaxis_title="Call #",
                yaxis_title="# Bins",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        mean_fill = data.get("mean_fill", [])
        if mean_fill:
            fig = go.Figure(
                go.Scatter(
                    x=calls,
                    y=[float(v) for v in mean_fill],
                    mode="lines+markers",
                    line={"color": "#FF9800", "width": 2},
                )
            )
            fig.update_layout(
                title="Mean Fill Level at Selection",
                xaxis_title="Call #",
                yaxis_title="Fill",
                yaxis_range=[0.0, 1.0],
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    day_vals = data.get("day", [])
    col3, col4 = st.columns(2)
    col3.metric("Total Calls", len(calls))
    n_sel = data.get("n_selected", [])
    if n_sel:
        total_selected = sum(int(v) for v in n_sel)
        col4.metric("Total Bins Selected", total_selected)
    if day_vals:
        st.caption(f"Days recorded: {min(day_vals)} – {max(day_vals)}")


# ---------------------------------------------------------------------------
# RLS / generic renderer
# ---------------------------------------------------------------------------


def _render_rls(data: Dict[str, List[Any]], height: int, smooth: int) -> None:
    st.markdown("**RLS / Generic Policy – Iteration Telemetry**")

    iterations = data.get("iteration", list(range(len(next(iter(data.values()), [])))))

    op_names = data.get("op_name", [])
    if op_names:
        col1, col2 = st.columns(2)
        with col1:
            counts = _bar_counts(op_names)
            fig = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color="#2196F3"))
            fig.update_layout(
                title="Operator Usage",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Any numeric series that isn't iteration or op_name
            numeric_keys = [
                k for k in data if k not in ("iteration", "op_name") and isinstance(data[k][0], (int, float))
            ]
            if numeric_keys:
                fig = go.Figure()
                colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
                for idx, key in enumerate(numeric_keys[:5]):
                    fig.add_trace(
                        go.Scatter(
                            x=iterations,
                            y=_ema([float(v) for v in data[key]], smooth),
                            mode="lines",
                            name=key,
                            line={"color": colors[idx % len(colors)], "width": 2},
                        )
                    )
                fig.update_layout(
                    title="Metric Trajectories",
                    xaxis_title="Iteration",
                    height=height,
                    margin={"t": 40, "b": 30, "l": 50, "r": 10},
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        # All numeric keys as line chart
        numeric_keys = [k for k in data if k != "iteration" and isinstance(data[k][0], (int, float))]
        if numeric_keys:
            fig = go.Figure()
            colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
            for idx, key in enumerate(numeric_keys[:5]):
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=_ema([float(v) for v in data[key]], smooth),
                        mode="lines",
                        name=key,
                        line={"color": colors[idx % len(colors)], "width": 2},
                    )
                )
            fig.update_layout(
                title="Policy Metrics",
                xaxis_title="Iteration",
                height=height,
                margin={"t": 40, "b": 30, "l": 50, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Recorded {len(iterations)} iterations across {len(data)} metric(s): {', '.join(sorted(data.keys()))}")

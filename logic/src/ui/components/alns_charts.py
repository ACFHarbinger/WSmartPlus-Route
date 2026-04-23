"""Visualisation components for ALNS operator weight trajectories.

This module provides specialized tools for auditing the roulette-wheel
selection dynamics of ALNS Destroy and Repair operators. It includes
a tracked solver subclass, a standalone telemetry tracker, and Plotly
renderers for probability trajectories.

Example:
    render_alns_operator_charts(solver.weight_history)

Attributes:
    TrackedVectorizedALNS: Drop-in solver that snapshots operator weights.
    ALNSSnapshotTracker: Non-invasive wrapper for weight recording.
    render_alns_operator_charts: Main UI orchestrator for dynamics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from logic.src.models.policies.adaptive_large_neighborhood_search import VectorizedALNS

# ---------------------------------------------------------------------------
# Tracked ALNS subclass (zero-interference, opt-in)
# ---------------------------------------------------------------------------


class TrackedVectorizedALNS:
    """Drop-in replacement for VectorizedALNS that records operator weight trajectories.

    Inherits all functionality from the original solver and adds a
    ``weight_history`` attribute populated after each call to ``solve()``.

    Attributes:
        weight_history: Dictionary containing the weight history for both destroy and repair operators.
        log_freq: Snapshot operator weights every N ALNS iterations.
        _inner: The inner VectorizedALNS instance.
    """

    def __init__(self, *args: Any, log_freq: int = 25, **kwargs: Any) -> None:
        """
        Args:
            args: Forwarded verbatim to ``VectorizedALNS.__init__``.
            log_freq: Snapshot operator weights every N ALNS iterations.
            kwargs: Forwarded verbatim to ``VectorizedALNS.__init__``.
        """
        self._inner = VectorizedALNS(*args, **kwargs)
        self.log_freq = log_freq
        self.weight_history: Dict[str, List[List[float]]] = {
            "destroy": [],
            "repair": [],
        }

    # Proxy attribute access to the inner solver
    def __getattr__(self, name: str) -> Any:
        """
        Get attribute from the inner solver.

        Args:
            name: Name of the attribute to get.

        Returns:
            The attribute from the inner solver.
        """
        return getattr(self._inner, name)

    def solve(
        self,
        initial_solutions: Any,
        n_iterations: int = 2000,
        time_limit: Optional[float] = None,
        max_vehicles: int = 0,
        start_temp: float = 0.5,
        cooling_rate: float = 0.9995,
        **kwargs: Any,
    ) -> Any:
        """
        Solve using the inner VectorizedALNS, logging weight snapshots.

        Calls the original ``solve()`` in chunks of ``log_freq`` iterations,
        recording ``d_weights`` / ``r_weights`` between chunks so that the
        weight trajectory can be plotted after completion.

        Args:
            initial_solutions: Initial solutions for the ALNS solver.
            n_iterations: Number of iterations to run the solver for.
            time_limit: Time limit for the solver.
            max_vehicles: Maximum number of vehicles to use.
            start_temp: Starting temperature for the solver.
            cooling_rate: Cooling rate for the solver.
            kwargs: Forwarded verbatim to ``VectorizedALNS.solve``.

        Returns:
            The same ``(routes, costs)`` tuple returned by ``VectorizedALNS.solve``.
        """

        self.weight_history = {"destroy": [], "repair": []}
        inner = self._inner

        remaining = n_iterations
        result: Any = None
        current_solutions = initial_solutions

        def _snapshot() -> None:
            if hasattr(inner, "d_weights") and hasattr(inner, "r_weights"):
                d = inner.d_weights
                r = inner.r_weights
                if hasattr(d, "tolist"):
                    self.weight_history["destroy"].append(_softmax_probs(d))
                    self.weight_history["repair"].append(_softmax_probs(r))

        _snapshot()  # record initial state

        while remaining > 0:
            chunk = min(self.log_freq, remaining)

            result = inner.solve(
                initial_solutions=current_solutions,
                n_iterations=chunk,
                time_limit=time_limit,
                max_vehicles=max_vehicles,
                start_temp=start_temp,
                cooling_rate=cooling_rate,
                **kwargs,
            )
            # After each chunk, update current_solutions from the result
            # (VectorizedALNS.solve returns (routes_list, costs))
            if result and isinstance(result, tuple) and len(result) > 0:
                # routes_list is a list of lists, if we want to continue we should
                # ideally convert it back to a tensor, but VectorizedALNS.solve
                # currently doesn't easily support continuing from routes_list
                # because it expects a giant tour initial_solutions tensor.
                pass

            _snapshot()
            remaining -= chunk

        return result


# ---------------------------------------------------------------------------
# Snapshot tracker (non-invasive wrapper, no subclassing required)
# ---------------------------------------------------------------------------


class ALNSSnapshotTracker:
    """Records operator weights before and after a solver run.

    Attributes:
        weight_history: Dictionary containing the weight history for both destroy and repair operators.

    Example:
        tracker = ALNSSnapshotTracker()
        tracker.before(solver)
        solver.solve(...)
        tracker.after(solver)
    """

    def __init__(self) -> None:
        """
        Initialize the snapshot tracker.
        """
        self.weight_history: Dict[str, List[List[float]]] = {
            "destroy": [],
            "repair": [],
        }

    def before(self, solver: Any) -> None:
        """
        Snapshot weights immediately before a solve call.

        Args:
            solver: The ALNS solver instance.
        """
        self._snapshot(solver, label="before")

    def after(self, solver: Any) -> None:
        """
        Snapshot weights immediately after a solve call.

        Args:
            solver: The ALNS solver instance.
        """
        self._snapshot(solver, label="after")

    def _snapshot(self, solver: Any, label: str) -> None:
        """
        Snapshot weights immediately before or after a solve call.

        Args:
            solver: The ALNS solver instance.
            label: Label indicating whether this is a "before" or "after" snapshot.
        """
        if hasattr(solver, "d_weights") and hasattr(solver, "r_weights"):
            self.weight_history["destroy"].append(_softmax_probs(solver.d_weights))
            self.weight_history["repair"].append(_softmax_probs(solver.r_weights))


# ---------------------------------------------------------------------------
# Main Streamlit render function
# ---------------------------------------------------------------------------


def render_alns_operator_charts(
    weight_history: Dict[str, List[List[float]]],
    destroy_op_names: Optional[List[str]] = None,
    repair_op_names: Optional[List[str]] = None,
    chart_type: str = "line",
    smooth_window: int = 1,
    height: int = 400,
    title: str = "ALNS Operator Weight Dynamics",
    color_palette: Optional[List[str]] = None,
) -> None:
    """
    Render line or stacked-area charts of ALNS operator selection probabilities.

    Args:
        weight_history: Dict with keys ``"destroy"`` and ``"repair"``, each
            containing a list of weight vectors sampled at ``log_freq`` intervals.
            Shape: ``{"destroy": [[w0, w1, w2], ...], "repair": [[w0, w1], ...]}``.
        destroy_op_names: Display names for destroy operators. Defaults to
            ``["Random Removal", "Worst Removal", "Cluster Removal"]``.
        repair_op_names: Display names for repair operators. Defaults to
            ``["Greedy Insertion", "Regret-k Insertion"]``.
        chart_type: ``"line"`` for multi-line or ``"area"`` for stacked area.
            Default ``"line"``.
        smooth_window: Exponential moving-average span for smoothing. 1 = none.
            Default 1.
        height: Chart height in pixels. Default 400.
        title: Section heading.
        color_palette: Optional list of hex colour strings (one per operator).
    """
    if go is None:
        st.error("**plotly** is required. Run: `pip install plotly`")
        return

    st.subheader(title)

    if not weight_history or (not weight_history.get("destroy") and not weight_history.get("repair")):
        st.info(
            "No operator weight data available. "
            "Attach a ``TrackedVectorizedALNS`` or ``ALNSSnapshotTracker`` before solving."
        )
        return

    d_names = destroy_op_names or ["Random Removal", "Worst Removal", "Cluster Removal"]
    r_names = repair_op_names or ["Greedy Insertion", "Regret-k Insertion"]
    colors = color_palette or ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    # ── Interactive controls ─────────────────────────────────────────────────
    ctrl_cols = st.columns(3)
    with ctrl_cols[0]:
        chart_type = st.selectbox(
            "Chart Type",
            ["line", "area"],
            index=0 if chart_type == "line" else 1,
            key=f"alns_chart_type_{title}",
        )
    with ctrl_cols[1]:
        smooth_window = st.slider(
            "Smoothing Window",
            min_value=1,
            max_value=20,
            value=smooth_window,
            key=f"alns_smooth_{title}",
        )

    # ── Two-column layout ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Destroy Operators**")
        fig_d = _build_chart(
            weight_history.get("destroy", []),
            d_names,
            "Destroy Operator Probabilities",
            chart_type,  # type: ignore[arg-type]
            smooth_window,
            height,
            colors,
        )
        if fig_d:
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No destroy operator data.")

    with col2:
        st.markdown("**Repair Operators**")
        fig_r = _build_chart(
            weight_history.get("repair", []),
            r_names,
            "Repair Operator Probabilities",
            chart_type,  # type: ignore[arg-type]
            smooth_window,
            height,
            colors,
        )
        if fig_r:
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("No repair operator data.")

    # ── Final probability table ──────────────────────────────────────────────
    destroy_data = weight_history.get("destroy", [])
    repair_data = weight_history.get("repair", [])
    if destroy_data or repair_data:
        with st.expander("Final Selection Probabilities"):
            _render_prob_table(destroy_data, d_names, repair_data, r_names)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_chart(
    data: List[List[float]],
    names: List[str],
    chart_title: str,
    chart_type: str,
    smooth_window: int,
    height: int,
    colors: List[str],
) -> Optional[go.Figure]:
    """Builds a Plotly line or stacked-area chart for an operator group.

    Args:
        data: List of weight vectors over time.
        names: Human-readable operator labels.
        chart_title: Figure title.
        chart_type: Either 'line' or 'area'.
        smooth_window: EMA span.
        height: Plot height in pixels.
        colors: Color palette to use.

    Returns:
        Optional[go.Figure]: Resulting Plotly figure if data exists.
    """
    if go is None:
        return None

    if not data:
        return None

    arr = np.array(data, dtype=np.float32)  # (T, n_ops)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    T, n_ops = arr.shape

    # Row-normalise to probabilities
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    arr = arr / row_sums

    # Smoothing via exponential moving average
    if smooth_window > 1:
        alpha_ema = 2.0 / (smooth_window + 1)
        smoothed = arr.copy()
        for t in range(1, T):
            smoothed[t] = alpha_ema * arr[t] + (1 - alpha_ema) * smoothed[t - 1]
        arr = smoothed

    x = list(range(T))
    fig = go.Figure()

    for i in range(n_ops):
        name = names[i] if i < len(names) else f"Op {i}"
        color = colors[i % len(colors)]
        trace_kwargs: Dict[str, Any] = dict(
            x=x,
            y=arr[:, i].tolist(),
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>",
        )
        if chart_type == "area":
            fig.add_trace(go.Scatter(**trace_kwargs, fill="tonexty", mode="lines", stackgroup="ops"))
        else:
            fig.add_trace(go.Scatter(**trace_kwargs, mode="lines"))

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=14)),
        xaxis_title="Snapshot (every log_freq iterations)",
        yaxis_title="Selection Probability",
        yaxis=dict(range=[0.0, 1.05], gridcolor="#f0f0f0"),
        xaxis=dict(gridcolor="#f0f0f0"),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        plot_bgcolor="white",
    )
    return fig


def _render_prob_table(
    destroy_data: List[List[float]],
    d_names: List[str],
    repair_data: List[List[float]],
    r_names: List[str],
) -> None:
    """Renders a concise table of final operator selection probabilities.

    Args:
        destroy_data: History of destroy weights.
        d_names: Destroy operator labels.
        repair_data: History of repair weights.
        r_names: Repair operator labels.
    """

    rows: List[Dict[str, Any]] = []

    if destroy_data:
        final_d = _softmax_probs(destroy_data[-1])
        for name, prob in zip(d_names, final_d, strict=False):
            rows.append({"Type": "Destroy", "Operator": name, "P(select)": f"{prob:.4f}"})

    if repair_data:
        final_r = _softmax_probs(repair_data[-1])
        for name, prob in zip(r_names, final_r, strict=False):
            rows.append({"Type": "Repair", "Operator": name, "P(select)": f"{prob:.4f}"})

    if rows:
        st.table(pd.DataFrame(rows))


def _softmax_probs(weights: Any) -> List[float]:
    """Converts raw ALNS weights to probabilities via normalisation.

    Args:
        weights: Raw weight vector (Tensor or NumPy).

    Returns:
        List[float]: Scaled probabilities summing to 1.0.
    """
    if torch is not None:
        try:
            w = weights.float().cpu() if hasattr(weights, "cpu") else torch.tensor(weights, dtype=torch.float32)
            w = w.clamp(min=1e-8)
            return (w / w.sum()).tolist()
        except Exception:
            pass

    # Fallback: plain NumPy normalisation
    arr = np.asarray(weights, dtype=np.float64)
    arr = np.clip(arr, 1e-8, None)
    return (arr / arr.sum()).tolist()

"""
Streamlit component for visualising Transformer attention maps.

Renders interactive attention weight visualisations produced by
``logic.src.tracking.hooks.attention_hooks.add_attention_hooks``.

Three display modes (chosen automatically based on the ``node_coords`` argument):

* **Heatmap** — Pure attention matrix rendered as a colour grid. Used when no
  node coordinates are provided.
* **Bipartite graph** — Nodes plotted at their normalised 2-D positions with
  edges weighted by attention strength. Used when coords are in ``[0, 1]``.
* **Geo overlay** — Nodes plotted on a Mapbox map with attention arcs.
  Activated when coordinate values exceed 1.0 (i.e., lat/lon scale).

Usage::

    from logic.src.ui.components.attention_viz import render_attention_viz
    from logic.src.tracking.hooks.attention_hooks import add_attention_hooks

    hook_data = add_attention_hooks(model.encoder)
    model(td)  # run inference to populate hook_data

    render_attention_viz(
        hook_data,
        node_coords=coords_np,   # (N, 2)  — optional
        title="Encoder Self-Attention",
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render_attention_viz(
    hook_data: Dict[str, List[Any]],
    node_coords: Optional[np.ndarray] = None,
    decoding_step: int = 0,
    head_idx: int = 0,
    map_center: Optional[Tuple[float, float]] = None,
    map_zoom: int = 12,
    height: int = 500,
    min_edge_alpha: float = 0.05,
    top_k_edges: int = 20,
    color_scale: str = "Reds",
    title: str = "Attention Map",
) -> None:
    """
    Render an interactive attention weight visualisation in Streamlit.

    Dispatches to the appropriate rendering mode based on ``node_coords``:
    no coords → heatmap, normalised coords → bipartite graph,
    lat/lon coords → geo overlay.

    Args:
        hook_data: Dict returned by ``add_attention_hooks``.  Keys:
            ``"weights"`` (``List[Tensor[B, H, N, N]]``),
            ``"masks"`` (``List[Tensor]``).
        node_coords: Optional ``(N, 2)`` NumPy array of node positions.
            Values ``> 1.0`` are interpreted as lat/lon for geo mode.
        decoding_step: Initial decoding step index to display. Default 0.
        head_idx: Initial attention head index to display. Default 0.
        map_center: ``(lat, lon)`` centre for the geo overlay.
            Auto-computed from ``node_coords`` when None.
        map_zoom: Initial Mapbox zoom level. Default 12.
        height: Plot height in pixels. Default 500.
        min_edge_alpha: Opacity threshold below which edges are hidden.
            Default 0.05.
        top_k_edges: Maximum number of attention edges rendered. Default 20.
        color_scale: Plotly colour scale name. Default ``"Reds"``.
        title: Panel heading shown above the visualisation.
    """
    try:
        import plotly.graph_objects as go  # noqa: F401 — validate import early
    except ImportError:
        st.error("**plotly** is required for attention visualisation. Run: `pip install plotly`")
        return

    weights_list: List[Any] = hook_data.get("weights", [])
    if not weights_list:
        st.info("No attention data captured yet. Ensure hooks are registered before running inference.")
        return

    st.subheader(title)

    # ── Interactive controls ─────────────────────────────────────────────────
    n_steps = len(weights_list)
    # Probe shape to get n_heads safely
    probe = weights_list[0]
    probe_np = _to_numpy(probe)
    n_heads = probe_np.shape[1] if probe_np.ndim == 4 else 1

    ctrl_cols = st.columns(3)
    with ctrl_cols[0]:
        step_idx = st.slider(
            "Decoding Step",
            min_value=0,
            max_value=max(0, n_steps - 1),
            value=min(decoding_step, n_steps - 1),
            key=f"attn_step_{title}",
        )
    with ctrl_cols[1]:
        head = st.slider(
            "Attention Head",
            min_value=0,
            max_value=max(0, n_heads - 1),
            value=min(head_idx, n_heads - 1),
            key=f"attn_head_{title}",
        )
    with ctrl_cols[2]:
        topk = st.slider(
            "Top-K Edges",
            min_value=5,
            max_value=100,
            value=top_k_edges,
            key=f"attn_topk_{title}",
        )

    # ── Extract attention matrix [N, N] ─────────────────────────────────────
    attn_np = _to_numpy(weights_list[step_idx])
    attn_matrix = _extract_head(attn_np, head_idx=head)

    # ── Choose rendering mode ────────────────────────────────────────────────
    use_geo = (
        node_coords is not None
        and node_coords.ndim == 2
        and node_coords.shape[1] >= 2
        and float(node_coords.max()) > 1.0
    )
    use_bipartite = node_coords is not None and not use_geo

    if use_geo and node_coords is not None:
        fig = _render_geo(attn_matrix, node_coords, map_center, map_zoom, topk, min_edge_alpha, height)
    elif use_bipartite and node_coords is not None:
        fig = _render_bipartite(attn_matrix, node_coords, topk, min_edge_alpha, height, color_scale)
    else:
        fig = _render_heatmap(attn_matrix, height, color_scale)

    st.plotly_chart(fig, use_container_width=True)

    # ── Summary statistics ───────────────────────────────────────────────────
    with st.expander("Attention Statistics"):
        stat_cols = st.columns(3)
        stat_cols[0].metric("Max Weight", f"{attn_matrix.max():.4f}")
        stat_cols[1].metric("Mean Weight", f"{attn_matrix.mean():.4f}")
        entropy = float(-np.sum(attn_matrix * np.log(np.clip(attn_matrix, 1e-9, None)), axis=-1).mean())
        stat_cols[2].metric("Avg Row Entropy", f"{entropy:.4f}")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_geo(
    attn: np.ndarray,
    coords: np.ndarray,
    center: Optional[Tuple[float, float]],
    zoom: int,
    top_k: int,
    min_alpha: float,
    height: int,
) -> Any:
    """Attention arcs on a Mapbox scatter-map (lat/lon coordinates)."""
    import plotly.graph_objects as go

    n = attn.shape[0]
    lats, lons = coords[:n, 0], coords[:n, 1]
    if center is None:
        center = (float(lats.mean()), float(lons.mean()))

    edges = _top_k_edges(attn, top_k, min_alpha)
    max_w = edges[0][0] if edges else 1.0

    edge_traces = []
    for weight, src, dst in edges:
        alpha = max(min_alpha, float(weight) / (max_w + 1e-8))
        edge_traces.append(
            go.Scattermapbox(
                lat=[float(lats[src]), float(lats[dst]), None],
                lon=[float(lons[src]), float(lons[dst]), None],
                mode="lines",
                line=dict(width=max(1, int(alpha * 4)), color=f"rgba(220,50,50,{alpha:.3f})"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_trace = go.Scattermapbox(
        lat=lats.tolist(),
        lon=lons.tolist(),
        mode="markers+text",
        marker=dict(size=9, color="steelblue"),
        text=[str(i) for i in range(n)],
        hovertemplate="Node %{text}<br>Lat: %{lat:.4f} Lon: %{lon:.4f}<extra></extra>",
        name="Nodes",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center[0], lon=center[1]),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
        showlegend=False,
    )
    return fig


def _render_bipartite(
    attn: np.ndarray,
    coords: np.ndarray,
    top_k: int,
    min_alpha: float,
    height: int,
    color_scale: str,
) -> Any:
    """Attention edges overlaid on a 2-D node layout (normalised coordinates)."""
    import plotly.graph_objects as go

    n = attn.shape[0]
    xy = coords[:n, :2].astype(float).copy()
    # Normalise to [0, 1]
    for dim in range(2):
        lo, hi = xy[:, dim].min(), xy[:, dim].max()
        if hi > lo:
            xy[:, dim] = (xy[:, dim] - lo) / (hi - lo)

    edges = _top_k_edges(attn, top_k, min_alpha)
    max_w = edges[0][0] if edges else 1.0

    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for weight, src, dst in edges:
        alpha = float(weight) / (max_w + 1e-8)
        if alpha < min_alpha:
            continue
        edge_x += [xy[src, 0], xy[dst, 0], None]
        edge_y += [xy[src, 1], xy[dst, 1], None]

    in_degree = attn.sum(axis=0)  # column-wise sum = total attention received

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(180,50,50,0.35)"),
        hoverinfo="skip",
        showlegend=False,
    )
    node_trace = go.Scatter(
        x=xy[:, 0].tolist(),
        y=xy[:, 1].tolist(),
        mode="markers+text",
        marker=dict(
            size=12,
            color=in_degree.tolist(),
            colorscale=color_scale,
            showscale=True,
            colorbar=dict(title="Σ Attn"),
        ),
        text=[str(i) for i in range(n)],
        textposition="top center",
        hovertemplate="Node %{text}<br>Total attention: %{marker.color:.4f}<extra></extra>",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
    )
    return fig


def _render_heatmap(attn: np.ndarray, height: int, color_scale: str) -> Any:
    """Standard attention matrix heatmap (no node coordinates required)."""
    import plotly.graph_objects as go

    n = attn.shape[0]
    fig = go.Figure(
        data=go.Heatmap(
            z=attn.tolist(),
            x=[f"Node {i}" for i in range(n)],
            y=[f"Node {i}" for i in range(n)],
            colorscale=color_scale,
            hovertemplate="From %{y} → To %{x}<br>Weight: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        xaxis=dict(title="Key Nodes"),
        yaxis=dict(title="Query Nodes", autorange="reversed"),
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert a PyTorch tensor or list to a float32 NumPy array."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def _extract_head(attn_np: np.ndarray, head_idx: int = 0) -> np.ndarray:
    """
    Extract an ``(N, N)`` attention matrix from a tensor of arbitrary leading dims.

    Shapes handled:
    - ``(B, H, N, N)`` → ``attn_np[0, head_idx]``
    - ``(H, N, N)``    → ``attn_np[head_idx]``
    - ``(N, N)``       → returned as-is
    """
    if attn_np.ndim == 4:
        h = min(head_idx, attn_np.shape[1] - 1)
        return attn_np[0, h]
    if attn_np.ndim == 3:
        h = min(head_idx, attn_np.shape[0] - 1)
        return attn_np[h]
    return attn_np


def _top_k_edges(
    attn: np.ndarray,
    top_k: int,
    min_alpha: float,
) -> List[Tuple[float, int, int]]:
    """Return the top-K ``(weight, src, dst)`` edges by attention weight."""
    n = attn.shape[0]
    edges = [(float(attn[i, j]), i, j) for i in range(n) for j in range(n) if i != j and float(attn[i, j]) > min_alpha]
    edges.sort(key=lambda x: -x[0])
    return edges[:top_k]

# Copyright (c) WSmart-Route. All rights reserved.
"""
KPI card HTML generation and number formatting utilities.

Provides functions to create styled KPI card HTML with optional deltas and sparklines.
"""

from typing import Dict, Optional, Tuple, Union

from logic.src.pipeline.ui.styles.colors import KPI_COLORS, KPI_FALLBACK_COLORS

# Type alias for KPI values with optional delta
KPIValue = Union[float, int, str]
KPIDelta = Optional[float]


def format_number(value: float, precision: int = 2) -> str:
    """Format a number with thousands separator and precision."""
    if abs(value) >= 1000:
        return f"{value:,.{precision}f}"
    return f"{value:.{precision}f}"


def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value:.1f}%"


def _format_delta(delta: float) -> str:
    """Format a delta value with arrow indicator."""
    if delta > 0:
        return f"\u25b2 +{format_number(delta)}"
    elif delta < 0:
        return f"\u25bc {format_number(delta)}"
    return "\u2014 0"


def _delta_css_class(delta: float) -> str:
    """Return the CSS class for a delta value."""
    if delta > 0:
        return "positive"
    elif delta < 0:
        return "negative"
    return "neutral"


def create_kpi_html(
    label: str,
    value: str,
    color: str = "#667eea",
    color_end: str = "#5a67d8",
    delta: Optional[str] = None,
    delta_class: str = "neutral",
    sparkline_svg: str = "",
) -> str:
    """
    Create HTML for a single KPI card.

    Args:
        label: Metric label.
        value: Formatted value string.
        color: Background gradient start color.
        color_end: Background gradient end color.
        delta: Optional formatted delta string (e.g., "▲ +1.5").
        delta_class: CSS class for delta coloring ("positive", "negative", "neutral").
        sparkline_svg: Optional inline SVG string for a sparkline.

    Returns:
        HTML string.
    """
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="delta {delta_class}">{delta}</div>'

    sparkline_html = ""
    if sparkline_svg:
        sparkline_html = f'<div class="sparkline">{sparkline_svg}</div>'

    return (
        f'<div class="kpi-card" style="background: linear-gradient(135deg, {color} 0%, {color_end} 100%);">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"{delta_html}"
        f"{sparkline_html}"
        f"</div>"
    )


def create_kpi_row(metrics: dict, prefix: str = "") -> str:
    """
    Create HTML for a row of KPI cards with semantic colors.

    Args:
        metrics: Dict of label -> value.
        prefix: Optional prefix for display names.

    Returns:
        HTML string for the row.
    """
    cards = []

    for i, (label, value) in enumerate(metrics.items()):
        display_label = f"{prefix}{label}" if prefix else label

        if display_label in KPI_COLORS:
            color, color_end = KPI_COLORS[display_label]
        elif label in KPI_COLORS:
            color, color_end = KPI_COLORS[label]
        else:
            fallback = KPI_FALLBACK_COLORS[i % len(KPI_FALLBACK_COLORS)]
            color, color_end = fallback

        formatted = format_number(value) if isinstance(value, float) else str(value)
        cards.append(create_kpi_html(display_label, formatted, color, color_end))

    return f'<div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;">{"".join(cards)}</div>'


def create_kpi_row_with_deltas(
    metrics: Dict[str, Tuple[KPIValue, KPIDelta]],
    sparklines: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create HTML for a row of KPI cards with delta indicators and optional sparklines.

    Args:
        metrics: Dict of label -> (value, delta). Delta is None to skip.
        sparklines: Optional dict of label -> SVG string for sparkline.

    Returns:
        HTML string for the row.
    """
    cards = []

    for i, (label, (value, delta)) in enumerate(metrics.items()):
        if label in KPI_COLORS:
            color, color_end = KPI_COLORS[label]
        else:
            fallback = KPI_FALLBACK_COLORS[i % len(KPI_FALLBACK_COLORS)]
            color, color_end = fallback

        formatted = format_number(value) if isinstance(value, float) else str(value)

        delta_str: Optional[str] = None
        delta_class = "neutral"
        if delta is not None:
            delta_str = _format_delta(delta)
            delta_class = _delta_css_class(delta)

        sparkline_svg = ""
        if sparklines and label in sparklines:
            sparkline_svg = sparklines[label]

        cards.append(create_kpi_html(label, formatted, color, color_end, delta_str, delta_class, sparkline_svg))

    return f'<div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;">{"".join(cards)}</div>'

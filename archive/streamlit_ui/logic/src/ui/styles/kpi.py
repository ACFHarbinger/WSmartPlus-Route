"""
KPI card HTML generation and number formatting utilities.

Provides functions to create styled KPI card HTML with optional deltas and sparklines.

Attributes:
    KPIValue: Type alias for KPI values (float, int, or str).
    KPIDelta: Type alias for optional float delta values.
    renderer: Global instance of KPIRenderer for HTML generation.

Example:
    >>> from logic.src.ui.styles.kpi import create_kpi_row
    >>> metrics = {"Profit": 1250.5, "Distance": 45.2}
    >>> html = create_kpi_row(metrics)
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import jinja2

from logic.src.ui.styles.colors import KPI_COLORS, KPI_FALLBACK_COLORS

# Type alias for KPI values with optional delta
KPIValue = Union[float, int, str]
KPIDelta = Optional[float]


def format_number(value: float, precision: int = 2) -> str:
    """Format a number with thousands separator and precision.

    Args:
        value (float): The numeric value to format.
        precision (int): Number of decimal places (default: 2).

    Returns:
        str: Formatted string (e.g., "1,250.50").
    """
    if abs(value) >= 1000:
        return f"{value:,.{precision}f}"
    return f"{value:.{precision}f}"


def format_percentage(value: float) -> str:
    """Format a value as percentage.

    Args:
        value (float): The numeric value (e.g., 85.5).

    Returns:
        str: Formatted percentage string (e.g., "85.5%").
    """
    return f"{value:.1f}%"


def _format_delta(delta: float) -> str:
    """Format a delta value with arrow indicator.

    Args:
        delta (float): The change value.

    Returns:
        str: Formatted delta string with ▲/▼ arrows.
    """
    if delta > 0:
        return f"\u25b2 +{format_number(delta)}"
    elif delta < 0:
        return f"\u25bc {format_number(delta)}"
    return "\u2014 0"


def _delta_css_class(delta: float) -> str:
    """Return the CSS class for a delta value.

    Args:
        delta (float): The change value.

    Returns:
        str: CSS class name ("positive", "negative", or "neutral").
    """
    if delta > 0:
        return "positive"
    elif delta < 0:
        return "negative"
    return "neutral"


class KPIRenderer:
    """Renderer for KPI components using Jinja2 templates.

    Attributes:
        env (jinja2.Environment): Jinja2 environment for template loading.
        card_template (jinja2.Template): Template for individual KPI cards.
        row_template (jinja2.Template): Template for a row container of cards.
    """

    def __init__(self):
        """Initialize the renderer and load templates."""
        # Setup Jinja environment to look in the same directory as this file
        template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
        self.env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        self.card_template = self.env.get_template("kpi_card.html")
        self.row_template = self.env.get_template("kpi_row.html")

    def render_card(self, label: str, value: str, **kwargs) -> str:
        """Renders a single KPI card using the HTML template.

        Args:
            label (str): Metric title.
            value (str): Formatted primary value.
            kwargs: Additional template variables (color, delta, etc.).

        Returns:
            str: Rendered HTML card content.
        """
        return self.card_template.render(label=label, value=value, **kwargs)

    def render_row(self, cards_html: List[str]) -> str:
        """Wraps multiple card strings into a flex container.

        Args:
            cards_html (List[str]): List of rendered card HTML strings.

        Returns:
            str: Wrapped HTML row content.
        """
        return self.row_template.render(cards=cards_html)


# Global renderer instance
renderer = KPIRenderer()


def create_kpi_html(
    label: str,
    value: str,
    color: str = "#667eea",
    color_end: str = "#5a67d8",
    delta: Optional[str] = None,
    delta_class: str = "neutral",
    sparkline_svg: str = "",
) -> str:
    """Wrapper function that now uses the template renderer.

    Args:
        label (str): Metric label.
        value (str): Display value.
        color (str): Start color for gradient.
        color_end (str): End color for gradient.
        delta (Optional[str]): Formatted delta string.
        delta_class (str): CSS class for delta.
        sparkline_svg (str): Raw SVG content for sparkline.

    Returns:
        str: Rendered HTML for a single KPI card.
    """
    return renderer.render_card(
        label=label,
        value=value,
        color=color,
        color_end=color_end,
        delta=delta,
        delta_class=delta_class,
        sparkline_svg=sparkline_svg,
    )


def create_kpi_row(metrics: Dict[str, Union[float, int, str]], prefix: str = "") -> str:
    """Logic for color selection and HTML generation for a row of metrics.

    Args:
        metrics (Dict[str, Union[float, int, str]]): Mapping of label to value.
        prefix (str): Optional prefix for all labels.

    Returns:
        str: HTML flexbox row containing the KPI cards.
    """
    cards = []
    for i, (label, value) in enumerate(metrics.items()):
        display_label = f"{prefix}{label}" if prefix else label

        # Color resolution logic from original kpi.py
        if display_label in KPI_COLORS:
            color, color_end = KPI_COLORS[display_label]
        elif label in KPI_COLORS:
            color, color_end = KPI_COLORS[label]
        else:
            color, color_end = KPI_FALLBACK_COLORS[i % len(KPI_FALLBACK_COLORS)]

        formatted = format_number(value) if isinstance(value, float) else str(value)
        cards.append(create_kpi_html(display_label, formatted, color, color_end))

    return renderer.render_row(cards)


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
        HTML string for the row container.
    """
    cards = []

    for i, (label, (value, delta)) in enumerate(metrics.items()):
        # 1. Resolve colors based on global registries or positional fallbacks
        if label in KPI_COLORS:
            color, color_end = KPI_COLORS[label]
        else:
            fallback = KPI_FALLBACK_COLORS[i % len(KPI_FALLBACK_COLORS)]
            color, color_end = fallback

        # 2. Format the primary value
        formatted = format_number(value) if isinstance(value, float) else str(value)

        # 3. Process delta indicators and their corresponding CSS classes
        delta_str: Optional[str] = None
        delta_class = "neutral"
        if delta is not None:
            delta_str = _format_delta(delta)
            delta_class = _delta_css_class(delta)

        # 4. Extract sparkline SVG if provided for this specific label
        sparkline_svg = ""
        if sparklines and label in sparklines:
            sparkline_svg = sparklines[label]

        # 5. Render individual card using the external template
        card_html = renderer.render_card(
            label=label,
            value=formatted,
            color=color,
            color_end=color_end,
            delta=delta_str,
            delta_class=delta_class,
            sparkline_svg=sparkline_svg,
        )
        cards.append(card_html)

    # 6. Finalize the row by wrapping all cards in the flex container template
    return renderer.render_row(cards)

"""
Styling package for the dashboard.

Re-exports all public names from submodules for convenient access.

Attributes:
    CHART_COLORS: List of default charting colors.
    CUSTOM_CSS: consolidated CSS string.
    KPI_COLORS: Mapping of KPI labels to colors.
    KPI_FALLBACK_COLORS: List of fallback colors for KPIs.
    KPIDelta: Delta type alias.
    KPIValue: Value type alias.
    ROUTE_COLORS: List of route colors.
    STATUS_COLORS: Mapping of status to colors.

Example:
    >>> from logic.src.ui.styles import CUSTOM_CSS
    >>> print(CUSTOM_CSS[:10])
    <style>
"""

from logic.src.ui.styles.colors import (
    CHART_COLORS,
    KPI_COLORS,
    KPI_FALLBACK_COLORS,
    ROUTE_COLORS,
    STATUS_COLORS,
    get_page_config,
)
from logic.src.ui.styles.css import CUSTOM_CSS
from logic.src.ui.styles.kpi import (
    KPIDelta,
    KPIValue,
    create_kpi_html,
    create_kpi_row,
    create_kpi_row_with_deltas,
    format_number,
    format_percentage,
)

__all__ = [
    "CHART_COLORS",
    "CUSTOM_CSS",
    "KPI_COLORS",
    "KPI_FALLBACK_COLORS",
    "KPIDelta",
    "KPIValue",
    "ROUTE_COLORS",
    "STATUS_COLORS",
    "create_kpi_html",
    "create_kpi_row",
    "create_kpi_row_with_deltas",
    "format_number",
    "format_percentage",
    "get_page_config",
]

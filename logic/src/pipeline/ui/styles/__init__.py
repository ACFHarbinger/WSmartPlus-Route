# Copyright (c) WSmart-Route. All rights reserved.
"""
Styling package for the dashboard.

Re-exports all public names from submodules for convenient access.
"""

from logic.src.pipeline.ui.styles.colors import (
    CHART_COLORS,
    KPI_COLORS,
    KPI_FALLBACK_COLORS,
    ROUTE_COLORS,
    STATUS_COLORS,
    get_page_config,
)
from logic.src.pipeline.ui.styles.css import CUSTOM_CSS
from logic.src.pipeline.ui.styles.kpi import (
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

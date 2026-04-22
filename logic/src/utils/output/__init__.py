"""
Output utilities package.

Attributes:
    excel_summary: Utility to aggregate simulation results JSONs into a single Excel log.

Example:
    >>> from logic.src.utils.output import excel_summary
    >>> df = excel_summary.discover_and_aggregate()
    >>> print(df.head())
"""

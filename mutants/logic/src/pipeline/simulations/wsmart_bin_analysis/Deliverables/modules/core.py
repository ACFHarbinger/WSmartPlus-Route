"""
Core components for Container management and TAG enum.
"""

from enum import Enum
from typing import Optional

import pandas as pd


class TAG(Enum):
    """Quality tags indicating container data reliability level."""

    LOW_MEASURES = 0
    INSIDE_BOX = 1
    OK = 2
    WARN = 3
    LOCAL_WARN = 4


class DataMixin:
    """Mixin providing core data storage for Container objects."""

    def __init__(self, my_df: pd.DataFrame, my_rec: pd.DataFrame, info: pd.DataFrame):
        """Initialize the data mixin with fill data, records, and info."""
        self.df: pd.DataFrame
        self.info: pd.DataFrame
        self.recs: pd.DataFrame
        self.id: int
        self.tag: Optional[TAG] = None

        self.df = my_df.set_index("Date")
        self.df.drop(["ID"], axis=1, inplace=True, errors="ignore")
        self.info = info
        self.id = info["ID"]
        self.recs = my_rec.set_index("Date")
        self.recs.drop(["ID"], axis=1, inplace=True, errors="ignore")

    def __del__(self):
        """Clean up instance attributes to free memory."""
        if hasattr(self, "df"):
            del self.df
        if hasattr(self, "info"):
            del self.info
        if hasattr(self, "recs"):
            del self.recs
        if hasattr(self, "id"):
            del self.id

    def get_keys(self):
        """Return dictionary keys for fill data, records, and info."""
        return {
            "FILL": list(self.df.keys()),
            "RECS": list(self.recs.keys()),
            "INFO": list(self.info.keys()),
        }

    def get_vars(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return the fill DataFrame, records DataFrame, and info."""
        return self.df, self.recs, self.info

    def set_tag(self, tag: TAG):
        """Set the containers tag manually"""
        self.tag = tag

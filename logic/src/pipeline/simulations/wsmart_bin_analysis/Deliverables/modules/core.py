"""
Core components for Container management.

Attributes:
    DataMixin (Mixin): Mixin providing core data storage for Container objects.

Example:
    None
"""

from typing import Optional

import pandas as pd
from logic.src.pipeline.simulations.wsmart_bin_analysis.Deliverables.enums.tag import TAG


class DataMixin:
    """Mixin providing core data storage for Container objects.

    Attributes:
        df (pd.DataFrame): DataFrame containing fill level data with a DatetimeIndex.
        info (pd.DataFrame): DataFrame containing container information.
        recs (pd.DataFrame): DataFrame containing collection event records with a DatetimeIndex.
        id (int): The identifier of the container.
        tag (Optional[TAG]): The tag assigned to the container.
    """

    def __init__(self, my_df: pd.DataFrame, my_rec: pd.DataFrame, info: pd.DataFrame) -> None:
        """Initialize the data mixin with fill data, records, and info.

        Args:
            my_df (pd.DataFrame): DataFrame containing fill level data with a DatetimeIndex.
            my_rec (pd.DataFrame): DataFrame containing collection event records with a DatetimeIndex.
            info (pd.DataFrame): DataFrame containing container information.

        Returns:
            None
        """
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

    def __del__(self) -> None:
        """Clean up instance attributes to free memory.

        Returns:
            None
        """
        if hasattr(self, "df"):
            del self.df
        if hasattr(self, "info"):
            del self.info
        if hasattr(self, "recs"):
            del self.recs
        if hasattr(self, "id"):
            del self.id

    def get_keys(self) -> dict[str, list[str]]:
        """Return dictionary keys for fill data, records, and info.

        Returns:
            dict[str, list[str]]: The dictionary keys for fill data, records, and info.
        """
        return {
            "FILL": list(self.df.keys()),
            "RECS": list(self.recs.keys()),
            "INFO": list(self.info.keys()),
        }

    def get_vars(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return the fill DataFrame, records DataFrame, and info.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The fill DataFrame, records DataFrame, and info.
        """
        return self.df, self.recs, self.info

    def set_tag(self, tag: TAG) -> None:
        """Set the containers tag manually

        Args:
            tag (TAG): The tag to set.

        Returns:
            None
        """
        self.tag = tag

"""
Container management module for wsmart_bin_analysis.
This module acts as a facade, using mixins to organize functionality.

Attributes:
    Container: The class that manages the background behind a container.

Example:
    >>> from logic.src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container import Container
    >>> c = Container(my_df, my_rec, info)
"""

import pandas as pd

from .modules.analysis_utils import AnalysisMixin
from .modules.core import TAG, DataMixin
from .modules.plotting_utils import VisualizationMixin
from .modules.processing_utils import ProcessingMixin


class Container(DataMixin, AnalysisMixin, ProcessingMixin, VisualizationMixin):
    """
    Overall object to deal with container data. It supports manipulation operations over the container,
    managing the fill levels and collection events.

    Attributes:
        my_df: pd.DataFrame with fill-level information for a container.
        my_rec: pd.DataFrame with collection-event information for a container.
        info: pd.DataFrame with additional information for a container.
    """

    def __init__(self, my_df: pd.DataFrame, my_rec: pd.DataFrame, info: pd.DataFrame):
        """Initialize the Container with data, records, and info.

        Args:
            my_df (pd.DataFrame):
                Dataframe with fill-level information for a container.
            my_rec (pd.DataFrame):
                Dataframe with collection-event information for a container.
            info (pd.DataFrame):
                Dataframe with additional information for a container.
        """
        # Call the first mixin's __init__ (DataMixin)
        super().__init__(my_df, my_rec, info)


__all__ = ["Container", "TAG"]

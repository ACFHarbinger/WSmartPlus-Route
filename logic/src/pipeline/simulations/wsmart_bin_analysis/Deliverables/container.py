"""
Container management module for wsmart_bin_analysis.
This module acts as a facade, using mixins to organize functionality.
"""
from .modules.analysis_utils import AnalysisMixin
from .modules.core import TAG, DataMixin
from .modules.plotting_utils import VisualizationMixin
from .modules.processing_utils import ProcessingMixin


class Container(DataMixin, AnalysisMixin, ProcessingMixin, VisualizationMixin):
    """
    Overall object to deal with container data. It supports manipulation operations over the container,
    managing the fill levels and collection events.
    """

    def __init__(self, my_df, my_rec, info):
        # Call the first mixin's __init__ (DataMixin)
        super().__init__(my_df, my_rec, info)


__all__ = ["Container", "TAG"]

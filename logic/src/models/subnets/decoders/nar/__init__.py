"""nar decoder package.

This package provides non-autoregressive (NAR) decoders that utilize
heatmaps for constructive routing.

Attributes:
    SimpleNARDecoder: Basic heatmap-based NAR decoder.

Example:
    >>> from logic.src.models.subnets.decoders.nar import SimpleNARDecoder
    >>> decoder = SimpleNARDecoder()
"""

from .decoder import SimpleNARDecoder as SimpleNARDecoder

"""tag.py module.

Attributes:
    TAG:
        An enum class that represents the quality tags indicating
        container data reliability level.

Example:
    >>> from logic.src.pipeline.simulations.wsmart_bin_analysis.Deliverables.enums.tag import TAG
    >>> TAG.LOW_MEASURES.value
    0
"""

from enum import Enum


class TAG(Enum):
    """
    Quality tags indicating container data reliability level.

    Attributes:
        LOW_MEASURES: Low number of measurements.
        INSIDE_BOX: Measurements taken inside a box.
        OK: Acceptable data quality.
        WARN: Warning level of data quality.
        LOCAL_WARN: Local warning level of data quality.
    """

    LOW_MEASURES = 0
    INSIDE_BOX = 1
    OK = 2
    WARN = 3
    LOCAL_WARN = 4

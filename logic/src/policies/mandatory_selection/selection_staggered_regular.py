"""
Staggered Regular Selection Module.

Extends the periodic RegularSelection by assigning each bin a fixed phase
offset derived from its position in the bin list:

    φ_i = i % X      (i is the 0-based bin index, X is the period)

Bin i is mandated on operational day t if and only if:

    (t - 1) % X == φ_i

This distributes collection load uniformly across the scheduling horizon:
on every day, approximately 1/X of all bins are mandated, eliminating the
all-or-nothing load spikes of the base RegularSelection (which mandates
every bin simultaneously every X days).

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_staggered_regular import StaggeredRegularSelection
    >>> strategy = StaggeredRegularSelection()
    >>> bins, ctx = strategy.select_bins(context)
"""

from typing import List, Optional, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import (
    MandatorySelectionRegistry,
)


@MandatorySelectionRegistry.register("staggered_regular")
class StaggeredRegularSelection(IMandatorySelectionStrategy):
    """
    Phase-staggered periodic collection strategy.

    Each bin i (0-indexed position in the bin list) is assigned a fixed phase
    offset:

        φ_i = i % X

    where X is the collection period (``threshold``). Bin i is mandated on
    day t if:

        (t - 1) % X == φ_i

    Design rationale
    ----------------
    The base ``RegularSelection`` mandates all bins simultaneously every X
    days, creating vehicle-fleet demand spikes that are incompatible with
    fixed-capacity routing. Staggered Regular resolves this by converting the
    global period into per-bin individual schedules with uniformly distributed
    offsets, ensuring the daily mandatory set has expected cardinality n/X.
    This models how real operators stagger collection across days of the week
    (e.g., zone A on Monday, zone B on Tuesday) without any spatial awareness.

    Compared to K-Means Sector Selection, Staggered Regular assigns phases
    purely by bin index order (no coordinate data required), making it
    lighter-weight but spatially blind.

    SelectionContext fields consumed
    --------------------------------
    threshold    (int)         : Collection period X ≥ 1. Defaults to 1.
    current_day  (int)         : Current operational day t. Defaults to 1.
    current_fill (ndarray[n]) : Per-bin fill ratios in [0, 1]. Optional.
    bin_ids      (ndarray[n]) : 0-based bin IDs used when fill is absent.

    Instance attributes
    -------------------
    min_fill (float): Minimum fill ratio for a bin to be eligible.
                      Defaults to 0.0.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Return bin IDs whose phase offset matches today's slot in the period.

        The active day-slot is ``(current_day - 1) % X``. A bin at 0-indexed
        position i is selected when ``i % X == day_slot`` and its fill level
        meets ``min_fill``.

        Args:
            context: SelectionContext providing current_day, threshold, and
                     optionally current_fill or bin_ids.

        Returns:
            A 2-tuple of:
            - List[int]: 1-based bin IDs due for collection today.
            - SearchContext: Populated with strategy name, the active day-slot
              within the period, period length, and number of bins selected.
        """
        current_day: int = getattr(context, "current_day", 1)
        threshold: int = max(1, int(getattr(context, "threshold", 1)))
        min_fill: float = float(getattr(self, "min_fill", 0.0))

        current_fill: Optional[np.ndarray] = getattr(context, "current_fill", None)
        bin_ids: Optional[np.ndarray] = getattr(context, "bin_ids", None)

        # The slot within [0, X) that today maps to. Bin i is due iff φ_i == day_slot.
        day_slot: int = (current_day - 1) % threshold

        eligible_bins: List[int] = []

        if current_fill is not None and len(current_fill) > 0:
            for i, fill in enumerate(current_fill):
                # Phase offset of bin i is i % X; collect iff it matches today's slot.
                if i % threshold != day_slot:
                    continue
                if fill < min_fill:
                    continue
                eligible_bins.append(i + 1)  # 1-based indexing for routing
        elif bin_ids is not None and len(bin_ids) > 0:
            # Fall back to position-in-array index when fill data is absent.
            for idx, raw_id in enumerate(bin_ids):
                if idx % threshold == day_slot:
                    eligible_bins.append(int(raw_id) + 1)

        metrics: dict = {
            "strategy": "StaggeredRegularSelection",
            "day_slot": day_slot,
            "threshold": threshold,
            "n_selected": len(eligible_bins),
        }
        return eligible_bins, SearchContext.initialize(selection_metrics=metrics)

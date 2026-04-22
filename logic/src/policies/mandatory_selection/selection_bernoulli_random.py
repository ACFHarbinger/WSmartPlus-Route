"""
Bernoulli Trial Random Selection Module.

Each bin is independently included in the mandatory set via a Bernoulli
trial with fixed probability p:

    X_i ~ Bernoulli(p),  i = 1, ..., n

A bin is mandated if X_i = 1. Unlike fixed-K random selection, the realised
set cardinality is random:

    |S| ~ Binomial(n, p)

with E[|S|] = n·p, providing a stochastic baseline whose variability in set
size mirrors a memoryless per-bin decision process.

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_bernoulli import BernoulliRandomSelection
    >>> strategy = BernoulliRandomSelection()
    >>> bins, ctx = strategy.select_bins(context)
"""

import random
from typing import List, Optional, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import (
    MandatorySelectionRegistry,
)


@MandatorySelectionRegistry.register("bernoulli_random")
class BernoulliRandomSelection(IMandatorySelectionStrategy):
    """
    Independent Bernoulli trial selection strategy.

    Each eligible bin i is independently mandated with probability p:

        X_i ~ Bernoulli(p)

    The ``threshold`` parameter is interpreted as the selection probability p
    and is clipped to [0, 1]. This strategy is the canonical stochastic null
    baseline: any deterministic strategy that does not significantly outperform
    it under identical conditions has a validity concern.

    Unlike Top-K random selection (which fixes |S| = K), the Bernoulli
    formulation decouples the selection decision at each bin, making the
    realised set size itself a random variable. This is the correct model when
    each bin's urgency is assessed independently without knowledge of how many
    others will be collected that day.

    SelectionContext fields consumed
    --------------------------------
    threshold    (float) : Selection probability p ∈ [0, 1]. Defaults to 0.5.
    current_day  (int)   : Used to derive a per-day RNG seed when no explicit
                           seed is provided, ensuring day-level reproducibility
                           while varying across days within a simulation episode.
    current_fill (ndarray[n]) : Per-bin fill ratios; used for min_fill filter.
    bin_ids      (ndarray[n]) : 0-based bin IDs used when fill is absent.
    seed         (int)   : Optional fixed RNG seed. When None (default), a
                           seed is derived from current_day to produce different
                           but deterministic draws each day.

    Instance attributes
    -------------------
    min_fill (float): Minimum fill ratio for a bin to be eligible for the
                      Bernoulli trial. Defaults to 0.0.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Sample each eligible bin independently via Bernoulli(p).

        Bins below ``min_fill`` are excluded before sampling (they are not
        eligible for the trial, not just rejected by it), so the effective
        population for the Binomial distribution is the eligible subset.

        Args:
            context: SelectionContext providing threshold (as p), current_fill
                     or bin_ids, and optionally seed.

        Returns:
            A 2-tuple of:
            - List[int]: 1-based bin IDs drawn as X_i = 1.
            - SearchContext: Populated with strategy name, p, number of
              eligible bins trialled, number selected, and the realised
              selection rate for diagnostics.
        """
        p: float = float(getattr(context, "threshold", 0.5))
        p = max(0.0, min(1.0, p))  # Clip to valid Bernoulli parameter range.

        min_fill: float = float(getattr(self, "min_fill", 0.0))

        # Seed resolution: prefer an explicit seed; fall back to current_day
        # so that each simulation day produces a different but reproducible draw.
        explicit_seed: Optional[int] = getattr(context, "seed", None)
        current_day: int = getattr(context, "current_day", 1)
        seed: Optional[int] = explicit_seed if explicit_seed is not None else current_day

        current_fill: Optional[np.ndarray] = getattr(context, "current_fill", None)
        bin_ids: Optional[np.ndarray] = getattr(context, "bin_ids", None)

        rng = random.Random(seed)

        eligible_bins: List[int] = []
        n_trials: int = 0  # Number of bins that entered the Bernoulli trial.

        if current_fill is not None and len(current_fill) > 0:
            for i, fill in enumerate(current_fill):
                if fill < min_fill:
                    continue  # Below eligibility threshold — not trialled.
                n_trials += 1
                if rng.random() < p:
                    eligible_bins.append(i + 1)  # 1-based indexing for routing
        elif bin_ids is not None and len(bin_ids) > 0:
            for raw_id in bin_ids:
                n_trials += 1
                if rng.random() < p:
                    eligible_bins.append(int(raw_id) + 1)

        metrics: dict = {
            "strategy": "BernoulliRandomSelection",
            "p": p,
            "n_trials": n_trials,
            "n_selected": len(eligible_bins),
            # Realised rate for diagnosing RNG quality in simulation logs.
            "realised_rate": len(eligible_bins) / n_trials if n_trials > 0 else 0.0,
        }
        return eligible_bins, SearchContext.initialize(selection_metrics=metrics)

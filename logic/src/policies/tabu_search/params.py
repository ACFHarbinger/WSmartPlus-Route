"""
Configuration parameters for the Tabu Search (TS) solver.

Based on Fred Glover's "Tabu Search Fundamentals and Uses" (1995).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TSParams:
    """
    Configuration for the Tabu Search solver.

    TS uses adaptive memory (short-term and long-term) and responsive
    exploration through recency-based and frequency-based memory structures.

    Attributes:
        # Short-term memory (Recency-based)
        tabu_tenure: Fixed tabu tenure (number of iterations).
        dynamic_tenure: Whether to use dynamic tenure based on solution quality.
        min_tenure: Minimum tenure for dynamic adjustment.
        max_tenure: Maximum tenure for dynamic adjustment.

        # Aspiration criteria
        aspiration_enabled: Enable aspiration criteria to override tabu.

        # Long-term memory (Frequency-based)
        intensification_enabled: Enable intensification strategies.
        diversification_enabled: Enable diversification strategies.
        intensification_interval: Iterations between intensification phases.
        diversification_interval: Iterations between diversification phases.
        elite_size: Number of elite solutions to maintain.
        frequency_penalty_weight: Weight for frequency-based penalties.

        # Candidate list strategies
        candidate_list_enabled: Use candidate list to restrict neighborhood.
        candidate_list_size: Maximum size of candidate list.

        # Strategic oscillation
        oscillation_enabled: Enable strategic oscillation.
        feasibility_tolerance: Tolerance for infeasibility during oscillation.

        # General search parameters
        max_iterations: Maximum number of iterations.
        max_iterations_no_improve: Restart if no improvement for this many iterations.
        n_removal: Number of nodes to remove per destroy step.
        n_llh: Number of low-level heuristics in the pool.
        time_limit: Wall-clock time limit in seconds.

        # Neighborhood structure
        use_swap: Enable swap neighborhood.
        use_relocate: Enable relocate neighborhood.
        use_2opt: Enable 2-opt neighborhood.
        use_insertion: Enable insertion-based neighborhoods.
    """

    # Short-term memory
    tabu_tenure: int = 7
    dynamic_tenure: bool = True
    min_tenure: int = 5
    max_tenure: int = 15

    # Aspiration criteria
    aspiration_enabled: bool = True

    # Long-term memory
    intensification_enabled: bool = True
    diversification_enabled: bool = True
    intensification_interval: int = 100
    diversification_interval: int = 200
    elite_size: int = 5
    frequency_penalty_weight: float = 0.1

    # Candidate list
    candidate_list_enabled: bool = True
    candidate_list_size: int = 20

    # Strategic oscillation
    oscillation_enabled: bool = False
    feasibility_tolerance: float = 0.1

    # General search
    max_iterations: int = 5000
    max_iterations_no_improve: int = 500
    n_removal: int = 3
    n_llh: int = 5
    time_limit: float = 60.0

    # Neighborhood structure
    use_swap: bool = True
    use_relocate: bool = True
    use_2opt: bool = True
    use_insertion: bool = True

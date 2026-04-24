"""Acceptance Criteria Package.

This package contains implementations of various move acceptance criteria
for metaheuristic search (e.g., SA, Late Acceptance, Great Deluge).

Attributes:
    adaptive_boltzmann_metropolis: Adaptive Boltzmann Metropolis criterion.
    all_moves_accepted: Criterion that accepts all moves.
    aspiration_criterion: Aspiration-based acceptance.
    binary_tournament_acceptance: Binary tournament selection for acceptance.
    boltzmann_metropolis_criterion: Standard Boltzmann Metropolis (SA).
    demon_algorithm: Demon algorithm acceptance.
    ensemble_move_acceptance: Ensemble-based acceptance.
    epsilon_dominance: Epsilon-dominance for multi-objective search.
    exponential_monte_carlo_counter: EMCC criterion.
    fitness_proportional: Fitness-proportional acceptance.
    generalized_tsallis_simulated_annealing: Tsallis-based SA.
    great_deluge: Standard Great Deluge algorithm.
    improving_and_equal: Accepts improving or equal cost moves.
    late_acceptance_hill_climbing: LAHC criterion.
    monte_carlo: General Monte Carlo acceptance.
    non_linear_great_deluge: NLGD variant.
    old_bachelor_acceptance: Old Bachelor acceptance.
    only_improving: Strict descent criterion.
    pareto_dominance: Pareto-dominance for multi-objective search.
    probabilistic_transition: Probabilistic transition criterion.
    record_to_record_travel: Record-to-record travel criterion.
    skewed_variable_neighborhood_search: Skewed VNS acceptance.
    step_counting_hill_climbing: SCHC criterion.
    threshold_accepting: Threshold accepting algorithm.

Example:
    >>> from logic.src.policies.acceptance_criteria import adaptive_boltzmann_metropolis
"""

from . import adaptive_boltzmann_metropolis as adaptive_boltzmann_metropolis
from . import all_moves_accepted as all_moves_accepted
from . import aspiration_criterion as aspiration_criterion
from . import binary_tournament_acceptance as binary_tournament_acceptance
from . import boltzmann_metropolis_criterion as boltzmann_metropolis_criterion
from . import demon_algorithm as demon_algorithm
from . import ensemble_move_acceptance as ensemble_move_acceptance
from . import epsilon_dominance as epsilon_dominance
from . import exponential_monte_carlo_counter as exponential_monte_carlo_counter
from . import fitness_proportional as fitness_proportional
from . import generalized_tsallis_simulated_annealing as generalized_tsallis_simulated_annealing
from . import great_deluge as great_deluge
from . import improving_and_equal as improving_and_equal
from . import late_acceptance_hill_climbing as late_acceptance_hill_climbing
from . import monte_carlo as monte_carlo
from . import non_linear_great_deluge as non_linear_great_deluge
from . import old_bachelor_acceptance as old_bachelor_acceptance
from . import only_improving as only_improving
from . import pareto_dominance as pareto_dominance
from . import probabilistic_transition as probabilistic_transition
from . import record_to_record_travel as record_to_record_travel
from . import skewed_variable_neighborhood_search as skewed_variable_neighborhood_search
from . import step_counting_hill_climbing as step_counting_hill_climbing
from . import threshold_accepting as threshold_accepting

__all__ = [
    "adaptive_boltzmann_metropolis",
    "all_moves_accepted",
    "aspiration_criterion",
    "boltzmann_metropolis_criterion",
    "demon_algorithm",
    "ensemble_move_acceptance",
    "epsilon_dominance",
    "exponential_monte_carlo_counter",
    "fitness_proportional",
    "generalized_tsallis_simulated_annealing",
    "great_deluge",
    "improving_and_equal",
    "late_acceptance_hill_climbing",
    "monte_carlo",
    "non_linear_great_deluge",
    "old_bachelor_acceptance",
    "only_improving",
    "pareto_dominance",
    "probabilistic_transition",
    "record_to_record_travel",
    "skewed_variable_neighborhood_search",
    "step_counting_hill_climbing",
    "threshold_accepting",
    "binary_tournament_acceptance",
]

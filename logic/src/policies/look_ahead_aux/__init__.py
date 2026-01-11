"""
Public exports for look-ahead auxiliary route construction and optimization.

This package provides a comprehensive suite of utilities for solving profit-
based vehicle routing problems, including mutation operators, feasibility 
checks, performance metrics, and high-level search orchestrators.
"""

from .routes import create_points as create_points
from .solutions import compute_initial_solution as compute_initial_solution, improved_simulated_annealing as improved_simulated_annealing, find_solutions as find_solutions
from .update import should_bin_be_collected as should_bin_be_collected, get_next_collection_day as get_next_collection_day, add_bins_to_collect as add_bins_to_collect, update_fill_levels_after_first_collection as update_fill_levels_after_first_collection

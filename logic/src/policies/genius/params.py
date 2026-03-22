"""
Configuration parameters for the GENIUS (GENI + US) meta-heuristic.

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GENIUSParams:
    """
    Configuration for the GENIUS meta-heuristic solver.

    GENIUS combines two procedures:
    - **GENI (Generalized Insertion)**: Efficient insertion heuristic that considers
      two types of insertions (Type I and Type II) and uses p-neighborhood restriction.
    - **US (Unstringing and Stringing)**: Post-optimization procedure that removes
      and reinserts vertices to further improve the solution.

    The algorithm follows these steps:
    1. Construct initial solution using GENI insertion
    2. Apply US post-optimization cycles to refine the solution
    3. Iterate until time limit or no improvement

    Attributes:
        neighborhood_size: Size of p-neighborhood for GENI insertion (p parameter).
            Restricts insertion search to p closest vertices to the candidate node.
            Typical values: 3-7. Paper recommends p=5 for good balance.
        us_cycles: Number of Unstringing-Stringing cycles to apply per iteration.
        unstring_type: Type of unstringing operator to use (1=Type I, 2=Type II,
            3=Type III, 4=Type IV). Each type differs in arc reconnection patterns.
        string_type: Type of stringing operator to use (1=Type I, 2=Type II,
            3=Type III, 4=Type IV). Should match unstring_type for consistency.
        n_iterations: Number of complete GENIUS iterations (GENI + US cycles).
        vrpp: Whether to use VRPP expanded pool (True).
        profit_aware_operators: Whether to use profit-aware operators (True) or
            cost-aware operators (False).
        time_limit: Wall-clock time limit in seconds. Set to 0 for no limit.
    """

    neighborhood_size: int = 5
    us_cycles: int = 10
    unstring_type: int = 1
    string_type: int = 1
    n_iterations: int = 1
    vrpp: bool = False
    profit_aware_operators: bool = False
    time_limit: float = 60.0

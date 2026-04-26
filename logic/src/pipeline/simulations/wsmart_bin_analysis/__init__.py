"""
WSmart Bin Analysis module for analyzing and predicting bin fill levels.

Attributes:
    GridBase: Class for representing a grid of bins.
    Simulation: Class for running simulations on the grid.
    OldGridBase: Class for representing a grid of bins (old implementation).

Example:
    >>> from wsmart_bin_analysis import GridBase, Simulation
    >>> grid = GridBase(num_rows=10, num_cols=10, bin_capacity=10)
    >>> sim = Simulation(grid)
    >>> sim.run_simulation()
"""

from .Deliverables.simulation import GridBase as GridBase
from .Deliverables.simulation import Simulation as Simulation
from .sample_gen import OldGridBase as OldGridBase

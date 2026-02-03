"""
Data Config module.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class DataConfig:
    """Data generation configuration.

    Attributes:
        name: Name to identify dataset. For training data, this will result in .td files.
        filename: Filename of the dataset to create (ignores datadir).
        data_dir: Create datasets in data.
        problem: Problem type selection. Should be 'vrpp', 'wcvrp', 'swcvrp', or 'all'.
        mu: Mean of Gaussian noise (implies Gaussian noise generation if set).
        sigma: Variance of Gaussian noise.
        data_distributions: Distributions to generate for problems.
        dataset_size: Size of the dataset.
        num_locs: Sizes of problem instances.
        penalty_factor: Penalty factor for problems.
        overwrite: Set true to overwrite.
        seed: Random seed.
        n_epochs: The number of epochs to generate data for.
        epoch_start: Start at epoch #.
        dataset_type: Set type of dataset to generate ('train', 'train_time', 'test_simulator').
        area: County area of the bins locations.
        waste_type: Type of waste bins selected for the optimization problem.
        focus_graphs: Path to the files with the coordinates of the graphs to focus on.
        focus_size: Number of focus graphs to include in the data.
        vertex_method: Method to transform vertex coordinates.
    """

    name: Optional[str] = None
    filename: Optional[str] = None
    data_dir: str = "datasets"
    problem: str = "all"
    mu: Optional[List[float]] = None
    sigma: Any = 0.6
    data_distributions: List[str] = field(default_factory=lambda: ["all"])
    dataset_size: int = 128_000
    val_size: int = 1280
    num_locs: List[int] = field(default_factory=lambda: [20, 50, 100])
    penalty_factor: float = 3.0
    overwrite: bool = False
    seed: int = 42
    n_epochs: int = 1
    epoch_start: int = 0
    dataset_type: Optional[str] = None
    area: str = "riomaior"
    waste_type: str = "plastic"
    focus_graphs: Optional[List[str]] = None
    focus_size: int = 0
    vertex_method: str = "mmn"
    # Environment generation params
    min_loc: float = 0.0
    max_loc: float = 1.0
    data_distribution: Optional[str] = None
    min_fill: float = 0.0
    max_fill: float = 1.0
    fill_distribution: str = "uniform"

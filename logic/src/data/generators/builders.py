"""
Builders for Vehicle Routing Problem (VRP) instances.

This module provides builder classes to construct VRP instances from raw data,
handling coordinate normalization, waste scaling, and feature extraction.
"""

from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from logic.src.constants import MAX_CAPACITY_PERCENT, MAX_WASTE
from logic.src.data.generators.waste import generate_waste
from logic.src.data.processor import process_coordinates
from logic.src.utils.data.loader import load_focus_coords, load_grid_base
from logic.src.utils.functions import get_path_until_string


class VRPInstanceBuilder:
    """
    Builder pattern for creating Vehicle Routing Problem (VRP) instances.

    This class provides a fluent interface to configure and generate VRP datasets
    with various parameters such as problem size, distribution, area, and waste type.
    """

    def __init__(
        self,
        data=None,
        depot_idx=0,
        vehicle_cap=100.0,
        customers=None,
        dimension=0,
        coords=None,
        device="cpu",
    ):
        """
        Initialize the VRPInstanceBuilder.

        Args:
            data: Raw problem data dictionary or dataframe.
            depot_idx (int): Index of the depot node.
            vehicle_cap (float): Capacity of the vehicles.
            customers: List of customer node indices.
            dimension (int): Total number of nodes (including depot).
            coords: Coordinate data (list or array).
            device (str): Device for random number generation.
        """
        self.data_dict = data
        self.depot_idx = depot_idx
        self.vehicle_cap = vehicle_cap
        self.customers = customers if customers is not None else []
        self.dimension = dimension
        self.coords = coords

        self._dataset_size = 10
        self._problem_size = 20
        self._waste_type = None
        self._distribution = "gamma1"
        self._area = "Rio Maior"
        self._focus_graph = None
        self._focus_size = 0
        self._method = None
        self._num_days = 1
        self._problem_name = None
        self._noise_mean = 0.0
        self._noise_variance = 0.0
        self._device = torch.device(device)
        self._seed = None
        self.np_rng = np.random.default_rng(42)
        self.generator = torch.Generator(device=self._device)

    def set_seed(self, seed: int):
        """Sets the random seed for reproducibility."""
        self._seed = seed
        self.np_rng = np.random.default_rng(seed)
        self.generator.manual_seed(seed)
        return self

    def set_dataset_size(self, size: int):
        """Sets the number of instances to generate."""
        self._dataset_size = size
        return self

    def set_problem_size(self, size: int):
        """Sets the number of nodes (graph size) for the problem."""
        self._problem_size = size
        return self

    def set_waste_type(self, waste_type: str):
        """Sets the type of waste (e.g., 'plastic', 'paper')."""
        self._waste_type = waste_type
        return self

    def set_distribution(self, distribution: str):
        """Sets the data distribution for generating waste levels."""
        self._distribution = distribution
        return self

    def set_area(self, area: str):
        """Sets the geographical area for the problem instance."""
        self._area = area
        return self

    def set_focus_graph(self, focus_graph: Optional[str] = None, focus_size: int = 0):
        """Sets parameters for focusing on a specific subgraph."""
        self._focus_graph = focus_graph
        self._focus_size = focus_size
        return self

    def set_method(self, method: Optional[str]):
        """Sets the method used for vertex generation/selection."""
        self._method = method
        return self

    def set_num_days(self, num_days: int):
        """Sets the number of simulation days."""
        self._num_days = num_days
        return self

    def set_problem_name(self, problem_name: str):
        """Sets the name of the problem (e.g., 'vrpp', 'wcvrp')."""
        self._problem_name = problem_name
        return self

    def set_noise(self, mean: float, variance: float):
        """Sets the mean and variance for noise injection."""
        self._noise_mean = mean
        self._noise_variance = variance
        return self

    def build(self):
        """
        Generates the simulation dataset as a dict of numpy arrays.

        Returns:
            dict: A dictionary of numpy arrays:
                - 'depot': np.ndarray of shape (dataset_size, coord_dim)
                - 'locs': np.ndarray of shape (dataset_size, problem_size, coord_dim)
                - 'waste': np.ndarray of shape (dataset_size, num_days, problem_size)
                - 'noisy_waste': np.ndarray of same shape as waste (equals waste when not SWCVRP)
                - 'max_waste': np.ndarray of shape (dataset_size,)
        """
        depot, loc, grid, idx = self._prepare_coordinates()

        # Generate waste/fill values over days
        fill_values = []
        coords = (depot, loc)

        for _ in range(self._num_days):
            waste = generate_waste(
                self._problem_size, self._distribution, coords, self._dataset_size, grid, rng=self.np_rng
            )
            if self._dataset_size == 1 and len(waste.shape) == 1:
                waste = waste[None, :]
            fill_values.append(waste)

        # Shape: (dataset_size, num_days, problem_size)
        fill_arr = np.transpose(np.array(fill_values), (1, 0, 2))

        # generate_waste returns values in [0, 1] normalized scale;
        # simulation datasets store in [0, 100] percentage scale to match Bins.
        fill_arr = fill_arr * MAX_CAPACITY_PERCENT

        if self._problem_name == "swcvrp":
            noise = np.random.normal(self._noise_mean, np.sqrt(self._noise_variance), fill_arr.shape)
            noisy_fill_arr = np.clip(fill_arr + noise, 0, MAX_CAPACITY_PERCENT)
        else:
            noisy_fill_arr = fill_arr.copy()

        return {
            "depot": np.flip(depot, axis=-1) if depot.shape[-1] == 2 else depot,
            "locs": np.flip(loc, axis=-1) if loc.shape[-1] == 2 else loc,
            "node_ids": np.tile(np.arange(1, self._problem_size + 1), (self._dataset_size, 1)),
            "waste": fill_arr,
            "noisy_waste": noisy_fill_arr,
            "max_waste": np.full(self._dataset_size, MAX_CAPACITY_PERCENT),
        }

    def build_td(self) -> TensorDict:
        """
        Generates the dataset as a batched TensorDict.

        Returns:
            TensorDict: A TensorDict with keys matching environment expectations.
        """
        depot, loc, grid, idx = self._prepare_coordinates()

        fill_values = []
        coords = (depot, loc)
        for _ in range(self._num_days):
            waste = generate_waste(
                self._problem_size, self._distribution, coords, self._dataset_size, grid, rng=self.generator
            )
            if self._dataset_size == 1 and len(waste.shape) == 1:
                waste = waste[None, :]
            fill_values.append(waste)

        fill_vals = np.array(fill_values)  # (num_days, dataset_size, num_loc)
        fill_vals = np.transpose(fill_vals, (1, 0, 2))  # (dataset_size, num_days, num_loc)

        bs = self._dataset_size
        device = "cpu"  # Generate on CPU

        depot_tensor = torch.tensor(depot, dtype=torch.float32, device=device)
        locs_tensor = torch.tensor(loc, dtype=torch.float32, device=device)

        # Handle noise for stochastic variants
        if self._problem_name == "swcvrp":
            real_waste = torch.tensor(fill_vals, dtype=torch.float32, device=device)
            noise = torch.randn_like(real_waste) * np.sqrt(self._noise_variance) + self._noise_mean
            noisy_waste = torch.clamp(real_waste + noise, 0, MAX_WASTE)

            # For TensorDict, we usually want (bs, num_loc) for waste.
            # If num_days == 1, squeeze it.
            if self._num_days == 1:
                real_waste = real_waste.squeeze(1)
                noisy_waste = noisy_waste.squeeze(1)

            td_data = {
                "depot": depot_tensor,
                "locs": locs_tensor,
                "real_waste": real_waste,
                "waste": noisy_waste,
            }
        else:
            waste = torch.tensor(fill_vals, dtype=torch.float32, device=device)
            if self._num_days == 1:
                waste = waste.squeeze(1)

            td_data = {
                "depot": depot_tensor,
                "locs": locs_tensor,
                "waste": torch.clamp(waste, 0, MAX_WASTE),
            }

        # Common attributes
        td_data.update(
            {
                "capacity": torch.full((bs,), self.vehicle_cap, device=device),
                "max_waste": torch.full((bs,), float(MAX_WASTE), device=device),
            }
        )

        return TensorDict(td_data, batch_size=[bs], device=device)

    def _prepare_coordinates(self):
        """Internal helper to prepare depot and location coordinates."""
        if self._focus_graph is not None:
            if self._focus_size <= 0:
                self._focus_size = self._dataset_size
            depot, loc, mm_arr, idx = load_focus_coords(
                self._problem_size,
                self._method,
                self._area,
                self._waste_type,  # type: ignore[arg-type]
                self._focus_graph,
                self._focus_size,
            )
            remaining_coords_size = self._dataset_size - self._focus_size
            if remaining_coords_size > 0:
                random_coords = np.random.uniform(
                    mm_arr[0],  # type: ignore[index]
                    mm_arr[1],  # type: ignore[index]
                    size=(remaining_coords_size, self._problem_size + 1, mm_arr.shape[-1]),  # type: ignore[union-attr]
                )
                depots, locs = process_coordinates(random_coords, self._method, col_names=None)
                depot = np.concatenate((depot, depots))  # type: ignore[assignment]
                loc = np.concatenate((loc, locs))  # type: ignore[assignment]
            grid = None
            if self._distribution == "emp":
                data_dir = get_path_until_string(self._focus_graph, "wsr_simulator")
                grid = load_grid_base(idx, self._area, data_dir)
        else:
            grid = None
            idx = np.array(range(self._problem_size))
            coord_size = 2 if self._method != "triple" else 3
            depot = np.random.uniform(size=(self._dataset_size, coord_size))
            loc = np.random.uniform(size=(self._dataset_size, self._problem_size, coord_size))
        return depot, loc, grid, idx

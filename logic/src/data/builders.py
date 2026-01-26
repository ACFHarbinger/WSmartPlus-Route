"""
Builders for Vehicle Routing Problem (VRP) instances.

This module provides builder classes to construct VRP instances from raw data,
handling coordinate normalization, demand scaling, and feature extraction.
"""

from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from logic.src.constants import MAX_WASTE
from logic.src.pipeline.simulations.bins import Bins
from logic.src.pipeline.simulations.processor import process_coordinates
from logic.src.utils.data.data_utils import generate_waste_prize, load_focus_coords
from logic.src.utils.functions.function import get_path_until_string


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

    def set_method(self, method: str):
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
        Generates the dataset based on configured parameters.

        Returns:
            list: A list of problem instances, where each instance is a tuple containing:
                  - depot (list): Coordinates of the depot.
                  - loc (list): Coordinates of customer nodes.
                  - waste (list): Waste levels (or demand) for nodes.
                  - max_waste (float): Maximum capacity or waste limit.
        """
        depot, loc, bins, idx = self._prepare_coordinates()

        # Generate waste/fill values over days
        fill_values = []
        coords = (depot, loc)

        for _ in range(self._num_days):
            waste = generate_waste_prize(self._problem_size, self._distribution, coords, self._dataset_size, bins)
            if self._dataset_size == 1 and len(waste.shape) == 1:
                waste = waste[None, :]
            fill_values.append(waste)

        # Transpose to (dataset_size, num_days, problem_size)
        fill_values = np.transpose(np.array(fill_values), (1, 0, 2))

        # Construct the output list
        if self._problem_name == "swcvrp":
            # SWCVRP Case: Generate Noisy Waste
            real_waste_list = fill_values.tolist()

            # Generate Noise
            noise = np.random.normal(self._noise_mean, np.sqrt(self._noise_variance), fill_values.shape)
            noisy_fill_values = np.clip(fill_values + noise, 0, MAX_WASTE)
            noisy_waste_list = noisy_fill_values.tolist()

            return list(
                zip(
                    depot.tolist(),
                    loc.tolist(),
                    real_waste_list,
                    noisy_waste_list,
                    np.full(self._dataset_size, MAX_WASTE).tolist(),
                )
            )
        else:
            # Standard WCVRP/VRPP Case
            waste_list = fill_values.tolist()

            return list(
                zip(
                    depot.tolist(),
                    loc.tolist(),
                    waste_list,
                    np.full(self._dataset_size, MAX_WASTE).tolist(),
                )
            )

    def build_td(self) -> TensorDict:
        """
        Generates the dataset as a batched TensorDict.

        Returns:
            TensorDict: A TensorDict with keys matching environment expectations.
        """
        # We reuse the logic but convert to tensors before returning
        # Note: self._num_days is handled by taking the first day for train data
        # or keeping it as (dataset_size, num_days, num_loc) for simulation data.

        # Reuse build for now, then convert? Or refactor build?
        # Let's refactor slightly to get raw arrays.
        depot, loc, bins, idx = self._prepare_coordinates()

        fill_values = []
        coords = (depot, loc)
        for _ in range(self._num_days):
            waste = generate_waste_prize(self._problem_size, self._distribution, coords, self._dataset_size, bins)
            if self._dataset_size == 1 and len(waste.shape) == 1:
                waste = waste[None, :]
            fill_values.append(waste)

        fill_vals = np.array(fill_values)  # (num_days, dataset_size, num_loc)
        fill_vals = np.transpose(fill_vals, (1, 0, 2))  # (dataset_size, num_days, num_loc)

        # For training data, we usually expect 1 day.
        # If num_days > 1, we keep the temporal dimension.
        # But for RL4COEnvBase reset, it usually expects 'demand' or 'waste' of shape (bs, num_loc)

        bs = self._dataset_size
        device = "cpu"  # Generate on CPU

        depot_tensor = torch.tensor(depot, dtype=torch.float32, device=device)
        locs_tensor = torch.tensor(loc, dtype=torch.float32, device=device)

        # Handle noise for stochastic variants
        if self._problem_name == "swcvrp":
            real_waste = torch.tensor(fill_values, dtype=torch.float32, device=device)
            noise = torch.randn_like(real_waste) * np.sqrt(self._noise_variance) + self._noise_mean
            noisy_waste = torch.clamp(real_waste + noise, 0, MAX_WASTE)

            # For TensorDict, we usually want (bs, num_loc) for demand.
            # If num_days == 1, squeeze it.
            if self._num_days == 1:
                real_waste = real_waste.squeeze(1)
                noisy_waste = noisy_waste.squeeze(1)

            td_data = {
                "depot": depot_tensor,
                "locs": locs_tensor,
                "real_waste": real_waste,
                "demand": noisy_waste,
                "waste": noisy_waste,
            }
        else:
            waste = torch.tensor(fill_values, dtype=torch.float32, device=device)
            if self._num_days == 1:
                waste = waste.squeeze(1)

            td_data = {
                "depot": depot_tensor,
                "locs": locs_tensor,
                "demand": waste,
                "waste": waste,
                "prize": waste.clone(),
            }

        # Common attributes
        td_data.update(
            {
                "capacity": torch.full((bs,), self.vehicle_cap, device=device),
                "max_waste": torch.full((bs,), float(MAX_WASTE), device=device),
                "max_length": torch.full((bs,), 2.4, device=device),  # Default or parameterized
            }
        )

        return TensorDict(td_data, batch_size=[bs], device=device)

    def _prepare_coordinates(self):
        """Internal helper to prepare depot and location coordinates."""
        if self._focus_graph is not None:
            assert self._focus_size > 0, "Focus size must be positive when using focus graph"
            depot, loc, mm_arr, idx = load_focus_coords(
                self._problem_size,
                self._method,
                self._area,
                self._waste_type,
                self._focus_graph,
                self._focus_size,
            )
            remaining_coords_size = self._dataset_size - self._focus_size
            if remaining_coords_size > 0:
                random_coords = np.random.uniform(
                    mm_arr[0],
                    mm_arr[1],
                    size=(remaining_coords_size, self._problem_size + 1, mm_arr.shape[-1]),
                )
                depots, locs = process_coordinates(random_coords, self._method, col_names=None)
                depot = np.concatenate((depot, depots))  # type ignore[assignment]
                loc = np.concatenate((loc, locs))  # type: ignore[assignment]
            bins = None
            if self._distribution == "emp":
                data_dir = get_path_until_string(self._focus_graph, "wsr_simulator")
                bins = Bins(
                    self._problem_size,
                    data_dir,
                    sample_dist=self._distribution,
                    area=self._area,
                    indices=idx[0],
                    grid=None,
                    waste_type=self._waste_type,
                )
        else:
            bins = None
            idx = [None]
            coord_size = 2 if self._method != "triple" else 3
            depot = np.random.uniform(size=(self._dataset_size, coord_size))
            loc = np.random.uniform(size=(self._dataset_size, self._problem_size, coord_size))
        return depot, loc, bins, idx

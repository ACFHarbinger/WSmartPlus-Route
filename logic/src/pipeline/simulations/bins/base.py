"""
Bin State Management and Waste Accumulation Simulation.
"""

import math
import os
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas
import torch

from logic.src.constants.routing import MAX_CAPACITY_PERCENT
from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params

from ..wsmart_bin_analysis import GridBase
from .prediction import calculate_frequency_and_level, predict_days_to_overflow


class Bins:
    """
    Manages the state and dynamics of a population of waste collection bins.

    This class simulates the temporal evolution of bin fill levels through:
    - Daily waste deposition (stochastic or pre-recorded)
    - Collection events that empty bins
    - Overflow detection and waste loss calculation
    - Online statistical learning (mean, variance) via Welford's algorithm

    The Bins object maintains two parallel state vectors:
        - real_c: Ground truth fill levels (0-100%)
        - c: Observed fill levels (may include sensor noise)
    """

    def __init__(
        self,
        n: int,
        data_dir: str,
        sample_dist: str = "gamma",
        grid: Optional[GridBase] = None,
        area: Optional[str] = None,
        waste_type: Optional[str] = None,
        indices: Optional[List[int]] = None,
        waste_file: Optional[str] = None,
        noise_mean: float = 0.0,
        noise_variance: float = 0.0,
    ):
        """
        Initializes the bin population with area-specific parameters.
        """
        assert sample_dist == "emp" or "gamma" in sample_dist
        self.n = n
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        _, revenue, density, expenses, bin_volume = load_area_and_waste_type_params(area, waste_type)
        self.revenue = revenue
        self.density = density
        self.expenses = expenses
        self.volume = bin_volume

        self.c = np.zeros((n))
        self.real_c = np.zeros((n))
        self.means = np.zeros((n))
        self.std = np.zeros((n))
        self.day_count = 0
        self.square_diff = np.zeros((n))
        self.start_with_fill = False

        self.lost = np.zeros((n))
        self.distribution = sample_dist
        self.dist_param1 = np.ones((n)) * 10
        self.dist_param2 = np.ones((n)) * 10
        self.inoverflow = np.zeros((n))
        self.collected = np.zeros((n))
        self.ncollections = np.zeros((n))
        self.history: List[np.ndarray] = []
        self.level_history: List[np.ndarray] = []
        self.travel: float = 0.0
        self.profit: float = 0.0
        self.ndays: int = 0
        self.collectdays = np.ones((n)) * 5
        self.collectlevl = np.ones((n)) * 80
        self.data_dir = data_dir

        if indices is None:
            self.indices = np.array(range(n))
        else:
            self.indices = np.array(indices)

        if grid is None and sample_dist == "emp":
            src_area = area.translate(str.maketrans("", "", "-_ ")).lower() if area is not None else ""
            waste_csv = f"out_rate_crude[{src_area}].csv"
            info_csv = f"out_info[{src_area}].csv"

            # Read info file to map indices to IDs
            info_df = pandas.read_csv(os.path.join(data_dir, "coordinates", info_csv))
            real_ids = info_df.iloc[self.indices]["ID"].tolist()

            # Check ID type in waste csv
            waste_path = os.path.join(data_dir, "bins_waste", waste_csv)
            waste_header = pandas.read_csv(waste_path, nrows=0).columns
            if pandas.api.types.is_string_dtype(waste_header):
                real_ids = [str(i) for i in real_ids]

            self.grid = GridBase(
                real_ids,
                data_dir,
                rate_type="crude",
                names=[waste_csv, info_csv, None],
                same_file=True,
            )
        else:
            self.grid = grid  # type: ignore[assignment]

        if waste_file is not None:
            with open(os.path.join(data_dir, waste_file), "rb") as file:
                self.waste_fills = pickle.load(file)
        else:
            self.waste_fills = None
        self.noisy_waste_fills = None

    def __get_stdev(self):
        """Computes current standard deviation."""
        if self.day_count > 1:
            variance = self.square_diff / (self.day_count - 1)
            return np.sqrt(variance)
        return np.zeros(self.n)

    def set_statistics(self, stats_file: str) -> None:
        """Loads pre-computed fill statistics."""
        data = pandas.read_csv(os.path.join(self.data_dir, stats_file))
        self.means = np.maximum(data["Mean"].values.astype(np.float64), 0)
        self.std = np.maximum(data["StD"].values.astype(np.float64), 0)
        self.day_count = np.maximum(data.at[0, "Count"].astype(np.int64), 0)
        self.square_diff = (self.std**2) * (self.day_count - 1)
        self.start_with_fill = True

    def is_stochastic(self) -> bool:
        """Checks if using stochastic filling."""
        return self.waste_fills is None

    def get_fill_history(self, device: Optional[torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        """Retrieves history of daily fill increments."""
        if device is not None:
            return torch.tensor(np.array(self.history), dtype=torch.float, device=device) / 100.0
        return np.array(self.history)

    def get_level_history(self, device: Optional[torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        """Retrieves history of daily absolute fill levels."""
        if device is not None:
            return torch.tensor(np.array(self.level_history), dtype=torch.float, device=device) / 100.0
        return np.array(self.level_history)

    def predictdaystooverflow(self, cl: float) -> np.ndarray:
        """Predicts days to overflow based on statistics."""
        return predict_days_to_overflow(self.means, self.std, self.c, cl)

    def set_indices(self, indices: Optional[Union[List[int], np.ndarray]] = None) -> None:
        """Sets subset of bin indices."""
        if indices is not None:
            self.indices = np.array(indices)
        else:
            self.indices = np.array(range(self.n))

    def set_sample_waste(self, sample_id: int) -> None:
        """Sets current waste profile from pre-recorded data."""
        if isinstance(self.waste_fills[sample_id], (list, tuple)) and len(self.waste_fills[sample_id]) == 2:
            self.noisy_waste_fills = self.waste_fills[sample_id][1]
            self.waste_fills = self.waste_fills[sample_id][0]
        else:
            self.waste_fills = self.waste_fills[sample_id]

        if self.start_with_fill:
            self.real_c = self.waste_fills[0].copy()
            if self.noisy_waste_fills is not None:
                self.c = self.noisy_waste_fills[0].copy()
            else:
                noise = (
                    np.random.normal(self.noise_mean, np.sqrt(self.noise_variance), self.n)
                    if self.noise_variance > 0
                    else np.zeros(self.n)
                )
                self.c = np.clip(self.real_c + noise, 0, 100)
            self.level_history.append(self.c.copy())

    def collect(self, idsfull: List[int], cost: float = 0) -> Tuple[np.ndarray, float, int, float]:
        """Processes waste collection from bins in the tour."""
        ids = set(idsfull)
        total_collected = np.zeros((self.n))
        if len(ids) < 2:
            return total_collected, 0, 0, 0

        ids.remove(0)
        self.ndays += 1
        bin_ids = np.array(list(ids)) - 1
        collected = (self.real_c[bin_ids] / 100) * self.volume * self.density
        self.collected[bin_ids] += collected
        self.ncollections[bin_ids] += 1
        total_collected[bin_ids] += collected
        self.real_c[bin_ids] = 0
        self.c[bin_ids] = 0
        self.travel += cost
        profit = np.sum(total_collected) * self.revenue - float(cost) * self.expenses
        self.profit += float(profit)
        return total_collected, float(np.sum(collected)), bin_ids.size, float(profit)

    def _process_filling(
        self, todaysfilling: np.ndarray, noisyfilling: Optional[np.ndarray] = None
    ) -> Tuple[int, np.ndarray, np.ndarray, float]:
        """Processes daily waste deposition."""
        self.history.append(todaysfilling)

        todaysfilling_arr = np.array(todaysfilling)
        old_means = self.means.copy()

        self.day_count += 1
        delta = todaysfilling_arr - old_means
        self.means += delta / self.day_count
        self.square_diff += delta * (todaysfilling_arr - self.means)
        self.std = self.__get_stdev()

        todays_lost = (
            (np.maximum(self.real_c + todaysfilling - MAX_CAPACITY_PERCENT, 0) / MAX_CAPACITY_PERCENT)
            * self.volume
            * self.density
        )
        todaysfilling = np.minimum(todaysfilling, MAX_CAPACITY_PERCENT)
        self.lost += todays_lost
        self.real_c = np.minimum(self.real_c + todaysfilling, MAX_CAPACITY_PERCENT)

        if noisyfilling is not None:
            self.c = np.minimum(self.c + noisyfilling, MAX_CAPACITY_PERCENT)
        elif self.noise_variance > 0:
            noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_variance), self.n)
            self.c = np.clip(self.real_c + noise, 0, MAX_CAPACITY_PERCENT)
        else:
            self.c = self.real_c.copy()

        self.level_history.append(self.c.copy())
        self.real_c = np.maximum(self.real_c, 0)
        inoverflow = self.real_c == MAX_CAPACITY_PERCENT
        self.inoverflow += inoverflow
        return (
            int(np.sum(inoverflow)),
            np.array(todaysfilling),
            np.array(self.c),
            float(np.sum(todays_lost)),
        )

    def stochasticFilling(
        self, n_samples: int = 1, only_fill: bool = False
    ) -> Union[np.ndarray, Tuple[int, np.ndarray, np.ndarray, float]]:
        """Generates fills by sampling from distributions."""
        todaysfilling = np.zeros(self.n)
        if self.distribution == "gamma":
            todaysfilling = np.random.gamma(self.dist_param1, self.dist_param2, size=(n_samples, self.n))
            if n_samples <= 1:
                todaysfilling = todaysfilling.squeeze(0)
        elif self.distribution == "emp":
            sampled_value = self.grid.sample(n_samples=n_samples)
            todaysfilling = np.maximum(sampled_value, 0)

        if only_fill:
            return np.minimum(todaysfilling, MAX_CAPACITY_PERCENT)
        return self._process_filling(todaysfilling)

    def deterministicFilling(self, date):
        """Loads deterministic fill levels from historical data."""
        todaysfilling = self.grid.get_values_by_date(date, sample=True)
        return self._process_filling(todaysfilling)

    def loadFilling(self, day: int) -> Tuple[int, np.ndarray, np.ndarray, float]:
        """Loads deterministic waste fills from pre-recorded data."""
        todaysfilling = self.waste_fills[day] if self.start_with_fill else self.waste_fills[day - 1]
        noisyfilling = None
        if self.noisy_waste_fills is not None:
            noisyfilling = self.noisy_waste_fills[day] if self.start_with_fill else self.noisy_waste_fills[day - 1]
        return self._process_filling(todaysfilling, noisyfilling)

    def __setDistribution(self, param1, param2):
        """Internal helper to set sampling distribution parameters."""
        if len(param1) == 1:
            self.dist_param1 = np.ones((self.n)) * param1
            self.dist_param2 = np.ones((self.n)) * param2
        else:
            self.dist_param1 = param1
            self.dist_param2 = param2
        self.setCollectionLvlandFreq()

    def setGammaDistribution(self, option=0):
        """Configures bins with predefined Gamma distribution profiles."""

        def __set_param(param):
            """set param.

            Args:
                    param (Any): Description of param.

            Returns:
                Any: Description of return value.
            """
            param_len = len(param)
            if self.n == param_len:
                return param
            param = param * math.ceil(self.n / param_len)
            if self.n % param_len != 0:
                param = param[: param_len - self.n % param_len]
            return param

        self.distribution = "gamma"
        if option == 0:
            k = __set_param([5, 5, 5, 5, 5, 10, 10, 10, 10, 10])
            th = __set_param([5, 2])
        elif option == 1:
            k = __set_param([2, 2, 2, 2, 2, 6, 6, 6, 6, 6])
            th = __set_param([6, 4])
        elif option == 2:
            k = __set_param([1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
            th = __set_param([8, 6])
        else:
            assert option == 3
            k = __set_param([5, 2])
            th = __set_param([10])
        self.__setDistribution(k, th)

    def setCollectionLvlandFreq(self, cf=0.9):
        """Sets visit frequency and level thresholds."""
        for ii in range(0, self.n):
            f2, lv2 = calculate_frequency_and_level(
                self.dist_param1[ii] * self.dist_param2[ii],
                self.dist_param1[ii] * self.dist_param2[ii] ** 2,
                cf,
            )
            self.collectdays[ii] = f2
            self.collectlevl[ii] = lv2

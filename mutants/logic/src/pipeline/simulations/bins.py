"""
Bin State Management and Waste Accumulation Simulation.

This module implements the physics of waste bin filling, collection, and overflow
tracking for the WSmart-Route simulator. It supports both stochastic (statistical)
and deterministic (empirical) waste generation patterns.

The Bins class maintains:
- Current fill levels (observed and real)
- Statistical profiles (mean, std dev) updated online via Welford's method
- Overflow tracking and waste loss calculations
- Collection history and performance metrics

Waste Generation Modes:
    - Stochastic: Samples from Gamma distributions or empirical grid data
    - Deterministic: Loads pre-recorded daily fill values from pickle files
    - Noisy: Optionally adds Gaussian noise to simulate sensor uncertainty

Classes:
    Bins: Core state manager for waste bin population
"""

import math
import os
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas
import torch
from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params
from scipy import stats

from .wsmart_bin_analysis import GridBase


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

    Attributes:
        n: Number of bins in the system
        distribution: 'gamma' or 'emp' for sampling strategy
        revenue: Revenue per kg of collected waste
        density: Waste density (kg/L)
        volume: Bin volume (L)
        expenses: Cost per km traveled
        c: Observed fill levels (0-100%) with optional noise
        real_c: True fill levels (0-100%)
        means: Running mean fill per day (Welford's algorithm)
        std: Running standard deviation of daily fills
        lost: Cumulative waste lost to overflows per bin (kg)
        collected: Cumulative waste collected per bin (kg)
        ncollections: Number of times each bin was collected
        history: Daily fill increments (List of np.ndarray)
        level_history: Daily absolute levels after filling
        travel: Total distance traveled (km)
        profit: Cumulative profit (revenue - costs)
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

        Args:
            n: Number of bins to manage
            data_dir: Root directory for simulation data
            sample_dist: Distribution type - 'gamma' or 'emp' (empirical)
            grid: Pre-configured GridBase object for empirical sampling
            area: Geographic area name (e.g., 'Rio Maior')
            waste_type: Waste stream type ('paper', 'plastic', 'glass')
            indices: Subset of bin IDs to use (None = all bins)
            waste_file: Path to pickle file with pre-recorded fill data
            noise_mean: Mean of Gaussian sensor noise
            noise_variance: Variance of Gaussian sensor noise

        Raises:
            AssertionError: If sample_dist not in ['emp', 'gamma']
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
        """
        Computes the current standard deviation of bin fills using the square differences.

        Returns:
            np.ndarray: Per-bin standard deviation.
        """
        if self.day_count > 1:
            variance = self.square_diff / (self.day_count - 1)
            return np.sqrt(variance)
        else:
            return np.zeros(self.n)

    def _predictdaystooverflow(self, ui: np.ndarray, vi: np.ndarray, f: np.ndarray, cl: float) -> np.ndarray:
        """
        Internal math for predicting days until a bin overflows.

        Uses the Gamma distribution CDF to estimate the probability of
        reaching 100% capacity within a 31-day window.

        Args:
            ui: Mean fill rate.
            vi: Variance of fill rate.
            f: Current fill level.
            cl: Confidence level (0-1).

        Returns:
            np.ndarray: Predicted days to overflow per bin (clipped at 31).
        """
        n = np.zeros(ui.shape[0]) + 31
        for ii in np.arange(1, 31, 1):
            k = ii * ui**2 / vi
            th = vi / ui
            aux = np.zeros(ui.shape[0]) + 31
            p = 1 - stats.gamma.cdf(100 - f, k, scale=th)
            aux[np.nonzero(p > cl)[0]] = ii
            n = np.minimum(n, aux)
            if np.all(p > cl):
                return n
        return n

    def set_statistics(self, stats_file: str) -> None:
        """
        Loads pre-computed fill statistics from a CSV file.

        Args:
            stats_file: Path to the statistics CSV relative to data_dir.
        """
        data = pandas.read_csv(os.path.join(self.data_dir, stats_file))
        self.means = np.maximum(data["Mean"].values.astype(np.float64), 0)
        self.std = np.maximum(data["StD"].values.astype(np.float64), 0)
        self.day_count = np.maximum(data.at[0, "Count"].astype(np.int64), 0)
        self.square_diff = (self.std**2) * (self.day_count - 1)
        self.start_with_fill = True

    def is_stochastic(self) -> bool:
        """
        Checks if the bins are using stochastic filling.

        Returns:
            bool: True if waste_fills is not provided (using sampling).
        """
        return self.waste_fills is None

    def get_fill_history(self, device: Optional[torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        """
        Retrieves the history of daily fill increments.

        Args:
            device: Optional torch.device to return data as a tensor.

        Returns:
            Union[np.ndarray, torch.Tensor]: Fill history.
        """
        if device is not None:
            return torch.tensor(np.array(self.history), dtype=torch.float, device=device) / 100.0
        else:
            return np.array(self.history)

    def get_level_history(self, device: Optional[torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        """
        Retrieves the history of daily absolute fill levels.

        Args:
            device: Optional torch.device to return data as a tensor.

        Returns:
            Union[np.ndarray, torch.Tensor]: Level history.
        """
        if device is not None:
            return torch.tensor(np.array(self.level_history), dtype=torch.float, device=device) / 100.0
        else:
            return np.array(self.level_history)

    def predictdaystooverflow(self, cl: float) -> np.ndarray:
        """
        Predicts days to overflow based on current running statistics.

        Args:
            cl: Confidence level for prediction.

        Returns:
            np.ndarray: Predicted days to overflow per bin.
        """
        return self._predictdaystooverflow(self.means, self.std, self.c, cl)

    def set_indices(self, indices: Optional[Union[List[int], np.ndarray]] = None) -> None:
        """
        Sets the subset of bin indices to be included in simulation logic.

        Args:
            indices: List or array of 0-based bin indices (optional).
        """
        if indices is not None:
            self.indices = np.array(indices)
        else:
            self.indices = np.array(range(self.n))

    def set_sample_waste(self, sample_id: int) -> None:
        """
        Sets the current waste profile from the loaded waste_fills based on sample ID.

        Args:
            sample_id: Integer index of the sample to use.
        """
        # Check if the data contains both real and noisy values
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
        """
        Processes waste collection from bins in the tour.

        Empties visited bins, updates collection statistics, and computes profit.
        Uses Welford's online algorithm to update mean and standard deviation
        of daily fills without storing full history.

        Args:
            idsfull: List of bin IDs visited in tour (includes depot as 0)
            cost: Total tour distance in km (default: 0)

        Returns:
            Tuple containing:
                - total_collected: Per-bin collected waste array (kg)
                - sum_collected: Total kg collected across all bins
                - n_collected: Number of bins collected
                - profit: Net profit = (kg × revenue) - (km × cost_per_km)

        Note:
            If tour has ≤2 nodes (only depot), no collection occurs.
            Visited bins are reset to 0% fill level (both real and observed).
        """
        # Check if tour has bins to collect
        ids = set(idsfull)
        total_collected = np.zeros((self.n))
        if len(ids) < 2:
            return total_collected, 0, 0, 0

        # Collect waste
        ids.remove(0)
        self.ndays += 1
        bin_ids = np.array(list(ids)) - 1
        collected = (self.real_c[bin_ids] / 100) * self.volume * self.density
        self.collected[bin_ids] += collected
        self.ncollections[bin_ids] += 1
        total_collected[bin_ids] += collected
        self.real_c[bin_ids] = 0
        self.c[bin_ids] = 0  # Observed bins are also emptied
        self.travel += cost
        profit = np.sum(total_collected) * self.revenue - float(cost) * self.expenses  # type: ignore[assignment]
        self.profit += float(profit)
        return total_collected, float(np.sum(collected)), bin_ids.size, float(profit)

    def _process_filling(
        self, todaysfilling: np.ndarray, noisyfilling: Optional[np.ndarray] = None
    ) -> Tuple[int, np.ndarray, np.ndarray, float]:
        """
        Processes daily waste deposition and updates bin states.

        Core simulation step that:
        1. Records today's fill in history
        2. Detects and logs overflows (>100% capacity)
        3. Updates real and observed fill levels
        4. Optionally injects sensor noise

        Args:
            todaysfilling: Array of waste added today (% of capacity)
            noisyfilling: Optional pre-computed noisy observations

        Returns:
            Tuple containing:
                - n_overflows: Number of bins that overflowed (int)
                - fill_increment: Today's fill increments (np.ndarray)
                - observed_levels: Current observed fill levels (np.ndarray)
                - waste_lost: Total kg of waste lost to overflows (float)

        Note:
            Waste exceeding 100% capacity is lost (environmental impact).
            Noise can be pre-recorded (noisyfilling) or generated on-the-fly.
        """
        self.history.append(todaysfilling)

        # Update mean and standard deviation using Welford's method
        # This is now done daily regardless of collection
        todaysfilling_arr = np.array(todaysfilling)
        old_means = self.means.copy()

        self.day_count += 1
        delta = todaysfilling_arr - old_means
        self.means += delta / self.day_count
        self.square_diff += delta * (todaysfilling_arr - self.means)
        self.std = self.__get_stdev()

        # Lost overflows
        todays_lost = (np.maximum(self.real_c + todaysfilling - 100, 0) / 100) * self.volume * self.density
        todaysfilling = np.minimum(todaysfilling, 100)
        self.lost += todays_lost

        # New depositions for the overflow calculation
        self.real_c = np.minimum(self.real_c + todaysfilling, 100)

        # Inject noise into observed c
        if noisyfilling is not None:
            self.c = np.minimum(self.c + noisyfilling, 100)
        elif self.noise_variance > 0:
            noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_variance), self.n)
            self.c = np.clip(self.real_c + noise, 0, 100)
        else:
            self.c = self.real_c.copy()

        self.level_history.append(self.c.copy())
        self.real_c = np.maximum(self.real_c, 0)
        inoverflow = self.real_c == 100
        self.inoverflow += self.real_c == 100
        return (
            int(np.sum(inoverflow)),
            np.array(todaysfilling),
            np.array(self.c),
            float(np.sum(todays_lost)),
        )

    def stochasticFilling(
        self, n_samples: int = 1, only_fill: bool = False
    ) -> Union[np.ndarray, Tuple[int, np.ndarray, np.ndarray, float]]:
        """
        Generates daily waste fills by sampling from statistical distributions.

        Supports two modes:
        - Gamma: Samples from configured Gamma(k, θ) per bin
        - Empirical: Samples from historical data via GridBase

        Args:
            n_samples: Number of samples to generate (default: 1)
            only_fill: If True, return only fill values without state update

        Returns:
            If only_fill=True:
                - np.ndarray: Sampled fill values (clipped to 100%)
            If only_fill=False:
                - Tuple: (n_overflows, fill, levels, waste_lost) from _process_filling

        Note:
            Negative samples are clipped to 0 (physical constraint).
        """
        todaysfilling = np.zeros(self.n)
        if self.distribution == "gamma":
            todaysfilling = np.random.gamma(self.dist_param1, self.dist_param2, size=(n_samples, self.n))
            if n_samples <= 1:
                todaysfilling = todaysfilling.squeeze(0)
        elif self.distribution == "emp":
            sampled_value = self.grid.sample(n_samples=n_samples)
            todaysfilling = np.maximum(sampled_value, 0)

        if only_fill:
            return np.minimum(todaysfilling, 100)
        else:
            return self._process_filling(todaysfilling)

    def deterministicFilling(self, date):
        """
        Loads deterministic fill levels for a specific date from historical data.

        Args:
            date: datetime or string identifier for the target date.

        Returns:
            Tuple: (n_overflows, fill, levels, waste_lost) from _process_filling.
        """
        todaysfilling = self.grid.get_values_by_date(date, sample=True)
        return self._process_filling(todaysfilling)

    def loadFilling(self, day: int) -> Tuple[int, np.ndarray, np.ndarray, float]:
        """
        Loads deterministic waste fills from pre-recorded data.

        Retrieves the exact fill values for a specific day from the loaded
        waste_fills array. Supports both real and noisy observation streams.

        Args:
            day: Simulation day index (int)

        Returns:
            Tuple: (n_overflows, fill, levels, waste_lost) from _process_filling

        Note:
            If start_with_fill=True, day 0 contains initial state.
            Otherwise, indexing is offset by 1 (day-1).
        """
        todaysfilling = self.waste_fills[day] if self.start_with_fill else self.waste_fills[day - 1]
        noisyfilling = None
        if self.noisy_waste_fills is not None:
            noisyfilling = self.noisy_waste_fills[day] if self.start_with_fill else self.noisy_waste_fills[day - 1]
        return self._process_filling(todaysfilling, noisyfilling)

    def __setDistribution(self, param1, param2):
        """
        Internal helper to set sampling distribution parameters.

        Args:
            param1: First parameter (e.g., k for Gamma).
            param2: Second parameter (e.g., theta for Gamma).
        """
        if len(param1) == 1:
            self.dist_param1 = np.ones((self.n)) * param1
            self.dist_param2 = np.ones((self.n)) * param2
        else:
            self.dist_param1 = param1
            self.dist_param2 = param2
        self.setCollectionLvlandFreq()

    def setGammaDistribution(self, option=0):
        """
        Configures the bins with one of the predefined Gamma distribution profiles.

        Profiles vary in variability and mean fill rates to simulate
        different urban environments.

        Args:
            option: Integer 0-3 selecting the predefined profile.
        """

        def __set_param(param):
            """Helper to broaden scalar params to vector of size n."""
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

    def freqvisit2(self, ui, vi, cf):
        """
        Calculates the recommended visit frequency and target overflow level.

        Args:
            ui: Mean daily fill rate.
            vi: Variance of daily fill rate.
            cf: Target confidence level (e.g., 0.9 for 90% service level).

        Returns:
            Tuple[int, float]: (Optimal days between visits, Target level at visit).
        """
        # a = gamma.cdf(30, k, scale=th)
        # c = gamma.ppf(a, k, scale=th)
        # print(a,c)
        for n in range(1, 50):
            k = n * ui**2 / vi
            th = vi / ui
            if n == 1:
                ov = 100 - stats.gamma.ppf(1 - cf, k, scale=th)

            v = stats.gamma.ppf(1 - cf, k, scale=th)
            if v > 100:
                return n, ov

    def setCollectionLvlandFreq(self, cf=0.9):
        """
        Propagates service level targets to visit frequency and level thresholds.

        Args:
            cf: Service level confidence factor (default: 0.9).
        """
        for ii in range(0, self.n):
            f2, lv2 = self.freqvisit2(
                self.dist_param1[ii] * self.dist_param2[ii],
                self.dist_param1[ii] * self.dist_param2[ii] ** 2,
                cf,
            )
            self.collectdays[ii] = f2
            self.collectlevl[ii] = lv2
        return

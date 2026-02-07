"""
Simulation engine for bin fill levels.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .grid import GridBase
from .predictors import Predictor


class Simulation(GridBase):
    """
    A simulation class that extends GridBase to provide time-stepping and collection logic.
    """

    def __init__(
        self,
        sim_type: str,
        ids: list,
        data_dir: str,
        train_split=None,
        start_date=None,
        end_date=None,
        rate_type=None,
        predictQ: bool = False,
        info_ver=None,
        names=None,
        savefit_name=None,
    ):  # type: ignore[assignment]
        """
        Initialize the simulation with a specific type and time range.

        Args:
            sim_type (str): Type of simulation ('sampled', 'real', or 'real+sampled').
            ids (list): List of bin IDs to simulate.
            data_dir (str): Root directory for data files.
            train_split (str, optional): Split date for training/testing.
            start_date (str, optional): Start date of the simulation.
            end_date (str, optional): End date of the simulation.
            rate_type (str, optional): 'mean' or 'crude'.
            predictQ (bool): Whether to use a predictor.
            info_ver (str, optional): Info file version suffix.
            names (str, optional): Manual filenames.
            savefit_name (str, optional): Filename for saved model weights.
        """
        Sim_Keys = ["sampled", "real", "real+sampled"]
        assert sim_type in Sim_Keys, f"sim_type {sim_type} is no acceptected. Must be one from {Sim_Keys}"
        super().__init__(
            ids=ids,
            rate_type=rate_type,
            info_ver=info_ver,
            data_dir=data_dir,
            names=names,
        )

        self.fill: np.ndarray = np.zeros(self.get_num_bins())
        self.sim_type: str = sim_type
        self.predictQ: bool = predictQ

        self.start_date: pd.Timestamp = pd.to_datetime(start_date, format="%d-%m-%Y", errors="raise")
        self.split: pd.Timestamp = pd.to_datetime(train_split, format="%d-%m-%Y", errors="raise")
        self.end_date: pd.Timestamp = pd.to_datetime(end_date, format="%d-%m-%Y", errors="raise")
        self.current_date: pd.Timestamp = self.start_date

        self.rates: pd.DataFrame = self.pre_simulate_rates()
        if self.predictQ:
            print("Getting in the Pedictor")
            self.predictor: Predictor = Predictor(
                self.values_by_date_range(end=self.split),
                self.rates[self.split :],
                savefit_name,
            )

    def pre_simulate_rates(self) -> pd.DataFrame:
        """
        Pre_simulates Rates according to set at init
        Returns
        -------
        rate: pd.Dataframe
            dataframe with simulated rates
        """
        date_range = pd.date_range(self.start_date, self.end_date)
        rate_list: list[np.ndarray] = []
        for date in date_range:
            if self.sim_type == "sample":
                rate_list.append(self.sample())
            elif self.sim_type == "real":
                rate_list.append(self.get_values_by_date(date=date))
            elif self.sim_type == "real+sampled":
                rate_list.append(self.get_values_by_date(date=date, sample=True))
            else:
                raise ValueError("self.type not recognised")

        rate = pd.DataFrame(np.vstack(rate_list))
        rate.index = date_range
        return rate

    def reset_simulation(self):
        "Reset the simulation witout changing the pre-simulated rates and predictions"
        self.current_date = self.start_date
        self.fill = np.zeros(self.get_num_bins())

    def get_current_step(
        self,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Gets rates and prediction at the given timestep
        Returns
        -------
        rate: np.ndarray
            current rate

        pred: np.ndarray
            current prediction
        error:
            current predicted error on the prediction
        """
        if self.predictQ and self.current_date > self.split:
            pred, p_error = self.predictor.get_pred_values(self.current_date)
            return self.rates.loc[self.current_date, :].to_numpy(), pred, p_error
        else:
            return self.rates.loc[self.current_date, :].to_numpy(), None, None

    def make_collections(self, bins_index_list: list[int] = None) -> np.ndarray:  # type: ignore[assignment]
        """
                Preforms collections on the bins specified by the index. The index is induced by the order of
                the dataframe.

                Returns
        -------
                collected_junk: np.ndarray
                    the collected trash in each of the collected bins using bins_index_list as an index map.
                    It corresponds to the percentage of the bin
        """
        if bins_index_list is None:
            bins_index_list = []

        idx = np.array(bins_index_list)
        collected_junk = self.fill[idx]
        self.fill[idx] = 0
        return collected_junk

    def advance_timestep(self, date=None) -> tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Advances the timeStep of the simulation by updating the fill level according to the rates the
        update procedure defined at init.

        Parameters
        ----------
        date: str, datetime
            date to fetch values. string format %d-%m-%Y or datetime

        Returns
        -------
        n_overflows: int
            the number of overflows occurred in that step
        prediction: np.ndarray
            returns an array of prediction made to the corresponding bins
        error: np.ndarray
            returns an array with the mean of the error predicted; i.e 50% of the times, the actual value will
            be in interval [prediction-error, prediction+error]

        """
        self.current_date = self.current_date + pd.Timedelta("1D")
        rate, prediction, p_error = self.get_current_step()

        self.fill = self.fill + rate
        self.fill[self.fill < 0] = 0

        overfmask = self.fill > 100
        self.fill[overfmask] = 100
        return int(np.sum(overfmask)), prediction, p_error

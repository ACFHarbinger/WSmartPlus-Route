"""
DEPRECATED: Background management for container data sampling.
This module is legacy code and has been replaced by GridBase in simulation.py.
"""

import os

import numpy as np
import pandas as pd


class OldGridBase:
    """
    Class that manages the background behind containers.

    It receives a csv file, and pre-processes the necessary data in order to allow sampling.
    """

    def __init__(self, data_dir: str, area: str) -> None:
        """
        Initialize the OldGridBase sampler.

        Args:
            data_dir (str): Root directory for data files.
            area (str): Area name (e.g., 'cascais') used to identify specific CSV files.
        """
        self.data_dir = data_dir
        self.src_area = area.translate(str.maketrans("", "", "-_ ")).lower()
        self.__data: pd.DataFrame = self.__data_preprocess(
            pd.read_csv(os.path.join(data_dir, "bins_waste", f"old_out_crude_rate[{self.src_area}].csv"))
        )
        self.__info: pd.DataFrame = pd.read_csv(
            os.path.join(data_dir, "coordinates", f"old_out_info[{self.src_area}].csv")
        )
        self.__freq_table: pd.DataFrame = self.__calc_freq_tables()

    def __data_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        @return: The dataframe with dates as row indexes and rates rounded to the nearest integer
        """
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
        data = data.set_index("Date")
        data = data.round()
        return data

    def __calc_freq_tables(self) -> pd.DataFrame:
        """
        @return: A dataframe whose columns are the distribution of each rate
        """
        freq_table = self.__data.select_dtypes(include="number").apply(lambda x: x.value_counts(normalize=True), axis=0)
        freq_table = freq_table.fillna(0).cumsum()
        freq_table = freq_table.apply(lambda s: fix10pad(s))
        return freq_table

    def get_mean_rate(self) -> np.ndarray:
        """
        @return: The mean for each bin
        """
        return self.__data.mean(axis=0, skipna=True, numeric_only=True).to_numpy()

    def get_var_rate(self) -> np.ndarray:
        """
        @return: The var of each bin
        """
        return self.__data.var(axis=0, skipna=True, numeric_only=True).to_numpy()

    def get_std_rate(self) -> np.ndarray:
        """
        @return: The stardard variation of each bin
        """
        return self.__data.var(axis=0, skipna=True, numeric_only=True).transform("sqrt").to_numpy()

    def get_daterange(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        @return: The start and end of real values
        """
        index = self.__data.index
        return pd.Timestamp(index[0]), pd.Timestamp(index[-1])

    def sample(self, n_samples=1) -> np.ndarray:
        """
        Sample N times from each bins' waste fill rate distribution.

        Args:
            n_samples: Number of samples to generate (default: 1)

        Returns:
            np.ndarray: Array of shape (n_samples, n_columns) containing values sampled from each distribution
        """
        assert n_samples > 0, "Number of samples must be a positive integer"

        index_values = np.array(self.__freq_table.index)  # Shape: (279,)
        freq_table = self.__freq_table.to_numpy()  # Shape: (279, 317)

        # Generate random values for all samples and columns
        rand_vals = np.random.random(size=(n_samples, len(self.__freq_table.columns)))  # Shape: (n_samples, 317)

        # Apply searchsorted column-wise to get indices
        indexes = np.zeros((n_samples, freq_table.shape[1]), dtype=int)  # Shape: (n_samples, 317)
        for col in range(freq_table.shape[1]):
            indexes[:, col] = np.searchsorted(freq_table[:, col], rand_vals[:, col], side="right")

        # Index into index_values to get samples
        sample = index_values[indexes]
        return sample if n_samples > 1 else sample.squeeze(0)

    def get_values_by_date(self, date, sample: bool = False) -> np.ndarray:
        """
        @param date: Datetime object (e.g., when looping through a daterange) or string in %Y-%m-%d format.
        @param sample: Whether The bins NaN values should be sampled or mantained.
        @return: Each container rate in required date.
        """
        try:
            date = pd.to_datetime(date, format="%Y-%m-%d", errors="raise")
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")

        rate = self.___values_by_date(date)
        if sample:
            samp = self.sample()
            mask = rate.isna()
            rate[mask] = samp[mask]
        return rate.to_numpy()

    def ___values_by_date(self, date: pd.Timestamp) -> pd.Series:
        """
        @return: The actual rate row of each bin
        """
        return self.__data.loc[date, :]

    def get_info(self, i: int) -> dict:
        """
        @return: The info of the container with the given index
        """
        return self.__info.iloc[i, :].to_dict()

    def get_num_bins(self) -> int:
        """
        @return: The number of bins the the grid
        """
        return len(self.__info.index)

    def load_data(self, processed=True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the fill rate data and container metadata.

        Args:
            processed (bool): If True, returns pre-processed data from memory.
                             If False, re-reads the raw CSV. Defaults to True.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (Fill rate data, Container metadata)
        """
        if processed:
            data_file = self.__data
        else:
            data_file = pd.read_csv(
                os.path.join(
                    self.data_dir,
                    "bins_waste",
                    f"old_out_crude_rate[{self.src_area}].csv",
                )
            )
        return data_file, self.__info


### UTILS
def fix10pad(s: pd.Series) -> pd.Series:
    """
    A value is verifiied by checkking if is one if 0.9999999>1.

    1-0.9999999 < 1/365*3 ~ 0.0001 Hard Coded, change for much bigger datasets; same goes for zero.
    @return: A series with all its 1 values padded to 2 without changing the first 1.
    """
    temp = s[s >= 0.9999999]
    temp.iloc[1:] = 2.0
    s.loc[s >= 0.9999999] = temp

    temp = s[s <= 0.0000001]
    temp.iloc[:-1] = -1
    s.loc[s <= 0.0000001] = temp
    return s


if __name__ == "__main__":
    PATH = "/mnt/c/Users/Utilizador/OneDriveUL/Desktop/Masters/Wsmart+Route/Initial_studies/"

    # First load the necessary files
    grid = OldGridBase(PATH, area="cascais")

    # create a array with random fill level for each bin
    fill = np.zeros(grid.get_num_bins())

    # start simulation
    for _i in range(10):
        # sample from the distribution
        rate = grid.sample()

        fill = fill + rate

        # be carefull with negative rate and overflows
        fill[fill < 0] = 0
        fill[fill > 100] = 100

        # simulate collections

    # Instead of sampling, also allow to iterate through a datarange
    fill = np.zeros(grid.get_num_bins())
    datarange = pd.date_range(start=pd.to_datetime("2022-03-01"), end=pd.to_datetime("2022-05-01"))
    for date in datarange:
        # get real rates, set sample to true to sample nans
        rate = grid.get_values_by_date(date, sample=True)

        fill = fill + rate

        # etc etc etc same as above

    ## If you want the mean/std/variance of each container
    grid.get_std_rate()
    grid.get_mean_rate()
    grid.get_var_rate()

    # also you can ask for static inforation about the container with the given index
    grid.get_info(4)

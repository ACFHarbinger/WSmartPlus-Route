"""
GridBase manager for container ensembles.
"""

import os
from typing import Union, cast

import numpy as np
import pandas as pd

from .save_load import load_info, load_rate_global_wrapper, load_rate_series


class GridBase:
    """
    Class that manages the background behind a containers ensemble.

    It receives a csv file, and pre-processes the necessary data in order to allow sampling.
    """

    def __init__(self, ids, data_dir, rate_type, info_ver=None, names=None, same_file=False) -> None:
        """
        Initialize the GridBase manager.

        Args:
            ids (list): List of bin IDs to load.
            data_dir (str): Root directory for data files.
            rate_type (str): Type of rate data ('mean' or 'crude').
            info_ver (str, optional): Suffix for info files. Defaults to None.
            names (list[str], optional): Manual filenames. Defaults to None.
            same_file (bool): Whether data is stored in a single combined file. Defaults to False.
        """
        self.data: pd.DataFrame = None
        self.__info: Union[dict, pd.DataFrame] = None
        self.__freq_table: pd.DataFrame = None
        self.__data_dir: str = None  # type: ignore[assignment]

        self.__data_dir = data_dir
        self.data, self.__info = self.load_data(
            processed=False,
            ids=ids,
            data_dir=self.__data_dir,
            rate_type=rate_type,
            info_ver=info_ver,
            names=names,
            same_file=same_file,
        )

        self.data = self.data.select_dtypes(include="number").round(0)
        self.__freq_table = self.cacl_freq_tables()

    def load_data(
        self,
        ids,
        data_dir,
        info_ver=None,
        names=None,
        rate_type=None,
        processed=True,
        same_file=False,
    ) -> tuple[pd.DataFrame, Union[dict, pd.DataFrame]]:
        """
                Parameters
                ----------
                ids:list[int],
                    list of container ids to load
                data_dir:str,
                    files directory
                rate_type:str,
                    Can be 'mean' or 'crude' depending on the the file previously saved
                info_ver:str,
                    ver suffix unused in info saving
                names:list[str]
                    Names to use instead of the names generator.
                    All should come already with the .csv attached in the end
                processed: bool ,True
                    set to True  by default; set if files hae been previously loaded
                same_file: bool, False,
                    if the files are in the same file

                Returns
        -------

                rates: pd.Dataframe
                    The dataframe of processed rates
                info:dict: dict
                    Ids dictionary with info from each bin
        """
        waste_dir = os.path.join(data_dir, "bins_waste")
        coords_dir = os.path.join(data_dir, "coordinates")
        if processed:
            return self.data, self.__info
        elif same_file:
            rate = self.__data_preprocess_same_file(pd.read_csv(os.path.join(waste_dir, names[0])))
            rate = rate.loc[:, ids]
            info = pd.read_csv(os.path.join(coords_dir, names[1]))
            return rate, info
        else:
            rate_list: list[dict] = []
            info_dict = {}
            if names is None:
                for id in ids:
                    rate_list = rate_list + [load_rate_series(id, rate_type=rate_type, path=waste_dir)]
                    info_dict[id] = load_info(id, ver=info_ver, path=coords_dir)
            else:
                for i, id in enumerate(ids):
                    if "rate" in names[i].keys():
                        rate_list = rate_list + [
                            load_rate_series(
                                id,
                                rate_type=rate_type,
                                path=waste_dir,
                                name=names[i]["rate"],
                            )
                        ]
                    else:
                        rate_list = rate_list + [load_rate_series(id, rate_type=rate_type, path=waste_dir)]

                    if "info" in names[i].keys():
                        info_dict[id] = load_info(id, ver=info_ver, path=coords_dir, name=names[i]["info"])
                    else:
                        info_dict[id] = load_info(id, ver=info_ver, path=coords_dir)
            print("All loaded")
            return load_rate_global_wrapper(rate_list=rate_list), info_dict

    def __data_preprocess_same_file(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        -------
        data: pd.Dataframe,
            The dataframe with dates before preprocessing

        Returns
        -------
        data: pd.Dataframe,
            The dataframe with dates as row indexes
        """
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
        data = data.set_index("Date")
        return data

    def cacl_freq_tables(self) -> pd.DataFrame:
        """
        Returns
        --------
        a dataframe whose columns are the distribution of each rate
        """

        def count(series: pd.Series):
            """Calculate the cumulative frequency table for a series."""
            return fix10pad(series.value_counts(normalize=True).fillna(0).cumsum())

        freq_table = self.data.agg(count, axis=0)
        return cast(pd.DataFrame, freq_table)

    def get_mean_rate(self) -> np.ndarray:
        """
        Returns
        -------
        mean: np.ndarray
            the mean of each bin
        """
        return self.data.mean(axis=0, skipna=True, numeric_only=True).to_numpy()

    def get_var_rate(self) -> np.ndarray:
        """
                Returns
        -------
                var: np.ndarray
                    the varaince of each bin
        """
        return self.data.var(axis=0, skipna=True, numeric_only=True).to_numpy()

    def get_std_rate(self) -> np.ndarray:
        """
        Returns
        -------
        std: np.ndarray
            the stardard variation of each bin
        """
        return self.data.var(axis=0, skipna=True, numeric_only=True).transform("sqrt").to_numpy()

    def get_datarange(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Returns
        -------
        (start, end): pd.timestamp,
            start and end of the actual rate values
        """
        index = self.data.index
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
        assert (
            self.__freq_table is not None
        ), "Freq tables should be calculated before calling this method. Call self.calc_freq_tables()"

        index_values = np.array(self.__freq_table.index)
        freq_table = self.__freq_table.to_numpy()

        # Generate random values for all samples and columns
        rand_vals = np.random.random(size=(n_samples, len(self.__freq_table.columns)))

        # Apply searchsorted column-wise to get indices
        indexes = np.zeros((n_samples, freq_table.shape[1]), dtype=int)
        for col in range(freq_table.shape[1]):
            indexes[:, col] = np.searchsorted(freq_table[:, col], rand_vals[:, col], side="right")

        sample = index_values[indexes]
        return sample if n_samples > 1 else sample.squeeze(0)

    def get_values_by_date(self, date, sample: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        date: timestamp, str
            Datetime obect (e.g when looping through a datarange) or string in %d-%m-%Y format.
        sample: bool,
            Weather The cointaners whose value is NaN should be filled with a sample form self.sample()

        Returns
        -------
        rate: np.ndarray
            each container rate in required date.
        """
        date = pd.to_datetime(date, format="%d-%m-%Y", errors="raise")
        rate = self.___values_by_date(date)
        if sample:
            assert (
                self.__freq_table is not None
            ), "Freq tables should be calculated before calling this method. Call self.calc_freq_tables()"

            samp = self.sample()
            mask = rate.isna()
            rate[mask] = samp[mask]
        return rate.to_numpy()

    def ___values_by_date(self, date: pd.Timestamp) -> pd.Series:
        """
        Returns
        -------
        rate: np.darray
            The actual rate row of each bin
        """
        return self.data.loc[date, :]

    def values_by_date_range(self, start: pd.Timestamp = None, end: pd.Timestamp = None) -> pd.DataFrame:
        """
        Returns
        -------
        rate: pd.Dataframe
            The actual rate row of each bin per date
        """
        if end is None:
            end = self.data.index[-1]
        else:
            end = pd.to_datetime(end, format="%d-%m-%Y", errors="raise")

        if start is None:
            start = self.data.index[0]
        else:
            start = pd.to_datetime(start, format="%d-%m-%Y", errors="raise")
        return self.data.loc[start:end, :]

    def get_info(self, i: int) -> Union[dict, pd.DataFrame, pd.Series]:
        """
        Parameters
        ----------
        i: int
            index of the container to fect information from

        Returns
        -------
        info: dict
            the info of the container with the given index
        """
        return self.__info[self.data.columns[i]]

    def get_num_bins(self) -> int:
        """
        Returns
        -------
        num_bins: int
            the number of bins in the grid
        """
        return len(self.data.columns)


def fix10pad(s: pd.Series) -> pd.Series:
    """
    returns a series will all its 1 values padded to 2 without changing the first 1

    a value is verifiied by checkking if is one if 0.9999999>1.

    1-0.9999999 < 1/365*3 ~ 0.0001 Hard Coded, change for much bigger datasets; same goes for zero
    """
    temp = s[s >= 0.9999999]
    temp.iloc[1:] = 2.0
    s.loc[s >= 0.9999999] = temp

    temp = s[s <= 0.0000001]
    temp.iloc[:-1] = -1
    s.loc[s <= 0.0000001] = temp
    return s

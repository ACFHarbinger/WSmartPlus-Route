"""
Container management module for wsmart_bin_analysis.
"""

import warnings
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import ConstantInputWarning, spearmanr


class TAG(Enum):
    """
    Enum representing the quality/status tag of a container's data series.

    Attributes:
        LOW_MEASURES: Insufficient data points.
        INSIDE_BOX: Data falls consistently within expected bounds.
        OK: Data is good.
        WARN: Data has inconsistencies.
        LOCAL_WARN: Localized inconsistency detected.
    """

    LOW_MEASURES = 0
    INSIDE_BOX = 1
    OK = 2
    WARN = 3
    LOCAL_WARN = 4


class Container:
    """
    Overall object to deal with container data. It supports manipulation operations over the container,
    managing the fill levels and collection events.
    """

    def __init__(self, my_df: pd.DataFrame, my_rec: pd.DataFrame, info: pd.DataFrame):
        """
        Initialize the Container object.

        Args:
            my_df (pd.DataFrame): Dataframe containing fill level measurements.
            my_rec (pd.DataFrame): Dataframe containing collection events.
            info (pd.DataFrame): Dataframe containing static container information (metadata).
        """
        self.df: pd.DataFrame
        self.info: pd.DataFrame
        self.recs: pd.DataFrame
        self.id: int
        self.tag: Optional[TAG] = None

        self.df = my_df.set_index("Date")
        self.df.drop(["ID"], axis=1, inplace=True, errors="ignore")

        self.info = info
        self.id = info["ID"]

        self.recs = my_rec.set_index("Date")
        self.recs.drop(["ID"], axis=1, inplace=True, errors="ignore")

    def __del__(self):
        """
        Usually Python does the grbase collection for you. But if using a jupyter notebook, calling
        save does note clear space, because jupyter notebooks keep all objects "alive"

        Call del Container to clenaup the pace occupied
        """
        del self.df
        del self.info
        del self.recs
        del self.id

    def get_keys(self):
        "Get the keys in the internal variables of the class"
        return {
            "FILL": list(self.df.keys()),
            "RECS": list(self.recs.keys()),
            "INFO": list(self.info.keys()),
        }

    def get_vars(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generic getter for internal vars of bin
        """
        return self.df, self.recs, self.info

    def get_collection_quantities(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generic getter for internal vars of bi

        Returns
        -------
        avg_dist:np.ndarray
            numpy array with 100 - average distance information for each collection event
        spearman:np.ndarray
            numpy array with spearman coeficieet information for each collection event scaled by 100
        """
        try:
            avgd = self.recs["Avg_Dist"].copy(deep=True)
            avgd.dropna(inplace=True)
            avgd = avgd.to_numpy()
        except Exception:
            avgd = None

        try:
            spear = self.recs.loc[:, ["Spearman"]].copy(deep=True)
            spear.dropna(inplace=True)
            spear = spear.to_numpy()
        except Exception:
            spear = None
        return avgd, spear

    def get_scan_linear_spline(self, key, interval) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Calculates the cumulative garbage of a bin, returing it has daily values. The difference between values
        that have a collection in the mideele is set to zero before getiing the cumulative garbege.

        Parameters
        ----------
        interval:str,
            frequency string indicator. For daily values set to '1D'. For another frequencies check
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        """
        assert key in ["Mean", "Fill"]
        "key can only be 'Mean' of 'Fill'"

        date_range = pd.date_range(
            start=self.recs.index[0].round("D"),
            end=self.df.index[-1].round("D"),
            freq=interval,
        )

        numeric_data_range = np.array(
            list(
                map(
                    lambda y: (y - date_range[0]).total_seconds() / (3600 * 24),
                    date_range,
                )
            )
        )

        self.df["Diff"] = self.df[key].diff()
        self.df.at[self.df.index[0], "Diff"] = 0

        self.df["Num_T"] = self.df.index.to_series().transform(
            lambda x: (x - self.df.index[0]).total_seconds() / (3600 * 24)
        )

        self.df.loc[self.df["Rec"] == 1, "Diff"] = 0
        self.df["Scan"] = self.df["Diff"].to_numpy().cumsum()

        spline = np.interp(numeric_data_range, self.df["Num_T"], self.df["Scan"])

        self.df.drop(["Diff", "Num_T", "Scan"], axis=1, inplace=True)
        return date_range, spline

    def get_monotonic_mean_rate_error_splines(self, interval):
        """
        NOT WORKING!!!!

        Get the upper and lower splines for the monotocnic aproximation. The error is calculated to be the max error
        Possible in each iteration, e.g the lower bound is the next day minimum aproximation minus todays max and reversely
        for the upper bound. If plotting the splines, be aware that ther errors bars will drif off because this funtion
        returns the cumulative differences. when applyig the diff operator one can get a grasp of the local empirical error.

        Parameters
        ----------
        interval:str,
            frequency string indicator. For daily values set to '1D'. For another frequencies check
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        """
        assert all(
            map(lambda x: x in list(self.df.keys()), ["Max", "Min"])
        ), "Max and Min aproximations are not calculated. Try calling Container.calc_max_min_avg()) before"

        date_range = pd.date_range(
            start=self.recs.index[0].round("D"),
            end=self.df.index[-1].round("D"),
            freq=interval,
        )

        numeric_data_range = np.array(
            list(
                map(
                    lambda y: (y - date_range[0]).total_seconds() / (3600 * 24),
                    date_range,
                )
            )
        )

        self.df["Num_T"] = self.df.index.to_series().transform(
            lambda x: (x - self.df.index[0]).total_seconds() / (3600 * 24)
        )

        self.df["Min_SHF"] = self.df["Min"].shift(1).bfill()
        self.df["Max_SHF"] = self.df["Max"].shift(1).bfill()
        self.df["Diff_Max"] = self.df["Min_SHF"] - self.df["Max"]
        self.df["Diff_Min"] = self.df["Max_SHF"] - self.df["Min"]

        self.df.loc[self.df["Rec"] == 1, ["Diff_Min", "Diff_Max"]] = 0

        upper_spline = np.interp(
            numeric_data_range,
            self.df["Num_T"],
            self.df["Diff_Max"].to_numpy().cumsum(),
        )
        lower_spline = np.interp(
            numeric_data_range,
            self.df["Num_T"],
            self.df["Diff_Min"].to_numpy().cumsum(),
        )

        self.df.drop(
            ["Min_SHF", "Max_SHF", "Num_T", "Diff_Min", "Diff_Max"],
            axis=1,
            inplace=True,
        )
        return date_range, upper_spline, lower_spline

    def get_monotonic_mean_rate(self, freq) -> pd.DataFrame:
        """
        Get the mean monotic rate (difference between the mean between max and min monotonic approximations).
        The rates are the average difference between consecutive periods in the spline of the cumulative differenes between the bins.
        This is similar to make an average of the rate asross the time period. The rates should be unterstood has
        rate per frequency period. The differce between values of that have a collection marked in between are set to zero.

        Parameters
        ----------
        freq:str,
            frequency string indicator. For daily values set to '1D'. For another frequencies check
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        """
        data_range, spline = self.get_scan_linear_spline("Mean", freq)
        # _ , upper_spline, lower_spline = self.get_monotonic_mean_rate_error_splines(freq)

        # df = pd.DataFrame(data = {'Rate': np.diff(spline),
        #                          'Upper_Bound': np.diff(upper_spline),
        #                          'Lower_Bound': np.diff(lower_spline)},
        #                         index = data_range[:-1])

        df = pd.Series(data=np.diff(spline), index=data_range[:-1], name="Rate")

        df.index.name = "Date"
        return df.to_frame()

    def get_crude_rate(self, freq) -> pd.Series:
        """
        Get the crude rate (difference between consecutive values) rate of a container
        The rates are the average difference between consecutive periods in the spline of the cumulative differenes between the bins.
        This is similar to make an average of the rate asross the time period. The rates should be unterstood has
        rate per frequency period. The differce between values of that have a collection marked in between are set to zero."

         freq:str,
            frequency string indicator. For daily values set to '1D'. For another frequencies check
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliasess
        """
        data_range, spline = self.get_scan_linear_spline("Fill", freq)

        s = pd.Series(data=np.diff(spline), index=data_range[:-1], name="Rate")
        s.index.name = "Date"
        return s

    def get_tag(self, window: int, mv_thresh: int, min_days: int, use: str) -> TAG:
        """
        Counts the number of times that the moving average of the spearman coeficinet changes from
        False to True or otherwise. Also tags according to the the number of datapoints present in the
        Series.

        Parameters
        ----------
        window:int
            window to compute the rigth moving average of spearman correlations
        mv_thresh: int
            threshold to for a the moving avaraege to be consigered good or bad
        min_days: int
            minimum number of days to consider a container to have a sufficient number of measures
        use: str
            weather to use averege distance or spearman for tagging. Can be "spear" or "avg_dist"

        Returns
        -------
        tag: TAG
            tag of the containers quality information
        """
        KEYS = ["spear", "avg_dist"]
        assert use in KEYS, f"The key name {use} must be one of {KEYS}"

        if len(self.df) == 0:
            return TAG.LOW_MEASURES

        if len(self.recs) // len(self.df) >= 1 or (self.recs.index[-1] - self.recs.index[0]).days < min_days:
            return TAG.LOW_MEASURES

        if self.tag == TAG.LOCAL_WARN:
            return TAG.LOCAL_WARN

        if use == "spear":
            mv = self.recs["Spearman"].rolling(window, center=True).mean().ffill().bfill()
        elif use == "avg_dist":
            mv = self.recs["Avg_Dist"].rolling(window, center=True).mean().ffill().bfill()
        else:
            raise ValueError(f"The key name {use} must be one of {KEYS}")

        cut_mask = mv < mv_thresh
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            avg_cont = sum(cut_mask.diff().bfill())

        if avg_cont < 1:
            self.tag = TAG.OK
            return TAG.OK
        elif avg_cont == 1:
            self.tag = TAG.INSIDE_BOX
            return TAG.INSIDE_BOX
        else:
            self.tag = TAG.WARN
            return TAG.WARN

    def get_collections_std(self):
        """Get the variance between the distance between collections in days"""
        return np.sqrt(self.recs.index.to_series().diff().dropna().dt.total_seconds().std() / 7464960000)

    def set_tag(self, tag: TAG):
        """Set the containers tag mannually"""
        self.tag = tag

    def mark_collections(self):
        """
        Marks the collection in both dataframes. Creates a mask if fill information marking
        the first fill reading after each collection. Also saves the index pointers for each collection
        allowing to subset the readings dataframe for each period in between adjacent colletions.
        The fatest way make transformations by event is to use the collection mask calculated in cidx.
        It points to the integer line index of teh collectiton in self.recs

        Fundamental pre-processing tool for algorithms and plotting to work.
        """
        pos = self.df.index.searchsorted(self.recs.index.to_numpy(), side="left")

        self.recs["End_Pointer"] = pos
        self.recs.drop_duplicates(subset="End_Pointer", keep="first", inplace=True)

        d = np.searchsorted(pos, len(self.df), side="left")
        pos = pos[:d]

        mask = np.zeros(len(self.df), dtype=bool)
        self.df["Rec"] = 0

        mask[pos] = True
        self.df.loc[mask, "Rec"] = 1
        try:
            idx = self.df.index[self.recs["End_Pointer"].iloc[0]]
        except:
            self.df["Cidx"] = None
            print(f"Container {self.info['ID'].item()} Collections and Measures do not intersect")
            raise
            return

        cidx = np.repeat(
            np.arange(len(self.recs)),
            np.diff(np.append(self.recs["End_Pointer"], len(self.df))),
        )
        self.df.loc[idx:, "Cidx"] = cidx

    def calc_spearman(self, start_idx: int = 0, end_idx: int = -1):
        """
        Calculates the 100 x the Spearman correlation between the datapoints within each collection event.
        100 mean monotonic incresing -100 is monotonic decreasing and 0 is not monotonic at all

        Parameters
        ----------
        start/end: int, Optional
            Start numeric index of the colletciton to be considered.
            Allows to (re)compute metrics for a given subset of collection events.
            Default is the whole series.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConstantInputWarning)

            groupDF = self.df[
                self.df.index[self.recs["End_Pointer"].iloc[start_idx]] : self.df.index[
                    self.recs["End_Pointer"].iloc[end_idx] - 1
                ]
            ].groupby("Cidx")

            res = groupDF["Fill"].agg(lambda group: 100 * spearmanr(np.arange(len(group)), group).statistic)
            res.fillna(0, inplace=True)

            self.recs.loc[self.recs.index[start_idx] : self.recs.index[end_idx - 1], "Spearman"] = res.to_numpy()

    def calc_avg_dist_metric(self, start_idx: int = 0, end_idx: int = -1):
        """
        Calculates the avarage distance between max and min approximations between two collections.
        This value is calculated 100-avg_dist for coherence with spearman correlation

        The quantities are saved in the recs dataframe because this a quantitiy relating
        a period between collections.

        Parameters
        ----------
        start/end: int, Optional
            Start numeric index of the colletciton to be considered.
            Allows to (re)compute metrics for a given subset of collection events.
            Default is the whole series.
        """
        assert all(
            map(lambda x: x in list(self.df.keys()), ["Max", "Min"])
        ), "Max and Min aproximations are not calculated. Try calling Container.calc_max_min_avg()) before"

        groupDF = self.df[
            self.df.index[self.recs["End_Pointer"].iloc[start_idx]] : self.df.index[
                self.recs["End_Pointer"].iloc[end_idx] - 1
            ]
        ].groupby("Cidx")

        res = groupDF.apply(
            lambda group: (100 - (group["Max"] - group["Min"]).sum() / len(group) if len(group) > 0 else 100)
        ).to_numpy()
        try:
            # Fix: Assign only to the slice we calculated for, matching start/end indices
            # Note: end_idx is exclusive in python slicing usually, and here inputs are passed to DF slice
            # calc_spearman usage: self.recs.loc[self.recs.index[start_idx] : self.recs.index[end_idx - 1], ...
            # The slicing logic seems to target up to end_idx - 1.
            self.recs.loc[self.recs.index[start_idx] : self.recs.index[end_idx - 1], "Avg_Dist"] = res
        except:
            print("\n\nCidx are not set tup properly")
            print(
                self.df[
                    self.df.index[self.recs["End_Pointer"].iloc[start_idx]] : self.df.index[
                        self.recs["End_Pointer"].iloc[end_idx] - 1
                    ]
                ]
            )
            print(self.recs[self.recs.index[start_idx] : self.recs.index[end_idx - 1]])
            raise

    def calc_max_min_mean(self, start_idx: int = 0, end_idx: int = -1):
        """
        Given the fill values of a series, calculates the max and min monotonic
        aproximations.

        The max can be viewed as transversing the series from left
        to rigth and seting the next value to the value of previous one if it is lower.
        The min is equivelent but makign a swipe from rigth to left, changing
        if the next value is bigger.

        As these to Series are monotonic, their average
        is monotonic too.

        Allows to choose of subset of colletions. Optimized
        for vectorized operations

        Parameters
        ----------
        start/end: int, Optional
            Start numeric index of the colletciton to be considered.
            Allows to (re)compute metrics for a given subset of collection events.
            Default is the whole series.
        """

        def max_lopp(vec: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, bool]:
            """Inner loop for monotonic max approximation."""
            shf = np.roll(vec, 1)
            shf[0] = 0

            tmask = (shf > vec) & mask
            vec[tmask] = shf[tmask]

            return vec, bool(tmask.any())

        def min_loop(vec: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, bool]:
            """Inner loop for monotonic min approximation."""
            shf = np.roll(vec, -1)
            if np.issubdtype(shf.dtype, np.floating):
                shf[-1] = np.finfo(shf.dtype).max
            else:
                shf[-1] = np.iinfo(shf.dtype).max

            tmask = (shf < vec) & mask
            vec[tmask] = shf[tmask]

            return vec, bool(tmask.any())

        # max loop
        my_df = self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], ["Fill", "Rec"]].copy(deep=True)

        vec = my_df["Fill"].to_numpy().copy()
        mask = (my_df["Rec"] == 0).to_numpy().copy()

        vec, cond = max_lopp(vec, mask)
        while cond:
            vec, cond = max_lopp(vec, mask)

        self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Max"] = vec

        # min loop
        vec = my_df["Fill"].to_numpy().copy()
        mask = np.roll((my_df["Rec"] == 0).to_numpy(), -1)
        mask[-1] = False

        vec, cond = min_loop(vec, mask)
        while cond:
            vec, cond = min_loop(vec, mask)

        self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Min"] = vec

        # mean series
        self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Mean"] = (
            self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Min"]
            + self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Max"]
        ) / 2

    def adjust_collections(self, dist_thresh: int, c_trash: int, max_fill: int):
        """
        Orchestrator of the collections adjustement. Touches each event only once to guarantee
        stopping. Iterates until either touching each event or no event has an Avg_Dist
        lower then the threshold. Haldles all the index changes when deleating adjusting collections.

        Goes from left to rigth

        Parameters
        ----------
        dist_thresh:int
            theshold for avg_dist to look at the surrounding collections
        c_trash:int,
            minimum collected thrash to count a collection. Must be >= 0
        max_fill:int
            maximum fill value to count has the first value after a collection. If the fill value
            is bigger than this, then the collection is discarded.
        """
        assert c_trash >= 0, "c_trash must be greater or equal than zero "

        mask = self.recs["Avg_Dist"] < dist_thresh
        ac_mask = mask.copy(deep=True)
        while mask.any():
            deleted_collection_indexes: list[int] = []
            for index in np.where(mask)[0]:
                idx = index - len(deleted_collection_indexes)
                if (index >= len(self.recs) - 2) or idx == 0:
                    continue

                deleted = self.adjust_one_collection(idx, c_trash=c_trash, max_fill=max_fill)
                deleted = self.adjust_one_collection(idx - 1, c_trash=c_trash, max_fill=max_fill)

                if deleted == 1:
                    deleted_collection_indexes += [index - 1]

            # upadte mask being carefull for deleating
            ac_mask.drop(ac_mask.index[deleted_collection_indexes], inplace=True)

            # only untouched indexes
            mask = ~ac_mask & (self.recs["Avg_Dist"] < dist_thresh)

            # previous plus new indexes
            ac_mask = ac_mask | mask

    def adjust_one_collection(self, idx: int, c_trash: int, max_fill: int) -> int:
        """
        Adjust a single collection event based on fill data.

        Parameters
        ----------
        idx:int,
            Index of the colletion to be adjusted
        c_trash:int,
            minimum collected thrash to count a collection
        max_fill:int
            maximum fill value to count has the first value after a collection
        """
        base_index = self.recs["End_Pointer"].iat[idx] + 1
        end_index = self.recs["End_Pointer"].iat[idx + 2] - 1

        data = self.df["Fill"].iloc[base_index:end_index].diff().copy(deep=True)
        data.loc[self.df["Fill"].iloc[base_index:end_index] >= max_fill] = np.NAN
        if data.isna().all():
            new_index = 0
            collected_trash = -1
        else:
            new_index = data.argmin(skipna=True)
            collected_trash = self.df["Fill"].iat[new_index + base_index - 1] - self.df["Fill"].iat[base_index - 1]

        if collected_trash >= c_trash and self.df["Fill"].iat[new_index + base_index] <= max_fill:
            if self.df["Rec"].iat[new_index + base_index] == 0:
                self.df.at[self.df.index[new_index + base_index], "Rec"] = 1
                new_date = self.df.index[new_index + base_index] - timedelta(seconds=1)
                self.df.at[self.df.index[self.recs["End_Pointer"].iat[idx + 1]], "Rec"] = 0

                old_date = self.recs.index.tolist()[idx + 1]
                self.recs.rename(index={old_date: new_date}, inplace=True)
                self.recs.at[new_date, "End_Pointer"] = new_index + base_index

                self.df.loc[
                    self.df.index[self.recs["End_Pointer"].iat[idx]] : self.df.index[
                        self.recs["End_Pointer"].iat[idx + 1] - 1
                    ],
                    "Cidx",
                ] = self.df["Cidx"].iat[self.recs["End_Pointer"].iat[idx]]

                self.df.loc[
                    self.df.index[self.recs["End_Pointer"].iat[idx + 1]] : self.df.index[
                        self.recs["End_Pointer"].iat[idx + 2] - 1
                    ],
                    "Cidx",
                ] = self.df["Cidx"].iat[self.recs["End_Pointer"].iat[idx + 2] - 1]

                self.calc_max_min_mean(idx, idx + 2)
                self.calc_avg_dist_metric(idx, idx + 2)
                if "Spearman" in self.recs.columns:
                    self.calc_spearman(idx, idx + 2)
            return 0
        else:
            self.df.at[self.df.index[self.recs["End_Pointer"].iat[idx + 1]], "Rec"] = 0
            self.df.loc[
                self.df.index[self.recs["End_Pointer"].iat[idx]] : self.df.index[
                    self.recs["End_Pointer"].iat[idx + 2] - 1
                ],
                "Cidx",
            ] = self.df["Cidx"].iat[self.recs["End_Pointer"].iat[idx]]

            self.recs = self.recs.drop(self.recs.index[idx + 1])

            self.calc_max_min_mean(idx, idx + 1)
            self.calc_avg_dist_metric(idx, idx + 1)
            if "Spearman" in self.recs.columns:
                self.calc_spearman(idx, idx + 1)
            return 1

    def place_collections(self, dist_thresh: int, c_trash: int, max_fill: int, spear_thresh=None):  # type: ignore[assignment]
        """
        Orchestrator of the collections placement. Touches each event only once to guarantee
        stopping. Interates from left to rigth only once touching in event that have an Avg_Dist
        and Spearman below/above the thresholds. Haldles all the index changes when introducing adjusting
        collections.

        Parameters
        ----------
        dist_thresh:int
            theshold for avg_dist to look at the surrounding collections
        c_trash:int,
            minimum collected thrash to count a collection
        max_fill:int
            maximum fill value to count has the first value after a collection
        spear_thresh:int, optional
            threshol for spearman to look at the surrounding collections
        """
        if spear_thresh is not None:
            mask = (self.recs["Avg_Dist"] < dist_thresh) & (self.recs["Spearman"] < spear_thresh)
        else:
            mask = self.recs["Avg_Dist"] < dist_thresh

        added_colections_counter = 1
        while mask.any() and added_colections_counter != 0:
            added_colections_counter = 0
            for index in np.where(mask)[0]:
                idx = index + added_colections_counter
                if idx >= (len(self.recs) - 1):
                    continue

                added_colections_counter += self.place_one_collection(idx, c_trash=c_trash, max_fill=max_fill)
            if spear_thresh is not None:
                mask = (self.recs["Avg_Dist"] < dist_thresh) & (self.recs["Spearman"] < spear_thresh)
            else:
                mask = self.recs["Avg_Dist"] < dist_thresh

    def place_one_collection(self, idx: int, c_trash: int, max_fill: int):
        """
        Insert a collection event if criteria are met.

        Parameters
        ----------
        idx:int,
            Index of the colletion to be adjusted
        c_trash:int,
            minimum collected thrash to count a collection
        max_fill:int
            maximum fill value to count has the first value after a collection
        """
        base_index = self.recs["End_Pointer"].iat[idx] + 1
        end_index = self.recs["End_Pointer"].iat[idx + 1] - 1

        data = self.df["Fill"].iloc[base_index:end_index].diff().copy(deep=True)
        data.loc[self.df["Fill"].iloc[base_index:end_index] >= max_fill] = np.NAN
        if data.isna().all():
            new_index = 0
            collected_trash = -1
        else:
            new_index = data.argmin(skipna=True)
            collected_trash = self.df["Fill"].iat[new_index + base_index - 1] - self.df["Fill"].iat[base_index - 1]

        if collected_trash >= c_trash and self.df["Fill"].iat[new_index + base_index] <= max_fill:
            if self.df["Rec"].iat[new_index + base_index] == 0:
                self.df.at[self.df.index[new_index + base_index], "Rec"] = 1
                new_date = self.df.index[new_index + base_index] - timedelta(seconds=1)
                new_row = pd.DataFrame({"Date": [new_date], "End_Pointer": [new_index + base_index]}).set_index("Date")

                self.recs = pd.concat([self.recs, new_row]).sort_index()
                self.df.loc[
                    self.df.index[self.recs["End_Pointer"].iat[idx + 1]] : self.df.index[
                        self.recs["End_Pointer"].iat[idx + 2] - 1
                    ],
                    "Cidx",
                ] = self.df["Cidx"].max(skipna=True) + 1

                self.calc_max_min_mean(idx, idx + 2)
                self.calc_avg_dist_metric(idx, idx + 2)
                if "Spearman" in self.recs.columns:
                    self.calc_spearman(idx, idx + 2)
                return 1
        return 0

    def clean_box(self, window: int, mv_thresh: int, use: str):
        """
        Filter out collection events that fall below a certain moving average threshold.

        Args:
            window (int): Moving average window size.
            mv_thresh (int): Threshold for the moving average.
            use (str): Metric to use ('spear' or 'avg_dist').

        Raises:
            AssertionError: If `use` is not a valid key.
        """
        KEYS = ["spear", "avg_dist"]
        if use == "spear":
            mv = self.recs["Spearman"].rolling(window, center=True).mean().ffill().bfill()
        elif use == "avg_dist":
            mv = self.recs["Avg_Dist"].rolling(window, center=True).mean().ffill().bfill()
        else:
            raise ValueError(f"The key name {use} must be one of {KEYS}")

        cut_mask = mv < mv_thresh
        self.recs = self.recs[~cut_mask]

        self.df = self.df.loc[self.df.index[self.recs["End_Pointer"].iloc[0]] :, :]
        self.mark_collections()

    def plot_fill(self, start_date: datetime, end_date: datetime, fig_size: tuple = (9, 6)):
        """
        Plots the fill level of the bins with associated collections

        Parameters
        ----------
        fig_size: tuple of integers
            alias for plt.figure(figsize=fig_size)
        end/start_date: string with format %d/%m/%Y.
            Period to be analysed. First day is inclusive staring at 00:01 last day is exclusive.
        """
        start_date = pd.to_datetime(start_date, format="%d-%m-%Y", errors="raise")
        end_date = pd.to_datetime(end_date, format="%d-%m-%Y", errors="raise")

        filtered_df = self.df[cast(Any, start_date) : cast(Any, end_date)]
        filtered_df2 = self.recs[cast(Any, start_date) : cast(Any, end_date)]
        colors = filtered_df["Rec"].map({1: "green", 0: "red"})

        plt.figure(figsize=fig_size)
        plt.plot(
            filtered_df.index,
            filtered_df["Fill"],
            linestyle="-",
            color="black",
            linewidth=0.2,
        )
        plt.scatter(filtered_df.index, filtered_df["Fill"], marker="o", color=colors, s=4.5)
        for c in filtered_df2.index:
            plt.axvline(x=c, color="green", linewidth=0.9)

        plt.xlabel("Date")
        plt.ylabel("Fill Level")
        plt.title(
            "Container ID:" + str(int(self.info["ID"].item())) + "; Freguesia: " + str(self.info["Freguesia"].item())
        )
        plt.xticks(rotation=45)

        green_marker = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="1st Measure",
            markerfacecolor="green",
            markersize=3,
        )
        red_marker = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Measures",
            markerfacecolor="red",
            markersize=3,
        )
        green_line = Line2D([0], [1], color="green", linewidth=1, label="Collections")

        # Updating the legend with the green line
        plt.legend(handles=[green_marker, red_marker, green_line], loc="upper left")
        plt.show()

    def plot_max_min(self, start_date: datetime, end_date: datetime, fig_size: tuple = (9, 6)):
        """
        Plots tha raw data with max, min monotonic aproximations as well as their mean

        Parameters
        ----------
        fig_size: tuple of integers
            alias for plt.figure(figsize=fig_size)
        end/start_date: string with format %d/%m/%Y.
            Period to be analysed. First day is inclusive staring at 00:01 last day is exclusive.
        """
        start_date = pd.to_datetime(start_date, format="%d-%m-%Y", errors="raise")
        end_date = pd.to_datetime(end_date, format="%d-%m-%Y", errors="raise")

        filtered_df = self.df[cast(Any, start_date) : cast(Any, end_date)]
        colors = filtered_df["Rec"].map({1: "green", 0: "red"})

        plt.figure(figsize=fig_size)
        plt.plot(
            filtered_df.index,
            filtered_df["Max"],
            linestyle="-.",
            color="blue",
            linewidth=0.4,
        )
        plt.plot(
            filtered_df.index,
            filtered_df["Min"],
            linestyle="-.",
            color="pink",
            linewidth=0.4,
        )
        plt.plot(
            filtered_df.index,
            filtered_df["Mean"],
            linestyle="-",
            color="grey",
            linewidth=0.8,
        )
        plt.scatter(filtered_df.index, filtered_df["Fill"], marker="o", c=colors, s=8)
        # for c in filtered_df2.index:
        #     plt.axvline(x=c, color='green', linewidth=0.9)

        plt.xlabel("Date")
        plt.ylabel("Fill Level")
        plt.title(
            "Container ID:" + str(int(self.info["ID"].item())) + "; Freguesia: " + str(self.info["Freguesia"].item())
        )
        plt.xticks(rotation=45)
        plt.legend(["max", "min", "mean", "raw_data", "collections"], loc="upper left")
        plt.show()

    def plot_collection_metrics(self, start_date: datetime, end_date: datetime, fig_size: tuple = (9, 6)):
        """
        Plots the metrics for each collection period between colletions

        Parameters
        ----------
        fig_size: tuple of integers
            alias for plt.figure(figsize=fig_size)
        end/start_date: string with format %d/%m/%Y.
            Period to be analysed. First day is inclusive staring at 00:01 last day is exclusive.
        """
        start_date = pd.to_datetime(start_date, format="%d-%m-%Y", errors="raise")
        end_date = pd.to_datetime(end_date, format="%d-%m-%Y", errors="raise")

        filtered_df2 = self.recs[cast(Any, start_date) : cast(Any, end_date)]
        colors = "orange"
        colors2 = "blue"

        plt.figure(figsize=fig_size)
        plt.xlabel("Date")
        plt.ylabel("Score")
        plt.title(
            "Container ID:" + str(int(self.info["ID"].item())) + "; Freguesia: " + str(self.info["Freguesia"].item())
        )
        plt.xticks(rotation=45)
        for c in filtered_df2.index:
            plt.axvline(x=c, color="green", linewidth=1, label="")

        idx = filtered_df2.index.copy(deep=True)
        idx = idx.to_numpy()
        if len(idx) > 0:
            idx[:-1] = idx[:-1] + (idx[1:] - idx[:-1]) / 2
            idx[-1] = idx[-1] + pd.Timedelta(days=2)

        try:
            plt.plot(idx, filtered_df2["Spearman"], linewidth=0.8, label="")
            plt.scatter(idx, filtered_df2["Spearman"], s=40, c=colors2, label="spearman")
        except Exception:
            pass

        try:
            plt.plot(idx, filtered_df2["Avg_Dist"], linewidth=0.8, label="")
            plt.scatter(idx, filtered_df2["Avg_Dist"], s=40, c=colors, label="100 - avg_dist")
        except Exception:
            pass

        plt.legend(loc="lower left")
        plt.grid()
        plt.ylim(-10, 105)
        plt.show()

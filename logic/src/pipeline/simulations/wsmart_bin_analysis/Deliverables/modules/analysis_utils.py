"""
Analysis utilities for Container fill level data.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

from .core import TAG


class AnalysisMixin:
    """Mixin providing analysis methods for Container fill level data."""

    def get_collection_quantities(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return average distance and Spearman correlation arrays for collections."""
        try:
            avgd = self.recs["Avg_Dist"].copy(deep=True).dropna().to_numpy()
        except Exception:
            avgd = None
        try:
            spear = self.recs.loc[:, ["Spearman"]].copy(deep=True).dropna().to_numpy()
        except Exception:
            spear = None
        return avgd, spear

    def get_scan_linear_spline(self, key, interval) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """Compute a linear spline interpolation of cumulative fill rate."""
        assert key in ["Mean", "Fill"]
        date_range = pd.date_range(
            start=self.recs.index[0].round("D"),
            end=self.df.index[-1].round("D"),
            freq=interval,
        )
        numeric_data_range = np.array([(y - date_range[0]).total_seconds() / (3600 * 24) for y in date_range])
        self.df["Diff"] = self.df[key].diff().fillna(0)
        self.df["Num_T"] = self.df.index.to_series().transform(
            lambda x: (x - self.df.index[0]).total_seconds() / (3600 * 24)
        )
        self.df.loc[self.df["Rec"] == 1, "Diff"] = 0
        self.df["Scan"] = self.df["Diff"].to_numpy().cumsum()
        spline = np.interp(numeric_data_range, self.df["Num_T"], self.df["Scan"])
        self.df.drop(["Diff", "Num_T", "Scan"], axis=1, inplace=True)
        return date_range, spline

    def get_monotonic_mean_rate(self, freq) -> pd.DataFrame:
        """Compute monotonic mean fill rate at the given frequency."""
        data_range, spline = self.get_scan_linear_spline("Mean", freq)
        df = pd.Series(data=np.diff(spline), index=data_range[:-1], name="Rate")
        df.index.name = "Date"
        return df.to_frame()

    def get_crude_rate(self, freq) -> pd.Series:
        """Compute crude fill rate at the given frequency."""
        data_range, spline = self.get_scan_linear_spline("Fill", freq)
        s = pd.Series(data=np.diff(spline), index=data_range[:-1], name="Rate")
        s.index.name = "Date"
        return s

    def get_tag(self, window: int, mv_thresh: int, min_days: int, use: str) -> TAG:
        """Compute and return a quality tag based on collection metrics."""
        KEYS = ["spear", "avg_dist"]
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
            raise ValueError(f"Use must be one of {KEYS}")

        cut_mask = mv < mv_thresh
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            avg_cont = sum(cut_mask.diff().bfill())

        if avg_cont < 1:
            self.tag = TAG.OK
        elif avg_cont == 1:
            self.tag = TAG.INSIDE_BOX
        else:
            self.tag = TAG.WARN
        return self.tag

    def get_collections_std(self):
        """Compute standard deviation of collection intervals."""
        return np.sqrt(self.recs.index.to_series().diff().dropna().dt.total_seconds().std() / 7464960000)

    def calc_spearman(self, start_idx: int = 0, end_idx: int = -1):
        """Calculate Spearman correlation for fill levels between collections."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConstantInputWarning)
            groupDF = self.df[
                self.df.index[self.recs["End_Pointer"].iloc[start_idx]] : self.df.index[
                    self.recs["End_Pointer"].iloc[end_idx] - 1
                ]
            ].groupby("Cidx")
            res = groupDF["Fill"].agg(lambda group: 100 * spearmanr(np.arange(len(group)), group).statistic).fillna(0)
            self.recs.loc[self.recs.index[start_idx] : self.recs.index[end_idx - 1], "Spearman"] = res.to_numpy()

    def calc_avg_dist_metric(self, start_idx: int = 0, end_idx: int = -1):
        """Calculate average distance metric for fill levels between collections."""
        groupDF = self.df[
            self.df.index[self.recs["End_Pointer"].iloc[start_idx]] : self.df.index[
                self.recs["End_Pointer"].iloc[end_idx] - 1
            ]
        ].groupby("Cidx")
        res = groupDF.apply(
            lambda group: (100 - (group["Max"] - group["Min"]).sum() / len(group) if len(group) > 0 else 100)
        ).to_numpy()
        self.recs.loc[self.recs.index[start_idx] : self.recs.index[end_idx - 1], "Avg_Dist"] = res

    def calc_max_min_mean(self, start_idx: int = 0, end_idx: int = -1):
        """Compute max, min, and mean fill levels between collections."""

        def loop(vec: np.ndarray, mask: np.ndarray, mode="max") -> tuple[np.ndarray, bool]:
            """Propagate max or min values through the fill vector."""
            if mode == "max":
                shf = np.roll(vec, 1)
                shf[0] = 0
                tmask = (shf > vec) & mask
            else:
                shf = np.roll(vec, -1)
                shf[-1] = np.finfo(shf.dtype).max if np.issubdtype(shf.dtype, np.floating) else np.iinfo(shf.dtype).max
                tmask = (shf < vec) & mask
            vec[tmask] = shf[tmask]
            return vec, bool(tmask.any())

        my_df = self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], ["Fill", "Rec"]].copy(deep=True)
        vec_max = my_df["Fill"].to_numpy().copy()
        mask_max = (my_df["Rec"] == 0).to_numpy().copy()
        cond = True
        while cond:
            vec_max, cond = loop(vec_max, mask_max, "max")
        self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Max"] = vec_max

        vec_min = my_df["Fill"].to_numpy().copy()
        mask_min = np.roll((my_df["Rec"] == 0).to_numpy(), -1)
        mask_min[-1] = False
        cond = True
        while cond:
            vec_min, cond = loop(vec_min, mask_min, "min")
        self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Min"] = vec_min

        self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Mean"] = (
            self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Min"]
            + self.df.loc[self.recs.index[start_idx] : self.recs.index[end_idx], "Max"]
        ) / 2

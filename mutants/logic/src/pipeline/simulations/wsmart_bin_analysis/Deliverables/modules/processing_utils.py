"""
Processing utilities for Container collection events.
"""

from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd


class ProcessingMixin:
    """Mixin providing collection event processing methods for Container."""

    df: pd.DataFrame
    recs: pd.DataFrame
    id: int

    def calc_max_min_mean(self, start_idx: int = 0, end_idx: int = -1):
        ...

    def calc_avg_dist_metric(self, start_idx: int = 0, end_idx: int = -1):
        ...

    def calc_spearman(self, start_idx: int = 0, end_idx: int = -1):
        ...

    def mark_collections(self):
        """Mark collection events in the fill DataFrame and update records."""
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
        except Exception:
            self.df["Cidx"] = None
            print(f"Container {self.id} Collections and Measures do not intersect")
            raise
        cidx = np.repeat(np.arange(len(self.recs)), np.diff(np.append(self.recs["End_Pointer"], len(self.df))))
        self.df.loc[idx:, "Cidx"] = cidx

    def adjust_collections(self, dist_thresh: int, c_trash: int, max_fill: int):
        """Iteratively adjust collection timestamps based on distance threshold."""
        mask = self.recs["Avg_Dist"] < dist_thresh
        ac_mask = mask.copy(deep=True)
        while mask.any():
            deleted_indexes: List[int] = []
            for index in np.where(mask)[0]:
                idx = index - len(deleted_indexes)
                if index >= len(self.recs) - 2 or idx == 0:
                    continue
                self.adjust_one_collection(idx, c_trash, max_fill)
                if self.adjust_one_collection(idx - 1, c_trash, max_fill) == 1:
                    deleted_indexes.append(index - 1)
            ac_mask.drop(ac_mask.index[deleted_indexes], inplace=True)
            mask = ~ac_mask & (self.recs["Avg_Dist"] < dist_thresh)
            ac_mask |= mask

    def adjust_one_collection(self, idx: int, c_trash: int, max_fill: int) -> int:
        """Adjust a single collection event or remove it if invalid."""
        base_idx, end_idx = self.recs["End_Pointer"].iat[idx] + 1, self.recs["End_Pointer"].iat[idx + 2] - 1
        data = self.df["Fill"].iloc[base_idx:end_idx].diff().copy(deep=True)
        data.loc[self.df["Fill"].iloc[base_idx:end_idx] >= max_fill] = np.NAN
        if data.isna().all():
            new_idx, collected = 0, -1
        else:
            new_idx = data.argmin(skipna=True)
            collected = self.df["Fill"].iat[new_idx + base_idx - 1] - self.df["Fill"].iat[base_idx - 1]

        if collected >= c_trash and self.df["Fill"].iat[new_idx + base_idx] <= max_fill:
            if self.df["Rec"].iat[new_idx + base_idx] == 0:
                self.df.at[self.df.index[new_idx + base_idx], "Rec"] = 1
                new_date = self.df.index[new_idx + base_idx] - timedelta(seconds=1)
                self.df.at[self.df.index[self.recs["End_Pointer"].iat[idx + 1]], "Rec"] = 0
                old_date = self.recs.index.tolist()[idx + 1]
                self.recs.rename(index={old_date: new_date}, inplace=True)
                self.recs.at[new_date, "End_Pointer"] = new_idx + base_idx
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

    def place_collections(self, dist_thresh: int, c_trash: int, max_fill: int, spear_thresh=None):
        """Place new collection events where significant fill drops are detected."""
        mask = (
            (self.recs["Avg_Dist"] < dist_thresh) & (self.recs["Spearman"] < spear_thresh)
            if spear_thresh
            else (self.recs["Avg_Dist"] < dist_thresh)
        )
        added = 1
        while mask.any() and added != 0:
            added = 0
            for index in np.where(mask)[0]:
                idx = index + added
                if idx >= len(self.recs) - 1:
                    continue
                added += self.place_one_collection(idx, c_trash, max_fill)
            mask = (
                (self.recs["Avg_Dist"] < dist_thresh) & (self.recs["Spearman"] < spear_thresh)
                if spear_thresh
                else (self.recs["Avg_Dist"] < dist_thresh)
            )

    def place_one_collection(self, idx: int, c_trash: int, max_fill: int) -> int:
        """Attempt to place a new collection event within a segment."""
        base_idx, end_idx = self.recs["End_Pointer"].iat[idx] + 1, self.recs["End_Pointer"].iat[idx + 1] - 1
        data = self.df["Fill"].iloc[base_idx:end_idx].diff().copy(deep=True)
        data.loc[self.df["Fill"].iloc[base_idx:end_idx] >= max_fill] = np.NAN
        if data.isna().all():
            new_idx, collected = 0, -1
        else:
            new_idx = data.argmin(skipna=True)
            collected = self.df["Fill"].iat[new_idx + base_idx - 1] - self.df["Fill"].iat[base_idx - 1]

        if collected >= c_trash and self.df["Fill"].iat[new_idx + base_idx] <= max_fill:
            if self.df["Rec"].iat[new_idx + base_idx] == 0:
                self.df.at[self.df.index[new_idx + base_idx], "Rec"] = 1
                new_date = self.df.index[new_idx + base_idx] - timedelta(seconds=1)
                new_row = pd.DataFrame({"Date": [new_date], "End_Pointer": [new_idx + base_idx]}).set_index("Date")
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
        """Remove low-quality collection events below the threshold."""
        mv = self.recs["Spearman" if use == "spear" else "Avg_Dist"].rolling(window, center=True).mean().ffill().bfill()
        self.recs = self.recs[~(mv < mv_thresh)]
        self.df = self.df.loc[self.df.index[self.recs["End_Pointer"].iloc[0]] :, :]
        self.mark_collections()

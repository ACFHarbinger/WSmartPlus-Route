"""
Plotting utilities for Container fill level and metrics.
"""

from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


class VisualizationMixin:
    """Mixin providing plotting methods for Container fill level visualization."""

    df: pd.DataFrame
    recs: pd.DataFrame
    info: pd.DataFrame

    def plot_fill(self, start_date: datetime, end_date: datetime, fig_size: tuple = (9, 6)):
        """Plot fill levels over time with collection markers."""
        start_date = pd.to_datetime(start_date, format="%d-%m-%Y", errors="raise")
        end_date = pd.to_datetime(end_date, format="%d-%m-%Y", errors="raise")
        filtered_df = self.df[cast(Any, start_date) : cast(Any, end_date)]
        filtered_recs = self.recs[cast(Any, start_date) : cast(Any, end_date)]
        colors = filtered_df["Rec"].map({1: "green", 0: "red"})

        plt.figure(figsize=fig_size)
        plt.plot(filtered_df.index, filtered_df["Fill"], linestyle="-", color="black", linewidth=0.2)
        plt.scatter(filtered_df.index, filtered_df["Fill"], marker="o", color=colors, s=4.5)
        for c in filtered_recs.index:
            plt.axvline(x=c, color="green", linewidth=0.9)

        plt.xlabel("Date")
        plt.ylabel("Fill Level")
        plt.title(f"Container ID: {int(self.info['ID'].item())}; Freguesia: {self.info['Freguesia'].item()}")
        plt.xticks(rotation=45)

        handles = [
            Line2D([0], [0], marker="o", color="w", label="1st Measure", markerfacecolor="green", markersize=3),
            Line2D([0], [0], marker="o", color="w", label="Measures", markerfacecolor="red", markersize=3),
            Line2D([0], [0], color="green", linewidth=1, label="Collections"),
        ]
        plt.legend(handles=handles, loc="upper left")
        plt.show()

    def plot_max_min(self, start_date: datetime, end_date: datetime, fig_size: tuple = (9, 6)):
        """Plot max, min, and mean fill levels over time."""
        start_date = pd.to_datetime(start_date, format="%d-%m-%Y", errors="raise")
        end_date = pd.to_datetime(end_date, format="%d-%m-%Y", errors="raise")
        filtered_df = self.df[cast(Any, start_date) : cast(Any, end_date)]
        colors = filtered_df["Rec"].map({1: "green", 0: "red"})

        plt.figure(figsize=fig_size)
        plt.plot(filtered_df.index, filtered_df["Max"], linestyle="-.", color="blue", linewidth=0.4)
        plt.plot(filtered_df.index, filtered_df["Min"], linestyle="-.", color="pink", linewidth=0.4)
        plt.plot(filtered_df.index, filtered_df["Mean"], linestyle="-", color="grey", linewidth=0.8)
        plt.scatter(filtered_df.index, filtered_df["Fill"], marker="o", c=colors, s=8)

        plt.xlabel("Date")
        plt.ylabel("Fill Level")
        plt.title(f"Container ID: {int(self.info['ID'].item())}; Freguesia: {self.info['Freguesia'].item()}")
        plt.xticks(rotation=45)
        plt.legend(["max", "min", "mean", "raw_data"], loc="upper left")
        plt.show()

    def plot_collection_metrics(self, start_date: datetime, end_date: datetime, fig_size: tuple = (9, 6)):
        """Plot Spearman and average distance metrics for collections."""
        start_date = pd.to_datetime(start_date, format="%d-%m-%Y", errors="raise")
        end_date = pd.to_datetime(end_date, format="%d-%m-%Y", errors="raise")
        filtered_recs = self.recs[cast(Any, start_date) : cast(Any, end_date)]

        plt.figure(figsize=fig_size)
        plt.xlabel("Date")
        plt.ylabel("Score")
        plt.title(f"Container ID: {int(self.info['ID'].item())}; Freguesia: {self.info['Freguesia'].item()}")
        plt.xticks(rotation=45)
        for c in filtered_recs.index:
            plt.axvline(x=c, color="green", linewidth=1)

        idx = filtered_recs.index.to_numpy()
        if len(idx) > 0:
            idx[:-1] = idx[:-1] + (idx[1:] - idx[:-1]) / 2
            idx[-1] = idx[-1] + pd.Timedelta(days=2)

        if "Spearman" in filtered_recs.columns:
            plt.plot(idx, filtered_recs["Spearman"], linewidth=0.8)
            plt.scatter(idx, filtered_recs["Spearman"], s=40, c="blue", label="spearman")
        if "Avg_Dist" in filtered_recs.columns:
            plt.plot(idx, filtered_recs["Avg_Dist"], linewidth=0.8)
            plt.scatter(idx, filtered_recs["Avg_Dist"], s=40, c="orange", label="100 - avg_dist")

        plt.legend(loc="lower left")
        plt.grid()
        plt.ylim(-10, 105)
        plt.show()

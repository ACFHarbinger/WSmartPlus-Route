"""
Visualization utilities for training logs and statistics.
Separated from log_utils to isolate heavy plotting dependencies (matplotlib).
"""

import os
from typing import Any, Dict, List

import logic.src.constants as udef
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


def log_training(loss_keys: List[str], table_df: pd.DataFrame, opts: Dict[str, Any]) -> None:
    """
    Logs comprehensive training history to Parquet, WandB, and generates Plots.

    Args:
        loss_keys: List of loss column names.
        table_df: DataFrame containing training stats.
        opts: Options dictionary.
    """
    xname: str = "day" if opts["train_time"] else "epoch"
    x_values: List[int] = list(range(table_df.shape[0]))
    log_dir: str = os.path.join(
        opts["log_dir"],
        os.path.relpath(opts["save_dir"], start=opts["checkpoints_dir"]),
    )
    os.makedirs(log_dir, exist_ok=True)
    table_df.to_parquet(os.path.join(log_dir, "table.parquet"), engine="pyarrow")
    swapped_df: pd.DataFrame = table_df.swaplevel(axis=1)
    swapped_df.columns = ["_".join(col).strip() for col in swapped_df.columns]
    if opts["wandb_mode"] != "disabled":
        wandb_table: Any = wandb.Table(dataframe=swapped_df)
        wandb.log({"training_table": wandb_table})

    for l_id, l_key in enumerate(loss_keys):
        mean_loss: np.ndarray = swapped_df[f"mean_{l_key}"].to_numpy()
        std_loss: np.ndarray = swapped_df[f"std_{l_key}"].to_numpy()
        if np.all(mean_loss == std_loss):
            continue
        max_loss: np.ndarray = swapped_df[f"max_{l_key}"].to_numpy()
        min_loss: np.ndarray = swapped_df[f"min_{l_key}"].to_numpy()
        lower_bound: np.ndarray = np.maximum(mean_loss - std_loss, min_loss)
        upper_bound: np.ndarray = np.minimum(mean_loss + std_loss, max_loss)

        fig: Any = plt.figure(
            l_id,
            figsize=(20, 10),
            facecolor="white",
            edgecolor="black",
            layout="constrained",
        )
        label: str = l_key if l_key in udef.LOSS_KEYS else f"{l_key}_cost"
        ax: Any = fig.add_subplot(1, 1, 1)
        ax.plot(x_values, mean_loss, label=f"{label} μ", linewidth=2)
        ax.fill_between(x_values, lower_bound, upper_bound, alpha=0.3, label=f"{label} clip(μ ± σ)")
        ax.scatter(x_values, mean_loss, color="red", s=50, zorder=5)
        ax.set_xlabel(xname)
        ax.set_ylabel(label)
        ax.set_title(f"{label} per {xname}")
        ax.legend()
        ax.grid(True, linestyle="-", alpha=0.9)
        fig_path: str = os.path.join(log_dir, f"{label}.png")
        fig.savefig(fig_path)
        plt.close(fig)
        if opts["wandb_mode"] != "disabled":
            wandb.log({label: wandb.Image(fig_path)})


def log_plot(visualize: bool = False, **kwargs: Any) -> None:
    """
    Execution function for saving static plots.

    Args:
        visualize: Whether to show the plot. Defaults to False.
        **kwargs: Must contain 'fig' (matplotlib Figure) and 'fig_filename' (str).
    """
    kwargs["fig"].savefig(kwargs["fig_filename"], bbox_inches="tight")
    if visualize:
        plt.show()
    plt.close(kwargs["fig"])

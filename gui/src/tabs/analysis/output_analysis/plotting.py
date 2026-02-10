"""
Plotting utilities for Output Analysis.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from matplotlib.figure import Figure

from .engine import calculate_pareto_front


def generate_plot(
    figure: Figure,
    json_data: Dict[str, Any],
    y_key: str,
    x_selection: str,
    chart_type: str,
    selected_dist: str,
    pareto_enabled: bool = False,
) -> None:
    """
    Generates a plot on the provided figure based on filtered json_data.
    """
    figure.clear()
    ax = figure.add_subplot(111)

    all_dists = json_data.get("__Distributions__", [])
    indices_to_plot = []

    for i, dist in enumerate(all_dists):
        if selected_dist in ("All", dist):
            indices_to_plot.append(i)

    if not indices_to_plot:
        ax.text(0.5, 0.5, f"No data for: {selected_dist}", ha="center")
        return

    def filter_data(key: str, indices: List[int]) -> List[Any]:
        """
        Filter a list in json_data by the given indices.

        Args:
            key (str): Key in json_data to filter.
            indices (List[int]): List of indices to keep.

        Returns:
            List[Any]: Filtered list of values.
        """
        full_list = json_data.get(key, [])
        return [full_list[i] for i in indices]

    y_data_filtered = filter_data(y_key, indices_to_plot)
    policy_names_filtered = filter_data("__Policy_Names__", indices_to_plot)
    dists_filtered = filter_data("__Distributions__", indices_to_plot)
    file_ids_filtered = filter_data("__File_IDs__", indices_to_plot)

    is_2d_metric_plot = (x_selection != "Policy Names") and (x_selection in json_data)

    if is_2d_metric_plot:
        x_plot = filter_data(x_selection, indices_to_plot)

        # Group data by Policy Name for PLOTTING THE LINES
        policy_groups = defaultdict(list)
        for i in range(len(policy_names_filtered)):
            name = policy_names_filtered[i]
            policy_groups[name].append(
                {
                    "x": x_plot[i],
                    "y": y_data_filtered[i],
                    "dist": dists_filtered[i],
                }
            )

        # Group data by File ID for PARETO CALCULATION
        file_groups = defaultdict(list)
        for i in range(len(file_ids_filtered)):
            file_id = file_ids_filtered[i]
            file_groups[file_id].append({"x": x_plot[i], "y": y_data_filtered[i]})

        for name, points in policy_groups.items():
            points.sort(key=lambda p: p["x"])
            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]
            ax.plot(xs, ys, marker="o", label=name, markersize=6, alpha=0.8)

        ax.set_xlabel(x_selection)
        ax.set_ylabel(y_key)
        ax.set_title(f"{y_key} vs {x_selection} (Grouped by Policy)")

        if len(policy_groups) < 15:
            ax.legend(fontsize="small")

        if pareto_enabled:
            all_pareto_points_found = False
            for file_id, points in file_groups.items():
                group_xs = [p["x"] for p in points]
                group_ys = [p["y"] for p in points]
                pareto_indices = calculate_pareto_front(group_xs, group_ys)

                if pareto_indices:
                    all_pareto_points_found = True
                    pareto_pts = sorted(
                        [(group_xs[i], group_ys[i]) for i in pareto_indices],
                        key=lambda p: p[0],
                    )
                    px, py = zip(*pareto_pts)
                    ax.plot(px, py, "--", color="black", linewidth=1.5, zorder=1)

            if all_pareto_points_found:
                ax.plot([], [], "--", color="black", linewidth=1.5, label="Pareto Front", zorder=1)
                ax.legend(fontsize="small")
    else:
        x_indices = range(len(policy_names_filtered))
        x_labels = [f"{n} [{d}]" for n, d in zip(policy_names_filtered, dists_filtered)]

        if chart_type == "Line Chart":
            ax.plot(x_indices, y_data_filtered, marker="o")
        elif chart_type == "Bar Chart":
            ax.bar(x_indices, y_data_filtered)
        elif chart_type == "Scatter Plot":
            ax.scatter(x_indices, y_data_filtered)
        elif chart_type == "Area Chart":
            ax.fill_between(x_indices, y_data_filtered, alpha=0.5)

        ax.set_xticks(list(x_indices))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Policy Name [Distribution]")
        ax.set_ylabel(y_key)
        ax.set_title(f"{y_key} Across Policies")

    ax.grid(True, linestyle="--", alpha=0.6)
    figure.tight_layout()

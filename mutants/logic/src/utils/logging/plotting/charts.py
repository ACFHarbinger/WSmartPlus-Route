"""
Line chart plotting utilities.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_linechart(
    output_dest,
    graph_log,
    plot_func,
    policies,
    x_label=None,
    y_label=None,
    title=None,
    fsave=True,
    scale="linear",
    x_values=None,
    linestyles=None,
    markers=None,
    annotate=True,
    pareto_front=False,
):
    """
    Plots a generic line chart, optionally handling multiple policies and Pareto fronts.

    Args:
        output_dest (str): File path to save the plot.
        graph_log (np.ndarray): Data to plot.
        plot_func (callable): The pyplot plotting function (e.g., plt.plot, plt.scatter).
        policies (list): List of policy names for the legend.
        x_label (str, optional): Label for x-axis.
        y_label (str, optional): Label for y-axis.
        title (str, optional): Chart title.
        fsave (bool, optional): Whether to save figure to file. Defaults to True.
        scale (str, optional): Axis scale ('linear', 'log'). Defaults to "linear".
        x_values (list, optional): Custom x-values.
        linestyles (list, optional): List of linestyles.
        markers (list, optional): List of markers.
        annotate (bool, optional): Whether to annotate points. Defaults to True.
        pareto_front (bool, optional): Whether to calculate and overlay Pareto front. Defaults to False.

    Returns:
        list or None: Pareto dominants if pareto_front is True, else None.
    """

    def plot_graphs_out(plot_func, graph_log, x_values, linestyles, markers):
        """
        Helper to plot graphs for different policies.

        Args:
            plot_func: Function to plot a single line.
            graph_log: Log data.
            x_values: X-axis values.
            linestyles: Line styles.
            markers: Markers.
        """
        points_by_nbins = {}
        for id, lg in enumerate(zip(*graph_log)):
            if x_values is None:
                to_plot = (*lg,)
            else:
                to_plot = (
                    x_values,
                    *lg,
                )

            if linestyles is not None:
                line = linestyles[id % len(linestyles)]
            else:
                line = False

            if markers is not None:
                mark = markers[id % len(markers)]
            else:
                mark = False

            if not line and not mark:
                plot_func(*to_plot)
            elif not mark:
                plot_func(*to_plot, linestyle=line)
            elif not line:
                plot_func(*to_plot, marker=mark)
            else:
                plot_func(*to_plot, linestyle=line, marker=mark)

            for id, (x, y) in enumerate(zip(list(zip(*lg))[0], list(zip(*lg))[5])):
                if id not in points_by_nbins:
                    points_by_nbins[id] = []
                points_by_nbins[id].append((x, y))
        return points_by_nbins

    plt.figure(dpi=200)
    if title is not None:
        plt.title(title)

    if len(graph_log.shape) == 2:
        points_by_nbins = {0: []}
        for id, ll in enumerate(graph_log):
            if markers is not None:
                mark = markers[id % len(markers)]
                plot_func(ll, mark)
            else:
                plot_func(ll)

            points_by_nbins[0].append((ll[0], ll[5]))
    elif len(graph_log.shape) == 3:
        # nbins = [20, 50, 100, 200]
        # for id in range(len(nbins)):
        #    graph_log[id, :, 0] /= nbins[id]
        points_by_nbins = plot_graphs_out(plot_func, graph_log, x_values, linestyles, markers)
        if annotate:
            for lg in zip(*graph_log):
                for id, xy in enumerate(zip(list(zip(*lg))[0], list(zip(*lg))[5])):
                    # plt.annotate(id, xy=xy, textcoords='data')
                    if id == graph_log.shape[0] - 1:
                        plt.scatter(
                            xy[0],
                            xy[1],
                            s=200,
                            marker="o",
                            facecolors="none",
                            edgecolors="black",
                            linewidths=1,
                            zorder=10,
                        )

    # Pareto front for minimizing x and maximizing y
    if pareto_front:
        pareto_dominants = []
        pareto_labels = []
        for id_nbins, points in points_by_nbins.items():
            pareto_points = []
            dominance_ls = [0] * len(points)
            for id_point, point in enumerate(points):
                dominated = False
                for other_point in points:
                    # Check if other_point dominates point
                    if (
                        other_point[0] <= point[0]
                        and other_point[1] >= point[1]
                        and (other_point[0] < point[0] or other_point[1] > point[1])
                    ):
                        dominated = True
                        break
                if not dominated:
                    dominance_ls[id_point] = 1
                    pareto_points.append(point)

            pareto_dominants.append(dominance_ls)

            # Sort Pareto points by x-coordinate for drawing the front
            pareto_points.sort(key=lambda p: p[0])

            pareto_x = [p[0] for p in pareto_points]
            pareto_y = [p[1] for p in pareto_points]

            # colors = ['red', 'blue', 'green', 'purple', 'orange']
            # color = colors[id % len(colors)]

            label = f"Pareto Front (ID {id_nbins})"
            pareto_labels.append(label)
            plt.plot(
                pareto_x,
                pareto_y,
                "--",
                color="black",
                linewidth=2,
                label=label,
                zorder=5,
            )

            # Add dots at Pareto points
            # plt.scatter(pareto_x, pareto_y, c=color, s=50, zorder=6)

    if scale != "linear":
        plt.yscale(scale)
        plt.xscale(scale)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    plt.legend(policies)
    if fsave:
        if x_values is not None:
            plt.savefig(output_dest)
        else:
            plt.savefig(output_dest, bbox_inches="tight")
    plt.show()
    return pareto_dominants if pareto_front else None

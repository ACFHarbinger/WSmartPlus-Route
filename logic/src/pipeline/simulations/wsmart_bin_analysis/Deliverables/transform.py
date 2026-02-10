"""
Transformation utilities for processing container and collection data.

This module provides functions to:
- Adjust collection events based on heuristics.
- Fix sensor data anomalies.
- Calculate and visualize statistics like Spearman correlations and distances.
"""

import copy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .container import TAG, Container


def fix_collections_sensor(
    container: Container,
    box_window: int,
    mv_thresh: int,
    min_days: int,
    dist_thresh: int,
    c_trash: int,
    max_fill: int,
    var_thresh: int,
    use: str,
    spear_thresh: Optional[int] = None,
) -> tuple[float, list[TAG], Container]:
    """
    Orchestrates the collections adjustment measures. Deletes the region of the container considered to be inactive (which has
    random measures) and adjust/places collections.

    Parameters
    ----------
    container:Container
        container for collections to be fixed
    box_window:int
        window to compute the right moving average of spearman correlations
    mv_thresh: int
        threshold to for a the moving average to be considered good or bad
    min_days: int
        minimum number of days to consider a container to have a sufficient number of measures
    dist_thresh:int
        threshold for avg_dist to look at the surrounding collections
    c_trash:int,
        minimum collected thrash to count a collection
    max_fill:int
        maximum fill value to count has the first value after a collection
    var_thresh: float
        threshold for the variance change in the distance between the collections in days to be considered
        bad and prevent container.place_collections() to act in containers marked with tag TAG.OK. If such
        rollback happens, it will appear in tag. If such happens, it is an indicative that the sensor was giving
        incoherent measures for a very short period of time.
    use: str
        weather to use average distance or spearman for tagging. Can be "spear" or "avg_dist"
    spear_thresh:int, optional
        threshold for spearman to look at the surrounding collections

    Returns
    -------

    dvar: float
        variance of the distance between the collection in number of days,
    tags: list[Tag]
        list of the tags of the container obtained after each filter
    container:
        copy of the adjusted container adjusted if inplace is false. A pointer to the actual container otherwise.
    """
    container = copy.deepcopy(container)
    tag = container.get_tag(window=box_window, mv_thresh=mv_thresh, min_days=min_days, use=use)
    tags = [tag]
    dvar = 0
    if tag == TAG.INSIDE_BOX:
        container.clean_box(window=box_window, mv_thresh=mv_thresh, use=use)
        tag = container.get_tag(window=box_window, mv_thresh=mv_thresh, min_days=min_days, use=use)
        tags.append(tag)

    var = container.get_collections_std()
    if tag != TAG.LOW_MEASURES:
        container.adjust_collections(dist_thresh=dist_thresh, c_trash=c_trash, max_fill=max_fill)

        tag = container.get_tag(window=box_window, mv_thresh=mv_thresh, min_days=min_days, use=use)
        tags.append(tag)
        if tag == TAG.INSIDE_BOX:
            container.clean_box(window=box_window, mv_thresh=mv_thresh, use=use)
            tag = container.get_tag(window=box_window, mv_thresh=mv_thresh, min_days=min_days, use=use)
            tags.append(tag)

        var_tmp = container.get_collections_std()
        if tag == TAG.OK:
            container_ = copy.deepcopy(container)  # Enable Rollback if the sensor malfunction is local

        container.place_collections(
            dist_thresh=dist_thresh,
            spear_thresh=spear_thresh,
            c_trash=c_trash,
            max_fill=max_fill,
        )
        var_2 = container.get_collections_std()
        if tag == TAG.OK and abs(var_2 - var) > var_thresh:
            container = container_
            tags.append(TAG.LOCAL_WARN)
            var_2 = var_tmp

        dvar = var_2 - var
        tag = container.get_tag(window=box_window, mv_thresh=mv_thresh, min_days=min_days, use=use)
        tags.append(tag)
    return dvar, tags, container


def get_overall_sensors_statistics(containers_dict: dict) -> tuple[dict, dict]:
    """
    Simple Wrapper to get all the containers statistics in an organized way.
    Returns Nones if things are undefined

    Parameters
    ----------
    containers_dict: dict
        Dictionary of container objects to extract statistics from

    Returns
    -------
    dict_dist: dict
        dictionary with ids from the bins and avg distance information
    dict_spear: dict
        dictionary with ids from the bins and spearman information
    """
    dict_dist = {}
    dict_spear = {}
    container: Container
    for id, container in containers_dict.items():
        dist, spear = container.get_collection_quantities()
        dict_dist[id] = dist
        dict_spear[id] = spear
    return dict_dist, dict_spear


def filter_containers(containers_dict: dict) -> dict:
    """
    Given a containers dictionary returns the ids list whose tags are good

    Parameters
    ----------
    containers_dict: dict
        Dictionary of container objects to extract statistics from


    Returns
    -------
    c_dict: dict
        dictionary with the filtered bins
    """
    return dict(filter(lambda c: c[1].tag == TAG.OK or c[1].tag is None, containers_dict.items()))


def pre_process_container_metrics(containers_dict: dict, calc_spearman: bool = True):
    """
    Wrapper to calculate all pre_processing metrics for each container"

    Parameters
    ----------
    containers_dict: dict
        Dictionary of container objects to extract statistics from
    calc_spearman:bool (True)
        Optional; if spearman correlation is to be calculated
    """
    container: Container

    i = 0
    thresh = 0
    step = len(containers_dict) // 20
    for _, container in containers_dict.items():
        container.calc_max_min_mean()
        container.calc_avg_dist_metric()
        if calc_spearman:
            container.calc_spearman()

        i += 1
        if i > thresh:
            print(f"Processed {i} of {len(containers_dict)} containers")
            thresh += step


def view_metrics(containers_dict: dict, box_window: int, mv_thresh: int, min_days: int, use: str):
    """
    Receives the parameters necessary for tagging. Makes 3 histograms with Spearman Average Distance distribution across bins across all bins as well as tags

    Parameters
    ----------
    container:Container
        container for collections to be fixed
    box_window:int
        window to compute the right moving average of spearman correlations
    mv_thresh: int
        threshold to for a the moving average to be considered good or bad
    min_days: int
        minimum number of days to consider a container to have a sufficient number of measures
    use: str
        weather to use average distance or spearman for tagging. Can be "spear" or "avg_dist"
    """
    tags: list[TAG] = []
    container: Container

    dict_dist, dict_spear = get_overall_sensors_statistics(containers_dict=containers_dict)
    for _, container in containers_dict.items():
        tags += [container.get_tag(window=box_window, mv_thresh=mv_thresh, min_days=min_days, use=use)]

    def make_histogram(data, bins, yaxis, tick_labels=None, histrange=None):
        """
        Create and display a histogram plot.

        Args:
            data: Input data for the histogram.
            bins: Number of bins or sequence of bin edges.
            yaxis: Label for the Y-axis.
            tick_labels (list, optional): Custom tick labels.
            histrange (tuple, optional): Range of the histogram (min, max).
        """
        hist, bin_edges = np.histogram(a=data, bins=bins, range=histrange, density=True)
        hist = hist * np.diff(bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot the histogram using plt.bar
        if tick_labels is None:
            plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.6)
        else:
            plt.bar(
                bin_centers,
                hist,
                width=(bin_edges[1] - bin_edges[0]),
                alpha=0.6,
                tick_label=tick_labels,
            )

        # Add labels and title
        plt.xticks(rotation=45)
        plt.xlabel("Value")
        plt.ylabel(yaxis)
        plt.show()

    if use == "spear":
        all_spear = np.concatenate(list(dict_spear.values()))
        make_histogram(all_spear, 20, "Spearman Distribution")

    all_dist = np.concatenate(list(dict_dist.values()))
    make_histogram(all_dist, 10, "Average Distance Distribution")

    make_histogram(
        [tag.value for tag in tags],
        range(len(TAG) + 1),
        "Percentage",
        tick_labels=[tag.name for tag in TAG],
        histrange=(0, max([tag.value for tag in TAG])),
    )

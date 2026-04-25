"""Hybrid Genetic Search with Adaptive Diversity Control (HGS-ADC) configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HGSADCConfig:
    """
    HGS-ADC configuration parameters.

    Attributes:
        pop_size (int): Population size.
        mu (int): Number of parents selected for reproduction.
        nb_close (int): Number of nearest neighbors for local search.
        generations (int): Number of generations.
        n_vehicles (int): Number of vehicles.
    """

    pop_size: int = 25
    mu: int = 25
    nb_close: int = 4
    generations: int = 50
    n_vehicles: int = 0

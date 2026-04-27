"""
Data generation and management module for WSmart-Route.

This package contains tools for creating synthetic VRP instances, including
builders for various problem types (VRPP, WCVRP, SWCVRP) and generation scripts.

Attributes:
    generate_datasets: Generate datasets for various problem types.

Example:
    >>> from logic.src.data import generate_datasets
    >>> generate_datasets()
"""

from .generators.datasets import generate_datasets as generate_datasets

__all__ = [
    "generate_datasets",
]

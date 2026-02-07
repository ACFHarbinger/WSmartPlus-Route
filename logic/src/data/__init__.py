"""
Data generation and management module for WSmart-Route.

This package contains tools for creating synthetic VRP instances, including
builders for various problem types (VRPP, WCVRP, SWCVRP) and generation scripts.
"""

from .generators.datasets import generate_datasets as generate_datasets

"""Data access and business logic services for the Streamlit dashboard.

This package provides a centralized interface for loading, parsing, and
analyzing simulation telemetry and training logs.

Attributes:
    data_loader: Core utilities for filesystem and database access.
    log_parser: Specialized logic for parsing simulation JSONL logs.
    simulation_analytics: Statistical computation services.
    tracking_service: Backend-agnostic experiment tracking.

Example:
    >>> from logic.src.ui.services import data_loader
    >>> root = data_loader.get_project_root()
"""

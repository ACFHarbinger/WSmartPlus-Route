# Copyright (c) WSmart-Route. All rights reserved.
"""
Configuration file for mutmut mutation testing.
"""


def pre_mutation(context):
    """
    Filter out files that should not be mutated.
    """
    filename = context.filename
    # Only mutate core logic and models
    if not (filename.startswith("logic/src/policies/") or filename.startswith("logic/src/models/")):
        context.skip = True

    # Skip tests, docs, and benchmarks
    if "test" in filename or "docs" in filename or "benchmark" in filename:
        context.skip = True

    # Skip specific files if needed
    if filename.endswith("__init__.py"):
        context.skip = True

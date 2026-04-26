"""
Sphinx configuration file for the WSmart-Route documentation.

Attributes:
    extensions: List of sphinx extensions to be used.
    templates_path: List of paths to the sphinx templates.
    exclude_patterns: List of patterns to be excluded from the documentation.
    html_theme: Theme to be used for the documentation.
    html_static_path: List of paths to the sphinx static files.
    release: Version of the project.
    author: Author of the project.
    copyright: Copyright information of the project.
    project: Name of the project.

Example:
    None
"""

# Configuration file for the Sphinx documentation builder.

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Smart+ Bin Forecast"
copyright = "2024, André Ribeiro"
author = "André Ribeiro"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

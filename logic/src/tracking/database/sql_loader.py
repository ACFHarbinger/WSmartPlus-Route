"""SQL loader utility for the tracking database CLI.

This module provides the mechanism for loading raw SQL strings from external
files located in the `sql/` sub-package. It supports loading entire file
contents or extracting specifically delimited sections from high-level SQL
templates.

Attributes:
    _SQL_DIR: The directory path where SQL files are stored.
    _SECTION_RE: Regex pattern to identify section markers like -- [name].

Example:
    >>> from logic.src.tracking.database.sql_loader import load_sections
    >>> queries = load_sections("stats.sql")
    >>> print(queries["table_sizes"])
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

_SQL_DIR = Path(__file__).parent / "sql"

_SECTION_RE = re.compile(r"^--\s*\[(\w+)\]\s*$", re.MULTILINE)


def load_sql(filename: str) -> str:
    """Returns the raw text of a SQL file from the sql/ directory.

    Args:
        filename: Name of the file within the sql/ directory (e.g., 'inspect.sql').

    Returns:
        str: The raw SQL content.
    """
    return (_SQL_DIR / filename).read_text(encoding="utf-8")


def load_sections(filename: str) -> Dict[str, str]:
    """Parses a SQL file with -- [section_name] markers into a dictionary.

    Sections appear in file order. Each body is stripped of leading/trailing
    whitespace. Empty bodies are omitted.

    Args:
        filename: Filename relative to the sql/ subdirectory.

    Returns:
        Dict[str, str]: Mapping of section name to SQL text.
    """
    text = load_sql(filename)
    parts = _SECTION_RE.split(text)
    # parts layout after split: [pre_text, name1, body1, name2, body2, ...]
    sections: Dict[str, str] = {}
    it = iter(parts[1:])  # skip any text before the first marker
    for name, body in zip(it, it):
        stripped = body.strip()
        if stripped:
            sections[name] = stripped
    return sections

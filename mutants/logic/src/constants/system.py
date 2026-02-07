"""
System-wide constants and operation mappings.
"""

from typing import Any, Callable, Dict, List, Optional

# Multi-core processing settings
CORE_LOCK_WAIT_TIME: int = 100
LOCK_TIMEOUT: int = CORE_LOCK_WAIT_TIME


def update_lock_wait_time(num_cpu_cores: Optional[int] = None) -> int:
    """
    Updates the global LOCK_TIMEOUT based on the number of CPU cores.

    Args:
        num_cpu_cores: Number of CPU cores to scale the timeout by.

    Returns:
        The new (or default) value of LOCK_TIMEOUT.
    """
    global LOCK_TIMEOUT
    global CORE_LOCK_WAIT_TIME
    if num_cpu_cores is None:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME
    else:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME * num_cpu_cores
    return LOCK_TIMEOUT


# File system settings
CONFIRM_TIMEOUT: int = 30

FS_COMMANDS: List[str] = ["create", "read", "update", "delete", "cryptography"]

OPERATION_MAP: Dict[str, Callable[[Any, Any], Any]] = {
    "=": lambda x, y: y,
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "+": lambda x, y: x + y,
    "+=": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "-=": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "*=": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "/=": lambda x, y: x / y,
    "**": lambda x, y: x**y,
    "**=": lambda x, y: x**y,
    "//": lambda x, y: x // y,
    "//=": lambda x, y: x // y,
    "%": lambda x, y: x % y,
    "%=": lambda x, y: x % y,
    "": lambda x, y: x,
    "<<": lambda x, y: x << y,
    "<<=": lambda x, y: x << y,
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y,
    ">>": lambda x, y: x >> y,
    ">>=": lambda x, y: x >> y,
    ">": lambda x, y: x > y,
    ">=": lambda x, y: x >= y,
    "|": lambda x, y: x | y,
    "|=": lambda x, y: x | y,
    "&": lambda x, y: x & y,
    "&=": lambda x, y: x & y,
    "^": lambda x, y: x ^ y,
    "^=": lambda x, y: x ^ y,
    "is": lambda x, y: x is y,
    "isnot": lambda x, y: x is not y,
    "in": lambda x, y: x in y,
    "notin": lambda x, y: x not in y,
    "@": lambda x, y: x @ y,
    "@=": lambda x, y: x @ y,
    "divmod": lambda x, y: divmod(x, y),
}

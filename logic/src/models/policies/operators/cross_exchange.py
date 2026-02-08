"""cross_exchange.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import cross_exchange
    """
from logic.src.models.policies.operators.swap_star import vectorized_swap_star

# Aliases
vectorized_cross_exchange = vectorized_swap_star

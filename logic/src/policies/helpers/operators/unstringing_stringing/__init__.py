"""
Unstringing and Stringing (US) operators package.

Based on the GENIUS algorithm and variants described in:
- Gendreau, M., Hertz, A., & Laporte, G. (1992). "New Insertion and
  Postoptimization Procedures for the Traveling Salesman Problem."
  Operations Research, 40(6), 1086-1094.
- Müller, L. F., & Bonilha, I. (2022). "Hyper-Heuristic Based on ACO
  and Local Search for Dynamic Optimization Problems."
  Algorithms, 15(1), 9. https://doi.org/10.3390/a15010009
"""

from logic.src.policies.helpers.operators.unstringing_stringing.stringing_i import (
    apply_type_i_s,
    apply_type_i_s_profit,
)
from logic.src.policies.helpers.operators.unstringing_stringing.stringing_ii import (
    apply_type_ii_s,
    apply_type_ii_s_profit,
)
from logic.src.policies.helpers.operators.unstringing_stringing.stringing_iii import (
    apply_type_iii_s,
    apply_type_iii_s_profit,
)
from logic.src.policies.helpers.operators.unstringing_stringing.stringing_iv import (
    apply_type_iv_s,
    apply_type_iv_s_profit,
)
from logic.src.policies.helpers.operators.unstringing_stringing.stringing_wrapper import (
    stringing_insertion,
    stringing_profit_insertion,
)
from logic.src.policies.helpers.operators.unstringing_stringing.unstringing_i import (
    apply_type_i_us,
    apply_type_i_us_profit,
)
from logic.src.policies.helpers.operators.unstringing_stringing.unstringing_ii import (
    apply_type_ii_us,
    apply_type_ii_us_profit,
)
from logic.src.policies.helpers.operators.unstringing_stringing.unstringing_iii import (
    apply_type_iii_us,
    apply_type_iii_us_profit,
)
from logic.src.policies.helpers.operators.unstringing_stringing.unstringing_iv import (
    apply_type_iv_us,
    apply_type_iv_us_profit,
)
from logic.src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper import (
    unstringing_profit_removal,
    unstringing_removal,
)

__all__ = [
    # Unstringing operators (removal)
    "apply_type_i_us",
    "apply_type_i_us_profit",
    "apply_type_ii_us",
    "apply_type_ii_us_profit",
    "apply_type_iii_us",
    "apply_type_iii_us_profit",
    "apply_type_iv_us",
    "apply_type_iv_us_profit",
    # Stringing operators (insertion)
    "apply_type_i_s",
    "apply_type_i_s_profit",
    "apply_type_ii_s",
    "apply_type_ii_s_profit",
    "apply_type_iii_s",
    "apply_type_iii_s_profit",
    "apply_type_iv_s",
    "apply_type_iv_s_profit",
    # Automated wrappers
    "stringing_insertion",
    "stringing_profit_insertion",
    "unstringing_removal",
    "unstringing_profit_removal",
]

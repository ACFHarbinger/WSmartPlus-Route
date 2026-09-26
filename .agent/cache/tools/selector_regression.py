"""Regression checks for the selector fixes DS-19/20/21 (B-cursor-02/03/04).

usage (repo root): .venv/bin/python .agent/cache/tools/selector_regression.py
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from tensordict import TensorDict

from logic.src.pipeline.rl.core.reinforce import REINFORCE
from logic.src.policies.mandatory_selection.selection_lookahead import LookaheadSelection
from logic.src.policies.vector.selection import LastMinuteSelector

# DS-20: percent threshold against fraction fills
sel = LastMinuteSelector()  # 70 %
mask = sel.select(torch.tensor([[0.0, 0.5, 0.75, 0.95]]))
assert mask.tolist() == [[False, False, True, True]], mask
print("DS-20 ok: threshold 70 (%) selects fills 0.75/0.95 only")

# DS-19: training path — first customer (fullest) can be mandatory
module = REINFORCE(env=SimpleNamespace(name="vrpp", num_loc=3), policy=torch.nn.Linear(1, 1),
                   baseline="none", mandatory_selector=LastMinuteSelector(70))
td = TensorDict({"waste": torch.tensor([[0.9, 0.1, 0.1]])}, [1])
out = module._apply_mandatory_selection(td)
assert out["mandatory"].tolist() == [[False, True, False, False]], out["mandatory"]
print("DS-19 ok: customer 1 (fill 0.9) is mandatory; depot column prepended")

# DS-21: scalar lookahead uses days relative to today
ls = LookaheadSelection.__new__(LookaheadSelection)
fill, rate = np.array([60.0, 40.0]), np.array([50.0, 20.0])
day0 = ls._add_bins_to_collect([0, 1], 3, [], fill, rate, current_collection_day=0)
day5 = ls._add_bins_to_collect([0, 1], 8, [], fill, rate, current_collection_day=5)
assert sorted(day0) == sorted(day5), (day0, day5)
print(f"DS-21 ok: same horizon length gives the same bins on day 0 and day 5 ({sorted(day5)})")
print("ALL SELECTOR REGRESSIONS PASS")

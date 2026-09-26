import sys
sys.path.insert(0, "/tmp/kimi-wsr")
import numpy as np
from omegaconf import OmegaConf
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.params import BPCParams
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import run_bpc

cfg = OmegaConf.load("logic/configs/policies/policy_bpc.yaml")
flat = {}
for item in cfg.bpc.custom:
    flat.update(OmegaConf.to_container(item))
params = BPCParams.from_config(flat)
print("shipped yaml params: lr_pre_pruning =", params.lr_pre_pruning,
      "| cutting_planes =", params.cutting_planes,
      "| exact_mode =", params.exact_mode,
      "| enable_dssr =", params.enable_dssr,
      "| time_limit =", params.time_limit)

dm = np.array([[0, 1, 1, 1],
               [1, 0, 5, 1],
               [1, 5, 0, 1],
               [1, 1, 1, 0]], dtype=float)
wastes = {1: 1.0, 2: 1.0, 3: 1.0}
routes, obj = run_bpc(dm, wastes, 2.0, 10.0, 1.0, params)
print(f"SHIPPED YAML CONFIG -> routes={routes} obj={obj:.4f}   (true optimum = 25.0)")
print("verdict:", "WRONG (invalid LCI cut pruned the multi-route optimum)" if obj < 25 - 1e-6 else "ok")

"""
BPC Policy Adapter.

Adapts the Branch-and-Price-and-Cut (BPC) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BPCConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import run_bpc

from .params import BPCParams


@PolicyRegistry.register("bpc")
class BPCPolicy(BaseRoutingPolicy):
    """
    Branch-and-Price-and-Cut policy class.

    Visits pre-selected 'mandatory' bins using exact or heuristic BPC solvers.
    """

    def __init__(self, config: Optional[Union[BPCConfig, Dict[str, Any]]] = None):
        """Initialize BPC policy with optional config.

        Args:
            config: BPCConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return BPCConfig

    def _get_config_key(self) -> str:
        """Return config key for BPC."""
        return "bpc"

    def _load_model(self, path: str, model_type: str) -> None:
        """Lazy load the XGBoost or Torch model."""
        if getattr(self, "_model_loaded", False):
            return

        import os

        if not os.path.exists(path):
            raise RuntimeError(f"BPCPolicy: ADP Model file not found at {path}")

        if model_type == "sklearn":
            import joblib

            self._model = joblib.load(path)
        elif model_type == "torch":
            import torch

            self._model = torch.load(path, map_location="cpu")
            self._model.eval()
        else:
            raise RuntimeError(f"BPCPolicy: Unsupported model_type {model_type}")

        self._model_loaded = True

    def _predict_V(self, features: np.ndarray, model_type: str) -> np.ndarray:
        """Predict the value (expected future cost or penalty) of the state."""
        if model_type == "sklearn":
            if hasattr(self._model, "predict_proba"):
                return self._model.predict_proba(features)[:, 1]
            else:
                return self._model.predict(features)
        elif model_type == "torch":
            import torch

            with torch.no_grad():
                input_tensor = torch.from_numpy(features).float()
                output = self._model(input_tensor)
                return output.squeeze().numpy()
        raise ValueError("Invalid model_type")

    def execute(self, **kwargs: Any) -> Tuple[Union[List[int], List[List[int]]], float, Any]:
        """
        Execute BPC.
        If multi_day_mode is True, evaluates for a single stage of the multi-period
        problem without standard subset extraction, and applies Approximate Dynamic Programming.
        """
        import logging

        logger = logging.getLogger(__name__)

        config_dict = kwargs.get("config", {}).get(self._get_config_key(), {})
        multi_day_mode = config_dict.get("multi_day_mode", False)

        if not multi_day_mode:
            # Standard single-day mode
            return super().execute(**kwargs)

        # Multi-day mode (BPC-ADP)
        dist_matrix = kwargs["model_ls"][1]
        bins = kwargs["bins"]
        profit_vars = kwargs["model_ls"][2]

        R = profit_vars.get("revenue_kg", 1.0)
        C = profit_vars.get("cost_km", 1.0)
        vehicle_limit = profit_vars.get("number_vehicles", None)

        wastes = {i: float(bins.c[i - 1]) for i in range(1, len(bins.c) + 1)}
        capacity = float(profit_vars.get("bin_capacity", 100.0))
        mandatory = set(kwargs.get("mandatory", []))

        params = BPCParams.from_config(config_dict)

        adp_model_path = config_dict.get("adp_model_path", "")
        adp_model_type = config_dict.get("adp_model_type", "sklearn")

        n_bins = len(bins.c)
        node_prizes: Dict[int, float] = {}

        if adp_model_path:
            self._load_model(adp_model_path, adp_model_type)

            fill_ratios = np.array(bins.c) / capacity
            acc_rate = bins.rate if hasattr(bins, "rate") and bins.rate is not None else np.zeros(n_bins)
            std_dev = getattr(bins, "rate_std", np.zeros(n_bins))
            dist_to_depot = dist_matrix[0, 1:]

            for i in range(1, n_bins + 1):
                idx = i - 1

                # Construct S_t+1 | leave bin i
                leave_fill = min(1.0, fill_ratios[idx] + (acc_rate[idx] / capacity if acc_rate[idx] > 0 else 0))
                feat_leave = np.array(
                    [
                        leave_fill,
                        acc_rate[idx],
                        std_dev[idx],
                        dist_to_depot[idx],
                        leave_fill * capacity * R,
                        (1.0 - leave_fill) * capacity / (acc_rate[idx] + 1e-6) if acc_rate[idx] > 0 else 99.0,
                    ]
                ).reshape(1, -1)

                v_leave = self._predict_V(feat_leave, adp_model_type)
                if isinstance(v_leave, (list, np.ndarray)):
                    v_leave = v_leave[0]

                # Construct S_t+1 | empty bin i
                empty_fill = min(1.0, 0.0 + (acc_rate[idx] / capacity if acc_rate[idx] > 0 else 0))
                feat_empty = np.array(
                    [
                        empty_fill,
                        acc_rate[idx],
                        std_dev[idx],
                        dist_to_depot[idx],
                        empty_fill * capacity * R,
                        (1.0 - empty_fill) * capacity / (acc_rate[idx] + 1e-6) if acc_rate[idx] > 0 else 99.0,
                    ]
                ).reshape(1, -1)

                v_empty = self._predict_V(feat_empty, adp_model_type)
                if isinstance(v_empty, (list, np.ndarray)):
                    v_empty = v_empty[0]

                # Augmented Node Prize
                rho_it = float(v_leave - v_empty)
                base_profit = wastes[i] * R
                node_prizes[i] = base_profit + rho_it
        else:
            logger.warning(
                "BPCPolicy executing multi-day mode without adp_model_path. Running as standard unaugmented."
            )
            for i in range(1, n_bins + 1):
                node_prizes[i] = wastes[i] * R

        routes, _ = run_bpc(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params,
            mandatory_indices=mandatory,
            vehicle_limit=vehicle_limit,
            node_prizes=node_prizes,
        )

        global_route = []
        if routes:
            for r in routes:
                global_route.extend([n for n in r if n != 0])
                global_route.append(0)

        model_env = kwargs.get("model_env")
        cost = model_env.compute_route_cost(global_route) if model_env is not None else 0.0

        return global_route, cost, {"policy_type": "bpc", "multi_day_mode": True, "node_prizes": node_prizes}

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run BPC solver.

        All nodes in mandatory_nodes are treated as mandatory for the solver.
        In VRPP mode, additional nodes from sub_wastes might be collected if profitable.

        Returns:
            Tuple of (routes, profit, solver_cost)
            - routes: List of routes (list of node indices).
            - profit: Objective value (collected revenue - distance cost) in $.
            - solver_cost: Raw travel distance (km), NOT multiplied by cost_unit.
              Callers needing monetary cost should compute solver_cost * cost_unit.
        """
        # Return contract for run_bpc:
        #   routes          — list of customer-node lists (depot excluded)
        #   objective_value — net profit = Σ(revenue_i) - travel_cost, in monetary units.
        #                     May be a greedy-fallback value if BPC found no integer solution.
        # Convert local mandatory indices to a set of mandatory nodes for the solver
        mandatory_indices: Set[int] = set(mandatory_nodes)

        # Initialize standardized params object (Phase 1 refactoring)
        params = BPCParams.from_config(values)

        # Extract vehicle limit from simulation context (sim.n_vehicles)
        n_vehicles = kwargs.get("n_vehicles")
        # Explicit int conversion and positive check. None and 0 both map to
        # unlimited fleet. False is rejected at the int() call (TypeError surfaced
        # to the caller) rather than silently treated as unlimited.
        vehicle_limit = None if n_vehicles is None else int(n_vehicles) if int(n_vehicles) > 0 else None

        # run_bpc returns (routes, objective_value) where objective_value is
        # net profit (revenue - travel_cost) in the problem's monetary units.
        # It is NOT a raw travel cost despite the variable name used in run_bpc's
        # return signature. Rename immediately to prevent future misreading.
        routes, objective_value = run_bpc(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_indices=mandatory_indices,
            vehicle_limit=vehicle_limit,
            env=kwargs.get("model_env"),
            node_coords=kwargs.get("node_coords"),
            recorder=kwargs.get("recorder"),
        )

        profit = objective_value

        # Compute raw travel distance (km)
        raw_distance = 0.0
        for route in routes:
            # Normalize: strip any leading/trailing depot index before wrapping.
            # Route.nodes stores customer-only sequences, but defensive stripping
            # guards against representation changes in run_bpc's return value.
            inner = [n for n in route if n != 0]
            if not inner:
                continue

            path = [0] + inner + [0]
            for i in range(len(path) - 1):
                raw_distance += sub_dist_matrix[path[i]][path[i + 1]]

        return routes, profit, raw_distance

"""
GPHH Policy Adapter.

Adapts the Genetic Programming Hyper-Heuristic (GPHH) constructive heuristic
generator to the agnostic BaseRoutingPolicy interface.

Generates synthetic training environments (random Euclidean distance matrices)
to provide true spatial-topology diversity during GP tree evolution.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.gphh import GPHHConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.genetic_programming_hyper_heuristic.params import GPHHParams
from logic.src.policies.genetic_programming_hyper_heuristic.solver import GPHHSolver

# ---------------------------------------------------------------------------
# Synthetic training environment factory
# ---------------------------------------------------------------------------


def _make_synthetic_training_envs(
    n_nodes: int,
    n_envs: int,
    capacity: float,
    R: float,
    C: float,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, Dict[int, float], List[int]]]:
    """
    Generate ``n_envs`` random Euclidean VRPP instances for GP training.

    Each environment has the same number of nodes as the test instance but
    with independently randomised 2-D coordinates and waste values, giving
    distinct spatial topologies.  This prevents the GP tree from overfitting
    to the test instance's geometry during evolution.

    Distance matrices are Euclidean (L2), scaled to the unit square.
    Waste values are drawn uniformly from (0, max_capacity / 4].

    Args:
        n_nodes: Number of customer nodes (excluding depot).
        n_envs: Number of training environments to generate.
        capacity: Vehicle capacity (for waste sampling).
        R: Revenue coefficient.
        C: Cost coefficient (unused here, kept for signature symmetry).
        rng: NumPy random generator for reproducibility.

    Returns:
        List of (dist_matrix, wastes, mandatory_nodes) triples.
    """
    envs: List[Tuple[np.ndarray, Dict[int, float], List[int]]] = []
    max_waste = max(capacity / 4.0, 1.0)

    for _ in range(n_envs):
        # Generate random 2-D coordinates for depot + n_nodes customers
        coords = rng.random((n_nodes + 1, 2)) * 100.0
        dm = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))

        wastes = {i: float(rng.uniform(0.1, max_waste)) for i in range(1, n_nodes + 1)}
        envs.append((dm, wastes, []))  # No mandatory nodes in synthetic envs

    return envs


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------


@PolicyRegistry.register("gphh")
class GPHHPolicy(BaseRoutingPolicy):
    """
    GPHH policy class.

    Evolves a GP tree that scores candidate insertions during constructive
    solution building, generating a learned construction heuristic.
    Provides synthetic training environments with diverse spatial topologies
    to promote generalizable GP trees.
    """

    def __init__(self, config: Optional[Union[GPHHConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GPHHConfig

    def _get_config_key(self) -> str:
        return "gphh"

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
        seed = values.get("seed", 42)
        n_envs = int(values.get("n_training_instances", 3))
        n_nodes = len(sub_dist_matrix) - 1

        params = GPHHParams(
            gp_pop_size=int(values.get("gp_pop_size", 20)),
            max_gp_generations=int(values.get("max_gp_generations", 30)),
            tree_depth=int(values.get("tree_depth", 3)),
            tournament_size=int(values.get("tournament_size", 3)),
            time_limit=float(values.get("time_limit", 60.0)),
            parsimony_coefficient=float(values.get("parsimony_coefficient", 0.0)),
            candidate_list_size=int(values.get("candidate_list_size", 10)),
            n_training_instances=n_envs,
            training_sample_ratio=float(values.get("training_sample_ratio", 0.5)),
            seed=seed,
            vrpp=values.get("vrpp", True),
        )

        # Generate synthetic training environments with diverse spatial topologies
        rng = np.random.default_rng(seed)
        training_envs = _make_synthetic_training_envs(
            n_nodes=n_nodes,
            n_envs=n_envs,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            rng=rng,
        )

        solver = GPHHSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
            training_environments=training_envs,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost

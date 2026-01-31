"""
VRPP Optimizer module - Unified interface for Gurobi and Hexaly solvers.

This module implements exact and heuristic solvers for the Vehicle Routing
Problem with Profits (VRPP) in the context of waste collection.

Solver Backends:
----------------
1. Gurobi: Mixed-Integer Programming with advanced cuts and heuristics
   - 2-index flow formulation
   - Capacity constraints (load variables)
   - Subtour elimination (flow variables)
   - Critical bin enforcement (must-go constraints)

2. Hexaly: Local Search Optimizer (formerly LocalSolver)
   - List-based modeling (route representation)
   - Automatic fleet sizing support
   - Early stopping callback for convergence
   - Efficient for large-scale instances

VRPP Objective:
    Maximize: Revenue(collected_waste) - Cost(distance) - Penalty(vehicles_used)

Key Features:
- Prize-collecting formulation (optional node visits)
- Must-go node enforcement (critical bins)
- Multi-vehicle support with automatic fleet sizing
- Time limits and solution quality controls
"""

import io
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import gurobipy as gp
import hexaly.optimizer as hx
import numpy as np
from gurobipy import GRB, quicksum
from numpy.typing import NDArray

from logic.src.constants.optimization import (
    HEURISTICS_RATIO,
    MIP_GAP,
    NODEFILE_START_GB,
)

from ..base_routing_policy import BaseRoutingPolicy
from .factory import PolicyRegistry


# Retaining the complex static solver functions outside the class
def run_vrpp_optimizer(
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    param: float,
    media: NDArray[np.float64],
    desviopadrao: NDArray[np.float64],
    values: Dict[str, float],
    binsids: List[int],
    must_go: List[int],
    env: Optional[gp.Env] = None,
    number_vehicles: int = 1,
    time_limit: int = 60,
    optimizer: str = "gurobi",
    max_iter_no_improv: int = 10,
):
    """
    Solve VRPP using either Gurobi or Hexaly optimizer.
    """
    if optimizer == "gurobi":
        return _run_gurobi_optimizer(
            bins=bins,
            distance_matrix=distance_matrix,
            env=env,
            param=param,
            media=media,
            desviopadrao=desviopadrao,
            values=values,
            binsids=binsids,
            must_go=must_go,
            number_vehicles=number_vehicles,
            time_limit=time_limit,
        )
    elif optimizer == "hexaly":
        return _run_hexaly_optimizer(
            bins=bins,
            distancematrix=distance_matrix,
            param=param,
            media=media,
            desviopadrao=desviopadrao,
            values=values,
            must_go=must_go,
            number_vehicles=number_vehicles,
            time_limit=time_limit,
            max_iter_no_improv=max_iter_no_improv,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def _run_gurobi_optimizer(
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    env: Optional[gp.Env],
    param: float,
    media: NDArray[np.float64],
    desviopadrao: NDArray[np.float64],
    values: Dict[str, float],
    binsids: List[int],
    must_go: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
):
    """
    Solve the Vehicle Routing Problem with Profits using Gurobi Optimizer.
    """
    Omega, delta, psi = values["Omega"], values["delta"], values["psi"]
    Q, R, B, C, V = values["Q"], values["R"], values["B"], values["C"], values["V"]

    n_bins = len(bins)
    enchimentos = np.insert(bins, 0, 0.0)
    pesos_reais = [(e / 100) * B * V for e in enchimentos]  # Altera os enchimentos de % pra valor real em KG
    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]
    S_dict = {i: pesos_reais[i] for i in nodes}

    criticos = [bin_id in must_go for bin_id in binsids]
    criticos_dict = {i: criticos[i] for i in nodes}

    max_dist = 6000
    pares_viaveis = [(i, j) for i in nodes for j in nodes if i != j and distance_matrix[i][j] <= max_dist]
    mdl = gp.Model("VRPP", env=env) if env else gp.Model("VRPP")
    mdl.Params.LogToConsole = 0

    x = mdl.addVars(pares_viaveis, vtype=GRB.BINARY, name="x")
    y = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="y")
    f = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="f")
    g = mdl.addVars(nodes, vtype=GRB.BINARY, name="g")
    k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k_var")
    for i, j in pares_viaveis:
        mdl.addConstr(y[i, j] <= Q * x[i, j])
        mdl.addConstr(f[i, j] <= len(nodes) * x[i, j])

    for i in nodes_real:
        mdl.addConstr(quicksum(y[i, j] - y[j, i] for j in nodes if (i, j) in y or (j, i) in y) == S_dict[i] * g[i])

    if number_vehicles == 0:
        number_vehicles = len(binsids)

    MAX_TRUCKS = number_vehicles
    mdl.addConstr(k_var <= MAX_TRUCKS)

    mdl.addConstr(k_var == quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x))
    mdl.addConstr(quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x) == k_var)
    mdl.addConstr(quicksum(x[j, idx_deposito] for j in nodes_real if (j, idx_deposito) in x) == k_var)

    for j in nodes_real:
        if (idx_deposito, j) in x:
            mdl.addConstr(x[idx_deposito, j] <= g[j])
        if (j, idx_deposito) in x:
            mdl.addConstr(x[j, idx_deposito] <= g[j])

    mdl.addConstr(
        quicksum(g[i] for i in nodes_real if criticos_dict[i])
        >= len([i for i in nodes_real if criticos_dict[i]]) - len(nodes_real) * delta
    )

    for i in nodes_real:
        if criticos_dict[i] or enchimentos[i] >= psi * 100:
            mdl.addConstr(g[i] == 1)
        if enchimentos[i] < 10 and not criticos[i]:
            g[i].UB = 0

    for j in nodes_real:
        mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == g[j])
        mdl.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in x) == g[j])

    mdl.addConstr(quicksum(f[0, j] for j in nodes_real if (0, j) in f) == quicksum(g[j] for j in nodes_real))
    for j in nodes_real:
        mdl.addConstr(
            quicksum(f[i, j] for i in nodes if (i, j) in f) - quicksum(f[j, k] for k in nodes if (j, k) in f) == g[j]
        )

    mdl.setObjective(
        R * quicksum(S_dict[i] * g[i] for i in nodes_real)
        - 0.5 * C * quicksum(x[i, j] * distance_matrix[i][j] for i, j in pares_viaveis)
        - Omega * k_var,
        GRB.MAXIMIZE,
    )

    mdl.Params.MIPFocus = 1
    mdl.Params.Heuristics = HEURISTICS_RATIO
    mdl.Params.Threads = 0
    mdl.Params.Cuts = 3
    mdl.Params.CliqueCuts = 2
    mdl.Params.CoverCuts = 2
    mdl.Params.FlowCoverCuts = 2
    mdl.Params.GUBCoverCuts = 2
    mdl.Params.Presolve = 1
    mdl.Params.NodefileStart = NODEFILE_START_GB
    mdl.setParam("MIPGap", MIP_GAP)
    mdl.Params.TimeLimit = time_limit

    contentores_coletados = []
    profit = 0.0
    cost = 0.0
    mdl.optimize()
    if mdl.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and mdl.SolCount > 0:
        id_map = {i: binsids[i] for i in nodes}
        arcos_ativos = [(i, j) for i in nodes for j in nodes if i != j and x[i, j].X > 0.5]

        rotas = []
        visitados = set()
        while True:
            rota = []
            atual = 0
            while True:
                prox = [j for (i, j) in arcos_ativos if i == atual and (i, j) not in visitados]
                if not prox:
                    break
                j = prox[0]
                visitados.add((atual, j))
                rota.append((atual, j))
                atual = j
                if j == 0:
                    break
            if rota:
                rotas.append(rota)
            else:
                break

        for rota in rotas:
            contentores_coletados.extend([id_map[j] for (i, j) in rota])

        profit = float(mdl.ObjVal)
        cost = sum([x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis])

    return [0] + contentores_coletados, profit, cost


def _run_hexaly_optimizer(
    bins: NDArray[np.float64],
    distancematrix: List[List[float]],
    param: float,
    media: NDArray[np.float64],
    desviopadrao: NDArray[np.float64],
    values: Dict[str, float],
    must_go: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    max_iter_no_improv: int = 10,
):
    """
    Solve the Vehicle Routing Problem with Profits using Hexaly Optimizer (Local Search).
    """
    SCALE = 1000
    Omega, delta, psi = values["Omega"], values["delta"], values["psi"]
    Q, R, B, C, V = values["Q"], values["R"], values["B"], values["C"], values["V"]

    Q_int = int(Q * SCALE)
    n_bins = len(bins)

    if number_vehicles == 0:
        number_vehicles = n_bins

    enchimentos = np.insert(bins, 0, 0.0)
    pesos_reais_float = [(e / 100) * B * V for e in enchimentos]
    pesos_reais_int = [int(w * SCALE) for w in pesos_reais_float]

    dist_matrix_int = [[int(dist * SCALE) for dist in row] for row in distancematrix]

    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]

    must_go_set = set(must_go)

    mandatory = set()
    for i in nodes_real:
        if (i in must_go_set) or (enchimentos[i] >= psi * 100):
            mandatory.add(i)

    forbidden = set()
    for i in nodes_real:
        if enchimentos[i] < 10 and i not in must_go_set:
            forbidden.add(i)

    max_dist_int = 6000 * SCALE
    num_nodes = len(nodes)

    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model

        dist_array = model.array(dist_matrix_int)
        weights_array = model.array(pesos_reais_int)

        routes = [model.list(num_nodes) for _ in range(number_vehicles)]

        if number_vehicles > 1:
            for k in range(number_vehicles - 1):
                model.constraint(model.count(routes[k]) >= model.count(routes[k + 1]))

        model.constraint(model.disjoint(routes))
        all_visited = model.union(routes)
        model.constraint(model.contains(all_visited, idx_deposito) == 0)

        for node in forbidden:
            model.constraint(model.contains(all_visited, node) == 0)

        total_profit_int = 0
        total_dist_int = 0
        vehicles_used_expr = 0

        for k in range(number_vehicles):
            route = routes[k]
            count = model.count(route)

            is_used = model.iif(count > 0, 1, 0)
            vehicles_used_expr += is_used

            route_load = model.sum(
                model.range(0, count),
                model.lambda_function(lambda i: model.at(weights_array, model.at(route, i))),
            )
            model.constraint(route_load <= Q_int)
            total_profit_int += route_load

            d_start = model.iif(count > 0, model.at(dist_array, idx_deposito, model.at(route, 0)), 0)
            d_path = model.sum(
                model.range(0, count - 1),
                model.lambda_function(lambda i: model.at(dist_array, model.at(route, i), model.at(route, i + 1))),
            )
            d_end = model.iif(
                count > 0,
                model.at(dist_array, model.at(route, count - 1), idx_deposito),
                0,
            )

            route_dist = d_start + d_path + d_end
            total_dist_int += route_dist

            if max_dist_int > 0:
                model.constraint(route_dist <= max_dist_int)

        for node in mandatory:
            model.constraint(model.contains(all_visited, node))

        must_go_list = list(must_go)
        if must_go_list:
            nb_must_go = len(must_go_list)
            visited_critical = model.sum([model.contains(all_visited, node) for node in must_go_list])
            limit = max(0, nb_must_go - int(len(nodes_real) * delta))
            model.constraint(visited_critical >= limit)

        revenue_term = (total_profit_int / SCALE) * R
        dist_cost_term = (total_dist_int / SCALE) * (0.5 * C)
        vehicle_cost_term = vehicles_used_expr * Omega

        obj = revenue_term - dist_cost_term - vehicle_cost_term

        model.maximize(obj)
        model.close()

        optimizer.param.time_limit = time_limit
        optimizer.param.verbosity = 0

        best_obj_val = [None]
        no_improv_iter = [0]

        def callback(opt, type):
            if opt.solution.status == hx.HxSolutionStatus.INFEASIBLE:
                return
            if type == hx.HxCallbackType.TIME_TICKED:
                val = opt.solution.get_value(obj)
                if best_obj_val[0] is None:
                    best_obj_val[0] = val
                    no_improv_iter[0] = 0
                else:
                    relative_improvement = abs(val - best_obj_val[0]) / max(abs(best_obj_val[0]), 1e-10)
                    if relative_improvement > 0.001:
                        best_obj_val[0] = val
                        no_improv_iter[0] = 0
                    else:
                        no_improv_iter[0] += 1
                    if no_improv_iter[0] >= max_iter_no_improv:
                        opt.stop()

        optimizer.add_callback(hx.HxCallbackType.TIME_TICKED, callback)

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            optimizer.solve()
            sys.stdout = old_stdout

            final_route_flat = [0]
            for k in range(number_vehicles):
                r_vals = list(cast(Any, routes[k]).value)
                if not r_vals:
                    continue
                final_route_flat.extend(r_vals)
                final_route_flat.append(0)

            profit = float(cast(Any, obj).value)
            cost = float(cast(Any, total_dist_int).value) / SCALE
            return final_route_flat, profit, cost
        except Exception:
            sys.stdout = old_stdout
            return [0, 0], 0.0, 0.0


# --- AGNOSTIC POLICY ADAPTER ---
@PolicyRegistry.register("vrpp")
class VRPPPolicy(BaseRoutingPolicy):
    """
    Agnostic VRPP Policy adapter.
    Delegates to run_vrpp_optimizer.
    """

    def _get_config_key(self) -> str:
        """Return config key for VRPP."""
        return "vrpp"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_demands: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float]:
        """Not used - VRPP requires specialized execute()."""
        return [[]], 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """Execute the VRPP policy."""
        must_go = kwargs.get("must_go", [])
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        policy_name = kwargs.get("policy", "gurobi_vrpp")
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        model_env = kwargs.get("model_env")
        waste_type = kwargs["waste_type"]
        area = kwargs["area"]
        n_vehicles = kwargs.get("n_vehicles", 1)
        config = kwargs.get("config", {})

        optimizer = "hexaly" if "hexaly" in policy_name else "gurobi"

        # Use base class to load params
        capacity, revenue, cost_unit, _ = self._load_area_params(area, waste_type, config)
        values = {"Q": capacity, "R": revenue, "C": cost_unit, "B": 0.0, "V": 1.0}

        from logic.src.utils.data.data_utils import load_area_and_waste_type_params

        Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)
        values = {"Q": Q, "R": R, "B": B, "C": C, "V": V}

        # Extracted config handling
        vrpp_cfg = config.get("vrpp", {})
        if optimizer in vrpp_cfg:
            opt_cfg = vrpp_cfg[optimizer]
            if isinstance(opt_cfg, list):
                for item in opt_cfg:
                    if isinstance(item, dict):
                        values.update(item)
            elif isinstance(opt_cfg, dict):
                values.update(opt_cfg)
        else:
            values.update(vrpp_cfg)

        values.setdefault("Omega", 0.1)
        values.setdefault("delta", 0.0)
        values.setdefault("psi", 1.0)
        time_limit = config.get("time_limit", 60)

        # Standardize distance matrix format
        dist_mat = distance_matrix.tolist() if hasattr(distance_matrix, "tolist") else distance_matrix
        binsids = np.arange(0, bins.n + 1).tolist()

        tour, profit, cost = run_vrpp_optimizer(
            bins=bins.c,
            distance_matrix=dist_mat,
            param=kwargs.get("threshold", 1.0),
            media=bins.means,
            desviopadrao=bins.std,
            values=values,
            binsids=binsids,
            must_go=must_go,
            env=model_env,
            number_vehicles=n_vehicles,
            time_limit=time_limit,
            optimizer=optimizer,
        )

        return tour, cost, None

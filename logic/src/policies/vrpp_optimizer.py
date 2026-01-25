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
from typing import Dict, List, Optional

import gurobipy as gp
import hexaly.optimizer as hx
import numpy as np
from gurobipy import GRB, quicksum
from numpy.typing import NDArray


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

    Unified interface for solving the Vehicle Routing Problem with Profits.
    Routes are constructed to maximize profit while respecting capacity and
    enforcing collection of critical (must-go) bins.

    Args:
        bins (NDArray[np.float64]): Current bin fill levels (0-100%)
        distance_matrix (List[List[float]]): Distance matrix (N+1 x N+1) with depot at 0
        param (float): Std deviation multiplier for must-go prediction (unused here,
            must_go list is pre-computed)
        media (NDArray[np.float64]): Mean accumulation rates (unused here)
        desviopadrao (NDArray[np.float64]): Std deviation of rates (unused here)
        values (Dict[str, float]): Problem parameters:
            - Q: Vehicle capacity
            - R: Revenue per kg of waste
            - B: Bin density (kg/m³)
            - C: Travel cost per distance unit
            - V: Bin volume (m³)
            - Omega: Penalty per vehicle used
            - delta: Tolerance for must-go violations
            - psi: Minimum fill threshold
        binsids (List[int]): List of bin IDs (0-indexed, includes depot at 0)
        must_go (List[int]): Bin IDs that must be collected (0-indexed)
        env (Optional[gp.Env]): Gurobi environment (Gurobi only)
        number_vehicles (int): Number of vehicles. If 0, automatic fleet sizing. Default: 1
        time_limit (int): Solver time limit in seconds. Default: 60
        optimizer (str): Solver backend: 'gurobi' or 'hexaly'. Default: 'gurobi'
        max_iter_no_improv (int): Early stopping iterations (Hexaly only). Default: 10

    Returns:
        Tuple[List[int], float, float]: Routes, profit, and cost
            - routes: Flattened tour [0, bin1, bin2, ..., 0]
            - profit: Total profit (revenue - cost - vehicle penalty)
            - cost: Total travel cost

    Raises:
        ValueError: If optimizer is not 'gurobi' or 'hexaly'
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

    Implements a 2-index flow formulation with MTZ subtour elimination
    equivalent and capacity constraints.

    Args:
        bins (NDArray[np.float64]): Bin fill levels.
        distance_matrix (List[List[float]]): Distance matrix.
        env (Optional[gp.Env]): Gurobi environment.
        param (float): Prediction parameter.
        media (NDArray[np.float64]): Mean rates.
        desviopadrao (NDArray[np.float64]): Std dev of rates.
        values (Dict[str, float]): Problem parameters.
        binsids (List[int]): Bin identifier mapping.
        must_go (List[int]): Must-collect bin indicators.
        number_vehicles (int): Max vehicles to use.
        time_limit (int): Solver time limit.

    Returns:
        Tuple[List[int], float, float]: (Route, profit, cost).
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

    # MUST GO passed as argument
    # must_go = []
    # binsids passed as argument
    criticos = [bin_id in must_go for bin_id in binsids]
    criticos_dict = {i: criticos[i] for i in nodes}

    max_dist = 6000
    pares_viaveis = [(i, j) for i in nodes for j in nodes if i != j and distance_matrix[i][j] <= max_dist]
    mdl = gp.Model("VRPP", env=env) if env else gp.Model("VRPP")

    x = mdl.addVars(
        pares_viaveis, vtype=GRB.BINARY, name="x"
    )  # diz se a gente usa ou não a estrada que vai do ponto i até o ponto j
    y = mdl.addVars(
        pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="y"
    )  # quanto de resíduo (kg) a gente está carregando nesse trecho entre i e j
    f = mdl.addVars(
        pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="f"
    )  # pra evitar que o modelo crie "ciclos pequenos" fora do caminho principal (subtours).
    g = mdl.addVars(nodes, vtype=GRB.BINARY, name="g")
    k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k_var")
    for i, j in pares_viaveis:
        mdl.addConstr(y[i, j] <= Q * x[i, j])  # limita que o trecho não tenha a capaciade maxima do caminhão
        mdl.addConstr(f[i, j] <= len(nodes) * x[i, j])  # evita subtours

    # Garante que o fluxo líquido em cada nó é igual ao resíduo gerado somente se o contentor for coletado
    for i in nodes_real:
        mdl.addConstr(quicksum(y[i, j] - y[j, i] for j in nodes if (i, j) in y or (j, i) in y) == S_dict[i] * g[i])

    # Teste de fixar o valor da quantidade de caminhões
    if number_vehicles == 0:
        number_vehicles = len(binsids)

    MAX_TRUCKS = number_vehicles
    mdl.addConstr(k_var <= MAX_TRUCKS)

    # Relaciona k_var com o número de rotas que partem e voltam do depósito.
    mdl.addConstr(k_var == quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x))
    mdl.addConstr(quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x) == k_var)
    mdl.addConstr(quicksum(x[j, idx_deposito] for j in nodes_real if (j, idx_deposito) in x) == k_var)

    # Se um contentor não for coletado (g[j]==0), não pode haver rota conectando-o ao depósito.
    for j in nodes_real:
        if (idx_deposito, j) in x:
            mdl.addConstr(x[idx_deposito, j] <= g[j])
        if (j, idx_deposito) in x:
            mdl.addConstr(x[j, idx_deposito] <= g[j])

    # Garante que pelo menos um número mínimo de contentores críticos seja visitado (ajustável por delta)
    mdl.addConstr(
        quicksum(g[i] for i in nodes_real if criticos_dict[i])
        >= len([i for i in nodes_real if criticos_dict[i]]) - len(nodes_real) * delta
    )

    for i in nodes_real:
        # Esses contentores devem ser coletados.
        if criticos_dict[i] or enchimentos[i] >= psi * 100:
            mdl.addConstr(g[i] == 1)
        # g[i] é forçado a ser 0 (não coleta).
        if enchimentos[i] < 10 and not criticos[i]:
            g[i].UB = 0

    # Se um contentor for visitado (g[j]==1), deve ter exatamente uma entrada e uma saída.
    for j in nodes_real:
        mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == g[j])
        mdl.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in x) == g[j])

    # Assegura conectividade entre os pontos e impede a criação de ciclos menores isolados (subtours).
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
    mdl.Params.Heuristics = 0.5
    mdl.Params.Threads = 0
    mdl.Params.Cuts = 3
    mdl.Params.CliqueCuts = 2  # força clique cuts
    mdl.Params.CoverCuts = 2  # força cuts de conjuntos
    mdl.Params.FlowCoverCuts = 2  # força cortes para fluxos
    mdl.Params.GUBCoverCuts = 2
    mdl.Params.Presolve = 1
    mdl.Params.NodefileStart = 0.5
    mdl.setParam("MIPGap", 0.01)
    mdl.Params.TimeLimit = time_limit

    contentores_coletados = []
    profit = 0.0
    cost = 0.0
    mdl.optimize()
    if mdl.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # resultados_y = []
        # resultados_g = []
        id_map = {i: binsids[i] for i in nodes}
        arcos_ativos = [(i, j) for i in nodes for j in nodes if i != j and x[i, j].X > 0.5]
        # final_gap = mdl.MIPGap

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

        # for i in nodes:
        #     for j in nodes:
        #         if i != j and y[i, j].X > 0:
        #             resultados_y.append((id_map[i], id_map[j], y[i, j].X))

        # Variáveis g[i]
        # for i in nodes:
        #     if g[i].X > 0.5:
        #         resultados_g.append((id_map[i], g[i].X))

        for idx, rota in enumerate(rotas, start=1):
            # df_rota = pd.DataFrame(
            #     [(id_map[i], id_map[j], x[i, j].X) for (i, j) in rota],
            #     columns=['i', 'j', 'x_ij']
            # )

            contentores_coletados.extend([id_map[j] for (i, j) in rota])

        profit = R * sum(S_dict[i] * g[i].X for i in nodes_real) - sum(
            x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis
        )

        cost = sum([x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis])
        contentores_coletados = [contentor for contentor in contentores_coletados]
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

    Args:
        bins (NDArray[np.float64]): Bin fill levels.
        distancematrix (List[List[float]]): Distance matrix.
        param (float): Prediction parameter.
        media (NDArray[np.float64]): Mean rates.
        desviopadrao (NDArray[np.float64]): Std dev of rates.
        values (Dict[str, float]): Problem parameters.
        must_go (List[int]): Must-collect bin indicators.
        number_vehicles (int): Max vehicles to use.
        time_limit (int): Solver time limit.
        max_iter_no_improv (int): Stop if no improvement for N ticks.

    Returns:
        Tuple[List[int], float, float]: (Route, profit, cost).
    """
    # ---------------------------------------------------------
    # 0. PARAMETERS & SCALING
    # ---------------------------------------------------------
    SCALE = 1000
    Omega, delta, psi = values["Omega"], values["delta"], values["psi"]
    Q, R, B, C, V = values["Q"], values["R"], values["B"], values["C"], values["V"]

    Q_int = int(Q * SCALE)
    n_bins = len(bins)

    # --- AUTO FLEET SIZING ---
    # Similar to Gurobi: If 0, allow worst-case max (1 vehicle per bin).
    # The solver will minimize usage via the Omega penalty.
    if number_vehicles == 0:
        number_vehicles = n_bins
    # ---------------------------------------------

    # ---------------------------------------------------------
    # 1. DATA PREPARATION
    # ---------------------------------------------------------
    enchimentos = np.insert(bins, 0, 0.0)
    pesos_reais_float = [(e / 100) * B * V for e in enchimentos]
    pesos_reais_int = [int(w * SCALE) for w in pesos_reais_float]

    dist_matrix_int = [[int(dist * SCALE) for dist in row] for row in distancematrix]

    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]

    # Must-Go & Mandatory Logic
    # must_go passed as argument
    # (previously calculated based on pred_value)
    must_go_set = set(must_go)

    mandatory = set()
    for i in nodes_real:
        if (i in must_go_set) or (enchimentos[i] >= psi * 100):
            mandatory.add(i)

    # Forbidden Logic
    forbidden = set()
    for i in nodes_real:
        if enchimentos[i] < 10 and i not in must_go_set:
            forbidden.add(i)

    max_dist_int = 6000 * SCALE
    num_nodes = len(nodes)

    # ---------------------------------------------------------
    # 2. HEXALY MODEL
    # ---------------------------------------------------------
    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model

        dist_array = model.array(dist_matrix_int)
        weights_array = model.array(pesos_reais_int)

        # Create available routes based on the updated number_vehicles
        routes = [model.list(num_nodes) for _ in range(number_vehicles)]

        # Symmetry Breaking: Force sequential usage of vehicles
        # This is crucial when number_vehicles is large (e.g. == n_bins)
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

            # 1. Vehicle Used? (Penalty applied later)
            is_used = model.iif(count > 0, 1, 0)
            vehicles_used_expr += is_used

            # 2. Load Calculation
            route_load = model.sum(
                model.range(0, count),
                model.lambda_function(lambda i: model.at(weights_array, model.at(route, i))),
            )
            model.constraint(route_load <= Q_int)
            total_profit_int += route_load

            # 3. Distance Calculation
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

        # ---------------------------------------------------------
        # 3. OBJECTIVE FUNCTION
        # ---------------------------------------------------------
        revenue_term = (total_profit_int / SCALE) * R
        dist_cost_term = (total_dist_int / SCALE) * (0.5 * C)

        # The penalty Omega will naturally minimize the number of vehicles used
        vehicle_cost_term = vehicles_used_expr * Omega

        obj = revenue_term - dist_cost_term - vehicle_cost_term

        model.maximize(obj)
        model.close()

        # ---------------------------------------------------------
        # 4. SOLVE WITH CALLBACK
        # ---------------------------------------------------------
        optimizer.param.time_limit = time_limit
        optimizer.param.verbosity = 0

        # --- EARLY STOPPING CALLBACK ---
        best_obj_val = [None]
        no_improv_iter = [0]

        def callback(opt, type):
            """
            Hexaly callback for early stopping based on convergence.
            """
            # Check feasibility first
            if opt.solution.status == hx.HxSolutionStatus.INFEASIBLE:
                return

            if type == hx.HxCallbackType.TIME_TICKED:
                # Get current objective value
                val = opt.solution.get_value(obj)

                if best_obj_val[0] is None:
                    best_obj_val[0] = val
                    no_improv_iter[0] = 0
                else:
                    # Check for significant relative improvement (e.g. 0.1%)
                    # Use max(..., 1e-10) to avoid division by zero
                    relative_improvement = abs(val - best_obj_val[0]) / max(abs(best_obj_val[0]), 1e-10)

                    if relative_improvement > 0.001:
                        best_obj_val[0] = val
                        no_improv_iter[0] = 0
                    else:
                        no_improv_iter[0] += 1

                    if no_improv_iter[0] >= max_iter_no_improv:
                        # Stop the optimizer if no improvement for 'max_iter_no_improv' ticks
                        opt.stop()

        # Register the callback
        optimizer.add_callback(hx.HxCallbackType.TIME_TICKED, callback)
        # -------------------------------

        # Output capture
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            optimizer.solve()
            sys.stdout = old_stdout

            final_route_flat = [0]
            calc_revenue = 0.0
            calc_dist = 0.0
            for k in range(number_vehicles):
                r_vals = list(routes[k].value)
                if not r_vals:
                    continue

                final_route_flat.extend(r_vals)
                final_route_flat.append(0)

                for node in r_vals:
                    calc_revenue += pesos_reais_float[node] * R

                if len(r_vals) > 0:
                    calc_dist += distancematrix[idx_deposito][r_vals[0]]
                    for i in range(len(r_vals) - 1):
                        calc_dist += distancematrix[r_vals[i]][r_vals[i + 1]]
                    calc_dist += distancematrix[r_vals[-1]][idx_deposito]

            profit = calc_revenue - calc_dist
            cost = calc_dist
            return final_route_flat, profit, cost
        except Exception:
            sys.stdout = old_stdout
            return [0], 0.0, 0.0

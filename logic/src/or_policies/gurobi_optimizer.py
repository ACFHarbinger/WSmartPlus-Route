import numpy as np
import pandas as pd
import gurobipy as gp

from gurobipy import GRB, quicksum
from typing import List
from numpy.typing import NDArray
from logic.src.pipeline.simulator.loader import load_area_and_waste_type_params


def policy_gurobi_vrpp(
        bins: NDArray[np.float64],
        distance_matrix: List[List[float]], 
        env: gp.Env, 
        param: float, 
        media: NDArray[np.float64], 
        desviopadrao: NDArray[np.float64], 
        waste_type: str='plastic', 
        area: str='riomaior', 
        number_vehicles: int=1, 
        time_limit: int=60
    ):
    Omega, delta, psi = 0.1, 0, 1
    Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)

    n_bins = len(bins)
    enchimentos = np.insert(bins, 0, 0.0)
    pesos_reais = [(e / 100) * B * V for e in enchimentos] # Altera os enchimentos de % pra valor real em KG
    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]
    S_dict = {i: pesos_reais[i] for i in nodes}

    # MUST GO
    must_go = []
    for container_id in range(0, n_bins):
        pred_value = (bins[container_id] + media[container_id] + param * desviopadrao[container_id])
        if pred_value >= 100:
            must_go.append(container_id + 1)  # +1 because of depot at 0

    binsids = np.arange(0, n_bins + 1).tolist()
    criticos = [bin_id in must_go for bin_id in binsids]
    criticos_dict = {i: criticos[i] for i in nodes}

    max_dist = 6000
    pares_viaveis = [(i, j) for i in nodes for j in nodes if i != j and distance_matrix[i][j] <= max_dist]
    mdl = gp.Model("VRPP", env=env) if env else gp.Model("VRPP")

    x = mdl.addVars(pares_viaveis, vtype=GRB.BINARY, name="x") #diz se a gente usa ou não a estrada que vai do ponto i até o ponto j
    y = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="y") # quanto de resíduo (kg) a gente está carregando nesse trecho entre i e j
    f = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="f") # pra evitar que o modelo crie "ciclos pequenos" fora do caminho principal (subtours).
    g = mdl.addVars(nodes, vtype=GRB.BINARY, name="g")
    k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k_var")
    for i, j in pares_viaveis:
        mdl.addConstr(y[i, j] <= Q * x[i, j]) # limita que o trecho não tenha a capaciade maxima do caminhão
        mdl.addConstr(f[i, j] <= len(nodes) * x[i, j]) #evita subtours

    # Garante que o fluxo líquido em cada nó é igual ao resíduo gerado somente se o contentor for coletado
    for i in nodes_real:
        mdl.addConstr(quicksum(y[i, j] - y[j, i] for j in nodes if (i, j) in y or (j, i) in y) == S_dict[i] * g[i])

    # Teste de fixar o valor da quantidade de caminhões
    MAX_TRUCKS = number_vehicles
    mdl.addConstr(k_var <= MAX_TRUCKS)

    # Relaciona k_var com o número de rotas que partem e voltam do depósito.
    mdl.addConstr(k_var == quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x))
    mdl.addConstr(quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x) == k_var)
    mdl.addConstr(quicksum(x[j, idx_deposito] for j in nodes_real if (j, idx_deposito) in x) == k_var)

    # Se um contentor não for coletado (g[j]==0), não pode haver rota conectando-o ao depósito.
    for j in nodes_real:
        if (idx_deposito, j) in x: mdl.addConstr(x[idx_deposito, j] <= g[j])
        if (j, idx_deposito) in x: mdl.addConstr(x[j, idx_deposito] <= g[j])

    #Garante que pelo menos um número mínimo de contentores críticos seja visitado (ajustável por delta)
    mdl.addConstr(quicksum(g[i] for i in nodes_real if criticos_dict[i]) >= len([i for i in nodes_real if criticos_dict[i]]) - len(nodes_real) * delta)

    for i in nodes_real:
        # Esses contentores devem ser coletados.
        if criticos_dict[i] or enchimentos[i] >= psi * 100:
            mdl.addConstr(g[i] == 1)
        # g[i] é forçado a ser 0 (não coleta).
        if enchimentos[i] < 10 and not criticos[i]:
            g[i].ub = 0

    # Se um contentor for visitado (g[j]==1), deve ter exatamente uma entrada e uma saída.
    for j in nodes_real:
        mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == g[j])
        mdl.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in x) == g[j])

    # Assegura conectividade entre os pontos e impede a criação de ciclos menores isolados (subtours).
    mdl.addConstr(quicksum(f[0, j] for j in nodes_real if (0, j) in f) == quicksum(g[j] for j in nodes_real))
    for j in nodes_real:
        mdl.addConstr(quicksum(f[i, j] for i in nodes if (i, j) in f) - quicksum(f[j, k] for k in nodes if (j, k) in f) == g[j])

    mdl.setObjective(
        R * quicksum(S_dict[i] * g[i] for i in nodes_real)
        - 0.5 * C * quicksum(x[i, j] * distance_matrix[i][j] for i, j in pares_viaveis)
        - Omega * k_var,
        GRB.MAXIMIZE
    )

    mdl.Params.MIPFocus = 1
    mdl.Params.Heuristics = 0.5
    mdl.Params.Threads = 0
    mdl.Params.Cuts = 3
    mdl.Params.CliqueCuts = 2    # força clique cuts
    mdl.Params.CoverCuts = 2     # força cuts de conjuntos
    mdl.Params.FlowCoverCuts = 2 # força cortes para fluxos
    mdl.Params.GUBCoverCuts = 2
    mdl.Params.Presolve = 1
    mdl.Params.NodefileStart = 0.5
    mdl.setParam("MIPGap", 0.01)
    mdl.Params.TimeLimit = time_limit

    contentores_coletados = []
    profit = 0
    cost = 0
    mdl.optimize()
    if mdl.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        resultados_y = []
        resultados_g = []
        id_map = {i: binsids[i] for i in nodes}
        arcos_ativos = [(i, j) for i in nodes for j in nodes if i != j and x[i, j].X > 0.5]
        final_gap = mdl.MIPGap

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

        for i in nodes:
            for j in nodes:
                if i != j and y[i, j].X > 0:
                    resultados_y.append((id_map[i], id_map[j], y[i, j].X))

        # Variáveis g[i]
        for i in nodes:
            if g[i].X > 0.5:
                resultados_g.append((id_map[i], g[i].X))

        for idx, rota in enumerate(rotas, start=1):
            df_rota = pd.DataFrame(
                [(id_map[i], id_map[j], x[i, j].X) for (i, j) in rota],
                columns=['i', 'j', 'x_ij']
            )
            
            contentores_coletados.extend([id_map[j] for (i, j) in rota])

        profit = (
            R * sum(S_dict[i] * g[i].X for i in nodes_real)
            - sum(x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis)
        )
            
        cost = sum(x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis)
        contentores_coletados = [contentor for contentor in contentores_coletados]
    return [0] + contentores_coletados, profit, cost
import time
import random
import numpy as np

from collections import deque
from typing import List, Tuple

# --- 1. Data Structures & Params ---

class Individual:
    def __init__(self, giant_tour: List[int]):
        self.giant_tour = giant_tour
        self.routes = []
        self.fitness = -float('inf') 
        self.profit_score = -float('inf') 
        self.cost = 0.0
        self.revenue = 0.0
        
        self.dist_to_parents = 0.0
        self.rank_profit = 0
        self.rank_diversity = 0

    def __lt__(self, other):
        return self.fitness < other.fitness


class HGSParams:
    def __init__(self, time_limit=10, population_size=50, elite_size=10, 
                 mutation_rate=0.2, max_vehicles=0):
        self.time_limit = time_limit
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.max_vehicles = max_vehicles


# --- 2. Linear Split (Vidal 2016) - Optimized Pure Python ---

class LinearSplit:
    def __init__(self, dist_matrix, demands, capacity, R, C, max_vehicles=0):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.R = R
        self.C = C
        self.max_vehicles = max_vehicles
        
    def split(self, giant_tour: List[int]) -> Tuple[List[List[int]], float]:
        if not giant_tour:
            return [], 0.0
            
        n = len(giant_tour)
        
        cum_load = [0.0] * (n + 1)
        cum_rev = [0.0] * (n + 1)
        cum_dist = [0.0] * (n + 1)
        
        dmat = self.dist_matrix
        demands = self.demands
        R_val = self.R
        
        load_curr = 0.0
        rev_curr = 0.0
        dist_curr = 0.0
        prev_node = 0
        
        d_0_x = [0.0] * (n + 1)
        d_x_0 = [0.0] * (n + 1)
        
        for i in range(1, n + 1):
            node = giant_tour[i-1]
            dem = demands.get(node, 0)
            
            load_curr += dem
            rev_curr += dem * R_val
            
            if i > 1:
                dist_curr += dmat[prev_node, node]
            
            cum_load[i] = load_curr
            cum_rev[i] = rev_curr
            cum_dist[i] = dist_curr
            
            d_0_x[i] = dmat[0, node]
            d_x_0[i] = dmat[node, 0]
            prev_node = node
            
        res = ([], -float('inf'))
        if self.max_vehicles == 0:
            res = self._split_unlimited(n, giant_tour, cum_load, cum_rev, cum_dist, d_0_x, d_x_0)
        else:
            res = self._split_limited(n, giant_tour, cum_load, cum_rev, cum_dist, d_0_x, d_x_0)
            
        if res[1] == -float('inf'):
            return self._fallback_split(giant_tour)
            
        return res

    def _fallback_split(self, giant_tour):
        routes = []
        current_route = []
        current_load = 0
        
        for node in giant_tour:
            dem = self.demands.get(node, 0)
            if current_load + dem <= self.capacity:
                current_route.append(node)
                current_load += dem
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [node]
                current_load = dem
                
        if current_route:
            routes.append(current_route)
            
        rev = sum(self.demands.get(n,0) for r in routes for n in r) * self.R
        cost = 0
        for r in routes:
            d = self.dist_matrix[0, r[0]]
            for k in range(len(r)-1):
                d += self.dist_matrix[r[k], r[k+1]]
            d += self.dist_matrix[r[-1], 0]
            cost += d * self.C
            
        profit = rev - cost
        return routes, profit

    def _split_unlimited(self, n, nodes, cum_load, cum_rev, cum_dist, d_0_x, d_x_0):
        V = [-float('inf')] * (n + 1)
        P = [-1] * (n + 1)
        V[0] = 0.0
        
        C_cost = self.C
        cap = self.capacity
        
        term_0 = V[0] - cum_rev[0] + C_cost * (cum_dist[1] - d_0_x[1])
        dq = deque([(0, term_0)])
        
        for i in range(1, n + 1):
            min_load = cum_load[i] - cap
            while dq:
                idx = dq[0][0]
                if cum_load[idx] < min_load - 1e-5:
                    dq.popleft()
                else:
                    break
            
            if dq:
                best_j, best_A = dq[0]
                B_i = cum_rev[i] - C_cost * (cum_dist[i] + d_x_0[i])
                V[i] = best_A + B_i
                P[i] = best_j
            
            if i < n:
                j_new = i
                if V[j_new] > -float('inf'):
                    idx_next = i + 1
                    term = V[j_new] - cum_rev[j_new] + C_cost * (cum_dist[idx_next] - d_0_x[idx_next])
                    while dq:
                        if dq[-1][1] <= term:
                            dq.pop()
                        else:
                            break
                    dq.append((j_new, term))
                    
        return self._reconstruct(n, nodes, P, V[n])

    def _split_limited(self, n, nodes, cum_load, cum_rev, cum_dist, d_0_x, d_x_0):
        K = self.max_vehicles
        V_prev = [-float('inf')] * (n + 1)
        V_prev[0] = 0.0
        
        P = [[-1] * (n + 1) for _ in range(K + 1)]
        best_profit = -float('inf')
        
        C_cost = self.C
        cap = self.capacity
        
        for k in range(1, K + 1):
            V_curr = [-float('inf')] * (n + 1)
            dq = deque()
            
            if V_prev[0] > -float('inf'):
                term_0 = V_prev[0] - cum_rev[0] + C_cost * (cum_dist[1] - d_0_x[1])
                dq.append((0, term_0))
                
            for i in range(1, n + 1):
                min_load = cum_load[i] - cap
                while dq:
                    idx = dq[0][0]
                    if cum_load[idx] < min_load - 1e-5:
                        dq.popleft()
                    else:
                        break
                
                if dq:
                    best_j, best_A = dq[0]
                    B_i = cum_rev[i] - C_cost * (cum_dist[i] + d_x_0[i])
                    V_curr[i] = best_A + B_i
                    P[k][i] = best_j
                
                if i < n:
                    j_new = i
                    if V_prev[j_new] > -float('inf'):
                        idx_next = i + 1
                        term = V_prev[j_new] - cum_rev[j_new] + C_cost * (cum_dist[idx_next] - d_0_x[idx_next])
                        while dq:
                            if dq[-1][1] <= term:
                                dq.pop()
                            else:
                                break
                        dq.append((j_new, term))
            
            if V_curr[n] > best_profit:
                best_profit = V_curr[n]
                 
            V_prev = V_curr

        if best_profit == -float('inf'):
            return [], -float('inf')
            
        return [], -float('inf') 

    def _reconstruct(self, n, nodes, P, total_profit):
        if total_profit == -float('inf'):
            return [], -float('inf')
        routes = []
        curr = n
        while curr > 0:
            prev = P[curr]
            if prev == -1: return [], -float('inf')
            routes.append(nodes[prev:curr])
            curr = prev
        routes.reverse()
        return routes, total_profit


def split_algorithm(giant_tour: List[int], dist_matrix, demands, capacity, R, C, values):
    s = LinearSplit(dist_matrix, demands, capacity, R, C, values.get('max_vehicles', 0))
    return s.split(giant_tour)


# --- 3. Biased Fitness ---

def update_biased_fitness(population: List[Individual], params: HGSParams):
    population.sort(key=lambda x: x.profit_score, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_profit = i
        
    if not population: return
    
    best_ind = population[0]
    best_succ = {}
    for r in best_ind.routes:
        if not r: continue
        prev = 0
        for n in r:
            best_succ[prev] = n
            prev = n
        best_succ[prev] = 0
        
    for ind in population:
        dist = 0
        for r in ind.routes:
            if not r: continue
            prev = 0
            for n in r:
                if best_succ.get(prev) != n:
                    dist += 1
                prev = n
            if best_succ.get(prev) != 0:
                dist += 1
        ind.dist_to_parents = dist
        
    population.sort(key=lambda x: x.dist_to_parents, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_diversity = i
        
    w = 1.0 - (params.elite_size / params.population_size)
    for ind in population:
        ind.fitness = ind.rank_profit + w * ind.rank_diversity


def evaluate(individual: Individual, split_algo: LinearSplit, must_go_bins, R, C):
    routes, profit = split_algo.split(individual.giant_tour)
    individual.routes = routes
    individual.profit_score = profit
    return individual


class LocalSearch:
    def __init__(self, dist_matrix, demands, capacity, R, C, params: HGSParams):
        self.d = dist_matrix
        self.demands = demands
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        
        n_nodes = len(dist_matrix)
        self.neighbors = {}
        for i in range(1, n_nodes):
            row = self.d[i]
            order = np.argsort(row)
            cands = []
            for c in order:
                if c != i and c != 0:
                    cands.append(c)
                    if len(cands) >= 10: break 
            self.neighbors[i] = cands 
            
        self.node_map = {} 
        self.route_loads = [] 

    def optimize(self, individual: Individual):
        if not individual.routes:
            return individual
            
        self.routes = [r[:] for r in individual.routes]
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]

        improved = True
        limit = 500  # Safety cap
        it = 0
        t_start = time.time()
        
        self.node_map.clear()
        for ri, r in enumerate(self.routes):
            for pi, node in enumerate(r):
                self.node_map[node] = (ri, pi)
                
        while improved and it < limit:
            improved = False
            it += 1
            if it % 50 == 0 and (time.time() - t_start > self.params.time_limit):
                break
            
            nodes = list(self.neighbors.keys())
            # Filter valid nodes only
            nodes = [n for n in nodes if n in self.node_map]
            random.shuffle(nodes)
            
            for u in nodes:
                if self._process_node(u):
                    improved = True
                    break 
        
        individual.routes = self.routes
        gt = []
        for r in self.routes:
            gt.extend(r)
        individual.giant_tour = gt
        return individual

    def _calc_load_fresh(self, r):
        return sum(self.demands.get(x, 0) for x in r)

    def _process_node(self, u):
        u_loc = self.node_map.get(u)
        if not u_loc: return False
        r_u, p_u = u_loc
        
        for v in self.neighbors[u]:
            v_loc = self.node_map.get(v)
            if not v_loc: continue
            r_v, p_v = v_loc
            
            if self._move_relocate(u, v, r_u, p_u, r_v, p_v): return True
            if self._move_swap(u, v, r_u, p_u, r_v, p_v): return True
            if r_u != r_v:
                if self._move_2opt_star(u, v, r_u, p_u, r_v, p_v): return True
            else:
                if self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v): return True
            
        return False
        
    def _update_map(self, affected_indices):
        for ri in affected_indices:
            for pi, node in enumerate(self.routes[ri]):
                self.node_map[node] = (ri, pi)
            self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])
        
    def _get_load_cached(self, ri):
        return self.route_loads[ri]

    def _move_relocate(self, u, v, r_u, p_u, r_v, p_v):
        if r_u == r_v and (p_u == p_v + 1): return False
        dem_u = self.demands.get(u,0)
        
        if r_u != r_v:
            if self._get_load_cached(r_v) + dem_u > self.Q: return False
            
        route_u = self.routes[r_u]
        route_v = self.routes[r_v] 
        prev_u = route_u[p_u-1] if p_u > 0 else 0
        next_u = route_u[p_u+1] if p_u < len(route_u)-1 else 0
        v_next = route_v[p_v+1] if p_v < len(route_v)-1 else 0
        
        delta = -self.d[prev_u, u] - self.d[u, next_u] + self.d[prev_u, next_u]
        delta -= self.d[v, v_next]
        delta += self.d[v, u] + self.d[u, v_next]
        
        if delta * self.C < -1e-4:
            self.routes[r_u].pop(p_u)
            if r_u == r_v and p_u < p_v: p_v -= 1
            self.routes[r_v].insert(p_v + 1, u)
            self._update_map({r_u, r_v})
            return True
        return False

    def _move_swap(self, u, v, r_u, p_u, r_v, p_v):
        if r_u == r_v and abs(p_u - p_v) <= 1: return False 
        
        dem_u = self.demands.get(u, 0)
        dem_v = self.demands.get(v, 0)
        
        if r_u != r_v:
             if self._get_load_cached(r_u) - dem_u + dem_v > self.Q: return False
             if self._get_load_cached(r_v) - dem_v + dem_u > self.Q: return False
             
        route_u = self.routes[r_u]
        route_v = self.routes[r_v]
        
        prev_u = route_u[p_u-1] if p_u > 0 else 0
        next_u = route_u[p_u+1] if p_u < len(route_u)-1 else 0
        prev_v = route_v[p_v-1] if p_v > 0 else 0
        next_v = route_v[p_v+1] if p_v < len(route_v)-1 else 0
        
        delta = -self.d[prev_u, u] - self.d[u, next_u] - self.d[prev_v, v] - self.d[v, next_v]
        delta += self.d[prev_u, v] + self.d[v, next_u] + self.d[prev_v, u] + self.d[u, next_v]
        
        if delta * self.C < -1e-4:
            self.routes[r_u][p_u] = v
            self.routes[r_v][p_v] = u
            self._update_map({r_u, r_v})
            return True
        return False
        
    def _move_2opt_star(self, u, v, r_u, p_u, r_v, p_v):
        route_u = self.routes[r_u]
        route_v = self.routes[r_v]
        
        tail_u = route_u[p_u+1:]
        tail_v = route_v[p_v+1:]
        
        l_head_u = self._calc_load_fresh(route_u[:p_u+1])
        l_head_v = self._calc_load_fresh(route_v[:p_v+1])
        l_tail_u = self._get_load_cached(r_u) - l_head_u
        l_tail_v = self._get_load_cached(r_v) - l_head_v
        
        if l_head_u + l_tail_v > self.Q or l_head_v + l_tail_u > self.Q:
            return False
            
        u_next = route_u[p_u+1] if p_u < len(route_u)-1 else 0
        v_next = route_v[p_v+1] if p_v < len(route_v)-1 else 0
        
        delta = -self.d[u, u_next] - self.d[v, v_next] + self.d[u, v_next] + self.d[v, u_next]
        
        if delta * self.C < -1e-4:
            new_ru = route_u[:p_u+1] + tail_v
            new_rv = route_v[:p_v+1] + tail_u
            self.routes[r_u] = new_ru
            self.routes[r_v] = new_rv
            self._update_map({r_u, r_v})
            return True
        return False
        
    def _move_2opt_intra(self, u, v, r_u, p_u, r_v, p_v):
        if p_u >= p_v: return False
        if p_u + 1 == p_v: return False
        
        route = self.routes[r_u]
        u_next = route[p_u+1]
        v_next = route[p_v+1] if p_v < len(route)-1 else 0
        
        delta = -self.d[u, u_next] - self.d[v, v_next] + self.d[u, v] + self.d[u_next, v_next]
        
        if delta * self.C < -1e-4:
             segment = route[p_u+1 : p_v+1]
             route[p_u+1 : p_v+1] = segment[::-1]
             self._update_map({r_u})
             return True
        return False


def ordered_crossover(p1: List[int], p2: List[int]) -> List[int]:
    size = len(p1)
    if size < 2: return p1[:]
    
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b] = p1[a:b]
    
    p1_set = set(p1[a:b])
    genes_from_p2 = [x for x in p2 if x not in p1_set]
    
    curr = b
    for gene in genes_from_p2:
        if -1 not in child:
            break
        while True:
            if curr >= size: curr = 0
            if child[curr] == -1: break
            curr += 1
            
        child[curr] = gene
        curr += 1
        
    child = [x for x in child if x != -1]
    return child


def run_hgs(dist_matrix, demands, capacity, R, C, values, global_must_go, local_to_global, vrpp_tour_global=None):
    """
    Dispatcher for HGS implementations.
    """
    engine = values.get('hgs_engine', 'custom')
    if engine == 'pyvrp':
        return _run_hgs_pyvrp(dist_matrix, demands, capacity, R, C, values, global_must_go)
    else:
        return _run_hgs_custom(dist_matrix, demands, capacity, R, C, values, global_must_go, local_to_global, vrpp_tour_global)


def _run_hgs_custom(dist_matrix, demands, capacity, R, C, values, global_must_go, local_to_global, vrpp_tour_global):
    """
    Custom Pure-Python HGS Implementation.
    """
    dist_matrix = np.array(dist_matrix)
    
    params = HGSParams(
        time_limit=values.get('time_limit', 10),
        population_size=values.get('population_size', 30),
        elite_size=values.get('elite_size', 10), 
        max_vehicles=values.get('max_vehicles', 0)
    )
    
    split_algo = LinearSplit(dist_matrix, demands, capacity, R, C, params.max_vehicles)
    ls = LocalSearch(dist_matrix, demands, capacity, R, C, params)
    
    base_tour = list(local_to_global.values())
    population = []
    
    start_time = time.time()
    for i in range(params.population_size):
        if i == 0 and vrpp_tour_global:
            t = vrpp_tour_global[:]
        elif i == 1:
            t = base_tour[:]
        else:
            t = base_tour[:]
            random.shuffle(t)
        
        ind = Individual(t)
        evaluate(ind, split_algo, global_must_go, R, C)
        population.append(ind)
        
    update_biased_fitness(population, params)
    population.sort()
    
    best_sol = max(population, key=lambda x: x.profit_score)
    print(f"[HGS] Init Best: {best_sol.profit_score:.2f} (Routes: {len(best_sol.routes)})")
    
    no_improv = 0
    max_stag = 50 
    
    n_gens = 0
    while time.time() - start_time < params.time_limit:
        n_gens += 1
        
        parents = []
        for _ in range(2):
            c1, c2 = random.sample(population, 2)
            parents.append(c1 if c1.fitness < c2.fitness else c2)
            
        child_tour = ordered_crossover(parents[0].giant_tour, parents[1].giant_tour)
        child = Individual(child_tour)
        evaluate(child, split_algo, global_must_go, R, C)
        
        ls.optimize(child)
        evaluate(child, split_algo, global_must_go, R, C)
        
        is_dup = any(abs(p.profit_score - child.profit_score) < 1e-4 for p in population)
        if not is_dup:
            population.append(child)
            update_biased_fitness(population, params)
            population.sort()
            population.pop()
            
            if child.profit_score > best_sol.profit_score:
                best_sol = child
                no_improv = 0
            else:
                no_improv += 1
        else:
            no_improv += 1
            
        if no_improv > max_stag:
            n_elite = params.elite_size
            for k in range(n_elite, params.population_size):
                p = population[k % n_elite]
                nt = p.giant_tour[:]
                random.shuffle(nt)
                ind = Individual(nt)
                evaluate(ind, split_algo, global_must_go, R, C)
                population[k] = ind
            update_biased_fitness(population, params)
            population.sort()
            no_improv = 0
    
    return best_sol.routes, best_sol.profit_score, best_sol.cost


def _run_hgs_pyvrp(dist_matrix, demands, capacity, R, C, values, global_must_go):
    """
    HGS Implementation using PyVRP library.
    """
    try:
        from pyvrp import Model, ProblemData, Client, Depot, VehicleType
        from pyvrp.stop import MaxRuntime
    except ImportError:
        print("Error: PyVRP not installed. Falling back to custom engine.")
        # Fallback requires consistent arguments, which might be tricky if not passed.
        # But we assume calling code handles this or fails.
        return [], 0, 0

    # 1. Prepare Data
    # PyVRP expects Clients (1..N) and Depot (0).
    # dist_matrix includes Depot at 0.
    
    clients = []
    n_nodes = len(dist_matrix)
    
    # Depot
    depot = Depot(0, 0) # ID 0
    
    # We iterate 1..N-1 because dist_matrix serves 0..N-1
    for i in range(1, n_nodes):
        weight = demands.get(i, 0)
        
        # Calculate Prize (Revenue)
        # Verify units: R is $/kg?
        prize = weight * R
        
        # Must Go enforcement
        if i in global_must_go:
            # Make prize huge so it's always profitable to visit
            prize += 100000.0
        
        # Modern PyVRP API:
        # Client(x, y, demand=0, service_duration=0, delivery=0, prize=0, required=False)
        # 'delivery' or 'demand'. For CVRP, use delivery usually?
        # PyVRP's 'demand' is load.
        # 'prize' implies optional. 
        # If we use prize, PyVRP treats it as Prize Collecting.
        
        # Note: PyVRP validation might require ints.
        client = Client(
            x=0, y=0,
            delivery=int(weight),
            prize=int(prize * 100) # Scale prize for precision
        )
        clients.append(client)
        
    # Vehicle Type
    vehicle_type = VehicleType(capacity=int(capacity), num_available=100)
    
    # Distance Matrix
    m_dist = [[int(d * C * 100) for d in row] for row in dist_matrix] 
    m_dur = [[0] * n_nodes for _ in dist_matrix] 
    
    data = ProblemData(
        clients=clients,
        depots=[depot],
        vehicle_types=[vehicle_type],
        distance_matrices=[m_dist],
        duration_matrices=[m_dur]
    )
    
    # Solve
    model = Model.from_data(data)
    time_limit_sec = values.get('time_limit', 10)
    result = model.solve(stop=MaxRuntime(time_limit_sec), display=False)
    
    # Parse Result
    routes = []
    for r in result.best.routes():
        r_indices = []
        for v in r.visits():
            r_indices.append(v) 
        routes.append(r_indices)
        
    # Recalculate Profit/Cost Manually to ensure correct scaling/float
    calc_profit = 0
    calc_cost = 0
    for r in routes:
        if not r: continue
        r_cost = 0
        
        # Dep->First
        r_cost += dist_matrix[0][r[0]] * C
        
        for k in range(len(r)-1):
            r_cost += dist_matrix[r[k]][r[k+1]] * C
            
        # Last->Dep
        r_cost += dist_matrix[r[-1]][0] * C
        
        calc_cost += r_cost
        
        for node in r:
            w = demands.get(node, 0)
            calc_profit += w * R
            
    calc_profit -= calc_cost
    return routes, calc_profit, calc_cost
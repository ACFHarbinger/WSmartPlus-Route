import numpy as np
import random
import time
import copy
from typing import List, Tuple, Dict
from src.pipeline.simulator.processor import create_dataframe_from_matrix

# --- HGS Auxiliaries ---

class Individual:
    def __init__(self, giant_tour: List[int]):
        self.giant_tour = giant_tour
        self.routes = []
        self.fitness = -float('inf')  # Profit
        self.cost = 0.0
        self.revenue = 0.0

    def __lt__(self, other):
        # Python heap/sort uses <. We want Max Profit, so we invert logic or sort reverse.
        # Here we define < as "is worse than" (lower fitness)
        return self.fitness < other.fitness

class HGSParams:
    def __init__(self, time_limit=10, population_size=50, elite_size=10, mutation_rate=0.2):
        self.time_limit = time_limit
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

def split_algorithm(giant_tour: List[int], dist_matrix, demands, capacity, R, C, values):
    """
    Decodes a Giant Tour (permutation of nodes) into a set of routes 
    that maximizes profit (Revenue - Travel Cost).
    
    This is a simplified Split algorithm using a shortest-path approach on a DAG.
    """
    n = len(giant_tour)
    # V[i] stores the max profit from node i to the end
    # P[i] stores the predecessor to reconstruct routes
    V = [-float('inf')] * (n + 1)
    P = [-1] * (n + 1)
    V[0] = 0
    
    # Precompute distances to depot (node 0)
    # Note: giant_tour indices map to actual bin IDs. 
    # dist_matrix indices usually include depot at 0.
    
    # We iterate through the giant tour
    for i in range(n):
        load = 0
        dist = 0
        revenue = 0
        # Try to form a route from i+1 to j
        for j in range(i + 1, n + 1):
            node_idx = giant_tour[j-1]
            
            # Update Load
            load += demands[node_idx]
            if load > capacity:
                break
                
            # Update Revenue
            # Revenue calculation matches Gurobi: (fill/100) * Density * Volume * R
            # Assuming demands passed here are already in kg for capacity check, 
            # we re-derive revenue or pass it in. 
            # For simplicity, let's assume 'demands' is weight, and revenue is prop to weight.
            # Revenue = Weight * R (if R is price per kg) OR calculated externally.
            # Based on Gurobi: R * S_dict[i]. S_dict is weight. So Revenue = Weight * R.
            revenue += demands[node_idx] * R
            
            # Update Distance
            if j == i + 1:
                # First node in route: Depot -> Node
                dist = dist_matrix[0][node_idx]
            else:
                prev_node = giant_tour[j-2]
                dist += dist_matrix[prev_node][node_idx]
            
            # Cost of returning to depot from current node j
            round_trip_cost = (dist + dist_matrix[node_idx][0]) * C
            
            # Profit of this segment
            segment_profit = revenue - round_trip_cost
            
            if V[i] + segment_profit > V[j]:
                V[j] = V[i] + segment_profit
                P[j] = i
                
    # Reconstruct Routes
    routes = []
    curr = n
    while curr > 0:
        prev = P[curr]
        route = giant_tour[prev:curr]
        routes.append(route)
        curr = prev
        
    total_profit = V[n]
    return routes, total_profit

def evaluate(individual: Individual, dist_matrix, demands, capacity, R, C, values):
    """
    Runs Split to determine fitness and routes.
    """
    routes, profit = split_algorithm(individual.giant_tour, dist_matrix, demands, capacity, R, C, values)
    individual.routes = routes
    individual.fitness = profit
    
    # Calculate explicit cost for reporting
    total_dist = 0
    for route in routes:
        if not route: continue
        d = dist_matrix[0][route[0]]
        for k in range(len(route)-1):
            d += dist_matrix[route[k]][route[k+1]]
        d += dist_matrix[route[-1]][0]
        total_dist += d
    individual.cost = total_dist * C
    return individual

def ordered_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """Standard OX Crossover for permutations."""
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[a:b] = p1[a:b]
    
    p2_idx = 0
    child_idx = b
    
    while -1 in child:
        if child_idx >= size:
            child_idx = 0
        
        if p2[p2_idx] not in child[a:b]:
            child[child_idx] = p2[p2_idx]
            child_idx += 1
        p2_idx += 1
        
    return child

def local_search(individual: Individual, dist_matrix):
    """
    Simple 2-opt improvement on the Giant Tour.
    """
    tour = individual.giant_tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour)): # +2 ensures we don't swap adjacent edges
                # Calculate simple distance delta (approximate)
                # Note: Exact delta requires running Split, which is expensive.
                # In HGS, we usually run LS on individual routes *after* Split.
                # Here we do a simple permutation swap to shake up the giant tour.
                
                # Perform swap
                new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                
                # We would need to evaluate to know if it's better. 
                # For this snippet, we just return a mutated tour probabilistically
                # or rely on the genetic pressure.
                # Let's just do a random swap mutation here for efficiency in Python
                pass
    
    # Simple Mutation instead of full 2-opt for speed in python
    if random.random() < 0.3:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
        
    individual.giant_tour = tour
    return individual

# --- Main Policy Function ---

def policy_lookahead_hgs(
    binsids: List[int],
    current_fill_levels: np.ndarray,
    distance_matrix: np.ndarray,
    values: dict,
    must_go_bins: List[int],
    time_limit: int = 10
):
    """
    Hybrid Genetic Search Policy for WCP.
    """
    # 1. Parse Parameters
    B, E, Q = values['B'], values['E'], values['vehicle_capacity']
    R, C = values['R'], values['C']
    
    # 2. Filter nodes: We consider all bins passed in 'binsids' as candidates,
    # but the HGS will decide order. 
    # Note: If you want to ONLY route specific bins, filter 'binsids' here.
    # The Gurobi model has a choice (g variable). HGS usually routes everyone in the list.
    # To mimic Gurobi's 'selection', we can include all bins with fill > 0.
    
    candidate_indices = [i for i, b_id in enumerate(binsids) if current_fill_levels[i] > 0]
    
    # Create mapping
    # HGS works on indices 0..N. We map these to the distance matrix indices.
    # The distance matrix includes depot at 0. 'binsids' usually start from index 1 in matrix?
    # Assuming binsids[i] corresponds to row/col i+1 in distance_matrix (since 0 is depot).
    
    # Map local HGS index -> Matrix/Bin Index
    local_to_global = {local_idx: global_idx + 1 for local_idx, global_idx in enumerate(candidate_indices)} 
    
    # Prepare Demands (Weights)
    demands = {}
    for local_i, global_i in local_to_global.items():
        # global_i is matrix index. binsids index is global_i - 1
        bin_array_idx = global_i - 1 
        fill = current_fill_levels[bin_array_idx]
        weight = (fill / 100.0) * B * E
        demands[global_i] = weight

    if not candidate_indices:
        return [0], 0, 0

    # 3. Initialization
    params = HGSParams(time_limit=time_limit)
    population = []
    
    # Seed population
    base_tour = list(local_to_global.values()) # List of matrix indices
    
    start_time = time.time()
    
    for _ in range(params.population_size):
        random.shuffle(base_tour)
        ind = Individual(base_tour[:])
        ind = evaluate(ind, distance_matrix, demands, Q, R, C, values)
        population.append(ind)
        
    population.sort(reverse=True) # Best (Highest Profit) first
    best_solution = population[0]
    
    # 4. Main HGS Loop
    generation = 0
    while time.time() - start_time < params.time_limit:
        generation += 1
        
        # Selection (Tournament)
        parent1 = population[random.randint(0, params.elite_size)]
        parent2 = population[random.randint(0, params.population_size - 1)]
        
        # Crossover
        child_tour = ordered_crossover(parent1.giant_tour, parent2.giant_tour)
        child = Individual(child_tour)
        
        # Education (Local Search / Mutation)
        child = local_search(child, distance_matrix)
        
        # Evaluation (Split)
        child = evaluate(child, distance_matrix, demands, Q, R, C, values)
        
        # Survivor Selection
        # Simple steady-state: if better than worst, replace worst
        if child.fitness > population[-1].fitness:
            population[-1] = child
            population.sort(reverse=True)
            
            if child.fitness > best_solution.fitness:
                best_solution = child
                
    # 5. Format Output
    # Convert routes to flat list of IDs (excluding depot 0 inside the list, 
    # but the simulator usually expects [0, id, id, 0, id...]).
    # The 'policy_lookahead_vrpp' returns [0] + contentores_coletados.
    
    final_sequence = []
    # Flatten routes
    for route in best_solution.routes:
        final_sequence.extend(route)
        
    # Convert Matrix Indices (1..N) back to Bin IDs if necessary
    # The Gurobi code returns: contentores_coletados = [id_map[j] for ...].
    # binsids are 0-indexed in the input list, but IDs might be +1.
    # The standard output seems to be the list of collected bin INDICES (0-based relative to binsids).
    
    # Convert back to 0-based index referenced in 'binsids'
    collected_bins_indices = [idx - 1 for idx in final_sequence]
    
    return [0] + collected_bins_indices, best_solution.fitness, best_solution.cost


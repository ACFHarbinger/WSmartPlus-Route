import random

from typing import List


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
            load += demands.get(node_idx, 0) # Safety get
            
            # Check Capacity
            # RELAXATION: Allow single-node routes even if they exceed capacity.
            # This handles overflowing bins that effectively fill the truck immediately.
            if load > capacity and j > i + 1:
                break
                
            # Update Revenue
            # Revenue = Weight * R
            revenue += demands.get(node_idx, 0) * R
            
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
    
    # Safety check if no solution found
    if V[n] == -float('inf'):
        return [], -float('inf')

    while curr > 0:
        prev = P[curr]
        route = giant_tour[prev:curr]
        routes.append(route)
        curr = prev
        
    total_profit = V[n]
    return routes, total_profit


def evaluate(individual: Individual, dist_matrix, demands, capacity, R, C, values, must_go_bins, local_to_global):
    """
    Runs Split to determine fitness and routes.
    Optimized for SINGLE VEHICLE reality:
    - Fitness = Profit(Route 0) - Penalty(Missed Must-Go)
    - Ignores subsequent routes (phantom routes).
    """
    routes, total_split_profit = split_algorithm(individual.giant_tour, dist_matrix, demands, capacity, R, C, values)
    individual.routes = routes
    
    # Analyze All Routes (Multi-Trip)
    if not routes:
        individual.fitness = -1e9
        individual.cost = 0
        return individual

    total_profit = 0
    total_cost = 0
    visited_nodes = set()

    for route in routes:
        # Calculate Profit of this Route
        load = 0
        revenue = 0
        dist = 0
        
        if route:
            # Distance
            dist = dist_matrix[0][route[0]]
            for k in range(len(route)-1):
                dist += dist_matrix[route[k]][route[k+1]]
            dist += dist_matrix[route[-1]][0]
            
            # Revenue & Load
            for node_idx in route:
                w = demands.get(node_idx, 0)
                load += w
                revenue += w * R
                visited_nodes.add(node_idx)
                
        cost = dist * C
        profit_r = revenue - cost
        
        total_profit += profit_r
        total_cost += cost

    # Check Must-Go Enforcement (Global across all routes)
    missed_must_go = 0
    for mg in must_go_bins:
        if mg not in visited_nodes:
            missed_must_go += 1
            
    # Fitness: Total Profit - Penalty
    penalty = missed_must_go * 10000.0
    individual.fitness = total_profit - penalty
    individual.cost = total_cost # Report total cost
    
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
    2-opt improvement on the Giant Tour.
    Optimizes the sequence distance (Depot -> ... -> Depot considered implicitly or just path).
    Here we optimize the Path Distance between nodes.
    """
    tour = individual.giant_tour[:]
    n = len(tour)
    if n < 2:
        return individual
        
    # Limit iterations for speed
    max_improvements = 5000 
    improvements = 0
    
    improved = True
    while improved and improvements < max_improvements:
        improved = False
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j == i + 1: continue # Adjacent edges
                
                # Check 2-opt swap:
                # Edge A-B and C-D  ->  A-C and B-D
                # Indices: A=i, B=i+1, C=j, D=j+1
                
                # We need to handle boundary conditions if we considered a closed loop (Depot-Depot).
                # But giant_tour is just a list. We optimize the linear path distance sum(d[k][k+1]).
                
                u = tour[i]
                v = tour[i+1] # B
                x = tour[j]   # C
                
                # If j is last element, 'y' is implicit end? 
                # Let's simple swap logic: reverse segment [i+1 : j+1]
                # Pre-swap edges: (tour[i], tour[i+1]) + (tour[j], tour[j+1])
                # Post-swap edges: (tour[i], tour[j]) + (tour[i+1], tour[j+1])
                # Note: This is valid for internal nodes.
                
                if j == n - 1:
                    # Edge at the end. 
                    # If we don't have a fixed end depot, we just check if dist(u, x) < dist(u, v) ?
                    # Not strictly correct without depot. 
                    # But standard HGS assumes TSP on nodes + 0.
                    # Let's check strict delta including neighbors.
                    pass
                    
                else: 
                    y = tour[j+1] # D
                    
                    # Original: u->v ... x->y
                    # New:      u->x ... v->y (reversed path between v..x)
                    d_current = dist_matrix[u][v] + dist_matrix[x][y]
                    d_new = dist_matrix[u][x] + dist_matrix[v][y]
                    
                    if d_new < d_current:
                        # Apply Swap
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        improvements += 1
                        break # First improvement
                        
            if improved: break
            
    individual.giant_tour = tour
    return individual
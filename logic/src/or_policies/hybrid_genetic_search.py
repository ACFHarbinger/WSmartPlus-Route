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
    
    This is a Split algorithm using a shortest-path approach on a DAG.
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
    
    # Optimization: Convert demands to list for faster access if keys are dense integers
    node_demands = [demands.get(node_id, 0) for node_id in giant_tour]
    
    # We iterate through the giant tour
    for i in range(n):
        load = 0
        dist = 0
        revenue = 0
        # Try to form a route from i+1 to j
        for j in range(i + 1, n + 1):
            node_idx = giant_tour[j-1]
            d = node_demands[j-1]
            
            # Update Load
            load += d
            
            # Check Capacity
            # RELAXATION: Allow single-node routes even if they exceed capacity.
            # This handles overflowing bins that effectively fill the truck immediately.
            if load > capacity and j > i + 1:
                break
                
            # Update Revenue
            # Revenue = Weight * R
            revenue += d * R
            
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


def local_search(individual: Individual, dist_matrix, demands, capacity, R, C, values, neighbors=None):
    """
    Route-Based Local Search (Granular).
    Optimizes the solution by performing Relocate and Swap moves on the decoded routes.
    """
    # 1. Decode to Routes (using Split)
    # We need to run split to get the current route structure
    routes, _ = split_algorithm(individual.giant_tour, dist_matrix, demands, capacity, R, C, values)
    
    if not routes:
        return individual

    # Helper to calculate route cost
    def get_route_cost(route):
        if not route: return 0
        d = dist_matrix[0][route[0]]
        for k in range(len(route)-1):
            d += dist_matrix[route[k]][route[k+1]]
        d += dist_matrix[route[-1]][0]
        return d * C

    def get_route_load(route):
        return sum(demands.get(n, 0) for n in route)

    # 2. Granular Search Setup
    # Flatten structure for easy iteration? Or iterate routes?
    # Iterate through all nodes U
    
    improved = True
    max_moves = 500 # Safety limit per call
    move_count = 0
    
    # Pre-calculate route loads/costs
    route_data = []
    for r_idx, route in enumerate(routes):
        route_data.append({
            'load': get_route_load(route),
            'cost': get_route_cost(route),
            'route': route
        })
        
    while improved and move_count < max_moves:
        improved = False
        
        # Iterate all routes and nodes
        # We make a list of (r_idx, node_idx_in_route, node_id) to iterate safely
        # But indices change on modification.
        
        # Strategy: Iterate u from 1..N. Find where u is.
        # Map: node -> (r_idx, pos)
        node_map = {}
        for r_idx, data in enumerate(route_data):
            for pos, node in enumerate(data['route']):
                node_map[node] = (r_idx, pos)
                
        # Iterate all nodes U
        all_nodes = list(node_map.keys())
        random.shuffle(all_nodes) # Shuffle to diversify order
        
        for u in all_nodes:
            if u not in node_map: continue # Moved/Removed?
            u_r_idx, u_pos = node_map[u]
            u_route = route_data[u_r_idx]['route']
            
            # Identify Neighbors V
            # If neighbors provided, use them. Else full scan (slow).
            search_space = neighbors[u] if neighbors else all_nodes
            
            for v in search_space:
                if u == v: continue
                if v not in node_map: continue
                
                v_r_idx, v_pos = node_map[v]
                v_route = route_data[v_r_idx]['route']
                
                # --- Operator 1: RELOCATE U -> V (Insert U after V) ---
                # Valid? Check Capacity of v_route (if u not in v_route)
                u_demand = demands.get(u, 0)
                
                if u_r_idx != v_r_idx:
                    if route_data[v_r_idx]['load'] + u_demand > capacity:
                        # Capacity violation (strict)
                        pass
                    else:
                        # Evaluate Move
                        # Remove U from U_Route
                        # Cost change U:
                        # ... prev_u -> u -> next_u ...  => ... prev_u -> next_u ...
                        # cost_diff_u = d(prev, next) - d(prev, u) - d(u, next)
                        
                        u_prev = u_route[u_pos-1] if u_pos > 0 else 0
                        u_next = u_route[u_pos+1] if u_pos < len(u_route)-1 else 0
                        
                        delta_u = dist_matrix[u_prev][u_next] - (dist_matrix[u_prev][u] + dist_matrix[u][u_next])
                        
                        # Insert U after V in V_Route
                        # ... v -> next_v ... => ... v -> u -> next_v ...
                        v_next = v_route[v_pos+1] if v_pos < len(v_route)-1 else 0
                        
                        delta_v = (dist_matrix[v][u] + dist_matrix[u][v_next]) - dist_matrix[v][v_next]
                        
                        total_delta = (delta_u + delta_v) * C
                        
                        if total_delta < -1e-4:
                            # Apply Move
                            u_route.pop(u_pos)
                            v_route.insert(v_pos + 1, u)
                            
                            # Update Loads/Costs metadata
                            route_data[u_r_idx]['load'] -= u_demand
                            route_data[v_r_idx]['load'] += u_demand
                            # We could treat costs exactly, but brute force update is safer for code simplicity
                            # (Cost is only used for final fitness, simplified delta logic above suffices for acceptance)
                            
                            improved = True
                            move_count += 1
                            break # Restart/Next U
                
                
                else:
                    # Intra-Route Relocate (Skipped for now)
                    pass 

                # --- Operator 2: SWAP U <-> V ---
                # Check Capacity
                # New Load U = Old Load U - u_dem + v_dem
                # New Load V = Old Load V - v_dem + u_dem
                v_demand = demands.get(v, 0)
                
                if u_r_idx != v_r_idx:
                    new_load_u = route_data[u_r_idx]['load'] - u_demand + v_demand
                    new_load_v = route_data[v_r_idx]['load'] - v_demand + u_demand
                    
                    if new_load_u <= capacity and new_load_v <= capacity:
                         # Evaluate Swap
                         # U Side: ... u_prev -> u -> u_next ...
                         u_prev = u_route[u_pos-1] if u_pos > 0 else 0
                         u_next = u_route[u_pos+1] if u_pos < len(u_route)-1 else 0
                         
                         v_prev = v_route[v_pos-1] if v_pos > 0 else 0
                         v_next = v_route[v_pos+1] if v_pos < len(v_route)-1 else 0

                         # Delta Remove U + Insert V
                         d_u_change = (dist_matrix[u_prev][v] + dist_matrix[v][u_next]) - \
                                      (dist_matrix[u_prev][u] + dist_matrix[u][u_next])
                                      
                         # Delta Remove V + Insert U
                         d_v_change = (dist_matrix[v_prev][u] + dist_matrix[u][v_next]) - \
                                      (dist_matrix[v_prev][v] + dist_matrix[v][v_next])

                         if (d_u_change + d_v_change) * C < -1e-4:
                             # Apply Swap
                             u_route[u_pos] = v
                             v_route[v_pos] = u
                             
                             route_data[u_r_idx]['load'] = new_load_u
                             route_data[v_r_idx]['load'] = new_load_v
                             
                             improved = True
                             move_count += 1
                             break
                             
                # --- Operator 3: 2-Opt* (Inter-Route Tail Swap) ---
                # Swap tails after u and v
                # u_route: [... u, u_next ...] -> [... u, v_next ...]
                # v_route: [... v, v_next ...] -> [... v, u_next ...]
                if u_r_idx != v_r_idx:
                    # Current Tails
                    # u_tail includes u_next...end
                    # v_tail includes v_next...end
                    
                    # Loads
                    # Calculating tail load might be O(L). For granular search, O(L) is fine (L~10).
                    # Actually, precomputing Prefix Loads would be faster, but let's do O(L) first.
                    
                    u_tail_load = sum(demands.get(n, 0) for n in u_route[u_pos+1:])
                    v_tail_load = sum(demands.get(n, 0) for n in v_route[v_pos+1:])
                    
                    u_head_load = route_data[u_r_idx]['load'] - u_tail_load
                    v_head_load = route_data[v_r_idx]['load'] - v_tail_load
                    
                    # New Loads
                    new_load_u = u_head_load + v_tail_load
                    new_load_v = v_head_load + u_tail_load
                    
                    if new_load_u <= capacity and new_load_v <= capacity:
                        # Cost Delta
                        # Edge changes:
                        # Remove (u, u_next) and (v, v_next)
                        # Add (u, v_next) and (v, u_next)
                        
                        u_next = u_route[u_pos+1] if u_pos < len(u_route)-1 else 0
                        v_next = v_route[v_pos+1] if v_pos < len(v_route)-1 else 0
                        
                        # Note: u and v are usually "neighbors", so crossing edges (u, v_next) might be short?
                        # Actually 2-opt* typically cuts edges (u, u_next) and (v, v_next) and connects u-v_next?
                        # Yes.
                        
                        curr_dist = dist_matrix[u][u_next] + dist_matrix[v][v_next]
                        new_dist = dist_matrix[u][v_next] + dist_matrix[v][u_next]
                        
                        delta = new_dist - curr_dist
                        
                        if delta * C < -1e-4:
                            # Apply 2-Opt*
                            # Tails
                            tail_u = u_route[u_pos+1:]
                            tail_v = v_route[v_pos+1:]
                            
                            # Construct new routes
                            # u_route becomes head_u + tail_v
                            # v_route becomes head_v + tail_u
                            
                            new_route_u = u_route[:u_pos+1] + tail_v
                            new_route_v = v_route[:v_pos+1] + tail_u
                            
                            # Update in place? Careful with list refs
                            route_data[u_r_idx]['route'] = new_route_u
                            route_data[v_r_idx]['route'] = new_route_v
                            
                            route_data[u_r_idx]['load'] = new_load_u
                            route_data[v_r_idx]['load'] = new_load_v
                            
                            improved = True
                            move_count += 1
                            break
                
            if improved: break
    
    # 3. 2-Opt (Intra-Route) - Quick check on all routes
    for data in route_data:
        route = data['route']
        if len(route) < 3: continue
        
        # Simple 2-opt pass on this route
        best_delta = 0
        best_move = None
        
        # Limit 2-opt search? O(L^2). Max L=100. Fast enough.
        for i in range(len(route)-1):
            for j in range(i+1, len(route)):
                if j == i+1: continue
                # u-v, x-y
                u = route[i]
                v = route[i+1]
                x = route[j]
                y = route[j+1] if j < len(route)-1 else 0
                
                curr = dist_matrix[u][v] + dist_matrix[x][y]
                new = dist_matrix[u][x] + dist_matrix[v][y]
                
                if new < curr - 1e-4:
                     # Apply immediately (First Improvement)
                     route[i+1:j+1] = route[i+1:j+1][::-1]
                     pass # Route modified in place
                     
    # 4. Re-Encode Giant Tour
    new_tour = []
    for data in route_data:
        new_tour.extend(data['route'])
        
    individual.giant_tour = new_tour
    return individual
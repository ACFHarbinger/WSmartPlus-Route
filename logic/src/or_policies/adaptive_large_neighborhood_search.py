
import random
import math
import time
import copy
from typing import List, Dict, Tuple

class ALNSParams:
    def __init__(self, time_limit=10, max_iterations=1000, 
                 start_temp=100, cooling_rate=0.995, 
                 reaction_factor=0.1, 
                 min_removal=1, max_removal_pct=0.3):
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate
        self.reaction_factor = reaction_factor # rho
        self.min_removal = min_removal
        self.max_removal_pct = max_removal_pct

class ALNSSolver:
    def __init__(self, dist_matrix, demands, capacity, R, C, params: ALNSParams):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        
        self.n_nodes = len(dist_matrix) - 1 # Assuming 0 is depot
        self.nodes = list(range(1, self.n_nodes + 1))
        
        # Operator weights
        self.destroy_ops = [self.random_removal, self.worst_removal, self.cluster_removal]
        self.repair_ops = [self.greedy_insertion, self.regret_2_insertion]
        
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)
        
        self.destroy_scores = [0.0] * len(self.destroy_ops)
        self.repair_scores = [0.0] * len(self.repair_ops)
        self.destroy_counts = [0] * len(self.destroy_ops)
        self.repair_counts = [0] * len(self.repair_ops)

    def solve(self, initial_solution=None):
        # Initial Solution Construction (if needed)
        if initial_solution:
            current_routes = initial_solution
        else:
            current_routes = self.build_initial_solution()
            
        best_routes = copy.deepcopy(current_routes)
        best_cost = self.calculate_cost(best_routes)
        current_cost = best_cost
        
        T = self.params.start_temp
        start_time = time.time()
        
        for it in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break
                
            # 1. Select Operators (Roulette Wheel)
            d_idx = self.select_operator(self.destroy_weights)
            r_idx = self.select_operator(self.repair_weights)
            
            destroy_op = self.destroy_ops[d_idx]
            repair_op = self.repair_ops[r_idx]
            
            # 2. Determine Removal Count
            n_remove = random.randint(self.params.min_removal, 
                                      max(self.params.min_removal, int(self.n_nodes * self.params.max_removal_pct)))
            
            # 3. Apply Operators
            # Destroy returns (partial_routes, removed_nodes)
            partial_routes, removed = destroy_op(copy.deepcopy(current_routes), n_remove)
            
            # Repair returns new_routes
            new_routes = repair_op(partial_routes, removed)
            
            # 4. Evaluate
            new_cost = self.calculate_cost(new_routes)
            
            # 5. Acceptance (Simulated Annealing)
            accept = False
            score = 0
            
            delta = new_cost - current_cost
            if delta < -1e-6: # Improvement
                accept = True
                if new_cost < best_cost - 1e-6: # New Global Best
                    best_routes = copy.deepcopy(new_routes)
                    best_cost = new_cost
                    score = 3 # Reward for global best
                else:
                    score = 1 # Reward for improvement
            else:
                # Metropolis
                prob = math.exp(-delta / T) if T > 0 else 0
                if random.random() < prob:
                    accept = True
                    score = 0 # No reward for worsening but accepted? Usually small reward
            
            if accept:
                current_routes = new_routes
                current_cost = new_cost
                
            # 6. Update Weights
            self.destroy_scores[d_idx] += score
            self.repair_scores[r_idx] += score
            self.destroy_counts[d_idx] += 1
            self.repair_counts[r_idx] += 1
            
            # Update weights periodically (simple smoothing)
            lambda_decay = 0.8
            self.destroy_weights[d_idx] = lambda_decay * self.destroy_weights[d_idx] + (1-lambda_decay) * max(0.1, score)
            self.repair_weights[r_idx] = lambda_decay * self.repair_weights[r_idx] + (1-lambda_decay) * max(0.1, score)

            # Cool down
            T *= self.params.cooling_rate
            
        return best_routes, best_cost

    def select_operator(self, weights):
        total = sum(weights)
        r = random.uniform(0, total)
        curr = 0
        for i, w in enumerate(weights):
            curr += w
            if curr >= r:
                return i
        return len(weights) - 1

    def calculate_cost(self, routes: List[List[int]]) -> float:
        total_dist = 0
        for route in routes:
            if not route: continue
            d = self.dist_matrix[0][route[0]]
            for i in range(len(route)-1):
                d += self.dist_matrix[route[i]][route[i+1]]
            d += self.dist_matrix[route[-1]][0]
            total_dist += d
        return total_dist * self.C # Cost Minimization

    # --- Operators ---
    
    def build_initial_solution(self):
        # Simple Savings or Greedy
        # Let's use a dummy one-route-per-node to start, usually ALNS fixes it fast
        # Or better: random shuffle giant tour and split
        nodes = self.nodes[:]
        random.shuffle(nodes)
        routes = []
        curr_route = []
        load = 0
        for node in nodes:
            d = self.demands.get(node, 0)
            if load + d <= self.capacity:
                curr_route.append(node)
                load += d
            else:
                if curr_route: routes.append(curr_route)
                curr_route = [node]
                load = d
        if curr_route: routes.append(curr_route)
        return routes

    def random_removal(self, routes, n_remove):
        removed = []
        # Flatten
        all_nodes = []
        for r_idx, r in enumerate(routes):
            for n_idx, node in enumerate(r):
                all_nodes.append((r_idx, n_idx, node))
        
        if n_remove >= len(all_nodes):
            return [[]], [n for _, _, n in all_nodes]
            
        targets = random.sample(all_nodes, n_remove)
        
        # Sort targets by r_idx, n_idx desc to pop safely
        targets.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        for r_idx, n_idx, node in targets:
            routes[r_idx].pop(n_idx)
            removed.append(node)
            
        # Clean empty routes
        routes = [r for r in routes if r]
        return routes, removed

    def worst_removal(self, routes, n_remove):
        # Remove nodes that contribute most to the cost
        costs = []
        for r_idx, route in enumerate(routes):
            if len(route) == 0: continue
            
            # Baseline cost
            base_d = self.dist_matrix[0][route[0]]
            for k in range(len(route)-1):
                base_d += self.dist_matrix[route[k]][route[k+1]]
            base_d += self.dist_matrix[route[-1]][0]
            
            for i, node in enumerate(route):
                # Calc cost without this node
                # Prev -> Next
                prev = 0 if i == 0 else route[i-1]
                nex = 0 if i == len(route)-1 else route[i+1]
                
                # We save dist(prev, node) + dist(node, nex)
                saved = self.dist_matrix[prev][node] + self.dist_matrix[node][nex]
                # We add dist(prev, nex)
                added = self.dist_matrix[prev][nex]
                
                cost_increase = added - saved # Negative usually, implies saving
                # We want Maximal Savings (Most negative cost_increase -> Max abs(saving))
                # Or simply: Highest cost contribution
                
                costs.append( (r_idx, i, node, saved - added) ) # Saving > 0
                
        costs.sort(key=lambda x: x[3], reverse=True) # Highest savings first
        removed = []
        
        # Need to be careful about indices changing. 
        # Easier strategy: stochastic worst removal or just recompute?
        # Recomputing is safer but slower. 
        # One-shot greedy:
        targets = costs[:n_remove]
        # Sort by index desc
        targets.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        for r_idx, n_idx, node, _ in targets:
            if n_idx < len(routes[r_idx]) and routes[r_idx][n_idx] == node:
                routes[r_idx].pop(n_idx)
                removed.append(node)
                
        routes = [r for r in routes if r]
        return routes, removed

    def cluster_removal(self, routes, n_remove):
        # Pick a random node, remove it and its k nearest neighbors
        # Simplified Shaw
        if not any(routes): return routes, []
        
        # Pick seed
        seed_route_idx = random.randint(0, len(routes)-1)
        if not routes[seed_route_idx]: return self.random_removal(routes, n_remove)
        
        seed_node = random.choice(routes[seed_route_idx])
        
        # Find nearest nodes in current solution NO, in general graph
        # Shaw removal usually removes related nodes (distance, time, etc)
        # Here just distance
        # We need to find where they are in current solution
        
        removed = [seed_node]
        
        # Get all nodes current pos
        node_map = {}
        for r_idx, r in enumerate(routes):
            for n_idx, node in enumerate(r):
                node_map[node] = (r_idx, n_idx)
                
        # Find neighbors
        candidates = []
        for v in self.nodes:
            if v == seed_node or v not in node_map: continue
            dist = self.dist_matrix[seed_node][v]
            candidates.append((v, dist))
            
        candidates.sort(key=lambda x: x[1])
        
        target_nodes = [x[0] for x in candidates[:n_remove-1]]
        removed.extend(target_nodes)
        
        # Now remove them from routes
        # Sort targets by position to remove safely
        to_remove_locs = []
        for node in removed:
            if node in node_map:
                to_remove_locs.append( (*node_map[node], node) )
        
        to_remove_locs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        final_removed = []
        for r_idx, n_idx, node in to_remove_locs:
            routes[r_idx].pop(n_idx)
            final_removed.append(node)
            
        routes = [r for r in routes if r]
        return routes, final_removed

    def greedy_insertion(self, routes, removed_nodes):
        # Insert each node in best position
        random.shuffle(removed_nodes) # Randomize order
        
        for node in removed_nodes:
            best_cost = float('inf')
            best_pos = None # (route_idx, insert_idx)
            
            # Try all positions
            for r_idx, route in enumerate(routes):
                load = sum(self.demands.get(n,0) for n in route)
                if load + self.demands.get(node,0) > self.capacity:
                    continue
                    
                # Try all slots: 0 to len(route)
                # Cost change: dist(prev, node) + dist(node, next) - dist(prev, next)
                for i in range(len(route) + 1):
                    prev = 0 if i == 0 else route[i-1]
                    nex = 0 if i == len(route) else route[i]
                    
                    cost_increase = self.dist_matrix[prev][node] + self.dist_matrix[node][nex] - self.dist_matrix[prev][nex]
                    
                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_pos = (r_idx, i)
            
            # Also consider new route
            if self.demands.get(node,0) <= self.capacity:
                cost_new = self.dist_matrix[0][node] + self.dist_matrix[node][0]
                if cost_new < best_cost:
                    best_cost = cost_new
                    best_pos = (len(routes), 0)
            
            # Apply
            if best_pos:
                r, i = best_pos
                if r == len(routes):
                    routes.append([node])
                else:
                    routes[r].insert(i, node)
            else:
                # Should not happen if capacity allows single trips, or create new route
                routes.append([node])
                
        return routes

    def regret_2_insertion(self, routes, removed_nodes):
        # Regret-2: max(2nd_best - best)
        # More computationally expensive
        # For simplify/speed, let's just stick to a specialized structure 
        # or implement full Regret
        
        # If too many nodes, fallback to greedy for speed
        if len(removed_nodes) > 30: 
            return self.greedy_insertion(routes, removed_nodes)
            
        pending = removed_nodes[:]
        
        while pending:
            max_regret = -1
            best_node_to_insert = None
            best_insert_pos = None # (r, i)
            
            # For each node, find Best and 2nd Best insertion scores
            for node in pending:
                valid_moves = [] # (cost_increase, r, i)
                
                # Check existing routes
                for r_idx, route in enumerate(routes):
                    load = sum(self.demands.get(n,0) for n in route)
                    if load + self.demands.get(node,0) > self.capacity: continue
                    
                    for i in range(len(route) + 1):
                        prev = 0 if i == 0 else route[i-1]
                        nex = 0 if i == len(route) else route[i]
                        cost = self.dist_matrix[prev][node] + self.dist_matrix[node][nex] - self.dist_matrix[prev][nex]
                        valid_moves.append((cost, r_idx, i))
                
                # Check new route
                if self.demands.get(node,0) <= self.capacity:
                    cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]
                    valid_moves.append((cost, len(routes), 0))
                
                if not valid_moves:
                    # Infeasible? Should force new route even if cap fail? No, capacity is hard.
                    # Just force new route ignoring checks if nothing else works (shouldn't happen with single node)
                    valid_moves.append((self.dist_matrix[0][node]*2, len(routes), 0))

                valid_moves.sort(key=lambda x: x[0])
                
                best = valid_moves[0]
                second = valid_moves[1] if len(valid_moves) > 1 else (best[0] * 1.5, -1, -1) # infinite regret surrogate
                
                regret = second[0] - best[0]
                
                if regret > max_regret:
                    max_regret = regret
                    best_node_to_insert = node
                    best_insert_pos = (best[1], best[2])
            
            # Apply Best Regret Move
            if best_node_to_insert:
                r, i = best_insert_pos
                if r == len(routes):
                    routes.append([best_node_to_insert])
                else:
                    routes[r].insert(i, best_node_to_insert)
                pending.remove(best_node_to_insert)
            else:
                break
                
        return routes

def run_alns(dist_matrix, demands, capacity, R, C, values):
    params = ALNSParams(
        time_limit=values.get('time_limit', 10),
        max_iterations=values.get('Iterations', 2000)
    )
    solver = ALNSSolver(dist_matrix, demands, capacity, R, C, params)
    routes, cost = solver.solve()
    return routes, cost

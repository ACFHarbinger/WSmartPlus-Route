"""
Adaptive Large Neighborhood Search (ALNS) policy module.

This module implements multiple variants of the Adaptive Large Neighborhood Search
metaheuristic for solving the Capacitated Vehicle Routing Problem with Profits (CVRPP).

The module provides three implementation variants:
1. Custom ALNS: Pure Python implementation with destroy/repair operators
2. ALNS Package: Integration with the `alns` Python package
3. OR-Tools ALNS: Uses Google OR-Tools with Guided Local Search metaheuristic

ALNS iteratively improves solutions by:
- Destroy operators: Remove nodes from current solution (random, worst, cluster removal)
- Repair operators: Re-insert removed nodes optimally (greedy, regret-2 insertion)
- Adaptive weighting: Operator selection based on historical performance
- Acceptance: Simulated annealing for accepting worse solutions

Reference:
    Pisinger, D., & Ropke, S. (2007). A general heuristic for vehicle routing problems.
    Computers & Operations Research, 34(8), 2403-2435.
"""
import math
import time
import copy
import random
import numpy as np

from typing import List, Tuple
from alns import ALNS
from alns.stop import MaxRuntime
from alns.select import RouletteWheel
from alns.accept import SimulatedAnnealing
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


class ALNSParams:
    """
    Configuration parameters for the ALNS solver.

    Attributes:
        time_limit (int): Maximum runtime in seconds. Default: 10
        max_iterations (int): Maximum number of ALNS iterations. Default: 1000
        start_temp (float): Initial temperature for simulated annealing. Default: 100
        cooling_rate (float): Temperature decay factor per iteration. Default: 0.995
        reaction_factor (float): Learning rate for operator weight updates (rho). Default: 0.1
        min_removal (int): Minimum number of nodes to remove. Default: 1
        max_removal_pct (float): Maximum percentage of nodes to remove. Default: 0.3
    """
    def __init__(self, time_limit=10, max_iterations=1000, 
                 start_temp=100, cooling_rate=0.995, 
                 reaction_factor=0.1, 
                 min_removal=1, max_removal_pct=0.3):
        """
        Initialize ALNS parameters.

        Args:
            time_limit (int): Max runtime in seconds.
            max_iterations (int): Max iterations.
            start_temp (float): Initial temperature.
            cooling_rate (float): Cooling rate.
            reaction_factor (float): Reaction factor.
            min_removal (int): Min nodes to remove.
            max_removal_pct (float): Max percentage of nodes to remove.
        """
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate
        self.reaction_factor = reaction_factor # rho
        self.min_removal = min_removal
        self.max_removal_pct = max_removal_pct

class ALNSSolver:
    """
    Custom implementation of Adaptive Large Neighborhood Search for CVRP.

    This solver uses destroy and repair operators to iteratively improve solutions.
    Operator selection is adaptive based on historical performance using roulette wheel selection.

    Attributes:
        dist_matrix (np.ndarray): Distance matrix where dist_matrix[i][j] is distance from i to j
        demands (dict): Demand/weight for each customer node {node_id: demand_value}
        capacity (float): Vehicle capacity constraint
        R (float): Revenue per unit of demand collected
        C (float): Cost per unit of distance traveled
        params (ALNSParams): Algorithm parameters
        destroy_ops (List): List of destroy operator functions
        repair_ops (List): List of repair operator functions
    """
    def __init__(self, dist_matrix, demands, capacity, R, C, params: ALNSParams):
        """
        Initialize the ALNS solver.

        Args:
            dist_matrix (np.ndarray): Distance matrix (N x N) including depot at index 0
            demands (dict): Dictionary mapping node IDs to demand values
            capacity (float): Maximum vehicle capacity
            R (float): Revenue per unit of collected demand
            C (float): Cost coefficient per distance unit
            params (ALNSParams): Configuration parameters for ALNS
        """
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
        """
        Execute the ALNS algorithm to find high-quality routing solutions.

        Args:
            initial_solution (List[List[int]], optional): Initial set of routes.
                If None, a random greedy solution is constructed.

        Returns:
            Tuple[List[List[int]], float]: Best routes found and their total cost
                - routes: List of routes where each route is a list of node IDs
                - cost: Total travel cost (distance * C)
        """
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

    def select_operator(self, weights: List[float]) -> int:
        """
        Select an operator index using roulette wheel selection.

        Args:
            weights (List[float]): Weights for each operator.

        Returns:
            int: Index of the selected operator.
        """
        total = sum(weights)
        r = random.uniform(0, total)
        curr = 0
        for i, w in enumerate(weights):
            curr += w
            if curr >= r:
                return i
        return len(weights) - 1

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """
        Calculate the total weighted travel cost for a set of routes.

        Args:
            routes (List[List[int]]): List of routes (node sequences).

        Returns:
            float: Total distance multiplied by the cost coefficient.
        """
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
    
    def build_initial_solution(self) -> List[List[int]]:
        """
        Construct an initial feasible solution using a greedy shuffle-and-split heuristic.

        Returns:
            List[List[int]]: Initial routing solution.
        """
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

    def random_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """
        Randomly remove n_remove nodes from the current routes.

        Args:
            routes (List[List[int]]): Current routes.
            n_remove (int): Number of nodes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
        """
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

    def worst_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """
        Remove nodes that contribute most to the current routing cost.

        Args:
            routes (List[List[int]]): Current routes.
            n_remove (int): Number of nodes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
        """
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

    def cluster_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """
        Remove a cluster of nodes based on spatial proximity (Shaw Removal variant).

        Args:
            routes (List[List[int]]): Current routes.
            n_remove (int): Number of nodes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
        """
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
            if v == seed_node or v not in node_map:
                continue
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

    def greedy_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """
        Insert removed nodes into their best (cheapest) positions greedily.

        Args:
            routes (List[List[int]]): Partial routes.
            removed_nodes (List[int]): Nodes to be re-inserted.

        Returns:
            List[List[int]]: New routes after insertion.
        """
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

    def regret_2_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """
        Insert removed nodes based on the regret-2 criterion.

        Nodes with the highest difference between their second-best and best
        insertion costs are prioritized.

        Args:
            routes (List[List[int]]): Partial routes.
            removed_nodes (List[int]): Nodes to be re-inserted.

        Returns:
            List[List[int]]: New routes after insertion.
        """
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
                    if load + self.demands.get(node,0) > self.capacity:
                        continue
                    
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
    """
    Run custom ALNS solver for CVRP.

    Args:
        dist_matrix (np.ndarray): Distance matrix (N x N) with depot at index 0
        demands (dict): Node demands {node_id: demand_value}
        capacity (float): Vehicle capacity constraint
        R (float): Revenue per unit demand
        C (float): Cost per unit distance
        values (dict): Configuration dictionary containing:
            - time_limit (int): Maximum runtime in seconds (default: 10)
            - Iterations (int): Maximum iterations (default: 2000)

    Returns:
        Tuple[List[List[int]], float]: Routes and total travel cost
    """
    params = ALNSParams(
        time_limit=values.get('time_limit', 10),
        max_iterations=values.get('Iterations', 2000)
    )
    solver = ALNSSolver(dist_matrix, demands, capacity, R, C, params)
    routes, cost = solver.solve()
    return routes, cost


# --- ALNS Package Implementation ---
class ALNSState:
    """
    State representation for the `alns` Python package.

    Encapsulates the current solution state including routes, unassigned nodes,
    and profit calculation for Prize-Collecting VRP.

    Attributes:
        routes (List[List[int]]): Current routing solution
        unassigned (List[int]): Nodes not yet assigned to routes
        dist_matrix: Distance matrix reference
        demands (dict): Node demand values
        capacity (float): Vehicle capacity
        R (float): Revenue coefficient
        C (float): Cost coefficient
        values (dict): Additional problem parameters
    """
    def __init__(self, routes: List[List[int]], unassigned: List[int], 
                 dist_matrix, demands, capacity, R, C, values):
        """
        Initialize the ALNS state.

        Args:
            routes (List[List[int]]): Current routes.
            unassigned (List[int]): Unassigned nodes.
            dist_matrix: Distance matrix.
            demands: Node demands.
            capacity: Vehicle capacity.
            R: Revenue coefficient.
            C: Cost coefficient.
            values: configuration values.
        """
        self.routes = routes
        self.unassigned = unassigned
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.R = R
        self.C = C
        self.values = values
        self._score = self.calculate_profit()

    def copy(self):
        """Returns a deep copy of the state."""
        return ALNSState(
            copy.deepcopy(self.routes), 
            copy.deepcopy(self.unassigned),
            self.dist_matrix, 
            self.demands, 
            self.capacity, 
            self.R, 
            self.C, 
            self.values
        )

    def objective(self):
        """
        Calculates the objective value for ALNS (minimization).

        Returns:
            float: Negative profit (since ALNS minimizes).
        """
        # ALNS package minimizes objective.
        # We want to maximize profit. So minimize negative profit.
        return -self._score

    def calculate_profit(self):
        """
        Calculates the total profit of the current solution.

        Returns:
            float: Total profit (Revenue - Cost).
        """
        total_profit = 0
        visited = set()
        
        for route in self.routes:
            if not route:
                continue
            
            dist = 0
            load = 0
            if len(route) > 0:
                dist += self.dist_matrix[0][route[0]]
                for i in range(len(route) - 1):
                    dist += self.dist_matrix[route[i]][route[i+1]]
                    load += self.demands[route[i]]
                dist += self.dist_matrix[route[-1]][0]
                load += self.demands[route[-1]]
            
            cost = dist * self.C
            
            revenue = 0
            for node in route:
                revenue += self.demands[node] * self.R
                visited.add(node)
            
            total_profit += revenue - cost
            
        return total_profit
        
    @property
    def cost(self) -> float:
        """
        The cost of the current state (identical to the negative profit).

        Returns:
            float: Current objective value.
        """
        return self.objective()

# Operators for ALNS package
def alns_pkg_random_removal(state: ALNSState, random_state: np.random.RandomState) -> ALNSState:
    """
    Random removal operator for the `alns` package.

    Args:
        state (ALNSState): Current state.
        random_state (np.random.RandomState): Random number generator.

    Returns:
        ALNSState: New state with nodes removed.
    """
    new_state = state.copy()
    all_nodes = [n for r in new_state.routes for n in r]
    if not all_nodes:
        return new_state
    n_remove = min(len(all_nodes), random_state.randint(1, min(len(all_nodes)+1, 10)))
    removed = random_state.choice(all_nodes, n_remove, replace=False)
    
    new_routes = []
    for r in new_state.routes:
        new_r = [n for n in r if n not in removed]
        if new_r:
            new_routes.append(new_r)
    
    new_state.routes = new_routes
    new_state.unassigned.extend(removed)
    new_state._score = new_state.calculate_profit()
    return new_state

def alns_pkg_worst_removal(state: ALNSState, random_state: np.random.RandomState) -> ALNSState:
    """
    Worst removal operator for the `alns` package.

    Args:
        state (ALNSState): Current state.
        random_state (np.random.RandomState): Random number generator.

    Returns:
        ALNSState: New state with nodes removed.
    """
    new_state = state.copy()
    all_nodes = [n for r in new_state.routes for n in r]
    if not all_nodes:
        return new_state
    
    costs = []
    for r_idx, route in enumerate(new_state.routes):
        for idx, node in enumerate(route):
            prev_node = 0 if idx == 0 else route[idx-1]
            next_node = 0 if idx == len(route)-1 else route[idx+1]
            cost = state.dist_matrix[prev_node][node] + state.dist_matrix[node][next_node] - state.dist_matrix[prev_node][next_node]
            costs.append((cost, node))
            
    costs.sort(key=lambda x: x[0], reverse=True)
    n_remove = min(len(all_nodes), random_state.randint(1, min(len(all_nodes)+1, 10)))
    removed = [x[1] for x in costs[:n_remove]]
    
    new_routes = []
    for r in new_state.routes:
        new_r = [n for n in r if n not in removed]
        if new_r:
            new_routes.append(new_r)
            
    new_state.routes = new_routes
    new_state.unassigned.extend(removed)
    new_state._score = new_state.calculate_profit()
    return new_state

def alns_pkg_greedy_insertion(state: ALNSState, random_state: np.random.RandomState) -> ALNSState:
    """
    Greedy insertion operator for the `alns` package.

    Args:
        state (ALNSState): Current state.
        random_state (np.random.RandomState): Random number generator.

    Returns:
        ALNSState: New state with nodes inserted.
    """
    new_state = state.copy()
    random_state.shuffle(new_state.unassigned)
    while new_state.unassigned:
        node = new_state.unassigned.pop(0)
        best_cost = float('inf')
        best_pos = None
        candidate_routes = new_state.routes + [[]]
        for r_idx, route in enumerate(candidate_routes):
            current_load = sum(state.demands[n] for n in route)
            if current_load + state.demands[node] > state.capacity:
                continue
            for i in range(len(route) + 1):
                prev_node = 0 if i == 0 else route[i-1]
                next_node = 0 if i == len(route) else route[i]
                delta_dist = state.dist_matrix[prev_node][node] + state.dist_matrix[node][next_node] - state.dist_matrix[prev_node][next_node]
                delta_leverage = (delta_dist * state.C) - (state.demands[node] * state.R)
                if delta_leverage < best_cost:
                    best_cost = delta_leverage
                    best_pos = (r_idx, i)
        if best_pos:
            r_idx, idx = best_pos
            if r_idx >= len(new_state.routes):
                new_state.routes.append([node])
            else:
                new_state.routes[r_idx].insert(idx, node)
    new_state._score = new_state.calculate_profit()
    return new_state

def run_alns_package(dist_matrix, demands, capacity, R, C, values):
    """
    Run ALNS using the `alns` Python package with Prize-Collecting VRP.

    This implementation uses the `alns` library's framework with custom
    destroy/repair operators and Simulated Annealing acceptance criterion.

    Args:
        dist_matrix (np.ndarray): Distance matrix (N x N) with depot at index 0
        demands (dict): Node demands {node_id: demand_value}
        capacity (float): Vehicle capacity constraint
        R (float): Revenue per unit demand
        C (float): Cost per unit distance
        values (dict): Configuration with 'time_limit' (default: 10 seconds)

    Returns:
        Tuple[List[List[int]], float]: Routes and total distance cost
    """
    n_nodes = len(dist_matrix) - 1
    nodes = list(range(1, n_nodes + 1))
    
    initial_routes = []
    for i in nodes:
        initial_routes.append([i])
        
    init_state = ALNSState(initial_routes, [], dist_matrix, demands, capacity, R, C, values)
    
    alns = ALNS(np.random.RandomState(42))
    
    alns.add_destroy_operator(alns_pkg_random_removal)
    alns.add_destroy_operator(alns_pkg_worst_removal)
    alns.add_repair_operator(alns_pkg_greedy_insertion)
    
    time_limit = values.get('time_limit', 10)
    select = RouletteWheel([25, 10, 1, 0], 0.8, 1, 1)
    accept = SimulatedAnnealing(start_temperature=1000, end_temperature=1, step=1 - 1e-3)
    stop = MaxRuntime(time_limit)
    
    result = alns.iterate(init_state, select, accept, stop)
    best_state = result.best_state
    
    total_dist_cost = 0
    for route in best_state.routes:
        if not route:
            continue
        d = dist_matrix[0][route[0]]
        for i in range(len(route)-1):
            d += dist_matrix[route[i]][route[i+1]]
        d += dist_matrix[route[-1]][0]
        total_dist_cost += d * C
        
    return best_state.routes, total_dist_cost


# --- OR-Tools ALNS Implementation ---
def run_alns_ortools(dist_matrix, demands, capacity, R, C, values):
    """
    Run ALNS using Google OR-Tools with Guided Local Search.

    Uses OR-Tools' constraint programming solver with the Guided Local Search
    metaheuristic, which is a variant of ALNS. Solves Capacitated VRP with
    active nodes only (nodes present in demands dictionary).

    Args:
        dist_matrix (np.ndarray): Full distance matrix (N x N) with depot at index 0
        demands (dict): Node demands {node_id: demand_value}. Only these nodes are active.
        capacity (float): Vehicle capacity constraint
        R (float): Revenue per unit demand (unused in this variant)
        C (float): Cost per unit distance
        values (dict): Configuration with 'time_limit' (default: 10 seconds)

    Returns:
        Tuple[List[List[int]], float]: Routes (in global node IDs) and total cost
    """
    # 1. Identify active nodes (Depot + keys in demands)
    active_nodes = [0] + sorted(list(demands.keys()))
    compact_to_global = {c: g for c, g in enumerate(active_nodes)}
    
    n_active = len(active_nodes)
    max_vehicles = n_active
    
    # 2. Create Distance Matrix for Compact Indices
    sub_matrix = np.zeros((n_active, n_active), dtype=int)
    for c1 in range(n_active):
        for c2 in range(n_active):
            g1 = compact_to_global[c1]
            g2 = compact_to_global[c2]
            sub_matrix[c1][c2] = int(dist_matrix[g1][g2])
            
    # 3. Create Demands Array
    sub_demands = [0] * n_active
    for c in range(1, n_active):
        g = compact_to_global[c]
        sub_demands[c] = int(demands[g])
        
    scale_factor = 1000
    scaled_capacity = int(capacity * scale_factor)
    scaled_demands = [int(d * scale_factor) for d in sub_demands]
    
    # 4. OR-Tools Setup
    manager = pywrapcp.RoutingIndexManager(n_active, max_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns distance between two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return sub_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        """Returns demand of a node."""
        from_node = manager.IndexToNode(from_index)
        return scaled_demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [scaled_capacity] * max_vehicles, 
        True,  # start cumul to zero
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    
    time_limit_sec = values.get('time_limit', 10)
    search_parameters.time_limit.FromSeconds(int(time_limit_sec))

    solution = routing.SolveWithParameters(search_parameters)

    routes = []
    total_cost = 0
    if solution:
        for vehicle_id in range(max_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue
                
            # Skip Start Node (Depot)
            index = solution.Value(routing.NextVar(index))
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                global_idx = compact_to_global[node_index]
                route.append(global_idx)
                index = solution.Value(routing.NextVar(index))
            
            if route:
                routes.append(route)
        
        # Calculate real cost with C
        total_cost = solution.ObjectiveValue() * C
        
    return routes, total_cost
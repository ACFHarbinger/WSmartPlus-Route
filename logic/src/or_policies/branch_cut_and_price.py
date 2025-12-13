
import gurobipy as gp
from gurobipy import GRB
import time
import math
from typing import List, Tuple

class BCPParams:
    def __init__(self, time_limit=30, max_iterations=50):
        self.time_limit = time_limit
        self.max_iterations = max_iterations

class BCPSolver:
    def __init__(self, dist_matrix, demands, capacity, R, C, params: BCPParams):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        
        # Columns (Routes)
        self.columns = [] 
        # Each column: (cost, [node_indices], capacity_used)
        
        # Optimization Model
        self.model = None
        self.constrs = {} # node -> constraint

    def solve(self):
        start_time = time.time()
        
        # 1. Initialize with Dummy/Heuristic Columns
        self.initialize_columns()
        
        # 2. Column Generation Loop
        self.model = gp.Model("BCP_Master")
        self.model.setParam('OutputFlag', 0)
        
        # Variables: y[r] = 1 if route r is used
        # Relaxed to continuous [0, 1] for CG
        self.vars = []
        for i, col in enumerate(self.columns):
            v = self.model.addVar(obj=col[0], vtype=GRB.CONTINUOUS, name=f"r_{i}")
            self.vars.append(v)
            
        # Constraints: Each customer visited exactly once (Set Partitioning)
        # Or at least once (Set Covering) - Covering is often numerically more stable 
        # and valid if triangular inequality holds (detours don't help).
        for i in self.nodes:
            # sum(a_ir * y_r) >= 1
            expr = gp.LinExpr()
            for r_idx, col in enumerate(self.columns):
                if i in col[1]:
                    expr.add(self.vars[r_idx], 1.0)
            self.constrs[i] = self.model.addConstr(expr >= 1, name=f"c_{i}")
            
        iter_count = 0
        while iter_count < self.params.max_iterations and (time.time() - start_time) < self.params.time_limit:
            iter_count += 1
            
            # Solve RMP (Restricted Master Problem)
            self.model.optimize()
            
            if self.model.status != GRB.OPTIMAL:
                print(f"BCP: Master problem status {self.model.status}")
                break
                
            # Get Duals
            duals = {i: self.constrs[i].Pi for i in self.nodes}
            
            # Solve Pricing Problem (ESPPRC)
            # Find route with Min Reduced Cost
            # Reduced Cost = Real Cost - sum(Duals of visited nodes)
            # Cost = Dist * C - sum(d_i) (since we minimize cost, and Duals are margin gain)
            # Actually, standard CG minimizes Cost. The constraint is >= 1.
            # Duals will be non-negative (>=0). Reduced Cost = C_path - sum(duals).
            # We want Reduced Cost < 0.
            
            new_routes = self.solve_pricing(duals)
            
            if not new_routes:
                break # Optimal relaxation found
                
            # Add columns
            added = False
            for route_nodes, route_cost in new_routes:
                exists = False
                # Simple check if already exists (optional, Gurobi handles dups but slower)
                # Ideally hash checks. Skipped for brevity.
                
                col = (route_cost, route_nodes, 0) # Capacity unused here
                self.columns.append(col)
                
                # Add var to model
                new_var = self.model.addVar(obj=route_cost, vtype=GRB.CONTINUOUS)
                self.vars.append(new_var)
                
                # Update constrs
                for node in route_nodes:
                    self.model.chgCoeff(self.constrs[node], new_var, 1.0)
                added = True
            
            if not added:
                break
                
        # 3. Final MIP Solve (Price and Branch heuristic)
        # Convert all variables to Binary and solve
        for v in self.vars:
            v.vtype = GRB.BINARY
            
        self.model.setParam('TimeLimit', max(5, self.params.time_limit - (time.time() - start_time)))
        self.model.optimize()
        
        final_routes = []
        final_cost = 0
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and self.model.SolCount > 0:
            for i, v in enumerate(self.vars):
                if v.X > 0.5:
                    final_routes.append(self.columns[i][1])
                    # Recalculate cost independently to be safe
                    # But columns[i][0] is the cost
                    # Note: Revenue calculation matches Evaluate function logic?
                    # The Evaluate function calculates Profit = Revenue - Cost.
                    # Here we minimize Cost.
                    
            # Compute optimization metric: Total Distance * C
            # The calling function expects "routes, total_profit" usually?
            # Or just routes.
            # LookAhead expects: routes, fitness/cost?
            # HGS returns routes, profit. Let's return consistent.
            pass
        
        total_dist_cost = self.calculate_total_cost(final_routes)
        return final_routes, total_dist_cost

    def initialize_columns(self):
        # Generate dummy routes (1 per node)
        for i in self.nodes:
            dist = (self.dist_matrix[0][i] + self.dist_matrix[i][0])
            cost = dist * self.C
            self.columns.append((cost, [i], self.demands.get(i,0)))

    def solve_pricing(self, duals):
        # Label Setting Algorithm for ESPPRC
        # Since 100 nodes is large for exact ESPPRC, we use a Heuristic Labeling
        # Or relax elementarity (SPPRC - allows cycles) -> easier
        # Or restricting search space (nearest neighbors)
        
        # Graph: 0 -> Nodes -> 0
        # Reduced Cost of edge (u, v) = Cost(u,v) - Dual(v)
        # Note: Dual(v) is associated with visiting node v.
        # So we can subtract Dual(v) upon arriving at v.
        
        # We need to find paths with Neg Reduced Cost.
        
        # Limitation: Full labeling is slow in Python.
        # We will implement a simplified "q-routes" or limited depth search.
        # Or just a Greedy randomized pricing for speed in this demo context.
        # Let's try heuristic construction.
        
        paths = []
        
        # Try finding improving columns by growing from each node
        # Using "nearest neighbor" lookups
        
        # Random multi-start greedy with duals
        # Sort edges by reduced cost?
        
        # For simplicity and speed in Python:
        # Construct paths greedily minimizing reduced cost
        
        for _ in range(50): # Try 50 heuristic paths
            curr = 0 # Depot
            load = 0
            path = []
            red_cost = 0.0
            real_cost = 0.0
            
            visited = set()
            
            while True:
                # Find best next node
                best_node = -1
                best_rc = float('inf')
                
                # Check candidates (all unvisited)
                # Optimization: only check K nearest neighbors
                
                candidates = [n for n in self.nodes if n not in visited]
                # To make it fast, maybe random sample if too large
                if len(candidates) > 20: 
                    candidates = list(range(1, self.n_nodes + 1))
                    # Wait, 'visited' check above is needed.
                    # Let's shuffle and pick subset
                    import random
                    random.shuffle(candidates)
                    candidates = candidates[:20]
                
                # Heuristic sort: likely good nodes have high duals
                candidates.sort(key=lambda x: duals.get(x,0), reverse=True)
                candidates = candidates[:10] # Top 10 high duals
                
                found = False
                for nxt in candidates:
                    if nxt in visited: continue
                    d_new = self.demands.get(nxt, 0)
                    if load + d_new > self.capacity: continue
                    
                    # Edge cost
                    dist = self.dist_matrix[curr][nxt]
                    rc_node = (dist * self.C) - duals.get(nxt, 0)
                    
                    if rc_node < best_rc:
                        best_rc = rc_node
                        best_node = nxt
                        found = True
                
                if found and best_node != -1:
                    # Look ahead: is returning to depot allowed?
                    # Reduced cost must be negative at end of tour (return to 0)
                    # Currently we are just greedily adding. 
                    # If we stop now:
                    closing_dist = self.dist_matrix[best_node][0]
                    closing_rc = closing_dist * self.C
                    
                    current_path_rc = red_cost + best_rc + closing_rc
                    
                    # Accept move
                    curr = best_node
                    path.append(curr)
                    visited.add(curr)
                    load += self.demands.get(curr, 0)
                    dist_step = self.dist_matrix[path[-2] if len(path)>1 else 0][curr]
                    
                    red_cost += (dist_step * self.C) - duals.get(curr, 0)
                    real_cost += dist_step * self.C
                    
                    # If path has neg reduced cost, save it!
                    if red_cost + (self.dist_matrix[curr][0] * self.C) < -1e-4:
                        final_real = real_cost + (self.dist_matrix[curr][0] * self.C)
                        paths.append((list(path), final_real))
                        # Don't break, keep extending to see if better?
                        # Usually BCP adds elementary paths.
                else:
                    break
        
        # Also, implement a simple SPPRC using `networkx` logic or simple DP if possible?
        # Python is too slow for 100 nodes exact label setting.
        # The heuristic above finds columns with negative reduced cost.
        return paths

    def calculate_total_cost(self, routes):
        total = 0
        for r in routes:
            d = self.dist_matrix[0][r[0]]
            for i in range(len(r)-1):
                d += self.dist_matrix[r[i]][r[i+1]]
            d += self.dist_matrix[r[-1]][0]
            total += d * self.C
        return total

def run_bcp(dist_matrix, demands, capacity, R, C, values):
    params = BCPParams(
        time_limit=values.get('time_limit', 30),
        max_iterations=values.get('Iterations', 50)
    )
    solver = BCPSolver(dist_matrix, demands, capacity, R, C, params)
    routes, cost = solver.solve()
    return routes, cost

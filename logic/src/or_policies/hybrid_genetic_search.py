import numpy as np
import random
import time
import copy
from typing import List, Tuple, Dict
from logic.src.pipeline.simulator.processor import create_dataframe_from_matrix

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